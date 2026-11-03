# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Causal confidence heads and prefix budget selection."""

import math
import heapq

import torch

from .linear import UnquantizedProjection


def causal_prefix(confidences, threshold):
    """Admit position i using only confidence through i and stop at the first failure.

    Confidence i must depend only on history and proposals preceding i. Looking
    ahead to a later confidence when deciding whether to admit i biases sampling.
    """
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("confidence threshold must be finite and in (0, 1]")
    product, count = 1., 0
    for confidence in confidences:
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        product *= confidence
        if product < threshold:
            break
        count += 1
    return count


def lagged_budget(confidences, costs):
    """Choose a width from a completed previous round's confidence snapshot.

    Costs cover widths one through len(confidences)+1 and include the entire
    proposal, verification and feedback step. Current sampled drafts must not
    influence this decision.
    """
    if len(costs) != len(confidences) + 1 or any(not math.isfinite(c) or c <= 0 for c in costs):
        raise ValueError("positive complete-step costs must cover every verification width")
    expected, survival = 1., 1.
    utilities = [expected / costs[0]]
    for confidence, cost in zip(confidences, costs[1:], strict=True):
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        survival *= confidence
        expected += survival
        utilities.append(expected / cost)
    return max(range(len(utilities)), key=utilities.__getitem__) + 1


def allocate_draft_budget(confidences, budget):
    """Allocate a fixed draft count across causal request prefixes.

    The caller chooses the budget independently of these sampled proposals.
    A request's next confidence enters the frontier only after its preceding
    position has been admitted. Ties prefer the earlier position, then request.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError("draft budget must be a nonnegative integer")
    rows = [tuple(row) for row in confidences]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for row in rows for p in row):
        raise ValueError("confidence must be finite and in [0, 1]")
    frontier = [(-row[0], 0, slot) for slot, row in enumerate(rows) if row]
    heapq.heapify(frontier)
    counts = [0] * len(rows)
    for _ in range(min(budget, sum(map(len, rows)))):
        probability, position, slot = heapq.heappop(frontier)
        counts[slot] += 1
        if position + 1 < len(rows[slot]):
            heapq.heappush(frontier, (probability * rows[slot][position + 1], position + 1, slot))
    return tuple(counts)


class PreparedDraftConfidence:
    """Preserve checkpoint storage precision while accumulating confidence in FP32."""

    def __init__(self, source, weight, output, *, name, bias=None, buffers=None):
        o, b, w = source.owner, source.batch, source.width
        hidden = source.draft.head_hidden
        markov = source.markov.embedding
        if weight.shape != (1, hidden.shape[1] + markov.shape[1]):
            raise ValueError("confidence weight must cover hidden and preceding-token Markov features")
        self.hidden = hidden.view(b, o.block_size, -1)[:, o.logit_offset:o.logit_offset + w - 1]
        def allocate(shape, dtype):
            return (torch.empty(shape, dtype=dtype, device=markov.device) if buffers is None
                    else buffers.zeros(shape, dtype))
        self.embeddings = allocate((b, w - 1, markov.shape[1]), markov.dtype)
        self.source = source
        self.canonical_embedding = None
        if (getattr(source, "proposal_block", None) is not None
                or getattr(source, "full_proposal", None) is not None):
            from .embedding import PreparedEmbedding

            self.preceding = allocate((b, w - 1), torch.int64)
            self.canonical_embedding = PreparedEmbedding(
                source.markov.embed.weight, self.preceding.view(-1),
                self.embeddings.view(-1, markov.shape[1]), session=o.session,
                name=name + ".canonical_predecessors")
        self.features = allocate((b, w - 1, weight.shape[1]), weight.dtype)
        self.logits = allocate((b, w - 1), torch.float32)
        self.output = output
        self.projection = UnquantizedProjection(self.features.view(b * (w - 1), -1), weight,
            self.logits.view(-1, 1), session=o.session, name=name, bias=bias)

    def record_embedding(self, position, embedding):
        if self.canonical_embedding is None:
            self.embeddings[:, position].copy_(embedding)

    def __call__(self):
        if self.canonical_embedding is not None:
            self.preceding[:, 0].copy_(torch.where(self.source.active, self.source.candidates[:, 0], 0))
            self.preceding[:, 1:].copy_(self.source.drafts[:, :-1])
            self.canonical_embedding()
        self.features[:, :, :self.hidden.shape[-1]].copy_(self.hidden)
        self.features[:, :, self.hidden.shape[-1]:].copy_(self.embeddings)
        self.projection(self.features.view(-1, self.features.shape[-1]))
        self.output.copy_(self.logits.sigmoid())
