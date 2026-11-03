# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Disposable GLM forward storage excluded from persistent tensor payloads."""

from ignition.preparation.disposable import storage_ranges
from ignition.components.prompt_logprobs import PreparedScoredPrefill
from .generation import PreparedGeneration
from .resume import PreparedResume
from .speculative import PreparedPrefill, PreparedRound


def disposable_ranges(scratch, arena, forwards):
    values = [scratch, arena.tensor]
    for forward in forwards.values():
        if isinstance(forward, PreparedScoredPrefill):
            forward = forward.forward
        if isinstance(forward, PreparedPrefill):
            values.extend((forward.feedback, forward.previous, forward.before))
            values.extend(_draft_scratch(forward.absorb))
            forward = forward.generation
        if isinstance(forward, (PreparedGeneration, PreparedResume)):
            sampling = forward.completion
            if hasattr(sampling, "scratch_tensors"):
                values.append(sampling.scratch_tensors)
        if isinstance(forward, PreparedRound):
            values.append(forward.owner.sampling_buffers.chunks)
            values.extend(getattr(forward, name) for name in (
                "draft_logits", "before", "pending", "previous", "active", "candidates", "drafts",
                "probabilities", "valid", "seed", "offset", "base_offset", "sampled", "tail_backup"))
            values.extend(_draft_scratch(forward.absorb))
            if forward.draft is not None:
                values.extend(_draft_scratch(forward.draft))
            target, proposals, verify, _ = forward.operations
            for distribution in (target, *proposals):
                values.extend(getattr(distribution, name) for name in (
                    "scores", "probabilities", "maximum", "valid",
                    "packed_top_k", "packed_top_p"))
            values.extend(getattr(verify, name) for name in (
                "safe_draft_probs", "safe_target_probs", "safe_ids", "seed_bits", "offset",
                "independent_accepted", "emitted_drafts", "output"))
    return storage_ranges(values)


def _draft_scratch(draft):
    values = [getattr(draft, name) for name in (
        "embedded", "joined", "residual", "normalized", "attention_output", "mlp_output", "hidden")]
    values.append(draft.attention.scratch)
    if not draft.cache_only:
        values.extend((draft.moe.scratch, draft.local_logits, draft.logits))
    return values
