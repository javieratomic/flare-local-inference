# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared FP8 sparse selection with shared scratch and caller-owned metadata."""

from functools import partial

from b12x.attention import dsa_indexer
from b12x.preparation import PreparedCall

from ignition.preparation.fixtures import indexer as fixtures
from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer


class PreparedSparseIndexer:
    """Select sparse cache records using the recipe's fixed geometry and index space.

    Prefill plans require one shared page table; decode plans accept a table per
    query row. Device lengths determine each row's visible candidate prefix.
    Inputs and output remain in caller-owned storage. Tuning uses private finite
    operands, then finalization binds the selected plan to the shared arena.
    """

    def __init__(self, query, weights, cache, page_table, lengths, active_width, output, *,
                 caps, session, name, workspace, scores=None):
        self.inputs = {"q_fp8": query, "query_weights": weights, "index_k_cache": cache,
            "page_table": page_table, "cache_lengths": lengths, "active_width": active_width,
            "output_indices": output, "output_scores": scores}
        self.output, self.scores = output, scores
        self.plan = dsa_indexer.plan(caps,
            invocation=dsa_indexer.invocation_from_tensors(caps, **self.inputs))

        def prepare(state):
            storage = planned_scratch(state.layout.scratch_specs(), workspace)
            runtime = fixtures.bind(state, self.inputs, storage)
            return PreparedCall(run=lambda: fixtures.run(state, runtime, self.inputs),
                                owners=prime_owners(workspace, storage, runtime))

        declare(session, (self.plan.request(name=name, prepare_call=prepare,
            benchmark_call=partial(fixtures.benchmark_call, arguments=self.inputs,
                                   session=session, workspace=workspace)),))

        def finalize():
            self.binding = dsa_indexer.bind(self.plan,
                scratch=planned_scratch(self.plan.scratch_specs(), workspace), **self.inputs)

        self._finalize = finalize
        register_finalizer(session, finalize, workspace=workspace)

    def rebind_cache(self, cache):
        """Bind a replacement index cache before capture; its page count is a runtime bound."""
        self.inputs["index_k_cache"] = cache
        self._finalize()

    def __call__(self):
        return dsa_indexer.run(self.binding)
