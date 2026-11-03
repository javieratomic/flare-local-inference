# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared paged latent-cache writes and sparse attention with shared scratch."""

import torch
from b12x.attention import sparse_mla
from b12x.preparation import PreparedCall

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer


class PreparedLatentCacheWriter:
    """Write GLM_NEXT latent records into fixed paged storage.

    Source, slots, and cache remain live graph inputs at their captured addresses.
    Preparation invokes the writer; the bootstrap must restore cache contents
    before admitting requests. The b12x writer has a fixed configuration.
    """

    def __init__(self, source, cache, slots, *, session, name):
        self.source, self.cache, self.slots = source, cache, slots
        self.plan = sparse_mla.plan_cache_writer(source, cache, slots)
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(source, cache, slots), owners=(source, cache, slots))),))

    def __call__(self):
        sparse_mla.concat_and_cache_glm_next_mla(self.source, self.cache, self.slots, plan=self.plan)


class PreparedSparseMLA:
    """Attend to caller-selected physical records and copy out before arena reuse.

    Recipes supply b12x geometry, paged storage, and device selection metadata.
    Plans use b12x's fixed native configuration and participate in session
    preparation and scratch finalization. Output storage must outlive this call;
    scratch can be overwritten by the next serialized operator.
    """

    def __init__(self, query, cache, selected, cache_lengths, selected_lengths, output, *,
                 caps, session, name, workspace, attention_sink=None, lse=None):
        if (output.shape != (*query.shape[:2], caps.v_head_dim)
                or output.dtype != torch.bfloat16 or output.device != query.device):
            raise ValueError("sparse MLA output requires BF16 [rows,heads,value_dim] on the query device")
        if bool(lse is not None) != caps.return_lse:
            raise ValueError("sparse MLA LSE output must match the plan's return_lse setting")
        if lse is not None and (lse.shape != query.shape[:2] or lse.dtype != torch.float32
                                or lse.device != query.device):
            raise ValueError("sparse MLA LSE output requires FP32 [rows,heads] on the query device")
        self.output, self.lse = output, lse
        self.plan = sparse_mla.plan(caps)
        self.inputs = {"q": query, "kv_cache": cache, "selected_indices": selected,
            "cache_lengths": cache_lengths, "selected_lengths": selected_lengths, "attention_sink": attention_sink}

        def prepare(state):
            storage = planned_scratch(state.scratch_specs(), workspace)
            runtime = state.bind(scratch=storage, q=query, kv_cache=cache, selected_indices=selected,
                cache_seqlens_int32=cache_lengths, nsa_cache_seqlens_int32=selected_lengths)
            return PreparedCall(run=lambda: state.run(runtime, kv_cache=cache, attention_sink=attention_sink),
                                owners=prime_owners(workspace, storage, runtime))

        declare(session, (self.plan.request(name=name, prepare_call=prepare),))

        def finalize():
            self.binding = sparse_mla.bind(self.plan,
                scratch=planned_scratch(self.plan.scratch_specs(), workspace), **self.inputs)

        self._finalize = finalize
        register_finalizer(session, finalize, workspace=workspace)

    def rebind_cache(self, cache):
        """Bind replacement paged storage before capture; the plan's max_kv_rows bounds its size."""
        self.inputs["kv_cache"] = cache
        self._finalize()

    def __call__(self):
        result = sparse_mla.run(self.binding)
        if isinstance(result, tuple):
            self.output.copy_(result[0])
            if self.lse is not None:
                self.lse.copy_(result[1])
        else:
            self.output.copy_(result)
        return self.output

    def attend_in_scratch(self):
        """Attend without copying out: the output (and LSE) alias arena scratch at fixed addresses.

        The caller must consume them before the next operator that uses the arena.
        """
        result = sparse_mla.run(self.binding)
        if isinstance(result, tuple):
            return result if self.lse is not None else (result[0], None)
        return result, None
