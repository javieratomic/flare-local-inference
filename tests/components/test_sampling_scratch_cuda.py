# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Aliased sampling operands preserve results across captured shapes and draft positions."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.parallel_draft import prepare_sampling_scratch
from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.components.speculation import (
    PreparedSpeculativeDistributions, PreparedStochasticVerification, SpeculativeRngState,
)
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import GraphBufferPool

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("vocab,batch_limit", [(10000, 4), (129280, 4), (129280, 32)])
def test_native_shared_sampling_survives_shape_switches_and_sequential_proposals(compiled, vocab, batch_limit):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    width_limit = 8 if vocab == 129280 else 6
    state = DeviceRequestState.allocate("cuda:0", vocab_size=vocab, capacity=batch_limit)
    for slot in range(batch_limit):
        state.admit(slot, RequestAdmission(generation=7, seed=93 + slot,
            max_new_tokens=100, temperature=1. if vocab == 129280 else .8,
            top_k=0 if vocab == 129280 else 20, top_p=.95 if vocab == 129280 else .9,
            repetition_penalty=1.1, presence_penalty=.2, eos_token_ids=()), [1, 2])
    state.status[1] = 0
    slots = torch.tensor([3, 0, 2, 1, *range(4, batch_limit)], dtype=torch.int32, device="cuda:0")
    epochs = torch.tensor([7, 7, 99, 7, *([7] * (batch_limit - 4))], dtype=torch.int64, device="cuda:0")
    live = torch.tensor([batch_limit], dtype=torch.int32, device="cuda:0")
    pool = GraphBufferPool("cuda:0", chunk_bytes=2 << 20)
    owners, graphs = [], {}
    generator = torch.Generator().manual_seed(17)
    for batch, width in ((batch_limit, width_limit), (1, 1), (batch_limit // 2, 3), (1, width_limit)):
        logits = torch.randn((batch, width, vocab), generator=generator).cuda()
        proposal_logits = torch.randn((batch, 1, vocab), generator=generator).cuda()
        for shared in (False, True):
            buffers = pool.cursor() if shared else None
            workspaces = prepare_sampling_scratch(batch, width, vocab, "cuda:0", buffers=buffers) if shared else {}
            ids = torch.zeros((batch, width - 1), dtype=torch.int64, device="cuda:0")
            probabilities = (buffers.zeros((batch, width - 1, vocab), torch.float32)
                             if shared else torch.empty((batch, width - 1, vocab), device="cuda:0"))
            valid = torch.zeros((batch, width - 1), dtype=torch.bool, device="cuda:0")
            seed = torch.arange(batch, dtype=torch.int64, device="cuda:0") + 51
            offset, sampled = torch.zeros_like(seed), torch.zeros_like(seed)
            proposals = [PreparedDraftDistribution(proposal_logits, state, slots[:batch], epochs[:batch], live, ids,
                prefix_offset=index, seed=seed, offset=offset, sampled=sampled,
                workspace=workspaces["proposal"] if shared else None) for index in range(width - 1)]
            for proposal in proposals:
                proposal.prepare()
            target = PreparedSpeculativeDistributions(logits, state, slots[:batch], epochs[:batch], live, ids,
                workspace=workspaces["target"] if shared else None)
            target.prepare()
            verify = PreparedStochasticVerification(probabilities, ids, target.probabilities, state,
                slots[:batch], epochs[:batch], live, rng=SpeculativeRngState.allocate("cuda:0", capacity=batch_limit),
                draft_valid=valid, target_valid=target.valid,
                workspace=workspaces["verify"] if shared else None)
            verify.prepare()
            owner = SimpleNamespace(proposals=proposals, target=target, verify=verify,
                ids=ids, probabilities=probabilities, valid=valid, offset=offset, sampled=sampled)
            def run(owner=owner):
                owner.ids.zero_()
                for index, proposal in enumerate(owner.proposals):
                    owner.offset.fill_(index * 4)
                    proposal()
                    owner.ids[:, index].copy_(owner.sampled)
                    owner.probabilities[:, index].copy_(proposal.probabilities[:, 0])
                    owner.valid[:, index].copy_(proposal.valid[:, 0])
                owner.target()
                owner.verify()
            call = run
            if compiled:
                from ignition.preparation.compilation import compile_forward
                call = compile_forward(run)
            call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            owners.append((owner, call))
            graphs[f"{batch}x{width}-{shared}"] = graph
    bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={}, resources=(owners, state, pool))
    worker = bundle.start_worker()
    try:
        for pair in (0, 3, 1, 0, 2, 1, 2, 3, 0):
            results = []
            for shared in (False, True):
                index = pair * 2 + int(shared)
                if shared:
                    for chunks in pool.chunks.values():
                        for tensor in chunks:
                            tensor.fill_(float("nan") if tensor.is_floating_point() else 255)
                torch.cuda.synchronize()
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                ticket = worker.submit(index)
                completion = worker.poll(timeout_ms=30000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                owner, _ = owners[index]
                out = owner.verify.output
                results.append([value.cpu().clone() for value in (owner.ids, owner.probabilities,
                    owner.target.probabilities, out.token_ids, out.accepted_drafts, out.proposed_count, out.valid)])
            for independent, shared in zip(*results, strict=True):
                torch.testing.assert_close(shared, independent, rtol=0, atol=0)
    finally:
        worker.close()
        for graph in graphs.values():
            graph.reset()
