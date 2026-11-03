# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Speculative metadata, cross-width checkpoints, and MTP prefix alignment."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.qwen38.draft_state import DraftMetadata
from ignition.recipes.qwen38.state import HybridStatePool


@pytest.mark.parametrize("shared", [False, True])
def test_draft_rows_pack_unequal_lengths_and_inactive_lanes(shared):
    backing = torch.full((16, 10240), 123, dtype=torch.bfloat16) if shared else None
    metadata = DraftMetadata(tokens=8, rows=4, context=256, device="cpu",
                             hidden=None if backing is None else backing[:8])
    slots = torch.tensor([2, 0, 3, -1], dtype=torch.int32)
    lengths = torch.tensor([2, 0, 1, 0])
    starts = torch.tensor([62, 71, 128, 0])
    tokens = torch.tensor([[901, 902], [100, 101], [401, 402], [0, 0]])
    hidden = torch.arange(8).to(torch.bfloat16)[:, None].expand(8, 10240).contiguous()
    rope = (torch.arange(8)[:, None] + torch.tensor([10, 20, 30])[None, :]).reshape(4, 2, 3)
    tables = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34], [41, 42, 43, 44]], dtype=torch.int32)
    metadata.set(slots, lengths, starts, tokens, hidden, rope, tables, capacity=2)
    assert metadata.qsa.query_start_loc.tolist() == [0, 2, 2, 3, 3]
    assert metadata.qsa.request_ids.tolist() == [0, 0, 2, -1, -1, -1, -1, -1]
    assert metadata.qsa.query_positions[:3].tolist() == [62, 63, 128]
    assert metadata.qsa.slot_mapping[:3].tolist() == [31 * 64 + 62, 31 * 64 + 63, 43 * 64]
    assert metadata.token_ids[:3].tolist() == [901, 902, 401]
    assert metadata.hidden[:3, 0].tolist() == [0, 1, 4]
    assert metadata.qsa.rope_positions[:3].tolist() == [[10, 20, 30], [11, 21, 31], [14, 24, 34]]
    assert metadata.qsa.raw_state_slot_ids.tolist() == [3, -1, 4, -1]
    assert metadata.qsa.sequence_lengths.tolist() == [64, 0, 129, 0]
    assert not metadata.hidden[3:].count_nonzero()
    if shared:
        assert metadata.hidden.data_ptr() == backing.data_ptr()
        assert backing[8:].eq(123).all()


def test_empty_draft_absorption_preserves_inactive_metadata():
    metadata = DraftMetadata(tokens=1, rows=1, context=64, device="cpu")
    metadata.set(torch.tensor([0], dtype=torch.int32), torch.tensor([0]), torch.tensor([0]),
        torch.tensor([7]), torch.zeros((1, 10240), dtype=torch.bfloat16), torch.zeros((1, 3), dtype=torch.int64),
        torch.full((4, 1), -1, dtype=torch.int32), capacity=1)
    assert metadata.qsa.query_start_loc.tolist() == [0, 0]
    assert metadata.qsa.request_ids.tolist() == [-1]
    assert metadata.qsa.raw_state_slot_ids.tolist() == [-1]
    assert metadata.qsa.slot_mapping.tolist() == [-1]


CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                         reason="requires an explicitly assigned CUDA device")


@CUDA
@torch.no_grad()
def test_live_row_state_survives_width_one_and_retains_accepted_endpoint():
    from ignition.recipes.qwen38.decoder_state import DecoderInputs
    from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
    from ignition.recipes.qwen38.speculative_state import PreparedVerificationState

    device = torch.device("cuda", 0)
    pool = HybridStatePool.allocate(34, device, speculative_tokens=7)
    requests = DeviceRequestState.allocate(device, consumed_tokens=pool.consumed_tokens[1:5, 0],
        ple_history=pool.ple_history[1:5])
    inputs = DecoderInputs.allocate(device, context=256, tokens=64)
    inputs.request_state_rows[:, 0] = torch.arange(1, 5, device=inputs.token_ids.device)
    inputs.request_state_rows[:, 1:] = torch.arange(6, 34, device=inputs.token_ids.device).view(4, 7)
    candidates = torch.arange(32, device=device, dtype=torch.int64).reshape(4, 8) + 901
    accepted = torch.ones(4, dtype=torch.int32, device=device)
    states = {width: PreparedVerificationState(inputs, requests, pool, candidates, accepted,
        width=width, rows=1) for width in range(1, 9)}
    counts = torch.tensor([5], dtype=torch.int32, device=device)
    before = torch.tensor([61], dtype=torch.int64, device=device)
    residual = torch.arange(8, device=device, dtype=torch.bfloat16)[:, None].expand(8, 10240).contiguous()
    hidden = residual[:, :2560].contiguous()
    states[8].prepare_commit(counts, before, residual, hidden)
    requests.admit(2, RequestAdmission(11, 42), list(range(128)), consumed_tokens=61)
    requests.pending_token[2] = 901
    inputs.row_slot_ids[0] = 2
    inputs.row_generations[0] = 11
    inputs.live_rows.fill_(1)
    inputs.page_tables[2].copy_(torch.tensor([5, 3, 1, 7], device=device, dtype=torch.int32))
    accepted[2] = 8
    states[1].begin()
    assert states[1].gdn.accepted_tokens.tolist() == [8]
    # Verification resumes from and commits into the live row; later columns stay null.
    assert states[1].gdn.state_indices.tolist() == [[3, 0, 0, 0, 0, 0, 0, 0]]
    assert states[1].token_ids.tolist() == [901]
    for width in range(2, 8):
        states[width].begin()
        assert states[width].token_ids.tolist() == list(range(901, 901 + width))
        assert states[width].qsa.query_positions.tolist() == list(range(61, 61 + width))
        assert states[width].gdn.accepted_tokens.tolist() == [8]
    states[8].begin()
    assert states[8].qsa.slot_mapping.tolist() == [381, 382, 383, 192, 193, 194, 195, 196]
    for layer in pool.gdn.values():
        layer.convolution[3].copy_(torch.arange(100, 110, device=device, dtype=torch.bfloat16))
    pool.ple_convolution[3].copy_(torch.arange(100, 116, device=device, dtype=torch.bfloat16))
    states[8].commit()
    assert accepted[2].item() == 5
    assert pool.last_hidden[3, 0].item() == 4
    assert pool.region("target_residual")[3, 0].item() == 4
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        states[1].begin()
    before_allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before_allocations
    assert states[1].gdn.accepted_tokens.tolist() == [5]
    assert states[1].gdn.state_indices.shape == (1, 8)
    graph.reset()


@pytest.mark.parametrize("temperature", [.7, 1.])
@pytest.mark.parametrize("copy_epoch", [12, 13])
def test_recursive_draft_reorders_residuals_and_physical_copy_hints(temperature, copy_epoch):
    from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
    from ignition.recipes.qwen38.speculative_model import PreparedSpeculativeRound

    batch, width = 2, 4
    requests = DeviceRequestState.allocate("cpu", vocab_size=128)
    requests.admit(3, RequestAdmission(13, 42, temperature=temperature, eos_token_ids=()), [3] * 64, consumed_tokens=64)
    requests.pending_token[3] = 17
    # Packed lane zero is inactive; lane one maps to physical slot three.
    slots = torch.tensor([1, 3], dtype=torch.int32)
    epochs = torch.tensor([0, 13], dtype=torch.int64)
    regions = {"target_residual": torch.zeros((5, 10240), dtype=torch.bfloat16),
               "target_residual_position": torch.zeros((5, 3), dtype=torch.int64)}
    pool = SimpleNamespace(region=lambda name: regions[name], rope_delta=torch.zeros((5, 1), dtype=torch.int64))
    owner = SimpleNamespace(requests=requests, pool=pool,
        copy_generation=torch.tensor([0, 0, 0, copy_epoch]),
        inputs=SimpleNamespace(request_state_rows=torch.arange(1, 5)[:, None].expand(4, 8), row_slot_ids=slots, row_generations=epochs, live_rows=torch.tensor([2], dtype=torch.int32)),
        cache={48: SimpleNamespace()},
        draft_rng_generation=torch.zeros(4, dtype=torch.int64), draft_rng_offset=torch.zeros(4, dtype=torch.int64),
        copy_expected=torch.tensor([-1, -1, -1, 64]), copy_count=torch.tensor([0, 0, 0, 1]),
        copy_ids=torch.full((4, 7), -1, dtype=torch.int64), candidates=torch.zeros((4, 8), dtype=torch.int64),
        draft_tables=torch.ones((4, 4), dtype=torch.int32))
    owner.copy_ids[3, 0] = 27
    operation = PreparedSpeculativeRound.__new__(PreparedSpeculativeRound)
    operation.owner, operation.batch, operation.width = owner, batch, width
    operation.token_limits = None
    operation.state = SimpleNamespace(_validate=lambda: None, token_ids=torch.zeros(batch * width, dtype=torch.int64),
        qsa=SimpleNamespace(rope_positions=torch.zeros((batch * width, 3), dtype=torch.int64)))
    for name in ("before", "pending", "seed", "offset", "base_offset", "sampled", "copy_mask"):
        setattr(operation, name, torch.zeros(batch, dtype=torch.int64))
    operation.previous = torch.zeros((batch, 10240), dtype=torch.bfloat16)
    operation.previous_rope = torch.zeros((batch, 3), dtype=torch.int64)
    operation.rope_delta = torch.zeros(batch, dtype=torch.int64)
    operation.active = torch.zeros(batch, dtype=torch.bool)
    operation.drafts = torch.zeros((batch, width - 1), dtype=torch.int64)
    operation.draft_valid = torch.zeros((batch, width - 1), dtype=torch.bool)
    operation.draft_probabilities = torch.zeros((batch, width - 1, 128), dtype=torch.float32)
    operation.raw_backup = {}
    operation.rows = torch.arange(batch)
    operation.target_residual = torch.zeros((batch * width, 10240), dtype=torch.bfloat16)
    feedback = torch.zeros((batch * width, 10240), dtype=torch.bfloat16)
    operation.draft_metadata = DraftMetadata(tokens=batch, rows=batch, context=256, device="cpu", hidden=feedback[:batch])
    operation.absorb_metadata = DraftMetadata(tokens=batch * width, rows=batch, context=256, device="cpu", hidden=feedback)

    class Draft:
        scratch = SimpleNamespace(residual=torch.zeros((batch, 10240), dtype=torch.bfloat16))
        inputs = []
        tokens = []

        def __call__(self):
            self.inputs.append(operation.draft_metadata.hidden.clone())
            self.tokens.append(operation.draft_metadata.token_ids.clone())
            self.scratch.residual.zero_()
            self.scratch.residual[0].fill_(99 + len(self.inputs))

    class Proposal:
        probabilities = torch.nn.functional.one_hot(torch.full((batch, 1), 21), 128).float()
        valid = torch.tensor([[False], [True]])
        vocab_ids = torch.arange(128)

        def __call__(self):
            # Draft proposals draw into the round's shared sample buffer.
            operation.sampled.copy_(self.probabilities[:, 0].argmax(-1))

    class Commit:
        emitted_count = torch.tensor([0, 1], dtype=torch.int32)

        def __call__(self):
            pass

    operation.draft = Draft()
    operation.decoder = lambda: operation.state.token_ids.copy_(owner.candidates[:batch, :width].reshape(-1))
    operation.absorb = lambda: None
    operation.operations = (Proposal(), (Proposal(),) * 3, lambda: None, Commit(), ())
    operation()
    chosen = 27 if copy_epoch == 13 else 21
    assert operation.drafts[1].tolist() == [chosen, 21, 21]
    assert operation.copy_mask.tolist() == [0, int(copy_epoch == 13)]
    assert operation.draft_probabilities[1, 0, chosen] == 1
    assert operation.draft_probabilities[1, 0].count_nonzero() == 1
    assert operation.draft.tokens[1][0] == chosen
    assert operation.draft.inputs[1][0].eq(100).all()
    assert operation.draft.inputs[2][0].eq(101).all()
    assert owner.draft_rng_offset.tolist() == [0, 0, 0, 12]
    assert owner.draft_rng_generation.tolist() == [0, 0, 0, 13]


@CUDA
@torch.no_grad()
def test_fixed_verification_masks_context_tail_without_changing_shape():
    from ignition.recipes.qwen38.decoder_state import DecoderInputs
    from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
    from ignition.recipes.qwen38.speculative_state import PreparedVerificationState

    pool = HybridStatePool.allocate(34, "cuda", speculative_tokens=7)
    requests = DeviceRequestState.allocate("cuda", consumed_tokens=pool.consumed_tokens[1:5, 0],
        ple_history=pool.ple_history[1:5])
    inputs = DecoderInputs.allocate("cuda", context=64, tokens=64)
    inputs.request_state_rows[:, 0] = torch.arange(1, 5, device=inputs.token_ids.device)
    inputs.request_state_rows[:, 1:] = torch.arange(6, 34, device=inputs.token_ids.device).view(4, 7)
    candidates = torch.zeros((4, 8), dtype=torch.int64, device="cuda")
    accepted = torch.ones(4, dtype=torch.int32, device="cuda")
    inputs.query_lengths.copy_(torch.tensor([2, 4, 1, 3], dtype=torch.int32, device="cuda"))
    for slot, position in enumerate([62, 60, 63, 61]):
        requests.admit(slot, RequestAdmission(1, 42), [7] * position, consumed_tokens=position)
        requests.pending_token[slot] = 7
    inputs.row_slot_ids.copy_(torch.arange(4, dtype=torch.int32, device="cuda"))
    inputs.row_generations.fill_(1)
    inputs.live_rows.fill_(4)
    inputs.page_tables.fill_(0)
    state = PreparedVerificationState(inputs, requests, pool, candidates, accepted,
        width=4, rows=4, token_limits=inputs.query_lengths)
    state.begin()
    assert requests.status.tolist() == [1] * 4
    assert state.qsa.query_positions.reshape(4, 4).tolist() == [
        [62, 63, -1, -1], [60, 61, 62, 63], [63, -1, -1, -1], [61, 62, 63, -1]]
    assert state.qsa.sequence_lengths.tolist() == [64] * 4
    assert state.gdn.query_start_loc.tolist() == [0, 4, 8, 12, 16]
    assert state.qsa.slot_mapping.reshape(4, 4).tolist() == [
        [62, 63, -1, -1], [60, 61, 62, 63], [63, -1, -1, -1], [61, 62, 63, -1]]
