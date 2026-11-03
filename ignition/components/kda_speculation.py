# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording KDA verification and accepted-prefix replay with one committed state per row."""

import itertools
import weakref
from dataclasses import dataclass

import torch
from b12x.sequence import gdn_decode

from ignition.preparation.scratch import planned_scratch
from ignition.preparation.tuning import register_finalizer

@dataclass(frozen=True)
class KDARecords:
    """Verification records of every KDA layer, so speculative rounds keep one recurrent row per request.

    Verification leaves each request's committed FP32 state unchanged and records, per verified
    token, its FP32 delta-rule correction and BF16 raw key and gate; accepting a prefix replays
    those records into the committed row for every layer in one launch. Records are indexed by
    state row like the recurrent pool. ``A_log`` and ``dt_bias`` stack every layer's gate
    parameters for that replay.
    """

    layers: tuple[int, ...]
    corrections: torch.Tensor
    keys_and_gates: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor

    @classmethod
    def allocate(cls, attention_weights, slots, heads, device, *, window=8):
        """``attention_weights`` maps each KDA layer's decoder index to its attention weights."""
        layers = tuple(sorted(attention_weights))
        count = len(layers)
        return cls(layers,
            torch.zeros((count, slots, heads, window, 128), dtype=torch.float32, device=device),
            torch.zeros((count, slots, heads, window, 256), dtype=torch.bfloat16, device=device),
            torch.stack([attention_weights[layer]["A_log"] for layer in layers]),
            torch.stack([attention_weights[layer]["dt_bias"].view(heads, 128) for layer in layers]))

    def layer(self, index):
        position = self.layers.index(index)
        return self.corrections[position], self.keys_and_gates[position]


_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


@torch.library.custom_op("flare::kda_record", mutates_args=("output", "corrections", "keys_gates"))
def _record(reads: list[torch.Tensor], output: torch.Tensor, corrections: torch.Tensor,
            keys_gates: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    gdn_decode.run_kda(owner.recurrence, **owner.options)


@_record.register_fake
def _fake_record(reads, output, corrections, keys_gates, handle):
    pass


def register_recording(owner):
    handle = next(_HANDLES)
    _OWNERS[handle] = owner
    return handle


@torch.library.custom_op("flare::kda_replay", mutates_args=("backing",))
def _replay(backing: torch.Tensor, reads: list[torch.Tensor], handle: int) -> None:
    gdn_decode.run_kda_commit(_OWNERS[handle].binding, lower_bound=-5.0)


@_replay.register_fake
def _fake_replay(backing, reads, handle):
    pass


class PreparedKDAReplay:
    def __init__(self, backing, states, records, plan, state_indices, *, session, workspace):
        self.backing, self.sources, self.plan, self.workspace = backing, state_indices, plan, workspace
        device = backing.device
        self.lengths = torch.zeros_like(state_indices, dtype=torch.int32)
        self.finals = torch.zeros_like(self.lengths)
        self.boundaries = torch.zeros_like(self.lengths)
        self.boundary_lengths = torch.zeros_like(self.lengths)
        self.handle = register_recording(self)
        corrections, kg = records.corrections, records.keys_and_gates
        def table(values):
            return torch.tensor(values, dtype=torch.int64, device=device)
        self.tables = dict(state_base_addrs=table([s.data_ptr() for s in states]),
            state_block_strides=table([s.stride(0) for s in states]),
            correction_cache_base_addrs=table([corrections[i].data_ptr() for i in range(len(states))]),
            correction_cache_block_strides=table([corrections.stride(1)] * len(states)),
            kg_cache_base_addrs=table([kg[i].data_ptr() for i in range(len(states))]),
            kg_cache_block_strides=table([kg.stride(1)] * len(states)),
            A_log=records.A_log, dt_bias=records.dt_bias)
        self.reads = [corrections, kg, self.sources, self.lengths, self.finals]
        register_finalizer(session, self.bind, workspace=workspace)

    def bind(self):
        self.binding = gdn_decode.bind_kda_commit(self.plan,
            scratch=planned_scratch(self.plan.scratch_specs(), self.workspace), **self.tables,
            state_indices=self.sources, commit_lens=self.lengths, final_state_indices=self.finals,
            boundary_state_indices=self.boundaries, boundary_recovery_lens=self.boundary_lengths)

    def __call__(self, counts):
        self.lengths.copy_(counts)
        self.finals.copy_(self.sources)
        _replay(self.backing, self.reads, self.handle)
