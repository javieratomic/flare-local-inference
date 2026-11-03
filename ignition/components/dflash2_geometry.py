# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grouped-query attention geometry for full-rotary DFlash2 drafts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DraftAttentionGeometry:
    heads: int
    kv_heads: int
    key_dim: int
    value_dim: int
    rotary_dim: int
    theta: float
    window: int

    def local(self, shard):
        shard.partition(self.heads)
        first, last = shard.replicated_partition(self.kv_heads)
        return self.heads // shard.world_size, last - first
