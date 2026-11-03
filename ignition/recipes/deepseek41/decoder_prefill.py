# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Pack a prefill chunk's decoder suffix while preserving absolute cache positions.

Decoder SWA bounded replay is an approximation described in section 3.2.2 of
the DeepSeek V4.1 technical report. Global KV must be produced from every
encoder row before gathering the suffix. Chunks that fit the replay window
retain their complete continuation history; a truncated chunk establishes a
fresh SWA boundary at its first retained row.
"""

import torch

from ignition.components.paged_batch import Metadata


class PreparedDecoderSuffix:
    def __init__(self, metadata, *, window=128):
        if metadata.starts.numel() != 1 or metadata.positions.numel() <= window or window <= 0:
            raise ValueError("decoder suffix packing requires one prefill chunk larger than its window")
        self.source = metadata
        self.metadata = Metadata.allocate(window, 1, metadata.page_table.shape[1], metadata.positions.device,
                                          samples=metadata.sample_indices.numel())
        self.offsets = torch.arange(window, device=metadata.positions.device)
        self.indices = torch.zeros_like(self.offsets)
        self.live = torch.zeros(window, dtype=torch.bool, device=metadata.positions.device)
        self.replay_start = torch.zeros_like(metadata.starts)

    def begin(self):
        source, target = self.source, self.metadata
        count = source.lengths[0].long()
        kept = count.clamp(max=self.offsets.numel())
        skipped = count - kept
        self.indices.copy_((source.query_start_loc[0] + skipped + self.offsets).clamp(
            0, source.positions.numel() - 1))
        self.live.copy_((self.offsets < kept) & (source.positions[self.indices] >= 0))
        target.token_ids.copy_(torch.where(self.live, source.token_ids[self.indices], 0))
        target.positions.copy_(torch.where(self.live, source.positions[self.indices], -1))
        target.starts.copy_(source.starts + skipped)
        target.lengths.copy_(kept.view(1))
        target.state_rows.copy_(source.state_rows)
        target.requests.zero_()
        target.query_start_loc[0].zero_()
        target.query_start_loc[1:].copy_(kept.view(1))
        target.sample_indices.copy_((source.sample_indices - skipped).clamp(0, self.offsets.numel() - 1))
        target.page_table.copy_(source.page_table)
        target.live_counts[0].copy_(kept)
        target.live_counts[1:].copy_(source.live_counts[1:])
        self.replay_start.copy_(torch.where(skipped > 0, target.starts, 0))

    def gather(self, source, output):
        live = self.live.view((-1,) + (1,) * (source.ndim - 1))
        output.copy_(torch.where(live, source[self.indices], 0))
