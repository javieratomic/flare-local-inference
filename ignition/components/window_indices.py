# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Read-only window-cache addresses shared by attention layers of one forward."""

import torch


class PreparedWindowIndices:
    """Refresh window addresses after their packed request metadata is produced.

    Layers may share these buffers while metadata, visibility, and replay bounds
    remain unchanged. Cache payloads and cache writes remain layer-specific.
    ``page_size`` is the cache's physical token extent, independent of the total
    packed row capacity. Draft blocks expose every live query to every other
    query in its request; other modes use causal windows.
    """

    def __init__(self, metadata, *, mode, tokens, page_size, width, visible=None,
                 replay_start=None, buffers=None):
        if mode not in ("decode", "verify", "prefill", "draft"):
            raise ValueError("window indices require a supported attention mode")
        if tokens != metadata.positions.numel() or page_size < 128 or width < 128:
            raise ValueError("window indices must match the packed rows and cache extent")
        if mode == "decode" and page_size != 128:
            raise ValueError("decode window indices require 128-token cache pages")
        self.metadata, self.mode, self.visible = metadata, mode, visible
        self.replay_start, self.window_page_size = replay_start, page_size
        self.offsets = torch.arange(width, device=metadata.positions.device)
        if buffers is None:
            device = metadata.positions.device
            self.window_slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
            self.window_indices = torch.full((tokens, width), -1, dtype=torch.int32, device=device)
            self.window_lengths = torch.zeros(tokens, dtype=torch.int32, device=device)
        else:
            self.window_slots = buffers.full((tokens,), -1, torch.int64)
            self.window_indices = buffers.full((tokens, width), -1, torch.int32)
            self.window_lengths = buffers.zeros((tokens,), torch.int32)

    def __call__(self):
        m = self.metadata
        live, position = m.positions >= 0, m.positions.clamp_min(0)
        rows, starts = m.state_rows[m.requests], m.starts[m.requests]
        left = (position - 127).clamp_min(0)
        if self.replay_start is not None:
            available = position + 1 - self.replay_start[m.requests]
            left = position - available.clamp(0, 128) + 1
        right = position
        if self.visible is not None:
            image_left, image_right = self.visible
            left = torch.minimum(left, position - image_left)
            right = position + image_right
        self.window_lengths.copy_(torch.where(live, right - left + 1, 0))
        positions = left[:, None] + self.offsets
        if self.mode == "decode":
            slots = rows * 128 + position % 128
            indices = rows[:, None] * 128 + positions % 128
        else:
            base = m.requests * self.window_page_size
            slots = base + 128 + position - starts
            indices = torch.where(
                positions < starts[:, None], positions % 128, 128 + positions - starts[:, None])
            indices = indices + base[:, None]
            if self.mode == "draft":
                context = starts.clamp(max=128)
                self.window_lengths.copy_(torch.where(live, context + m.lengths[m.requests], 0))
                indices = base[:, None] + torch.where(
                    self.offsets < context[:, None], self.offsets, 128 + self.offsets - context[:, None])
        self.window_slots.copy_(torch.where(live, slots, -1))
        self.window_indices.copy_(torch.where(self.offsets < self.window_lengths[:, None], indices, -1))
