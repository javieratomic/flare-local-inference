# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed request metadata for windowed and compressed-history attention."""

from dataclasses import dataclass

import torch

from ignition.components.sampling import ERROR, RUNNING
from ignition.components.token_state import PreparedTokenState

PAGE_TOKENS = 128


@dataclass(frozen=True)
class BatchInputs:
    request_state_rows: torch.Tensor
    row_slot_ids: torch.Tensor
    row_generations: torch.Tensor
    live_rows: torch.Tensor
    query_lengths: torch.Tensor
    token_ids: torch.Tensor
    page_tables: torch.Tensor
    sample_enabled: torch.Tensor

    @classmethod
    def allocate(cls, tokens, blocks, device, *, capacity):
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros((capacity, 8), torch.int64),
            zeros(capacity) - 1,
            zeros(capacity, torch.int64),
            zeros(1),
            zeros(capacity),
            zeros(tokens, torch.int64),
            zeros((capacity, blocks)) - 1,
            zeros(capacity),
        )

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class Metadata:
    token_ids: torch.Tensor
    sample_indices: torch.Tensor
    query_start_loc: torch.Tensor
    state_rows: torch.Tensor
    starts: torch.Tensor
    lengths: torch.Tensor
    requests: torch.Tensor
    positions: torch.Tensor
    page_table: torch.Tensor
    token_page_table: torch.Tensor
    live_counts: torch.Tensor
    max_query_tokens: int

    @classmethod
    def allocate(cls, tokens, rows, blocks, device, *, samples=None, buffers=None, max_query_tokens=None):
        query_capacity = (tokens + rows - 1) // rows if max_query_tokens is None else max_query_tokens
        if type(query_capacity) is not int or query_capacity < 1:
            raise ValueError("per-request query capacity must be a positive integer")
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device) if buffers is None else buffers.zeros(shape, dtype)

        pages = zeros((rows, blocks)).fill_(-1)
        per_token = pages.expand(tokens, -1) if rows == 1 else zeros((tokens, blocks)).fill_(-1)
        return cls(
            zeros(tokens, torch.int64),
            zeros(rows if samples is None else samples, torch.int64),
            zeros(rows + 1),
            zeros(rows, torch.int64),
            zeros(rows, torch.int64),
            zeros(rows),
            zeros(tokens, torch.int64),
            zeros(tokens, torch.int64).fill_(-1),
            pages,
            per_token,
            zeros(2),
            query_capacity,
        )

    def set(self, starts, lengths, state_rows, pages, tokens, live_rows):
        rows, width = tokens.shape
        offsets = torch.arange(self.token_ids.numel(), device=tokens.device)
        ends = lengths.cumsum(0)
        begins = ends - lengths
        request = (offsets[:, None] >= ends[None, :]).sum(1).clamp(max=rows - 1)
        column = offsets - begins[request]
        live = offsets < ends[-1]
        self.query_start_loc[0].zero_()
        self.query_start_loc[1:].copy_(ends)
        self.lengths.copy_(lengths)
        self.starts.copy_(starts)
        self.state_rows.copy_(torch.where(lengths > 0, state_rows, 0))
        self.page_table.copy_(pages)
        if rows > 1:
            self.token_page_table.copy_(pages[request])
        self.live_counts[0:1].copy_(ends[-1:])
        self.live_counts[1:].copy_(live_rows.clamp(0, rows))
        self.requests.copy_(request)
        self.positions.copy_(torch.where(live, starts[request] + column, -1))
        indices = (request * width + column).clamp(0, tokens.numel() - 1)
        self.token_ids.copy_(torch.where(live, tokens.reshape(-1)[indices], 0))
        columns = torch.arange(self.sample_indices.numel() // rows, device=tokens.device)
        self.sample_indices.copy_(
            (begins[:, None] + columns).clamp(0, self.token_ids.numel() - 1).reshape(-1)
        )


class PreparedBatch:
    def __init__(
        self,
        inputs,
        requests,
        metadata,
        *,
        mode,
        shard,
        pool,
        verification_width=1,
        target_token_capacity=None,
        input_vocabulary=None,
        validate_prefill=None,
        page_tokens=PAGE_TOKENS,
        buffers=None,
    ):
        self.inputs, self.requests, self.metadata, self.shard = inputs, requests, metadata, shard
        self.rows, self.tokens, self.mode = metadata.starts.numel(), metadata.token_ids.numel(), mode
        if mode not in ("prefill", "decode") or mode == "prefill" and self.rows != 1:
            raise ValueError("Paged prefill requires one packed sequence")
        self.pool = pool
        self.page_tokens = page_tokens
        self.verification_width = verification_width
        self.target_token_capacity = self.rows * verification_width if target_token_capacity is None else target_token_capacity
        if not self.rows <= self.target_token_capacity <= self.rows * verification_width:
            raise ValueError("target token capacity must fit the request rows and verification width")
        self.input_vocabulary = requests.vocab_size if input_vocabulary is None else input_vocabulary
        self.validate_prefill = validate_prefill
        device, capacity = metadata.token_ids.device, requests.capacity
        self.row_ids = torch.arange(self.rows, device=device)
        self.slot_ids = torch.arange(capacity, device=device)
        self.token_offsets = torch.arange(self.tokens, device=device)
        self.block_ids = torch.arange(inputs.page_tables.shape[1], device=device)
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device) if buffers is None else buffers.zeros(shape, dtype)
        self.validation_input = zeros((1, max(8, 2 * capacity)), torch.bfloat16)
        self.validation_sum = zeros(self.validation_input.shape, torch.bfloat16)
        self.present = zeros((capacity,), torch.bool)
        self.token_state = PreparedTokenState(
            requests,
            metadata.token_ids,
            metadata.query_start_loc,
            inputs.row_slot_ids[: self.rows],
            inputs.row_generations[: self.rows],
            inputs.live_rows,
            buffers=buffers,
        )
        self.token_state.prepare()

    def validate(self):
        b, r = self.inputs, self.requests
        slots = b.row_slot_ids[: self.rows].long()
        safe = slots.clamp(0, r.capacity - 1)
        resident = (self.row_ids < b.live_rows) & (slots >= 0) & (slots < r.capacity)
        resident &= b.row_generations[: self.rows] == r.generation[safe]
        duplicate = ((slots[:, None] == slots[None, :]) & resident[:, None] & resident[None, :]).sum(1) > 1
        lengths = (
            b.query_lengths[: self.rows]
            if self.mode == "prefill" or self.verification_width > 1
            else torch.ones_like(safe)
        )
        position = r.consumed_tokens[safe]
        end = position + lengths
        capacity = self.tokens if self.mode == "prefill" else self.verification_width
        valid = ~duplicate & (b.live_rows <= self.rows) & (lengths > 0) & (lengths <= capacity)
        if self.mode == "decode":
            offered = torch.where(resident & (r.status[safe] == RUNNING), lengths, 0)
            valid &= offered.sum() <= self.target_token_capacity
        valid &= (position >= 0) & (end <= b.page_tables.shape[1] * self.page_tokens)
        pages = b.page_tables[safe]
        valid &= (
            ((pages >= 0) & (pages < self.pool.memory_capacity[1]))
            | (self.block_ids >= (end[:, None] + self.page_tokens - 1) // self.page_tokens)
        ).all(1)
        state = b.request_state_rows[safe, 0]
        valid &= (state > 0) & (state < self.pool.memory_capacity[0])
        pending, tokens = r.pending_token[safe], b.token_ids[: self.tokens]
        if self.mode == "prefill":
            invalid = ((tokens < 0) | (tokens >= self.input_vocabulary)) & (self.token_offsets < lengths[0])
            valid &= (pending < 0) & ~invalid.any()
            if self.validate_prefill is not None:
                valid &= self.validate_prefill(tokens, lengths)
        else:
            valid &= (pending >= 0) & (pending < r.vocab_size)
        present = resident & (r.status[safe] == RUNNING)
        bad = present & ~valid
        bad_slots = ((self.slot_ids[:, None] == slots[None, :]) & bad[None, :]).any(1)
        self.present.copy_(((self.slot_ids[:, None] == slots[None, :]) & present[None, :]).any(1))
        self.validation_input[0, : r.capacity].copy_(bad_slots)
        self.validation_input[0, r.capacity : 2 * r.capacity].copy_(self.present)
        return self.validation_input

    def begin(self, totals):
        b, r, m, t = self.inputs, self.requests, self.metadata, self.token_state
        participants = totals[0, r.capacity : 2 * r.capacity]
        bad = self.present & (
            (totals[0, : r.capacity] > 0) | ((participants > 0) & (participants < self.shard.world_size))
        )
        r.status.copy_(torch.where(bad, ERROR, r.status))
        r.error_code.copy_(torch.where(bad, 32, r.error_code))
        t.gather()
        active = t.active.bool()
        slots = b.row_slot_ids[: self.rows].long().clamp(0, r.capacity - 1)
        lengths = b.query_lengths[: self.rows] if self.mode == "prefill" else torch.ones_like(m.lengths)
        lengths = torch.where(active, lengths, 0)
        ends = lengths.cumsum(0, dtype=torch.int32)
        starts = ends - lengths
        m.query_start_loc[0].zero_()
        m.query_start_loc[1:].copy_(ends)
        m.lengths.copy_(lengths)
        m.starts.copy_(t.positions)
        m.state_rows.copy_(torch.where(active, b.request_state_rows[slots, 0], 0))
        m.page_table.copy_(torch.where(active[:, None], b.page_tables[slots], -1))
        m.live_counts[0:1].copy_(ends[-1:])
        m.live_counts[1:2].copy_(b.live_rows.clamp(0, self.rows))
        m.sample_indices.copy_((ends - 1).clamp(min=0))
        request = (self.token_offsets[:, None] >= ends[None, :]).sum(1).clamp(max=self.rows - 1)
        live = self.token_offsets < ends[-1]
        m.requests.copy_(request)
        if self.rows > 1:
            m.token_page_table.copy_(m.page_table[request])
        m.positions.copy_(torch.where(live, t.positions[request] + self.token_offsets - starts[request], -1))
        if self.mode == "prefill":
            tokens = b.token_ids[:self.tokens]
        else:
            tokens = t.pending_tokens[request]
        m.token_ids.copy_(torch.where(live, tokens, 0))

    def finish(self):
        self.token_state.consume()
