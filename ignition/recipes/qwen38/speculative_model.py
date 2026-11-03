# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured MTP proposals with optional history augmentation, target verification, and commit rounds."""

from __future__ import annotations

import torch

from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.state import gather_rows
from ignition.preparation.scratch import GraphBufferPool

from .decoder import PreparedDecoder
from .draft_state import DRAFT_RAW_FIELDS, DraftMetadata
from ignition.components.gdn_recovery import GDNLayerRecords, PreparedGDNCommit
from .mtp import MTPScratch, PreparedMTP
from ignition.components.speculation import (
    DRAFT_RNG_DOMAIN, PreparedSpeculativeCommit,
    PreparedSpeculativeDistributions, PreparedStochasticVerification, SpeculativeRngState,
)
from .speculative_state import PreparedVerificationState


class SpeculativePreparation:
    """Shared persistent owners for all prepared speculative graph capacities."""

    def __init__(self, *, weights, mtp_weights, pool, requests, inputs, scratch,
                 workspaces, rotary, cache, table, session, completion, frame, sequence,
                 context, prefill_capacity, max_cache_pages,
                 runtime=None, vocabulary_runtime=None, coordinator=None, verification_width=8,
                 sampling_workspace=None, graph_buffers=None):
        self.collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator)
        self.weights, self.mtp_weights, self.pool, self.requests = weights, mtp_weights, pool, requests
        self.inputs, self.scratch, self.workspaces, self.rotary = inputs, scratch, workspaces, rotary
        self.cache, self.table, self.session = cache, table, session
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.context, self.max_cache_pages = context, max_cache_pages
        device = pool.backing.device
        if type(verification_width) is not int or not 1 <= verification_width <= 8:
            raise ValueError("verification width must be between one and eight")
        self.verification_width = verification_width
        self.sampling_workspace = sampling_workspace
        b, w = requests.capacity, verification_width
        self.sampling_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        self.verification_residual = torch.zeros((b * w, 10240), dtype=torch.bfloat16, device=device)
        self.mtp_scratch = MTPScratch.allocate(prefill_capacity, requests.capacity, device, world_size=weights.shard.world_size,
            buffers=None if graph_buffers is None else graph_buffers.cursor(),
            hyperconnection_lowrank=mtp_weights.module.hyper_connection_mixer.input_mix_weight_down.weight.shape[0])
        # Feedback is consumed before the MTP mixer writes normalized activations.
        self.feedback_hidden = self.mtp_scratch.hyperconnection.normalized
        self.prefill_residual = scratch.residual
        self.raw_backup = {name: torch.empty_like(getattr(cache[48], name)) for name in DRAFT_RAW_FIELDS}
        self.previous_accepted = torch.ones(requests.capacity, dtype=torch.int32, device=device)
        self.candidates = torch.zeros((requests.capacity, 8), dtype=torch.int64, device=device)
        self.draft_tables = torch.full((requests.capacity, context // 64), -1, dtype=torch.int32, device=device)
        self.copy_ids = torch.full((requests.capacity, 7), -1, dtype=torch.int64, device=device)
        self.copy_count = torch.zeros(requests.capacity, dtype=torch.int32, device=device)
        self.copy_expected = torch.full((requests.capacity,), -1, dtype=torch.int64, device=device)
        self.copy_generation = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        self.rng = SpeculativeRngState.allocate(device, capacity=requests.capacity)
        self.draft_rng_generation = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        self.draft_rng_offset = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        # Verification records of every target GDN layer, indexed by model slot.
        heads = next(iter(pool.gdn.values())).recurrent.shape[1] if pool.gdn else 0
        self.gdn_records = (GDNLayerRecords.allocate(pool.gdn, requests.capacity, device,
                                                     key_heads=heads // 3, value_heads=heads)
                            if pool.gdn else None)
        self.owners, self.decoders, self.mtp_modules = [], [], []

    def draft_metadata(self, tokens, rows):
        return DraftMetadata(tokens=tokens, rows=rows, context=self.context, device=self.pool.backing.device,
                             hidden=self.feedback_hidden[:tokens])

    def buffers(self):
        buffers = {"copy_token_ids": self.copy_ids, "copy_token_count": self.copy_count,
                   "copy_expected_consumed": self.copy_expected, "copy_generation": self.copy_generation,
                   "draft_page_tables": self.draft_tables}
        return buffers

    def mtp(self, metadata, name, *, cache_only=False):
        module = PreparedMTP(self.mtp_weights, self.mtp_scratch.rows(metadata.tokens, metadata.rows),
            metadata.token_ids, metadata.hidden, self.cache[48], metadata.qsa, self.rotary,
            self.workspaces, output_indices=metadata.output_indices, session=self.session,
            name=name, draft_tokens=7, max_cache_pages=self.max_cache_pages, cache_only=cache_only, **self.collectives)
        self.mtp_modules.append(module)
        return module

    def prefill(self, decoder, name):
        operation = PreparedPrefillAbsorption(self, decoder, name)
        self.owners.append(operation)
        return operation

    def resume(self, resume):
        return resume

    def round(self, batch, width, name):
        operation = PreparedSpeculativeRound(self, batch, width, name, decoder_factory=PreparedDecoder)
        self.owners.append(operation)
        return operation

    def rebind_pages_for_preparation(self, cache):
        self.cache = cache
        for module in self.mtp_modules:
            module.rebind_pages_for_preparation(cache[48])


class PreparedPrefillAbsorption:
    """Absorb exactly P-1 target-derived pairs before publishing a P-token prefix."""

    def __init__(self, owner, decoder, name, *, residual_output=None):
        self.owner, self.decoder = owner, decoder
        tokens, device = decoder.state.tokens, owner.pool.backing.device
        self.metadata = owner.draft_metadata(tokens, 1)
        self.mtp = owner.mtp(self.metadata, f"{name}.mtp_absorb", cache_only=True)
        self.target_residual = (owner.prefill_residual[:tokens]
                                if residual_output is None else residual_output)
        decoder.residual_output = self.target_residual
        self.previous = torch.zeros((1, 10240), dtype=torch.bfloat16, device=device)
        self.previous_rope = torch.zeros((1, 3), dtype=torch.int64, device=device)
        self.before = torch.zeros(1, dtype=torch.int64, device=device)
        self.indices = torch.arange(tokens, device=device)

    def __call__(self):
        self.begin()
        self.decoder()
        self.finish()

    def begin(self):
        o = self.owner
        slots = o.inputs.row_slot_ids[:1]
        safe = slots.long().clamp(0, o.requests.capacity - 1)
        self.before.copy_(o.requests.consumed_tokens[safe])
        gather_rows(o.pool.region("target_residual"), o.inputs.request_state_rows[safe, 0], self.previous)
        gather_rows(o.pool.region("target_residual_position"), o.inputs.request_state_rows[safe, 0], self.previous_rope)

    def finish(self):
        o, state, m = self.owner, self.decoder.state, self.metadata
        slots = o.inputs.row_slot_ids[:1]
        active = state.token_state.active.bool()
        length = (state.gdn.query_start_loc[1:] - state.gdn.query_start_loc[:-1]).long()
        shift = (self.before == 0).long()
        indices = (self.indices + shift[0]).clamp_max(self.indices.numel() - 1)
        hidden = self.target_residual[(indices - 1).clamp_min(0)]
        rope = state.qsa.rope_positions[(indices - 1).clamp_min(0)]
        first = (self.indices == 0) & (self.before[0] > 0)
        hidden = torch.where(first[:, None], self.previous, hidden)
        rope = torch.where(first[:, None], self.previous_rope, rope)
        lengths = torch.where(active, (length - shift).clamp_min(0), 0)
        m.set(slots, lengths, (self.before - 1).clamp_min(0), state.ple.token_ids[indices],
              hidden, rope, o.draft_tables, capacity=self.indices.numel())
        self.mtp()



class PreparedSpeculativeRound:
    """One graph contains every draft, all target rows, acceptance and state commit."""

    def __init__(self, owner, batch, width, name, *, decoder_factory, residual_output=None):
        self.owner, self.batch, self.width = owner, batch, width
        if not 1 <= width <= owner.verification_width:
            raise ValueError("round exceeds prepared verification scratch capacity")
        device = owner.pool.backing.device
        self.token_limits = owner.inputs.query_lengths[:batch]
        self.state = PreparedVerificationState(owner.inputs, owner.requests, owner.pool,
            owner.candidates, owner.previous_accepted, width=width, rows=batch, token_limits=self.token_limits)
        scratch = owner.scratch.rows(batch * width, batch * width)
        self.target_residual = (owner.verification_residual[:batch * width]
                                if residual_output is None else residual_output)
        self.decoder = decoder_factory(owner.weights, scratch, self.state, owner.cache, owner.rotary,
            owner.table, owner.workspaces, owner.completion, owner.frame, owner.sequence,
            session=owner.session, name=f"{name}.target", max_cache_pages=owner.max_cache_pages,
            residual_output=self.target_residual, verification=True,
            gdn_records=owner.gdn_records if width > 1 else None,
            gdn_in_place=width == 1 and owner.gdn_records is not None, **owner.collectives)
        owner.decoders.append(getattr(self.decoder, "forward", self.decoder))
        self.draft_metadata = owner.draft_metadata(batch, batch)
        self.draft = owner.mtp(self.draft_metadata, f"{name}.draft") if width > 1 else None
        self.absorb_metadata = owner.draft_metadata(batch * width, batch)
        self.absorb = owner.mtp(self.absorb_metadata, f"{name}.absorb", cache_only=True)
        self.before = torch.zeros(batch, dtype=torch.int64, device=device)
        self.pending = torch.zeros(batch, dtype=torch.int64, device=device)
        self.previous = torch.zeros((batch, 10240), dtype=torch.bfloat16, device=device)
        self.previous_rope = torch.zeros((batch, 3), dtype=torch.int64, device=device)
        self.rope_delta = torch.zeros(batch, dtype=torch.int64, device=device)
        self.active = torch.zeros(batch, dtype=torch.bool, device=device)
        self.drafts = torch.zeros((batch, width - 1), dtype=torch.int64, device=device)
        from ignition.components.parallel_draft import prepare_sampling_scratch

        buffers = owner.sampling_buffers.cursor()
        self.sampling_scratch = prepare_sampling_scratch(batch, width, owner.requests.vocab_size, device,
                                                         buffers=buffers, proposal=True)
        self.draft_probabilities = buffers.zeros((batch, width - 1, owner.requests.vocab_size), torch.float32)
        self.draft_valid = torch.zeros((batch, width - 1), dtype=torch.bool, device=device)
        self.offered = torch.full((batch,), width - 1, dtype=torch.int32, device=device)
        self.copy_mask = torch.zeros(batch, dtype=torch.int64, device=device)
        self.raw_backup = owner.raw_backup
        self.rows = torch.arange(batch, device=device)
        self.columns = torch.arange(width, device=device)
        self.seed = torch.zeros(batch, dtype=torch.int64, device=device)
        self.offset = torch.zeros(batch, dtype=torch.int64, device=device)
        self.base_offset = torch.zeros(batch, dtype=torch.int64, device=device)
        self.sampled = torch.zeros(batch, dtype=torch.int64, device=device)
        self.broadcast_buffers, self.name = buffers, name
        self.operations = None

    def prepare(self):
        if self.operations is not None:
            return
        o, b, w = self.owner, self.batch, self.width
        slots, epochs = o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b]
        target = PreparedSpeculativeDistributions(self.decoder.scratch.logits.view(b, w, -1),
            o.requests, slots, epochs, o.inputs.live_rows, self.drafts,
            workspace=self.sampling_scratch["target"])
        proposals = tuple(PreparedDraftDistribution(self.draft.scratch.logits.view(b, 1, -1),
            o.requests, slots, epochs, o.inputs.live_rows, self.drafts, prefix_offset=index,
            seed=self.seed, offset=self.offset, sampled=self.sampled,
            workspace=self.sampling_scratch["proposal"]) for index in range(w - 1))
        target.prepare()
        for proposal in proposals:
            proposal.prepare()
        verify = PreparedStochasticVerification(self.draft_probabilities, self.drafts, target.probabilities,
            o.requests, slots, epochs, o.inputs.live_rows, rng=o.rng,
            draft_valid=self.draft_valid, target_valid=target.valid, token_limits=self.token_limits,
            workspace=self.sampling_scratch["verify"])
        verify.prepare()
        from ignition.components.parallel_draft import prepare_sampling_broadcasts
        prepare_sampling_broadcasts(proposals, verify, collectives=o.collectives, session=o.session,
                                    name=self.name, buffers=self.broadcast_buffers)
        commit = PreparedSpeculativeCommit(o.requests, verify.output,
            self.state.token_ids.view(b, w), slots, epochs, o.inputs.live_rows,
            o.completion, o.frame, o.sequence, logits=self.decoder.scratch.logits.view(b, w, -1), rng=o.rng, copy_offered_mask=self.copy_mask,
            draft_context_lag=1, token_limits=self.token_limits)
        commit.prepare()
        self.state.prepare_commit(commit.emitted_count, commit.consumed_before,
                                  self.target_residual, self.decoder.scratch.sample_hidden)
        model_commit = tuple(self.state._commit)
        if o.gdn_records is not None and w > 1:
            heads = o.gdn_records.delta.shape[2]
            model_commit += (PreparedGDNCommit(o.pool.backing,
                {layer: state.recurrent for layer, state in o.pool.gdn.items()}, o.gdn_records, self.state.gdn,
                commit.emitted_count, key_heads=heads // 3, value_heads=heads),)
        self.operations = (target, proposals, verify, commit, model_commit)

    def __call__(self):
        o, b, w = self.owner, self.batch, self.width
        if self.operations is None:
            raise RuntimeError("speculative round prepare() must complete before capture")
        target, proposals, verify, commit, model_commit = self.operations
        slots = o.inputs.row_slot_ids[:b]
        self.state._validate()
        safe = slots.long().clamp(0, o.requests.capacity - 1)
        self.active.copy_((self.rows < o.inputs.live_rows[0]) & (slots >= 0) & (slots < o.requests.capacity)
            & (o.requests.generation[safe] == o.inputs.row_generations[:b]) & (o.requests.status[safe] == 1))
        self.before.copy_(o.requests.consumed_tokens[safe])
        self.pending.copy_(o.requests.pending_token[safe])
        gather_rows(o.pool.region("target_residual"), o.inputs.request_state_rows[safe, 0], self.previous)
        gather_rows(o.pool.region("target_residual_position"), o.inputs.request_state_rows[safe, 0], self.previous_rope)
        for name, backup in self.raw_backup.items():
            backup.copy_(getattr(o.cache[48], name))
        self.copy_mask.zero_()
        self.drafts.zero_()
        self.draft_valid.zero_()
        self.base_offset.copy_(torch.where(o.draft_rng_generation[safe] == o.requests.generation[safe],
                                          o.draft_rng_offset[safe], 0))
        self.seed.copy_(o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN)
        delta = gather_rows(o.pool.rope_delta, o.inputs.request_state_rows[safe, 0], self.rope_delta)
        for index, proposal in enumerate(proposals):
            hidden = self.previous if index == 0 else self.draft.scratch.residual[self.draft_metadata.output_indices]
            tokens = self.pending if index == 0 else self.drafts[:, index - 1]
            rope = self.previous_rope if index == 0 else (self.before + index - 1 + delta)[:, None].expand(-1, 3)
            draft_active = self.active
            if self.token_limits is not None:
                draft_active = draft_active & (index < self.token_limits)
            self.draft_metadata.set(slots, draft_active.long(), self.before - 1 + index,
                tokens, hidden, rope, o.draft_tables, capacity=1)
            self.draft()
            self.offset.copy_(self.base_offset + 4 * index)
            proposal()
            copy = (self.active & (o.copy_generation[safe] == o.requests.generation[safe])
                    & (o.copy_expected[safe] == self.before)
                    & (index < o.copy_count[safe]) & (o.copy_ids[safe, index] >= 0)
                    & (o.copy_ids[safe, index] < o.requests.vocab_size))
            point_mass = (target.vocab_ids == o.copy_ids[safe, index, None]).float()
            self.draft_probabilities[:, index].copy_(torch.where(copy[:, None], point_mass,
                                                                proposal.probabilities[:, 0]))
            self.sampled.copy_(torch.where(copy, o.copy_ids[safe, index], self.sampled))
            self.copy_mask.bitwise_or_(copy.long() * (1 << index))
            self.drafts[:, index].copy_(self.sampled)
            self.draft_valid[:, index].copy_(copy | proposal.valid[:, 0])
        if w > 1:
            # One masked update per tensor instead of one storage write per request slot.
            slot_ids = torch.arange(o.requests.capacity, device=slots.device)
            match = self.active[None, :] & (slots[None, :] == slot_ids[:, None])
            matched = match.any(1)
            base = torch.where(match, self.base_offset[None, :], 0).sum(1)
            o.draft_rng_offset.copy_(torch.where(matched, base + 4 * (w - 1), o.draft_rng_offset))
            o.draft_rng_generation.copy_(torch.where(matched, o.requests.generation, o.draft_rng_generation))
        if self.token_limits is not None:
            self.drafts.masked_fill_(self.columns[None, :-1] >= self.token_limits[:, None] - 1, 0)
        o.candidates[:b, 0].copy_(self.pending)
        o.candidates[:b, 1:w].copy_(self.drafts)
        self.decoder()
        target()
        verify()
        commit()
        for operation in model_commit:
            operation()
        for name, backup in self.raw_backup.items():
            getattr(o.cache[48], name).copy_(backup)
        residual = self.target_residual.view(b, w, 10240)
        hidden = torch.cat((self.previous[:, None, :], residual[:, :-1]), dim=1)
        rope = torch.cat((self.previous_rope[:, None, :], self.state.qsa.rope_positions.view(b, w, 3)[:, :-1]), dim=1)
        self.absorb_metadata.set(slots, commit.emitted_count.long(), self.before - 1,
            self.state.token_ids, hidden, rope, o.draft_tables, capacity=w)
        self.absorb()
