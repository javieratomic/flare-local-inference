# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Collective text execution and request completion."""

from ignition.components.collectives import PreparedAllReduce
from ignition.components.sampling import PreparedSampling
from ignition.components.state import save_hidden
from ignition.preparation.scratch import bind_scratch


class PreparedGeneration:
    def __init__(
        self,
        decoder,
        batch,
        records,
        frame,
        sequence,
        *,
        runtime,
        coordinator,
        session,
        name,
        workspace,
        context_factory=None,
        buffers=None,
    ):
        self.decoder, self.batch = decoder, batch
        self.validation = PreparedAllReduce(
            batch.validation_input,
            batch.validation_sum,
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".validation",
        )
        b, rows = batch.inputs, batch.rows
        args = (
            batch.requests,
            b.row_slot_ids[:rows],
            b.row_generations[:rows],
            b.live_rows,
            records,
            frame,
            sequence,
        )
        sampling_workspace = None if buffers is None else bind_scratch(
            PreparedSampling.scratch_layout(rows, batch.requests.vocab_size),
            device=decoder.scratch.logits.device, workspace=buffers)
        self.completion = PreparedSampling(decoder.scratch.logits, *args, sample_enabled=b.sample_enabled[:rows], workspace=sampling_workspace)
        self.completion.prepare()
        self.completion.prepare_broadcast(runtime=decoder.head.gather.runtime, session=session,
            coordinator=coordinator, name=name + ".sampling", buffers=buffers)
        self.compilation_variant = "sample"
        self.context = None
        if decoder.weights.draft is not None:
            if context_factory is None:
                raise ValueError("speculative generation requires a draft context factory")
            self.context = context_factory(
                decoder.weights,
                decoder.auxiliary,
                batch.metadata,
                decoder.pool,
                session=session,
                name=name + ".context",
                workspace=workspace,
            )

    def __call__(self):
        self.batch.begin(self.validation(self.batch.validate()))
        self.decoder()
        if self.context is not None:
            self.context(self.decoder.auxiliary, self.batch.metadata.lengths)
        save_hidden(
            self.decoder.scratch.hidden,
            self.decoder.pool.last_hidden,
            self.batch.metadata.state_rows,
            self.batch.token_state.active,
        )
        self.batch.finish()
        self.completion()
