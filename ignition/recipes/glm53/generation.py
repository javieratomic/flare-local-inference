# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compose a GLM rank's text forward, request consumption, and completion."""

from ignition.components.collectives import PreparedAllReduce
from ignition.components.sampling import PreparedSampling
from ignition.components.state import save_hidden

from .batch import PreparedBatch
from .stack import PreparedDecoder


class PreparedGeneration:
    """Gather GPU-owned tokens and commit root-broadcast sampling results on every rank.

    Native execution waits for every rank before retiring a step or reusing its
    cache pages.
    """

    def __init__(self, decoder: PreparedDecoder, batch: PreparedBatch, completion,
                 frame_index, submission_sequence, *, pool, runtime, coordinator, session, name):
        if decoder.metadata is not batch.metadata or decoder.weights.shard != batch.shard:
            raise ValueError("GLM generation requires the decoder's metadata and rank")
        if decoder.state is not pool.decoder:
            raise ValueError("GLM generation requires the decoder's continuation pool")
        self.decoder, self.batch, self.pool = decoder, batch, pool
        self.validation = PreparedAllReduce(batch.validation_input, batch.validation_sum,
            runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.validation")
        b, rows = batch.inputs, batch.rows
        arguments = (batch.requests, b.row_slot_ids[:rows], b.row_generations[:rows], b.live_rows,
                     completion, frame_index, submission_sequence)
        self.completion = PreparedSampling(
            decoder.scratch.logits, *arguments, sample_enabled=b.sample_enabled[:rows])
        self.completion.prepare()
        self.completion.prepare_broadcast(runtime=decoder.lm_head.gather.runtime, session=session,
            coordinator=coordinator, name=name + ".sampling")
        self.compilation_variant = "sample"

    def rebind_pages(self, pages):
        self.decoder.rebind_pages()
        self.batch.pages = self.pool.memory_capacity[1:]

    def __call__(self):
        self.batch.begin(self.validation(self.batch.validate()))
        self.decoder()
        save_hidden(self.decoder.scratch.hidden, self.pool.last_hidden,
                    self.batch.metadata.recurrent.convolution_slots, self.batch.token_state.active)
        self.batch.finish()
        self.completion()
