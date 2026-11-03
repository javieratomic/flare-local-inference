# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Cached-prefix vocabulary projection preserves its requested logit precision."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import collectives, paged_resume, vocabulary
from ignition.preparation.scratch import GraphBufferPool


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("pooled", [False, True])
def test_resume_projection_matches_gather_byte_geometry(monkeypatch, dtype, pooled):
    from b12x.comm import pcie

    observed = {}

    class State:
        def __init__(self, *args, **kwargs):
            self.hidden = torch.zeros(1, 64, dtype=torch.bfloat16)
            self.validation_input = torch.zeros(1, 8, dtype=torch.bfloat16)
            self.validation_sum = torch.zeros_like(self.validation_input)

    class Projection:
        def __init__(self, source, weight, output, **kwargs):
            assert source.dtype == weight.dtype == torch.bfloat16
            self.output = output

    class Sampling:
        def __init__(self, logits, *args, **kwargs):
            self.logits = logits

        @staticmethod
        def scratch_layout(rows, vocab):
            return {"scores": ((rows, vocab), torch.float32)}

        def prepare(self):
            pass

        def prepare_broadcast(self, **kwargs):
            pass

    heads = 4 if dtype == torch.float32 else 2
    runtime = SimpleNamespace(
        world_size=4,
        device=torch.device("cpu"),
        max_batch_size=1,
        heads_per_rank=heads,
        query_head_dim=64,
        total_heads=heads * 4,
        exchange_group=object(),
    )

    def query(owner, *, surface, call):
        observed.update(call)
        return object()

    plan = SimpleNamespace(request=lambda **kwargs: object())
    monkeypatch.setattr(paged_resume, "PreparedResumeState", State)
    monkeypatch.setattr(paged_resume, "PreparedSampling", Sampling)
    monkeypatch.setattr(vocabulary, "UnquantizedProjection", Projection)
    monkeypatch.setattr(collectives, "_roce", lambda runtime: False)
    monkeypatch.setattr(collectives, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(pcie, "query_from_runtime", query)
    monkeypatch.setattr(pcie, "plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: [0, 1, 2, 3]
    )
    inputs = SimpleNamespace(
        row_slot_ids=torch.zeros(1, dtype=torch.int32),
        row_generations=torch.ones(1, dtype=torch.int64),
        live_rows=torch.ones(1, dtype=torch.int32),
        request_state_rows=torch.zeros(4, 8, dtype=torch.int64),
    )
    weights = SimpleNamespace(
        head=torch.zeros(128, 64, dtype=torch.bfloat16),
        shard=SimpleNamespace(world_size=4),
    )
    pool = SimpleNamespace(
        backing=torch.empty(1, dtype=torch.uint8),
        last_hidden=torch.zeros(5, 64, dtype=torch.bfloat16),
    )
    options = {} if dtype == torch.bfloat16 else {"logits_dtype": torch.float32}
    operation = paged_resume.PreparedResume(
        weights,
        SimpleNamespace(vocab_size=512),
        pool,
        inputs,
        None,
        None,
        None,
        rows=1,
        runtime=None,
        vocabulary_runtime=runtime,
        coordinator=None,
        session=None,
        name="resume",
        buffers=GraphBufferPool("cpu").cursor() if pooled else None,
        **options,
    )
    assert operation.local_logits.dtype == operation.logits.dtype == dtype
    assert operation.projection.projection.output.dtype == dtype
    assert operation.completion.logits is operation.logits
    assert observed["local_input"].shape == (1, heads, 64)
    assert observed["out"].shape == (1, heads * 4, 64)
    assert observed["local_input"].dtype == observed["out"].dtype == torch.bfloat16
    assert (
        observed["local_input"].numel() * 2
        == operation.local_logits.numel() * dtype.itemsize
    )
