# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prompt-scoring vocabulary tiles honor the caller's output precision."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import prompt_logprobs
from ignition.preparation.scratch import GraphBufferPool


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_prompt_projection_and_reduction_share_requested_dtype(monkeypatch, dtype):
    observed = []

    class Projection:
        def __init__(self, source, weight, output, **kwargs):
            assert weight.dtype == torch.bfloat16
            observed.append(("projection", output.dtype))

        def __call__(self, source):
            pass

    def reduction(logits, *args, **kwargs):
        observed.append(("reduction", logits.dtype))
        return lambda: None

    monkeypatch.setattr(prompt_logprobs, "UnquantizedProjection", Projection)
    monkeypatch.setattr(prompt_logprobs, "PreparedPromptReduction", reduction)
    owner = SimpleNamespace(
        outputs=SimpleNamespace(
            scratch=GraphBufferPool("cpu"),
            targets=torch.zeros(9, dtype=torch.int32),
            values=torch.zeros(9, 4),
        ),
        frame=torch.zeros(1, dtype=torch.int32),
        rank=0,
        runtime=None,
    )
    source = torch.zeros(9, 128, dtype=torch.bfloat16)
    weight = torch.zeros(64, 128, dtype=torch.bfloat16)
    operation = prompt_logprobs.PromptScoring.prepare(
        owner,
        source,
        weight,
        vocabulary=64,
        session=None,
        coordinator=None,
        name="prompt-head",
        logits_dtype=dtype,
    )
    assert operation.source.dtype == torch.bfloat16
    assert observed == [("projection", dtype), ("reduction", dtype)]
