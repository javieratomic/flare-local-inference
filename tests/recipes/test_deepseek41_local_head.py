# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed target head construction exposes expanded local rows to its consumers."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.recipes.deepseek41.decoder import PreparedDecoder
from ignition.recipes.deepseek41.bootstrap import PreparationConfig
from ignition.preparation.scratch import GraphBufferPool


class CpuProjection:
    """Replace only GPU operator declaration in CPU geometry checks."""
    def __init__(self, source, weight, output, **kwargs):
        self.weight, self.output = weight, output

    def __call__(self, source):
        self.output.copy_(torch.nn.functional.linear(source.float(), self.weight.float()))
        return self.output


@pytest.mark.parametrize("batch,physical", PreparationConfig(max_active=4,
    speculative_widths=tuple(range(1, 9)), speculative_token_capacity=32).packed_speculative_shapes)
def test_packed_head_factory_uses_disjoint_logical_local_output(monkeypatch, batch, physical):
    monkeypatch.setattr("ignition.components.vocabulary.UnquantizedProjection", CpuProjection)
    decoder = PreparedDecoder.__new__(PreparedDecoder)
    logical = batch * 8
    cursor = GraphBufferPool("cpu", chunk_bytes=1 << 20).cursor()
    decoder.scratch = SimpleNamespace(hidden=torch.arange(logical * 4).reshape(logical, 4).bfloat16(),
        local_logits=cursor.full((logical, 3), float("nan"), torch.float32),
        logits=cursor.full((logical, 12), float("nan"), torch.float32))
    assert decoder.scratch.local_logits.untyped_storage().data_ptr() == decoder.scratch.logits.untyped_storage().data_ptr()
    decoder.weights = SimpleNamespace(head=torch.arange(3 * 4).reshape(3, 4).bfloat16())
    indices = torch.arange(logical, dtype=torch.int64).flip(0) % physical
    decoder.output_metadata = SimpleNamespace(token_ids=torch.zeros(physical, dtype=torch.int64), sample_indices=indices)
    decoder._prepare_head(vocabulary_parallel=True, vocabulary_runtime=None,
                          coordinator=None, session=None, name="target")
    decoder.head(decoder.head_hidden)
    expected = torch.nn.functional.linear(decoder.head_hidden.float(), decoder.weights.head.float())
    if physical < logical:
        expected = expected[indices]
        assert decoder.local_logits.data_ptr() == decoder.scratch.logits.data_ptr()
        assert decoder.scratch.local_logits[physical:].isnan().all()
    else:
        assert decoder.local_logits.data_ptr() == decoder.scratch.local_logits.data_ptr()
    assert decoder.local_logits.shape == (logical, 3) and decoder.local_logits.is_contiguous()
    torch.testing.assert_close(decoder.local_logits, expected)


@pytest.mark.parametrize("invalid", ["dtype", "rows", "columns", "overlap"])
def test_local_expansion_rejects_invalid_geometry_and_aliases(monkeypatch, invalid):
    monkeypatch.setattr("ignition.components.vocabulary.UnquantizedProjection", CpuProjection)
    source, weight = torch.zeros(2, 4), torch.zeros(3, 4)
    local = torch.empty(8, 3)
    output = torch.empty(8, 3)
    indices = torch.zeros(8, dtype=torch.int64)
    if invalid == "dtype":
        indices = indices.int()
    elif invalid == "rows":
        indices = indices[:4]
    elif invalid == "columns":
        output = torch.empty(8, 6)
    else:
        output = local
    with pytest.raises(ValueError):
        PreparedVocabularyProjection(source, weight, local[:2], output, runtime=None,
            session=None, coordinator=None, name="local", row_indices=indices, gather=False)
