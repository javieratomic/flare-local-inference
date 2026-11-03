# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark geometry selects complete production speculative rounds."""

from dataclasses import replace
from pathlib import Path

import pytest

from ignition import planning
from ignition.errors import UsageError
from ignition.preparation.selection import BenchmarkSelection
from ignition.recipes import resolve


def _plan():
    model = resolve("qwen3.8-27b")
    gpu = planning.Gpu(0, "NVIDIA GB10", "GPU-test", 128 << 30, 120 << 30, (12, 1))
    return planning.plan(model.recipe, model.checkpoint, (gpu,), selected=(0,), concurrency=32,
                         reserve_bytes=0, speculation=model.recipe.speculation)


def test_complete_rounds_and_context_backing():
    selection = BenchmarkSelection()
    assert selection.speculative_round_shapes == ((1, 8, 8), (2, 8, 16), (4, 8, 32))
    assert selection.prefill_capacities(4096) == (512, 2048, 4096)
    assert selection.physical_pages(128) == 517
    assert selection.physical_pages(128) * 128 > 4 * (16384 + 8)


@pytest.mark.parametrize("values", [dict(batches=()), dict(batches=(1, 3)), dict(batches=(1, 8)),
    dict(batches=(2, 1)), dict(contexts=(512, 512)), dict(contexts=(513,)),
    dict(contexts=(0,)), dict(width=0), dict(width=True), dict(width=4), dict(width=9)])
def test_invalid_selection(values):
    with pytest.raises(ValueError):
        BenchmarkSelection(**values)


def test_plan_retains_explicit_concurrency_and_token_budget():
    plan = replace(_plan(), benchmark_selection=BenchmarkSelection())
    assert plan.concurrency == 32
    assert plan.speculative_token_capacity == 128
    assert plan.to_dict()["benchmark_selection"]["speculative_round_shapes"] == ((1, 8, 8), (2, 8, 16), (4, 8, 32))


@pytest.mark.parametrize("changes,match", [
    (dict(concurrency=2), "max-concurrency"),
    (dict(speculation=None), "speculative width"),
    (dict(prompt_logprobs=True), "scored or mixed"),
    (dict(benchmark_selection=BenchmarkSelection(contexts=(262144,))), "prepared context"),
])
def test_plan_rejects_unavailable_geometry(changes, match):
    values = dict(benchmark_selection=BenchmarkSelection())
    values.update(changes)
    with pytest.raises(UsageError, match=match):
        replace(_plan(), **values)


def test_recipe_config_selects_rounds_without_changing_default_catalog():
    from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig

    normal = PreparationConfig(draft_snapshot=Path("draft"))
    selected = replace(normal, max_active=4, benchmark_selection=BenchmarkSelection())
    assert selected.speculative_round_shapes == ((1, 8, 8), (2, 8, 16), (4, 8, 32))
    assert (selected.decode_tokens, selected.decode_samples) == (32, 32)
    assert selected.speculative_token_capacity == 128
    assert (32, 1, 32) in normal.speculative_round_shapes
    assert len(normal.speculative_round_shapes) > len(selected.speculative_round_shapes)


def test_subset_contract_without_decode():
    import torch
    from types import SimpleNamespace
    from ignition.components.state import StateLayout, StateStorage
    from ignition.preparation.execution import GraphSignature
    from ignition.recipes.qwen38_dense.contract import prepare

    layout = StateLayout.from_specs((("consumed_tokens", (1,), torch.int64),))
    storage = StateStorage.allocate(layout, 9, "cpu")
    pool = SimpleNamespace(layout=layout, pages={})
    buffers = {"state_pool": storage.backing, "request_pending_token": torch.zeros(4, dtype=torch.int64)}
    signature = GraphSignature("speculative", 4, 8, speculation_enabled=True, packed_token_capacity=32)
    contract = prepare({"speculative4x8p32": signature}, buffers, pool, vocabulary=128)
    assert [graph["graph"] for graph in contract["graphs"]] == ["speculative4x8p32"]
