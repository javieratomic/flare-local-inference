# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Red Hat DSpark scratch ownership and checkpoint selection."""

import torch


def test_draft_scratch_export_ranges():
    import ctypes
    from dataclasses import fields, is_dataclass
    from types import SimpleNamespace
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.disposable import storage_ranges
    from ignition.recipes.kimi3.dspark import DraftScratch as DSparkScratch

    config = SimpleNamespace(hidden=128, vocabulary=512, draft=SimpleNamespace(
        heads=2, kv_heads=1, head_dim=64, window=128, text={"intermediate_size": 256}))
    scratch = DSparkScratch.allocate(config, TensorParallelShard(0, 1), 32, 8, "cpu")

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif is_dataclass(value):
            for field in fields(value):
                yield from tensors(getattr(value, field.name))
        elif isinstance(value, dict):
            for child in value.values():
                yield from tensors(child)

    owned = tuple(tensors(scratch))
    for tensor in owned:
        tensor.fill_(1)
    ranges = storage_ranges(scratch.scratch_tensors())
    assert len(ranges) == len({tensor.untyped_storage().data_ptr() for tensor in owned})
    assert ranges == storage_ranges(scratch.rows(8, 8).scratch_tensors())
    for address, size in ranges:
        ctypes.memset(address, 0, size)
    assert all(not tensor.count_nonzero() for tensor in owned)
