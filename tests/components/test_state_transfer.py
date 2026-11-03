# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordinary execution preserves unused draft bytes during continuation transfers."""

import os

import pytest
import torch
from triton.runtime.interpreter import InterpretedFunction

from ignition.components.state_transfer import _copy_rows


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an explicitly assigned CUDA device"))])
def test_ordinary_state_copy_and_reset_leave_draft_regions_untouched(device):
    backing = torch.full((4, 8192), 231, dtype=torch.uint8, device=device)
    backing[1].fill_(17)
    source = torch.tensor([1, -1], device=device)
    destination = torch.tensor([2, 3], device=device)
    operation = torch.tensor([0, 1], dtype=torch.int32, device=device)
    count = torch.tensor([2], dtype=torch.int32, device=device)
    errors = torch.zeros(2, dtype=torch.int32, device=device)
    args = (backing, source, destination, operation, count, errors, 8192, 1024, ((768, 1792), (4096, 5120)))
    kernel = _copy_rows if device == "cuda" else InterpretedFunction(_copy_rows.fn)
    kernel[(2, 8)](*args)
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel[(2, 8)](*args)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        graph.reset()
    wanted = torch.full((4, 8192), 231, dtype=torch.uint8)
    wanted[1:3].fill_(17)
    wanted[3].zero_()
    for start, end in args[-1]:
        wanted[2:, start:end].fill_(231)
    assert torch.equal(backing.cpu(), wanted)
