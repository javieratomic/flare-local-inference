# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Finite strided operands shared by a projection's serial tuning candidates."""

from contextlib import nullcontext

import torch

from .resources import candidate_inputs


class Inputs:
    def __init__(self, source, output):
        self.source = torch.empty_strided(source.shape, source.stride(),
                                         dtype=source.dtype, device=source.device)
        self.output = torch.empty_strided(output.shape, output.stride(),
                                         dtype=output.dtype, device=output.device)
        generator = torch.Generator(device=source.device).manual_seed(0xB12)
        self.seed = torch.randn(source.shape, device=source.device, dtype=source.dtype,
                                generator=generator).mul_(0.125)

    def produce(self):
        self.source.copy_(self.seed)


def input_factory(source, output, *, storage=None):
    def allocate():
        with (nullcontext() if storage is None else storage.transient()):
            return Inputs(source, output)

    return candidate_inputs(allocate)
