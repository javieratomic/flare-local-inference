# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit complete-round geometry for diagnostic graph preparation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSelection:
    batches: tuple[int, ...] = (1, 2, 4)
    contexts: tuple[int, ...] = (512, 2048, 8192, 16384)
    width: int = 8

    def __post_init__(self):
        for name, values in (("batches", self.batches), ("contexts", self.contexts)):
            if not isinstance(values, tuple) or not values or any(type(v) is not int for v in values):
                raise ValueError(f"benchmark {name} must be a nonempty tuple of integers")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"benchmark {name} must be distinct and increasing")
        if any(batch not in (1, 2, 4) for batch in self.batches):
            raise ValueError("benchmark batches must be selected from 1, 2, 4")
        if any(context < 128 or context % 128 for context in self.contexts):
            raise ValueError("benchmark contexts must be positive multiples of 128")
        if type(self.width) is not int or self.width != 8:
            raise ValueError("benchmark verification width must be 8")

    @property
    def max_batch(self):
        return self.batches[-1]

    @property
    def max_context(self):
        return self.contexts[-1]

    @property
    def speculative_round_shapes(self):
        return tuple((batch, self.width, batch * self.width) for batch in self.batches)

    def prefill_capacities(self, chunk):
        return tuple(sorted({min(context, chunk) for context in self.contexts}))

    def physical_pages(self, page_tokens):
        return 1 + self.max_batch * ((self.max_context + self.width + page_tokens - 1) // page_tokens)

    def to_dict(self):
        return dict(batches=self.batches, contexts=self.contexts, width=self.width,
                    speculative_round_shapes=self.speculative_round_shapes)
