# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused gated RMS normalization of recurrent attention outputs before their output projection."""

import torch
import triton
import triton.language as tl

from ignition.preparation.triton import launch_triton, warmup_triton


@triton.jit
def _gated_rms_norm(Hidden, Gate, Weight, Output, Live, gate_row_stride, EPS: tl.constexpr, HEADS: tl.constexpr,
                    DIM: tl.constexpr, HEAD_BLOCK: tl.constexpr, ROUND_NORMALIZED: tl.constexpr,
                    BF16_WEIGHT: tl.constexpr, SILU_GATE: tl.constexpr,
                    NORM_BEFORE_GATE: tl.constexpr = True, WEIGHT_PER_HEAD: tl.constexpr = False):
    # FP32 RMS normalization per head with configurable gate placement.
    # ROUND_NORMALIZED rounds the normalized values to BF16 before weighting, and BF16_WEIGHT also
    # rounds their product with a BF16 weight, as a module computing in its input dtype does.
    # Rows at or after the live count are written as zeros.
    row = tl.program_id(0)
    head = tl.program_id(1) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)[:, None]
    column = tl.arange(0, DIM)[None, :]
    mask = head < HEADS
    offset = row * (HEADS * DIM) + head * DIM + column
    hidden = tl.load(Hidden + offset, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(Gate + row.to(tl.int64) * gate_row_stride + head * DIM + column, mask=mask, other=0.0).to(tl.float32)
    gate = gate * tl.sigmoid(gate) if SILU_GATE else tl.sigmoid(gate)
    if not NORM_BEFORE_GATE:
        hidden = hidden * gate
    weight = tl.load(Weight + (head * DIM if WEIGHT_PER_HEAD else 0) + column, mask=mask, other=0).to(tl.float32)
    variance = tl.sum(hidden * hidden, axis=1)[:, None] / DIM
    # A constexpr epsilon keeps the arithmetic FP32 under whole-forward compilation,
    # which would pass a float argument as FP64.
    hidden = hidden * tl.rsqrt(variance + EPS)
    if ROUND_NORMALIZED:
        hidden = hidden.to(tl.bfloat16)
        if BF16_WEIGHT:
            hidden = (hidden * weight).to(tl.bfloat16).to(tl.float32)
        else:
            hidden = weight * hidden.to(tl.float32)
    else:
        hidden = weight * hidden
    value = hidden * gate if NORM_BEFORE_GATE else hidden
    live = row < tl.load(Live)
    tl.store(Output + offset, tl.where(live, value, 0.0).to(Output.dtype.element_ty), mask=mask)


class PreparedGatedRMSNorm:
    """Normalize BF16 groups and gate them with sigmoid(gate) or silu(gate).

    Matches an FP32 gated RMSNorm (normalize, scale by the weight, multiply by
    the sigmoid gate, round to BF16) such as Transformers' RMSNormGated with a
    sigmoid activation. ``round_normalized`` instead matches a module that rounds
    the normalized values to its BF16 input dtype before weighting them, as
    Qwen's gated norms do. Rows at or after the device live count become zeros.
    Inputs and output are caller-owned storage with contiguous rows; the gate
    may be a column slice of a wider row. The output may feed a projection directly.
    ``norm_before_gate=False`` gates in FP32 before computing the variance.
    A [heads, dim] weight gives each group separate affine parameters. Passing a
    preparation session declares compilation and tuning with the model's batch.
    """

    def __init__(self, hidden, gate, weight, output, live, *, eps, round_normalized=False, activation="sigmoid",
                 norm_before_gate=True, session=None, name=None):
        rows, heads, dim = hidden.shape
        gate_rows_contiguous = gate.stride()[1:] == (dim, 1)
        if (gate.shape != hidden.shape or output.shape != hidden.shape or weight.shape not in ((dim,), (heads, dim))
                or any(t.dtype != torch.bfloat16 for t in (hidden, gate, output))
                or weight.dtype not in (torch.bfloat16, torch.float32)
                or not all(t.is_contiguous() for t in (hidden, output, weight)) or not gate_rows_contiguous
                or live.shape != (1,) or dim & (dim - 1) or activation not in ("sigmoid", "silu")):
            raise ValueError("gated RMS norm requires BF16 [rows,heads,dim] tensors with contiguous rows "
                             "and a [dim] or [heads,dim] weight")
        self.head_block = min(8, triton.next_power_of_2(heads))
        self.args = (hidden, gate, weight, output, live, gate.stride(0))
        self.grid = (rows, triton.cdiv(heads, self.head_block), 1)
        self.options = {"EPS": float(eps), "HEADS": heads, "DIM": dim, "HEAD_BLOCK": self.head_block,
                        "ROUND_NORMALIZED": bool(round_normalized),
                        "BF16_WEIGHT": weight.dtype == torch.bfloat16, "SILU_GATE": activation == "silu",
                        "NORM_BEFORE_GATE": bool(norm_before_gate), "WEIGHT_PER_HEAD": weight.ndim == 2,
                        "num_warps": 4}
        self.plan = None
        if session is None:
            self.compiled = warmup_triton(_gated_rms_norm, self.args, self.grid, **self.options)
        else:
            from .gated_norm_preparation import declare_norm

            self.plan = declare_norm(self.args, self.options, session=session, name=name)

    def __call__(self):
        if self.plan is None:
            launch_triton(_gated_rms_norm, self.compiled, self.args, self.grid, **self.options)
        else:
            from .gated_norm_preparation import execute

            execute(*self.args[:5], self.plan.handle)
