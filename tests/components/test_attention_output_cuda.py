"""Prepared grouped attention projection with compiled, allocation-stable replay."""

import os

import pytest
import torch
from b12x.gemm import wo_projection
from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch
from b12x.preparation import PreparationSession

from ignition.components.attention_output import PreparedGroupedOutputProjection
from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


# DeepSeek V4.1 ranks own two output groups with four ranks and three with three.
@pytest.mark.parametrize("groups", [2, 3])
@pytest.mark.parametrize("tokens", [1, 4, 17, 4096])
@torch.no_grad()
def test_grouped_output_compiled_replay(tokens, groups):
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(4123)
    width, rank, hidden = 4096, 1024, 5120
    def weight(shape):
        values = (torch.randn(shape, device=device, generator=generator) * 16).to(torch.float8_e4m3fn)
        scales = torch.full((shape[0] // 32, shape[1] // 32), 117, dtype=torch.uint8, device=device)
        return values, scales.view(torch.float8_e8m0fnu)
    a, sa = weight((groups * rank, width))
    b, sb = weight((hidden, groups * rank))
    packed = wo_projection.pack_weights(a, sa, b, sb, groups=groups, group_width=width,
                                       rank=rank, hidden=hidden, block_size=(32, 32))
    source = torch.randn((tokens, groups, width), dtype=torch.bfloat16, device=device, generator=generator) * .125
    output = torch.empty((tokens, hidden), dtype=source.dtype, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        operation = PreparedGroupedOutputProjection(source, packed, output,
            session=session, name="wo", workspace=None)
        forward = compile_forward(lambda: operation(source))
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        torch.cuda.synchronize()
        for gain in (1, -.5, 2):
            source.mul_(gain)
            operation.scratch.fill_(255)
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            x = wo_projection.quantize_input(source, plan=operation.plan)
            x_deq = dequantize_mxfp8_rows_torch(x.values, x.scale_rows)
            a_deq = dequantize_mxfp8_rows_torch(packed.wo_a.values, packed.wo_a.scale_rows)
            intermediate = torch.einsum("tdg,rdg->trg", x_deq, a_deq).bfloat16()
            quantized = wo_projection.quantize_input_b(intermediate, plan=operation.plan)
            values = dequantize_mxfp8_rows_torch(quantized.values, quantized.scale_rows)
            weights = dequantize_mxfp8_rows_torch(packed.wo_b.values, packed.wo_b.scale_rows)
            expected = (values @ weights.T).bfloat16()
            error = (output.float() - expected.float()).square().mean().sqrt()
            assert error < expected.float().square().mean().sqrt() * .003
            torch.testing.assert_close(output, expected, rtol=.02, atol=.003)
            del x, x_deq, a_deq, intermediate, quantized, values, weights, expected, error
        graph.reset()


@pytest.mark.parametrize("tokens", [1, 7, 512])
@torch.no_grad()
def test_padded_output_group_adds_nothing(tokens):
    """A zero-weight third group leaves the two checkpoint groups' projection unchanged."""
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(4124)
    width, rank, hidden = 4096, 1024, 5120
    def weight(shape):
        values = (torch.randn(shape, device=device, generator=generator) * 16).to(torch.float8_e4m3fn)
        scales = torch.randint(110, 120, (shape[0] // 32, shape[1] // 32), dtype=torch.uint8, device=device,
                               generator=generator)
        return values, scales
    a, sa = weight((2 * rank, width))
    b, sb = weight((hidden, 2 * rank))
    padded_a = torch.cat((a, torch.zeros((rank, width), dtype=a.dtype, device=device)))
    padded_sa = torch.cat((sa, torch.full((rank // 32, width // 32), 127, dtype=torch.uint8, device=device)))
    padded_b = torch.cat((b, torch.zeros((hidden, rank), dtype=b.dtype, device=device)), dim=1)
    padded_sb = torch.cat((sb, torch.full((hidden // 32, rank // 32), 127, dtype=torch.uint8, device=device)), dim=1)
    source = torch.randn((tokens, 3, width), dtype=torch.bfloat16, device=device, generator=generator) * .125
    outputs = []
    for groups, parts, values in ((2, (a, sa, b, sb), source[:, :2].contiguous()),
                                  (3, (padded_a, padded_sa, padded_b, padded_sb), source)):
        wa, swa, wb, swb = parts
        packed = wo_projection.pack_weights(wa, swa.view(torch.float8_e8m0fnu), wb, swb.view(torch.float8_e8m0fnu),
            groups=groups, group_width=width, rank=rank, hidden=hidden, block_size=(32, 32))
        output = torch.empty((tokens, hidden), dtype=values.dtype, device=device)
        with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
            operation = PreparedGroupedOutputProjection(values, packed, output,
                session=session, name="wo", workspace=None)
            forward = compile_forward(lambda: operation(values))
            forward()
            torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        outputs.append(output)
    # The padded group's activations are arbitrary; its zero weights must remove them exactly.
    # Its quantization blocks and products are separate from the checkpoint groups', and adding
    # exact zeros preserves every accumulation, so both shapes produce identical bits.
    assert torch.equal(outputs[1], outputs[0])
