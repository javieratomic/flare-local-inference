"""Column gathering retains FP8 checkpoint bytes and square-block scale values."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from ignition.components.block_fp8_weights import BlockFP8WeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.reference.qwen38_quant import decode_mxfp8


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("block", [32, 128])
@torch.no_grad()
def test_interleaved_projection_columns_preserve_values_and_scales(tmp_path, block):
    generator = torch.Generator().manual_seed(41017)
    weight = torch.randn((256, 6144), generator=generator).to(torch.float8_e4m3fn)
    scales = torch.randint(123, 130, (256 // block, 6144 // block), generator=generator, dtype=torch.uint8)
    tensors = {"projection.weight": weight, "projection.scale": scales.view(torch.float8_e8m0fnu)}
    save_file(tensors, str(tmp_path / "weights.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(value.nbytes for value in tensors.values())},
        "weight_map": {name: "weights.safetensors" for name in tensors}}))
    (tmp_path / "config.json").write_text("{}")
    manifest = inspect_indexed_checkpoint(tmp_path)
    loader = BlockFP8WeightLoader(manifest, "cuda:0")
    loader.config = SimpleNamespace(quantization={"weight_block_size": [block, block]})
    expanded = scales.repeat_interleave(block, 0).repeat_interleave(block // 32, 1)
    reference = decode_mxfp8(weight, expanded)
    source = torch.randn((9, 6144), generator=generator)
    partials = []
    with loader:
        for rank in range(4):
            ranges = [(head * 256, (head + 1) * 256) for head in range(rank, 24, 4)]
            columns = torch.tensor([column for first, last in ranges for column in range(first, last)])
            packed = loader.load_linear_columns("projection", ranges)
            actual = decode_mxfp8(packed.weight.values.cpu(), packed.weight.scale_rows[0].view(torch.uint8).cpu())
            torch.testing.assert_close(actual, reference[:, columns], rtol=0, atol=0)
            assert torch.equal(packed.weight.values.cpu().view(torch.uint8), weight.view(torch.uint8)[:, columns])
            assert torch.equal(packed.weight.scale_rows[0].cpu().view(torch.uint8),
                               expanded[:, columns[::32] // 32])
            partials.append(source[:, columns] @ actual.T)
    torch.testing.assert_close(sum(partials), source @ reference.T, rtol=2e-5, atol=2e-3)


def _checkpoint(tmp_path, tensors):
    save_file(tensors, str(tmp_path / "weights.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(value.nbytes for value in tensors.values())},
        "weight_map": {name: "weights.safetensors" for name in tensors}}))
    (tmp_path / "config.json").write_text("{}")
    return inspect_indexed_checkpoint(tmp_path)


@torch.no_grad()
def test_padded_ranges_append_inert_rows_and_columns_to_checkpoint_values(tmp_path):
    from ignition.components.parallel import TensorParallelShard

    generator = torch.Generator().manual_seed(41018)
    # Eight 32-row output groups pad to nine across three ranks, as DeepSeek's grouped output does.
    weight = torch.randn((256, 256), generator=generator).to(torch.float8_e4m3fn)
    scales = torch.randint(118, 132, (256 // 32, 256 // 32), generator=generator, dtype=torch.uint8)
    head = torch.randn((200, 96), generator=generator).bfloat16()
    sink = torch.randn(64, generator=generator)
    manifest = _checkpoint(tmp_path, {"projection.weight": weight, "projection.scale": scales.view(torch.float8_e8m0fnu),
                                      "head.weight": head, "sink": sink})
    loader = BlockFP8WeightLoader(manifest, "cuda:0")
    loader.config = SimpleNamespace(quantization={"weight_block_size": [32, 32]})
    expanded = scales.repeat_interleave(32, 0)
    reference = decode_mxfp8(weight, expanded)
    source = torch.randn((5, 256), generator=generator)
    rows, partials = [], []
    with loader:
        for rank in range(3):
            shard = TensorParallelShard(rank, 3)
            first, last = shard.padded_partition(256, alignment=32)
            assert last - first == 96
            packed = loader.load_padded_linear("projection", rows=(first, last))
            values = packed.weight.values.cpu()
            unit = packed.weight.scale_rows[0].view(torch.uint8).cpu()
            stored = max(0, min(last, 256) - first)
            assert torch.equal(values[:stored].view(torch.uint8), weight[first:first + stored].view(torch.uint8))
            assert torch.equal(values[stored:].view(torch.uint8), torch.zeros_like(values[stored:].view(torch.uint8)))
            assert torch.equal(unit[:stored], expanded[first:first + stored])
            assert unit[stored:].eq(127).all()
            rows.append(decode_mxfp8(values, unit))
            # Column padding feeds a row-parallel projection whose padded inputs are arbitrary;
            # b12x packing pads K to 128 beyond the 96 logical columns.
            first, last = shard.padded_partition(256, alignment=32)
            columns = loader.load_padded_linear("projection", columns=(first, last))
            column_values = decode_mxfp8(columns.weight.values.cpu(),
                                         columns.weight.scale_rows[0].view(torch.uint8).cpu())[:, :96]
            stored = max(0, min(last, 256) - first)
            assert column_values[:, stored:].eq(0).all()
            padded_source = torch.cat((source, torch.randn((5, 32), generator=generator)), dim=1)
            partials.append(padded_source[:, first:last] @ column_values.T)
            head_rows = shard.padded_partition(200, alignment=8)
            padded_head = loader.load_padded_tensor("head.weight", head_rows).cpu()
            assert padded_head.shape == (head_rows[1] - head_rows[0], 96)
            stored = max(0, min(head_rows[1], 200) - head_rows[0])
            assert torch.equal(padded_head[:stored], head[head_rows[0]:head_rows[0] + stored])
            assert padded_head[stored:].eq(0).all()
            sink_rows = shard.padded_partition(64, alignment=8)
            padded_sink = loader.load_padded_tensor("sink", sink_rows, fill=-1.5).cpu()
            stored = max(0, min(sink_rows[1], 64) - sink_rows[0])
            assert torch.equal(padded_sink[:stored], sink[sink_rows[0]:sink_rows[0] + stored])
            assert padded_sink[stored:].eq(-1.5).all()
        with pytest.raises(ValueError):
            loader.load_padded_linear("projection", rows=(16, 48))
        with pytest.raises(ValueError):
            loader.load_linear("projection", rows=(192, 288))
    padded = torch.cat(rows)
    torch.testing.assert_close(padded[:256], reference, rtol=0, atol=0)
    assert padded[256:].eq(0).all()
    torch.testing.assert_close(sum(partials), source @ reference.T, rtol=2e-5, atol=2e-3)
