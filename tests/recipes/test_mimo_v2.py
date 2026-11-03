# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiMo partitions retain each checkpoint row's original quantization scale."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.mimo_v2.config import ModelConfig
from ignition.recipes.mimo_v2.weights import MiMoWeightLoader


def config():
    return ModelConfig.from_dict({
        "model_type": "mimo_v2", "architectures": ["MiMoV2ForCausalLM"],
        "attention_projection_layout": "fused_qkv", "attention_bias": False,
        "hidden_size": 128, "hidden_act": "silu", "num_hidden_layers": 2,
        "hybrid_layer_pattern": [0, 1], "moe_layer_freq": [0, 1],
        "num_attention_heads": 128, "num_key_value_heads": 8, "head_dim": 192,
        "v_head_dim": 128, "swa_num_attention_heads": 128, "swa_num_key_value_heads": 8,
        "swa_head_dim": 192, "swa_v_head_dim": 128, "partial_rotary_factor": .334,
        "rope_theta": 10000000, "swa_rope_theta": 10000, "sliding_window_size": 128,
        "add_swa_attention_sink_bias": True, "n_shared_experts": None,
        "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 1, "topk_group": 1,
        "quantization_config": {"quant_method": "fp8", "store_dtype": "mxfp4",
            "weight_block_size": [128, 128], "mxfp4_block_size": 32,
            "activation_scheme": "dynamic", "fmt": "e4m3",
            "ignored_layers": ["model.layers.0.self_attn.o_proj"]},
    }, checkpoint_ranks=8)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_qkv_keeps_training_chunk_padding_and_scale_ownership(world):
    c = config()
    # Every training rank stores 3392 weight rows and 27 scale blocks (3456 padded rows).
    source = torch.ones((27136, 128), dtype=torch.float8_e4m3fn)
    scales = torch.arange(1, 217, dtype=torch.float32).reshape(216, 1)
    restored = []
    for rank in range(world):
        loader = object.__new__(MiMoWeightLoader)
        loader.config, loader.shard, loader.device = c, TensorParallelShard(rank, world), "cpu"
        loader._session = SimpleNamespace(flush=lambda: None)
        loader._sources = {"model.layers.0.self_attn.qkv_proj.weight": source,
                           "model.layers.0.self_attn.qkv_proj.weight_scale_inv": scales}
        loader._copy = lambda target, value: target.copy_(value)
        result = loader.load_qkv(0)
        chunks = c.qkv_chunks(0, loader.shard)
        offset = 0
        for chunk in chunks:
            n = chunk.weight_rows[1] - chunk.weight_rows[0]
            values = result.values[offset:offset + chunk.padded_rows].float()
            scale = result.scales[offset // 128:(offset + chunk.padded_rows) // 128].repeat_interleave(128, 0)
            restored.append((values * scale)[:n])
            assert torch.count_nonzero(values[n:]) == 0
            assert (chunk.query_rows, chunk.key_rows, chunk.value_rows) == (3072, 192, 128)
            offset += chunk.padded_rows
    reference = torch.cat([scales[i * 27:(i + 1) * 27].repeat_interleave(128, 0)[:3392]
                           for i in range(8)]).expand(-1, 128)
    torch.testing.assert_close(torch.cat(restored), reference, rtol=0, atol=0)


def test_format_follows_metadata():
    c = config()
    assert c.projection_format("model.layers.0.self_attn.o_proj") == "bf16"
    assert c.projection_format("model.layers.1.self_attn.o_proj") == "block_fp8"
    assert c.projection_format("model.layers.1.mlp.experts.0.gate_proj") == "mxfp4"


def test_attention_schedule_has_independent_rotary_and_window_geometry():
    global_attention, window = config().attention(0), config().attention(1)
    assert (global_attention.window, global_attention.theta, global_attention.sinks) == (-1, 10000000, False)
    assert (window.window, window.theta, window.sinks) == (128, 10000, True)
    assert global_attention.rotary_dim == window.rotary_dim == 64
    assert global_attention.local(TensorParallelShard(7, 8)) == (16, 1)


def test_checkpoint_chunk_geometry_rejects_inexact_sharding():
    with pytest.raises(ValueError, match="checkpoint's fused-QKV shard count"):
        config().qkv_chunks(0, TensorParallelShard(0, 3))
    with pytest.raises(ValueError, match="checkpoint's fused-QKV shard count"):
        config().qkv_chunks(0, TensorParallelShard(0, 16))
