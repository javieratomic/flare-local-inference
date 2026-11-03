# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-27B DFlash2 geometry and target compatibility."""

from dataclasses import dataclass
import json
from pathlib import Path

from ignition.components.dflash2_geometry import DraftAttentionGeometry

REPOSITORY = "incoai/Qwen3.8-27B-DFlash2"
REVISION = "015e795645c74b1a0eeef3b570031fb62e769bc5"


@dataclass(frozen=True)
class DraftConfig:
    text: dict

    @classmethod
    def load(cls, snapshot, target):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()), target)

    @classmethod
    def from_dict(cls, text, target):
        required = dict(model_type="qwen3", architectures=["DFlash2DraftModel"], hidden_size=5120,
            intermediate_size=17408, num_hidden_layers=5, num_attention_heads=32, num_key_value_heads=8,
            head_dim=128, vocab_size=248320, num_target_layers=64, hidden_act="silu", dtype="bfloat16",
            rms_norm_eps=1e-6, attention_bias=False, attention_dropout=0., tie_word_embeddings=False,
            sliding_window=2048, use_sliding_window=True, is_causal=False,
            layer_types=["sliding_attention"] * 5,
            rope_parameters=dict(rope_type="default", rope_theta=10000000))
        if any(text.get(name) != value for name, value in required.items()):
            raise ValueError("unsupported Qwen3.8 DFlash2 decoder geometry or arithmetic")
        options = dict(block_size=8, conv_kernel_size=2, conv_group_size=16, selector_rank=256,
            selector_top_k=16, mask_token_id=248070, target_layer_ids=[5, 19, 33, 47, 61])
        if any(text.get("dflash_config", {}).get(name) != value for name, value in options.items()):
            raise ValueError("unsupported Qwen3.8 DFlash2 projector, selector, or context geometry")
        for name, expected in (("output_multiplier", 1.), ("final_logit_softcapping", None)):
            if text.get(name, expected) != expected or text["dflash_config"].get(name, expected) != expected:
                raise ValueError("unsupported Qwen3.8 DFlash2 output transform")
        if (text["hidden_size"] != target.hidden or text["vocab_size"] != target.vocabulary
                or text["num_target_layers"] != len(target.layers)):
            raise ValueError("Qwen3.8 DFlash2 draft does not match the target model")
        return cls(text)

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def target_layers(self):
        return tuple(self.text["dflash_config"]["target_layer_ids"])

    @property
    def block_size(self):
        return self.text["dflash_config"]["block_size"]

    @property
    def noise_token(self):
        return self.text["dflash_config"]["mask_token_id"]

    @property
    def attention(self):
        c = self.text
        return DraftAttentionGeometry(c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"],
            c["head_dim"], c["head_dim"], c["rope_parameters"]["rope_theta"], c["sliding_window"])

    def validate_shard(self, shard):
        self.attention.local(shard)
        shard.partition(self.text["hidden_size"], alignment=64)
        shard.partition(self.text["intermediate_size"], alignment=128)
