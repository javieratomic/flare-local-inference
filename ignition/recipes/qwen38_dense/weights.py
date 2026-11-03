# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical dense decoder parameters and checkpoint-calibrated projections."""

from dataclasses import dataclass, replace

import torch

from ignition.components.model import canonical_tree
from ignition.components.parallel import TensorParallelShard
from ignition.components.tensor_fp8 import TensorFP8Weight
from ignition.components.weights import DirectWeightLoader

from .config import ModelConfig, TEXT_PREFIX


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    tree: torch.nn.Module
    projections: dict
    head: object
    allocated_bytes: int
    shard: TensorParallelShard = TensorParallelShard(0, 1)
    draft: object = None

    @classmethod
    def load(cls, manifest, device, *, progress=None, mtp=False, draft_snapshot=None):
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

        config = ModelConfig.load(manifest.snapshot)
        if mtp:
            from .draft_config import DraftConfig
            config = replace(config, draft=DraftConfig.load(draft_snapshot, config))
        tree = canonical_tree(Qwen3_5TextModel, Qwen3_5TextConfig.from_dict(config.text))
        projections = {}
        before = torch.cuda.memory_allocated(device)
        with DirectWeightLoader(manifest, device, prefixes=(TEXT_PREFIX + ".", "lm_head."),
                                progress=progress) as loader:
            for name, parameter in tree.named_parameters():
                prefix = TEXT_PREFIX + "." + name.removesuffix(".weight")
                if name.endswith(".weight") and prefix in config.quantization:
                    weight = loader.load_linear(prefix)
                    packed = weight.packed if isinstance(weight, TensorFP8Weight) else weight
                    if (packed.out_features, packed.in_features) != tuple(parameter.shape):
                        raise ValueError(f"{prefix}: checkpoint projection shape differs from the model")
                    projections[name.removesuffix(".weight")] = weight
                    continue
                tensor = loader.load_tensor(TEXT_PREFIX + "." + name)
                if tensor.shape != parameter.shape:
                    raise ValueError(f"{name}: checkpoint parameter shape differs from the model")
                parent, _, field = name.rpartition(".")
                setattr(tree.get_submodule(parent), field, torch.nn.Parameter(tensor, requires_grad=False))
            head = loader.load_linear("lm_head")
            if (head.out_features, head.in_features) != (config.vocabulary, config.hidden):
                raise ValueError("Qwen vocabulary projection shape differs from the model")
        for index, layer in enumerate(tree.layers):
            if layer.block_type == "linear_attention":
                module = layer.linear_attn
                projections[f"layers.{index}.linear_attn.in_proj_ba"] = torch.cat(
                    (module.in_proj_b.weight, module.in_proj_a.weight))
        draft = None
        if mtp:
            from ignition.components.dflash2_weights import DraftWeights
            draft = DraftWeights.load(draft_snapshot, config.draft, device, TensorParallelShard(0, 1), progress=progress)
        return cls(config, tree, projections, head, torch.cuda.memory_allocated(device) - before, draft=draft)
