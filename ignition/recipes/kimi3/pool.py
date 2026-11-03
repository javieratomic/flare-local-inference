# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""KDA continuation rows and paged, unrotated E4M3 MLA latent records."""

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.recurrent import DeltaState
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS
from .dcp import RECORD_BYTES


def page_specs(config):
    """Per-rank storage for one page of target latent records."""
    # E4M3 records with a unit scale, as vLLM's Kimi K3 serving stores its latent cache (--kv-cache-dtype fp8).
    for index, kind in enumerate(config.layers):
        if kind == "mla":
            yield f"layer{index}.latent", (PAGE_TOKENS * RECORD_BYTES,)


def page_bytes(config):
    return sum(shape[0] for _, shape in page_specs(config))


def continuation_layout(config, world_size, *, speculative_tokens=0):
    """One continuation row; speculation widens each KDA convolution history by the verified drafts, DSpark
    adds every draft layer's key and value window, so restored
    continuations resume drafting. A copy of each page region holds a snapshot's partial page."""
    heads = config.text["linear_attn_config"]["num_heads"] // world_size
    regions = []
    for index, kind in enumerate(config.layers):
        if kind == "kda":
            regions.extend(((f"layer{index}.convolution", (3 * heads * 128, 3 + speculative_tokens), torch.bfloat16),
                            (f"layer{index}.recurrent", (heads, 128, 128), torch.float32)))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config))
    if speculative_tokens and config.draft is not None:
        d = config.draft
        for layer in range(d.layers):
            for kind in ("key", "value"):
                regions.append((f"draft{layer}.{kind}", (d.window, d.kv_heads // world_size, d.head_dim),
                                torch.bfloat16))
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(cls, config, world_size, slots, pages, device, *, speculative_tokens=0, **kwargs):
        return cls.allocate(continuation_layout(config, world_size, speculative_tokens=speculative_tokens),
                            page_specs(config), slots, pages, device, **kwargs)

    def recurrent(self, layer):
        return DeltaState(self.reserved_region(f"layer{layer}.convolution"), self.reserved_region(f"layer{layer}.recurrent"))

    def attention(self, layer):
        return self.pages[f"layer{layer}.latent"].view(torch.float8_e4m3fn).view(-1, PAGE_TOKENS, RECORD_BYTES)

    def share_pages(self, storage, group):
        """Every rank's address of each MLA layer's page pool, {layer: [world] int64}, mapped through CUDA IPC.

        Context-parallel prefill reads earlier positions from the ranks that own them.
        """
        names = [name for name in self.pages if name.startswith("layer") and name.endswith(".latent")]
        regions = {region.address: region for region in storage.dynamic_regions}
        addresses = storage.peer_storage.share_regions(group, [regions[self.pages[name].data_ptr()] for name in names])
        return {int(name.split(".")[0].removeprefix("layer")): torch.tensor(pointers, dtype=torch.int64,
                device=self.backing.device) for name, pointers in zip(names, addresses, strict=True)}
