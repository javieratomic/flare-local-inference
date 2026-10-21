"""Three padded attention ranks reproduce the unpartitioned checkpoint attention."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.components.ring_commit import PreparedRingCommit
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek41.attention import AttentionScratch, PreparedAttention, Rotary
from ignition.recipes.deepseek41.config import RankGeometry
from ignition.recipes.deepseek41.pool import ContinuationPool
from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1"
    or not os.environ.get("FLARE_TEST_DS41_CHECKPOINT"), reason="requires an assigned GPU and a DS41 checkpoint")

LAYER = 0
# One bfloat16 rounding step. Partitioning changes projection shapes, hence their tuned
# accumulation order, and each rank rounds its partial output to bfloat16.
BF16_STEP = 2. ** -8


def _load(manifest, device, world, rank):
    shard = TensorParallelShard(rank, world)
    prefix = f"layers.{LAYER}.attn"
    with DeepSeekWeightLoader(manifest, device, shard=shard, prefixes=(prefix + ".",)) as loader:
        heads = shard.padded_partition(64 * 512, alignment=8 * 512)
        groups = shard.padded_partition(8 * 1024, alignment=1024)
        weights = dict(qkv_a=loader.load_linear_group((prefix + ".wq_a", prefix + ".wkv")),
            q_b=loader.load_padded_linear(prefix + ".wq_b", rows=heads),
            output=loader.load_attention_output(prefix, groups, padded=True),
            q_norm=loader.load_tensor(prefix + ".q_norm.weight"),
            kv_norm=loader.load_tensor(prefix + ".kv_norm.weight"),
            sink=loader.load_padded_tensor(prefix + ".attn_sink", shard.padded_partition(64, alignment=8)))
        return weights, loader.config


def _partials(manifest, device, world, source, lengths):
    """Run each rank's prepared attention on shared inputs and return its partial outputs."""
    tokens, batch, width = sum(lengths), len(lengths), max(lengths)
    partials, attended = [], []
    for rank in range(world):
        weights, config = _load(manifest, device, world, rank)
        geometry = RankGeometry(world)
        assert weights["sink"].shape == (geometry.heads,) and weights["output"].groups == geometry.groups
        pool = ContinuationPool.allocate(config, batch + 1, batch * 2 + 1, device)
        metadata = Metadata.allocate(tokens, batch, 2, device, max_query_tokens=width)
        counts = torch.tensor(lengths, dtype=torch.int32, device=device)
        # Each request continues a 126-token history whose window rows are zero records.
        starts = torch.full((batch,), 126, dtype=torch.int64, device=device)
        states = torch.arange(1, batch + 1, dtype=torch.int64, device=device)
        pages = torch.arange(1, batch * 2 + 1, dtype=torch.int32, device=device).view(batch, 2)
        metadata.set(starts, counts, states, pages, torch.zeros((batch, width), dtype=torch.int64, device=device),
                     torch.tensor([batch], dtype=torch.int32, device=device))
        scratch = AttentionScratch.allocate(tokens, world, device)
        commit = PreparedRingCommit(pool.reserved_backing, [pool.layout.region(f"layer{LAYER}.window").offset],
            metadata.state_rows, metadata.starts, token_capacity=width, window_tokens=128, record_bytes=528)
        arena = ScratchArena(device, overlap=True)
        with RecordedPreparationSession(device=device, checkpoint=f"ds41-tp{world}-attention",
                workload_schema="padded-attention", numerical_policy="checkpoint", compile_workers=8) as session:
            session.register_scratch(arena)
            attention = PreparedAttention(LAYER, weights, source, scratch, metadata, pool, Rotary(config, device),
                mode="verify", world=world, session=session, name=f"tp{world}.rank{rank}", workspace=arena,
                window_staging=commit.staging[0], mla_compute_mode="bf16")
            session.tune(show_progress=False)
            pool.clear_for_preparation()
            attention(source)
            torch.cuda.synchronize()
        assert torch.isfinite(scratch.attended).all() and torch.isfinite(scratch.output).all()
        partials.append(scratch.output.float().clone())
        attended.append(scratch.attended.float().clone())
        del attention, session, weights, pool, scratch, arena, commit
        torch.cuda.empty_cache()
    return partials, torch.cat(attended, dim=1)


@pytest.mark.parametrize("lengths", [(1, 1, 1, 1), (6, 3, 1)])
@torch.no_grad()
def test_three_padded_ranks_match_unpartitioned_attention(lengths):
    torch._dynamo.reset()
    device = torch.device("cuda:0")
    manifest = inspect_indexed_checkpoint(Path(os.environ["FLARE_TEST_DS41_CHECKPOINT"]))
    generator = torch.Generator(device=device).manual_seed(41003)
    source = torch.randn((sum(lengths), 5120), dtype=torch.bfloat16, device=device, generator=generator) * .1
    reference, reference_heads = _partials(manifest, device, 1, source, lengths)
    reference = reference[0]

    def relative(value, expected):
        return ((value - expected).square().mean().sqrt() / expected.square().mean().sqrt()).item()

    deviations = {}
    for world in (4, 3):
        partials, heads = _partials(manifest, device, world, source, lengths)
        assert heads.shape[1] == RankGeometry(world).heads * world
        deviations[world] = relative(heads[:, :64], reference_heads), relative(sum(partials), reference)
    # BF16 MLA arithmetic pins the attention computation, so the padded three-rank partition
    # must reproduce the unpartitioned layer as closely as the qualified four-rank one does.
    for tp3, tp4 in zip(deviations[3], deviations[4], strict=True):
        assert tp3 <= tp4 + BF16_STEP, deviations
