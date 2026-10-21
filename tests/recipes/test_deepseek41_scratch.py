"""Shared activation storage preserves persistent selections and concurrent expert outputs."""

import pytest
import torch

from ignition.preparation.disposable import storage_ranges
from ignition.preparation.scratch import GraphBufferPool
from ignition.recipes.deepseek41.bootstrap import PreparationConfig
from ignition.recipes.deepseek41.decoder import Scratch


def _overlap(left, right):
    return max(left.data_ptr(), right.data_ptr()) < min(
        left.data_ptr() + left.nbytes, right.data_ptr() + right.nbytes)


def test_serial_stages_share_storage_without_overwriting_live_residuals_or_selections():
    scratch = Scratch.allocate(128, 1, 4, "cpu", candidate_tokens=16)
    retained = (scratch.residual, scratch.engram_residual, scratch.attention_hc.residual,
                scratch.ffn_hc.residual, scratch.attention.selected, scratch.attention.candidates,
                scratch.attention.candidate_counts, scratch.mlp_output)
    concurrent = (scratch.shared.gate_up, scratch.shared.intermediate, scratch.shared.output,
                  scratch.routed, scratch.mlp_partial)
    for index, value in enumerate(retained):
        value.fill_(index + 1)
    for index, value in enumerate(concurrent):
        assert all(not _overlap(value, other) for other in (*retained, *concurrent[:index]))
    assert any(_overlap(value, other) for value in concurrent
               for other in (scratch.embedding, scratch.engram.gathered, scratch.engram.projected))
    engram = (scratch.engram.gathered, scratch.engram.local, scratch.engram.partial, scratch.engram.projected)
    for index, value in enumerate(engram):
        assert all(not _overlap(value, other) for other in engram[:index])
    for value in (scratch.embedding, *engram, *concurrent):
        value.fill_(42)
    for index, value in enumerate(retained):
        assert value.eq(index + 1).all()
    assert scratch.rows(64, 1).attention.candidates.shape == (16, 16384)
    assert scratch.rows(8, 1).attention.candidates.shape == (8, 16384)


def test_suffix_destinations_and_variant_private_buffers_preserve_activation_storage():
    pool = GraphBufferPool("cpu", chunk_bytes=2 << 20)
    prefill = Scratch.allocate(128, 1, 4, "cpu", buffers=pool.cursor())
    suffix = Scratch.allocate(32, 1, 4, "cpu", buffers=prefill.buffers)
    pairs = [(getattr(prefill.attention_hc, name), getattr(suffix.attention_hc, name))
             for name in ("normalized", "residual", "post", "comb")]
    pairs.append((prefill.attn_mix, suffix.attn_mix))
    for source, destination in pairs:
        assert not _overlap(source, destination)
        source.fill_(7)
        destination.fill_(9)
        assert source.eq(7).all()
    first = prefill.rows(128, 1)
    private = first.buffers.zeros((128, 512), torch.bfloat16)
    capacity = pool.nbytes
    second = prefill.rows(32, 1)
    reused = second.buffers.zeros((128, 512), torch.bfloat16)
    assert reused.data_ptr() == private.data_ptr()
    reused.fill_(33)
    assert private[:32].eq(33).all()
    for source, destination in pairs:
        assert source.eq(7).all() and destination.eq(9).all()
    assert pool.nbytes == capacity
    assert storage_ranges((prefill, suffix, first, second)) == storage_ranges(pool)


@pytest.mark.parametrize("capacity,dspark,draft_width,expected", [
    (32, True, 6, 160), (32, False, 1, 32), (1, True, 6, 6),
])
def test_oneshot_capacity_excludes_prefills_assigned_to_dma(capacity, dspark, draft_width, expected):
    config = PreparationConfig(max_active=capacity, dspark=dspark,
        speculative_widths=tuple(range(1, draft_width + 1)), speculative_token_capacity=64)
    assert config.all_reduce_tokens == expected


def test_short_prefill_still_fits_the_oneshot_channel():
    config = PreparationConfig(max_active=1, dspark=False, prefill_chunk_size=64)
    assert config.all_reduce_tokens == 64


@pytest.mark.parametrize("world,heads,groups,vocabulary", [(4, 16, 2, 32320), (3, 24, 3, 43136)])
def test_rank_scratch_holds_whole_padded_head_groups_and_vocabulary(world, heads, groups, vocabulary):
    scratch = Scratch.allocate(64, 6, world, "cpu", candidate_tokens=16)
    assert scratch.attention.query.shape == scratch.attention.attended.shape == (64, heads, 512)
    assert scratch.attention.attended.view(64, groups, 4096).shape == (64, groups, 4096)
    assert scratch.local_logits.shape == (6, vocabulary) and scratch.logits.shape == (6, 129280)
    assert vocabulary * world >= 129280 and vocabulary % 64 == 0
    # Expert intermediates divide exactly: 2304 / 3 = 768 and 2304 / 4 = 576.
    assert scratch.shared.intermediate.shape[1] == 2304 // world
