"""Explicit recipe partitions preserve fused projections and grouped KV heads."""

import pytest

from ignition.components.parallel import TensorParallelShard


@pytest.mark.parametrize("world", (1, 2, 3, 4, 6, 8, 16))
def test_fused_projections_partition_each_component(world):
    shards = [TensorParallelShard(rank, world) for rank in range(world)]
    extents = (3072, 6144, 12288)
    partitions = [shard.fused_partitions(extents) for shard in shards]
    offset = 0
    for index, extent in enumerate(extents):
        assert [partitions[rank][index] for rank in range(world)] == [
            (offset + rank * extent // world, offset + (rank + 1) * extent // world)
            for rank in range(world)
        ]
        offset += extent


@pytest.mark.parametrize("world,expected", [(1, [(0, 2)]), (2, [(0, 1), (1, 2)]),
                                           (4, [(0, 1), (0, 1), (1, 2), (1, 2)])])
def test_query_rank_groups_share_kv_heads(world, expected):
    assert [TensorParallelShard(rank, world).replicated_partition(2) for rank in range(world)] == expected


def test_unrepresentable_partitions_fail_before_loading():
    with pytest.raises(ValueError):
        TensorParallelShard(0, 4).partition(640, alignment=128)
    with pytest.raises(ValueError):
        TensorParallelShard(0, 4).replicated_partition(3)
    assert TensorParallelShard(3, 4).partition(640, alignment=16) == (480, 640)


@pytest.mark.parametrize("world,rank", [(0, 0), (-1, 0), (True, 0), (8, -1), (8, 8), (8, True)])
def test_invalid_rank_identity(world, rank):
    with pytest.raises(ValueError):
        TensorParallelShard(rank, world)


@pytest.mark.parametrize("world", (3, 8, 16))
def test_replicated_kv_partitions_cover_query_rank_groups(world):
    for heads in (1, world):
        ranges = [TensorParallelShard(rank, world).replicated_partition(heads) for rank in range(world)]
        assert ranges == [(rank // (world // heads), rank // (world // heads) + 1)
                          for rank in range(world)]


@pytest.mark.parametrize("world", (1, 2, 4, 8))
@pytest.mark.parametrize("extent,alignment", [(64 * 512, 8 * 512), (8 * 1024, 1024), (64, 8), (2304, 32)])
def test_padded_partitions_match_exact_partitions_when_extents_divide(world, extent, alignment):
    for rank in range(world):
        shard = TensorParallelShard(rank, world)
        if extent % (world * alignment):
            continue
        assert shard.padded_partition(extent, alignment=alignment) == shard.partition(extent, alignment=alignment)
        assert shard.padded_width(extent, alignment=alignment) == extent // world


@pytest.mark.parametrize("extent,alignment,width", [
    # DeepSeek V4.1 query heads pad 64 -> 72 by whole eight-head output groups.
    (64 * 512, 8 * 512, 24 * 512), (64, 8, 24), (8 * 1024, 1024, 3 * 1024),
    # The vocabulary pads 129280 -> 129408 in 64-row blocks.
    (129280, 64, 43136),
    # Already divisible extents keep their exact widths.
    (2304, 32, 768), (6144, 256, 2048)])
def test_three_ranks_pad_by_whole_aligned_units(extent, alignment, width):
    shards = [TensorParallelShard(rank, 3) for rank in range(3)]
    ranges = [shard.padded_partition(extent, alignment=alignment) for shard in shards]
    assert ranges == [(rank * width, (rank + 1) * width) for rank in range(3)]
    assert all(shard.padded_width(extent, alignment=alignment) == width for shard in shards)
    assert width % alignment == 0 and 0 <= 3 * width - extent < 3 * alignment
    # Padding never displaces checkpoint positions: every real index keeps its rank-ordered place.
    covered = [index for first, last in ranges for index in range(first, min(last, extent))]
    assert covered == list(range(extent))


@pytest.mark.parametrize("extent,alignment", [(0, 1), (-8, 1), (8, 0), (8.0, 1), (True, 1), (8, True)])
def test_invalid_padded_extents_fail(extent, alignment):
    with pytest.raises(ValueError):
        TensorParallelShard(0, 3).padded_partition(extent, alignment=alignment)
    with pytest.raises(ValueError):
        TensorParallelShard(0, 3).padded_width(extent, alignment=alignment)
