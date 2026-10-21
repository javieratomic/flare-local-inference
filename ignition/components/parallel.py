"""Explicit tensor partitions for single-rank and b12x tensor-parallel recipes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorParallelShard:
    rank: int
    world_size: int

    def __post_init__(self):
        if (type(self.world_size) is not int or self.world_size < 1
                or type(self.rank) is not int or not 0 <= self.rank < self.world_size):
            raise ValueError("tensor parallelism requires a positive world size and a rank within it")

    def partition(self, extent: int, *, alignment: int = 1) -> tuple[int, int]:
        if (type(extent) is not int or extent <= 0 or type(alignment) is not int
                or alignment <= 0 or extent % (self.world_size * alignment)):
            raise ValueError("tensor extent must divide into aligned rank partitions")
        width = extent // self.world_size
        return self.rank * width, (self.rank + 1) * width

    def padded_width(self, extent: int, *, alignment: int = 1) -> int:
        """Return each rank's width after padding the axis to whole aligned units per rank.

        Ranks keep identical shapes, so compiled forwards and rank zero's kernel
        selections remain valid on every rank. The width equals `partition`'s
        when the extent already divides into aligned rank partitions.
        """
        if (type(extent) is not int or extent <= 0 or type(alignment) is not int or alignment <= 0):
            raise ValueError("padded tensor extent and alignment must be positive integers")
        units = -(-extent // (self.world_size * alignment))
        return units * alignment

    def padded_partition(self, extent: int, *, alignment: int = 1) -> tuple[int, int]:
        """Partition an axis padded to equal aligned rank widths.

        The range may extend past `extent`. Loaders fill the positions beyond the
        checkpoint with values that leave the model's result unchanged, so a
        padded alignment unit, such as an attention head group, is never split
        between checkpoint data and padding unless the extent itself does so.
        """
        width = self.padded_width(extent, alignment=alignment)
        return self.rank * width, (self.rank + 1) * width

    def fused_partitions(self, extents: tuple[int, ...], *, alignment: int = 1) -> tuple[tuple[int, int], ...]:
        """Slice each independently partitioned projection inside a fused axis."""
        offset, ranges = 0, []
        for extent in extents:
            first, last = self.partition(extent, alignment=alignment)
            ranges.append((offset + first, offset + last))
            offset += extent
        return tuple(ranges)

    def replicated_partition(self, extent: int) -> tuple[int, int]:
        """Partition an axis, replicating each element when ranks outnumber it.

        Grouped-query attention uses this for KV heads: contiguous rank groups
        share a KV head while each rank owns distinct query heads.
        """
        if type(extent) is not int or extent <= 0:
            raise ValueError("replicated extent must be a positive integer")
        if extent >= self.world_size:
            return self.partition(extent)
        if self.world_size % extent:
            raise ValueError("replicated extent must divide the rank count")
        first = self.rank // (self.world_size // extent)
        return first, first + 1
