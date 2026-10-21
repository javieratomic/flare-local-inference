"""Lossless expansion of square-block UE8M0 scales into native MXFP8 rows."""

import torch

from ignition.components.weights import DirectWeightLoader, _aligned_range, _padded_range
from ignition.preparation.tensor_metadata import record_numeric_tensor


# UE8M0 encodes 2**(code - 127); padded blocks hold FP8 zeros under a unit scale.
_UNIT_SCALE = 127


class BlockFP8WeightLoader(DirectWeightLoader):
    @property
    def block_size(self):
        shape = self.config.quantization["weight_block_size"]
        if shape not in ([32, 32], [128, 128]):
            raise ValueError("native FP8 loading requires square block-32 or block-128 scales")
        return shape[0]

    def load_linear_group(self, prefixes, *, row_ranges=None, column_range=None):
        prefixes = tuple(prefixes)
        ranges = (None,) * len(prefixes) if row_ranges is None else row_ranges
        parts = [
            self._dense_blocks(prefix, rows=rows, columns=column_range)
            for prefix, rows in zip(prefixes, ranges, strict=True)
        ]
        return self._pack_blocks(parts, dim=0)

    def load_padded_linear(self, prefix, *, rows=None, columns=None):
        """Load one projection whose block-aligned range may extend past the checkpoint.

        Positions beyond the checkpoint hold FP8 zeros with unit scales, so padded
        output rows are exactly zero and padded input columns contribute nothing.
        """
        return self._pack_blocks([self._dense_blocks(prefix, rows=rows, columns=columns, padded=True)], dim=0)

    def load_linear_columns(self, prefix, column_ranges, *, rows=None):
        """Gather aligned input-column ranges without changing checkpoint values or scales."""
        parts = [self._dense_blocks(prefix, rows=rows, columns=columns) for columns in column_ranges]
        if not parts:
            raise ValueError("a projection requires at least one input-column range")
        return self._pack_blocks(parts, dim=1)

    def _pack_blocks(self, parts, *, dim):
        from b12x.gemm.blockscaled import pack_weight

        self._session.flush()
        values = torch.cat([p[0] for p in parts], dim=dim) if len(parts) > 1 else parts[0][0]
        scales = torch.cat(
            [
                p[1].repeat_interleave(self.block_size, dim=0).repeat_interleave(self.block_size // 32, dim=1)
                for p in parts
            ], dim=dim,
        )
        return pack_weight(values, scales, recipe="mxfp8")

    def _dense_blocks(self, prefix, *, rows=None, columns=None, padded=False):
        self._require_open()
        weight, scale = self._sources[f"{prefix}.weight"], self._sources[f"{prefix}.scale"]
        block = self.block_size
        if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
            raise ValueError(f"{prefix}: expected native FP8 dense weights")
        n, k = weight.shape
        if scale.dtype != torch.float8_e8m0fnu or scale.shape != (
            (n + block - 1) // block,
            (k + block - 1) // block,
        ):
            raise ValueError(f"{prefix}: expected UE8M0 scales for {block}-by-{block} blocks")
        select = _padded_range if padded else _aligned_range
        r0, r1 = select(rows, n, block)
        c0, c1 = select(columns, k, block)
        # Checkpoint rows and columns precede any padding within the selected range.
        stored_rows, stored_columns = max(0, min(r1, n) - r0), max(0, min(c1, k) - c0)
        if (stored_rows, stored_columns) == (r1 - r0, c1 - c0):
            values = torch.empty((r1 - r0, c1 - c0), dtype=weight.dtype, device=self.device)
            scales = torch.empty(((r1 - r0) // block, (c1 - c0) // block), dtype=torch.uint8, device=self.device)
        else:
            values = torch.zeros((r1 - r0, c1 - c0), dtype=weight.dtype, device=self.device)
            scales = torch.full(((r1 - r0) // block, (c1 - c0) // block), _UNIT_SCALE,
                                dtype=torch.uint8, device=self.device)
            record_numeric_tensor(values)
            record_numeric_tensor(scales)
        if stored_rows and stored_columns:
            self._copy(values[:stored_rows, :stored_columns], weight[r0:r0 + stored_rows, c0:c0 + stored_columns])
            self._copy(scales[:stored_rows // block, :stored_columns // block],
                       scale[r0 // block:(r0 + stored_rows) // block,
                             c0 // block:(c0 + stored_columns) // block].view(torch.uint8))
        return values, scales

    def load_attention_output(self, prefix, groups, *, padded=False):
        """Pack whole low-rank output groups; padded groups are zero in both projections."""
        from b12x.gemm import wo_projection

        c = self.config.text
        first, last = groups
        if (type(first) is not int or type(last) is not int or first % c["o_lora_rank"]
                or last % c["o_lora_rank"]):
            raise ValueError("attention output ranges must hold whole low-rank groups")
        a, sa = self._dense_blocks(prefix + ".wo_a", rows=groups, padded=padded)
        b, sb = self._dense_blocks(prefix + ".wo_b", columns=groups, padded=padded)
        self._session.flush()
        return wo_projection.pack_weights(
            a,
            sa.view(torch.float8_e8m0fnu),
            b,
            sb.view(torch.float8_e8m0fnu),
            groups=(last - first) // c["o_lora_rank"],
            group_width=c["num_attention_heads"] * c["head_dim"] // c["o_groups"],
            rank=c["o_lora_rank"],
            hidden=c["hidden_size"],
            block_size=(self.block_size, self.block_size),
        )
