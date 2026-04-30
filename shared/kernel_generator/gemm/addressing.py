# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Address computation for GEMM kernel data movement.

Translates tile-hierarchy positions into concrete LDS and global memory
offsets.  Each method operates on a ``TileContext``, reading named
bindings (thread indices, kernel arguments) and emitting stinkytofu
ALU instructions that compute the final address into a new binding.

The address formulas assume:
  - A is stored row-major (M-major):  A[m, k] at offset ``m * lda + k``
  - B is stored column-major when ``trans_b=True``:  B[k, n] at ``n * ldb + k``
  - D is stored row-major: D[m, n] at offset ``m * ldd + n``
  - LDS layout: A occupies ``[0, wg_m * unroll_k * elem_bytes)``,
    B starts at ``wg_m * unroll_k * elem_bytes``.

For the ``mfma_f32_16x16x16_f16`` instruction the lane-to-element
mapping within one MFMA tile is:
  - lane_row = lane_id % 16   (each lane owns one row)
  - lane_col = lane_id / 16   (4 groups of 16 lanes span 4 K-columns,
    but each group reads a packed pair so effective K=16)

All computations are expressed via the ``Embed`` transform so the index
math is inspectable even in dry-run mode.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .context import TileContext
from .problem import GemmProblem, TileConfig, MfmaConfig
from .transforms import Dim, Embed

__all__ = ["AddressComputer"]


@dataclass
class AddressComputer:
    """Emit address-computation instructions into a ``TileContext``.

    Instantiate once per kernel, then call individual methods at the
    appropriate point in codegen.

    Args:
        problem: The GEMM problem specification.
        tile:    Tile configuration (sizes, MFMA variant, etc.).
    """
    problem: GemmProblem
    tile: TileConfig

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _level_offset(index: int, stride: int) -> int:
        """Single tile-level contribution: index * stride."""
        return index * stride

    @staticmethod
    def _sum_levels(levels: list[tuple[int, int]]) -> int:
        """Sum tile-level contributions.

        Each entry is ``(index, stride)``; the result is
        ``sum(index * stride for index, stride in levels)``.

        This makes the hierarchical decomposition of row/col coordinates
        explicit: workgroup, wave, mfma-tile, and lane levels each
        contribute one ``(index, stride)`` pair.
        """
        return sum(AddressComputer._level_offset(idx, s) for idx, s in levels)


    @property
    def _elem(self) -> float:
        return self.problem.element_bytes

    @property
    def _cluster_k(self) -> int:
        """Number of K-elements each thread loads contiguously (vector width)."""
        return self.tile.vector_width

    @property
    def _cluster_m(self) -> int:
        """Number of threads along M for the global-load cluster."""
        return min(self.tile.block_size, self.tile.wg_m)

    @property
    def _cluster_n(self) -> int:
        return min(self.tile.block_size, self.tile.wg_n)

    @property
    def _lds_offset_b(self) -> int:
        """Byte offset where B's LDS region starts."""
        return int(self.tile.wg_m * self.tile.unroll_k * self._elem)

    # -- pure-Python offset calculators (for testing / dry run) -------------

    def global_load_thread_coords_a(self, tid: int) -> tuple:
        """Return ``(thread_row, thread_col)`` for A's global-load cluster."""
        ck = self._cluster_k
        cm = self._cluster_m
        row = (tid // ck) % cm
        col = tid % ck
        return row, col

    def global_load_thread_coords_b(self, tid: int) -> tuple:
        """Return ``(thread_row, thread_col)`` for B's global-load cluster."""
        cn = self._cluster_n
        ck = self._cluster_k
        row = (tid // ck) % cn
        col = tid % ck
        return row, col

    def lds_write_offset_a(self, tid: int) -> int:
        """LDS write byte-offset for thread *tid* writing A."""
        row, col = self.global_load_thread_coords_a(tid)
        offset = self._sum_levels([
            (row, self.tile.unroll_k),   # row in LDS
            (col, 1),                    # column (K)
        ])
        return int(offset * self._elem)

    def lds_write_offset_b(self, tid: int) -> int:
        """LDS write byte-offset for thread *tid* writing B."""
        row, col = self.global_load_thread_coords_b(tid)
        offset = self._sum_levels([
            (row, self.tile.unroll_k),
            (col, 1),
        ])
        return int(self._lds_offset_b + offset * self._elem)

    def lds_read_offset_a(self, wave_m: int, mfma_mi: int, ki: int,
                          lane_id: int) -> int:
        """LDS read byte-offset for one A MFMA operand element.

        For mfma_f32_16x16x16_f16:
          row = wave_m * m_per_wave + mfma_mi * mfma_m + (lane_id % mfma_m)
          col = ki * mfma_k + (lane_id // mfma_m) * vec_width_lds

        We simplify to just lane_id % 16 for row and use ki*k for the
        K offset, since the LDS read width packs K automatically.
        """
        mfma = self.tile.mfma
        row = self._sum_levels([
            (wave_m, self.tile.m_per_wave),  # wave level
            (mfma_mi, mfma.m),               # mfma tile level
            (lane_id % mfma.m, 1),           # lane level
        ])
        col = self._sum_levels([
            (ki, mfma.k),                    # unroll iteration
        ])
        offset = self._sum_levels([
            (row, self.tile.unroll_k),
            (col, 1),
        ])
        return int(offset * self._elem)

    def lds_read_offset_b(self, wave_n: int, mfma_ni: int, ki: int,
                          lane_id: int) -> int:
        """LDS read byte-offset for one B MFMA operand element."""
        mfma = self.tile.mfma
        row = self._sum_levels([
            (wave_n, self.tile.n_per_wave),
            (mfma_ni, mfma.n),
            (lane_id % mfma.n, 1),
        ])
        col = self._sum_levels([
            (ki, mfma.k),
        ])
        offset = self._sum_levels([
            (row, self.tile.unroll_k),
            (col, 1),
        ])
        return int(self._lds_offset_b + offset * self._elem)

    def global_store_offset_d(self, wg_m: int, wg_n: int,
                              wave_m: int, wave_n: int,
                              mfma_mi: int, mfma_ni: int,
                              lane_id: int, ldd: int) -> int:
        """Global byte-offset for storing one element of D."""
        mfma = self.tile.mfma
        row = self._sum_levels([
            (wg_m, self.tile.wg_m),         # workgroup level
            (wave_m, self.tile.m_per_wave),  # wave level
            (mfma_mi, mfma.m),              # mfma tile level
            (lane_id % mfma.m, 1),          # lane level
        ])
        col = self._sum_levels([
            (wg_n, self.tile.wg_n),
            (wave_n, self.tile.n_per_wave),
            (mfma_ni, mfma.n),
            (lane_id // mfma.m, 1),
        ])
        return int((row * ldd + col) * self._elem)

    # -- Embed transforms (for introspection) ------------------------------

    def lds_embed_a(self) -> Embed:
        """Embed transform for A's LDS layout: offset = row * unroll_k + col."""
        return Embed(
            [Dim("row", self.tile.wg_m), Dim("col", self.tile.unroll_k)],
            Dim("lds_a_offset", self.tile.wg_m * self.tile.unroll_k),
            [self.tile.unroll_k, 1],
        )

    def lds_embed_b(self) -> Embed:
        """Embed transform for B's LDS layout: offset = row * unroll_k + col."""
        return Embed(
            [Dim("row", self.tile.wg_n), Dim("col", self.tile.unroll_k)],
            Dim("lds_b_offset", self.tile.wg_n * self.tile.unroll_k),
            [self.tile.unroll_k, 1],
        )


    def global_d_embed(self, ldd: int) -> Embed:
        """Build Embed transform for D's global offset from tile-level decomposition.

        Maps ``(row, col)`` to a linearized element offset via
        ``offset = row * ldd + col``.  The row/col values are already
        the fully-resolved coordinates produced by ``_sum_levels`` in
        ``global_store_offset_d``.
        """
        total_m = self.tile.wg_m  # rows covered by one workgroup tile
        total_n = self.tile.wg_n  # cols covered by one workgroup tile
        return Embed(
            [Dim("row", total_m), Dim("col", total_n)],
            Dim("d_offset", total_m * ldd),
            [ldd, 1],
        )

    # -- instruction emission -----------------------------------------------

    def emit_global_load_addr_a(self, ctx: TileContext) -> None:
        """Emit instructions to compute v_addr_a for global loads of A.

        Computes the 64-bit global address of A's tile for this thread.
        Requires bindings: ``v_tid``, ``srd_A``, ``s_lda``.
        Produces binding: updates ``v_addr_a`` (already allocated).
        """
        if ctx.module is None:
            return
        import stinkytofu as st

        ck = self._cluster_k
        cm = self._cluster_m

        # Allocate temporaries (scoped)
        ctx.alloc_vgpr(1, "_tmp_row_a")
        ctx.alloc_vgpr(1, "_tmp_col_a")
        ctx.alloc_vgpr(1, "_tmp_off_a")

        # thread_row = (tid / cluster_k) % cluster_m
        ctx.module.add(st.VLShiftRightB32(
            ctx.vgpr("_tmp_row_a"), ctx.vgpr("v_tid"),
            st.Register(int(math.log2(ck))),
            comment=f"tid / {ck}",
        ))
        if cm < self.tile.block_size:
            ctx.module.add(st.VAndB32(
                ctx.vgpr("_tmp_row_a"), ctx.vgpr("_tmp_row_a"),
                st.Register(cm - 1),
                comment=f"% {cm}",
            ))

        # thread_col = tid % cluster_k
        ctx.module.add(st.VAndB32(
            ctx.vgpr("_tmp_col_a"), ctx.vgpr("v_tid"),
            st.Register(ck - 1),
            comment=f"tid % {ck}",
        ))

        # offset = thread_row * lda + thread_col (in elements)
        ctx.module.add(st.VMulLOU32(
            ctx.vgpr("_tmp_off_a"), ctx.vgpr("_tmp_row_a"),
            ctx.sgpr("s_lda"),
            comment="row * lda",
        ))
        ctx.module.add(st.VAddU32(
            ctx.vgpr("_tmp_off_a"), ctx.vgpr("_tmp_off_a"),
            ctx.vgpr("_tmp_col_a"),
            comment="+ col",
        ))

        # Convert to bytes: offset * element_bytes
        if self._elem == 2:
            ctx.module.add(st.VLShiftLeftB32(
                ctx.vgpr("_tmp_off_a"), ctx.vgpr("_tmp_off_a"),
                st.Register(1),
                comment="* 2 (f16 bytes)",
            ))
        elif self._elem == 4:
            ctx.module.add(st.VLShiftLeftB32(
                ctx.vgpr("_tmp_off_a"), ctx.vgpr("_tmp_off_a"),
                st.Register(2),
                comment="* 4 (f32 bytes)",
            ))

        # Add base pointer: v_addr_a = srd_A + byte_offset
        # v_addr_a[0] = srd_A[0] + offset  (low 32 bits)
        ctx.module.add(st.VAddU32(
            ctx.vgpr("v_addr_a", 0, 1),
            ctx.sgpr("srd_A", 0, 1),
            ctx.vgpr("_tmp_off_a"),
            comment="addr_A_lo = base_lo + offset",
        ))
        # v_addr_a[1] = srd_A[1] (high 32 bits, carry not handled for simplicity)
        ctx.module.add(st.VMovB32(
            ctx.vgpr("v_addr_a", 1, 1),
            ctx.sgpr("srd_A", 1, 1),
            comment="addr_A_hi = base_hi",
        ))

    def emit_global_load_addr_b(self, ctx: TileContext) -> None:
        """Emit instructions to compute v_addr_b for global loads of B.

        For trans_b=True (column-major B), B[k,n] is at n * ldb + k.
        Thread cluster: thread_row indexes N, thread_col indexes K.
        """
        if ctx.module is None:
            return
        import stinkytofu as st

        ck = self._cluster_k
        cn = self._cluster_n

        ctx.alloc_vgpr(1, "_tmp_row_b")
        ctx.alloc_vgpr(1, "_tmp_col_b")
        ctx.alloc_vgpr(1, "_tmp_off_b")

        # thread_row = (tid / cluster_k) % cluster_n
        ctx.module.add(st.VLShiftRightB32(
            ctx.vgpr("_tmp_row_b"), ctx.vgpr("v_tid"),
            st.Register(int(math.log2(ck))),
            comment=f"tid / {ck}",
        ))
        if cn < self.tile.block_size:
            ctx.module.add(st.VAndB32(
                ctx.vgpr("_tmp_row_b"), ctx.vgpr("_tmp_row_b"),
                st.Register(cn - 1),
                comment=f"% {cn}",
            ))

        # thread_col = tid % cluster_k
        ctx.module.add(st.VAndB32(
            ctx.vgpr("_tmp_col_b"), ctx.vgpr("v_tid"),
            st.Register(ck - 1),
            comment=f"tid % {ck}",
        ))

        # For trans_b: offset = thread_row * ldb + thread_col
        ctx.module.add(st.VMulLOU32(
            ctx.vgpr("_tmp_off_b"), ctx.vgpr("_tmp_row_b"),
            ctx.sgpr("s_ldb"),
            comment="row * ldb",
        ))
        ctx.module.add(st.VAddU32(
            ctx.vgpr("_tmp_off_b"), ctx.vgpr("_tmp_off_b"),
            ctx.vgpr("_tmp_col_b"),
            comment="+ col",
        ))

        # To bytes
        if self._elem == 2:
            ctx.module.add(st.VLShiftLeftB32(
                ctx.vgpr("_tmp_off_b"), ctx.vgpr("_tmp_off_b"),
                st.Register(1), comment="* 2 (f16)",
            ))
        elif self._elem == 4:
            ctx.module.add(st.VLShiftLeftB32(
                ctx.vgpr("_tmp_off_b"), ctx.vgpr("_tmp_off_b"),
                st.Register(2), comment="* 4 (f32)",
            ))

        ctx.module.add(st.VAddU32(
            ctx.vgpr("v_addr_b", 0, 1),
            ctx.sgpr("srd_B", 0, 1),
            ctx.vgpr("_tmp_off_b"),
            comment="addr_B_lo = base_lo + offset",
        ))
        ctx.module.add(st.VMovB32(
            ctx.vgpr("v_addr_b", 1, 1),
            ctx.sgpr("srd_B", 1, 1),
            comment="addr_B_hi = base_hi",
        ))

    def emit_lds_write_addr(self, ctx: TileContext) -> None:
        """Compute LDS write addresses for both A and B.

        After this, ``v_lds_write_a`` and ``v_lds_write_b`` contain
        the byte offsets into LDS where this thread writes its data.
        """
        if ctx.module is None:
            return
        import stinkytofu as st

        ck = self._cluster_k
        cm = self._cluster_m

        ctx.alloc_vgpr(1, "_tmp_lw_row")
        ctx.alloc_vgpr(1, "_tmp_lw_col")

        # -- A: lds_write_a = (thread_row * unroll_k + thread_col) * elem_bytes
        ctx.module.add(st.VLShiftRightB32(
            ctx.vgpr("_tmp_lw_row"), ctx.vgpr("v_tid"),
            st.Register(int(math.log2(ck))),
            comment=f"tid / {ck} (row for A)",
        ))
        if cm < self.tile.block_size:
            ctx.module.add(st.VAndB32(
                ctx.vgpr("_tmp_lw_row"), ctx.vgpr("_tmp_lw_row"),
                st.Register(cm - 1), comment=f"% {cm}",
            ))
        ctx.module.add(st.VAndB32(
            ctx.vgpr("_tmp_lw_col"), ctx.vgpr("v_tid"),
            st.Register(ck - 1), comment=f"tid % {ck} (col for A)",
        ))
        ctx.module.add(st.VMulLOU32(
            ctx.vgpr("v_lds_write_a"), ctx.vgpr("_tmp_lw_row"),
            st.Register(self.tile.unroll_k),
            comment=f"row * {self.tile.unroll_k}",
        ))
        ctx.module.add(st.VAddU32(
            ctx.vgpr("v_lds_write_a"), ctx.vgpr("v_lds_write_a"),
            ctx.vgpr("_tmp_lw_col"),
            comment="+ col",
        ))
        if self._elem == 2:
            ctx.module.add(st.VLShiftLeftB32(
                ctx.vgpr("v_lds_write_a"), ctx.vgpr("v_lds_write_a"),
                st.Register(1), comment="* 2 (bytes)",
            ))
        elif self._elem == 4:
            ctx.module.add(st.VLShiftLeftB32(
                ctx.vgpr("v_lds_write_a"), ctx.vgpr("v_lds_write_a"),
                st.Register(2), comment="* 4 (bytes)",
            ))

        # -- B: lds_write_b = lds_offset_b + (thread_row * unroll_k + thread_col) * elem
        cn = self._cluster_n
        ctx.module.add(st.VLShiftRightB32(
            ctx.vgpr("_tmp_lw_row"), ctx.vgpr("v_tid"),
            st.Register(int(math.log2(ck))),
            comment=f"tid / {ck} (row for B)",
        ))
        if cn < self.tile.block_size:
            ctx.module.add(st.VAndB32(
                ctx.vgpr("_tmp_lw_row"), ctx.vgpr("_tmp_lw_row"),
                st.Register(cn - 1), comment=f"% {cn}",
            ))
        # Reuse _tmp_lw_col (same tid % ck)
        ctx.module.add(st.VMulLOU32(
            ctx.vgpr("v_lds_write_b"), ctx.vgpr("_tmp_lw_row"),
            st.Register(self.tile.unroll_k),
            comment=f"row * {self.tile.unroll_k}",
        ))
        ctx.module.add(st.VAddU32(
            ctx.vgpr("v_lds_write_b"), ctx.vgpr("v_lds_write_b"),
            ctx.vgpr("_tmp_lw_col"),
            comment="+ col",
        ))
        if self._elem == 2:
            ctx.module.add(st.VLShiftLeftB32(
                ctx.vgpr("v_lds_write_b"), ctx.vgpr("v_lds_write_b"),
                st.Register(1), comment="* 2 (bytes)",
            ))
        elif self._elem == 4:
            ctx.module.add(st.VLShiftLeftB32(
                ctx.vgpr("v_lds_write_b"), ctx.vgpr("v_lds_write_b"),
                st.Register(2), comment="* 4 (bytes)",
            ))
        # Add B's base offset in LDS
        ctx.module.add(st.VAddU32(
            ctx.vgpr("v_lds_write_b"), ctx.vgpr("v_lds_write_b"),
            st.Register(self._lds_offset_b),
            comment=f"+ lds_offset_b ({self._lds_offset_b})",
        ))

    def emit_lds_read_addr(self, ctx: TileContext) -> None:
        """Compute LDS read addresses for the current MFMA tile.

        Reads ``wave.mi``, ``wave.ni``, ``wave.ki`` from ``ctx.indices``.
        Updates ``v_lds_read_a`` and ``v_lds_read_b``.
        """
        if ctx.module is None:
            return
        import stinkytofu as st

        mfma = self.tile.mfma
        mi = ctx.indices.get("wave.mi", 0)
        ni = ctx.indices.get("wave.ni", 0)
        ki = ctx.indices.get("wave.ki", 0)

        # wave_m and wave_n are already computed in v_wave_m / v_wave_n.
        # The LDS read address for A:
        #   row = wave_m * m_per_wave + mi * mfma_m + lane_row
        #   col = ki * mfma_k
        #   lds_read_a = (row * unroll_k + col) * elem_bytes

        ctx.alloc_vgpr(1, "_tmp_lr_row")

        # row base = wave_m * m_per_wave + mi * mfma_m
        row_base = mi * mfma.m
        ctx.module.add(st.VMulLOU32(
            ctx.vgpr("_tmp_lr_row"), ctx.vgpr("v_wave_m"),
            st.Register(self.tile.m_per_wave),
            comment=f"wave_m * {self.tile.m_per_wave}",
        ))
        if row_base > 0:
            ctx.module.add(st.VAddU32(
                ctx.vgpr("_tmp_lr_row"), ctx.vgpr("_tmp_lr_row"),
                st.Register(row_base),
                comment=f"+ mi*mfma_m ({row_base})",
            ))

        # + lane_row (lane_id % mfma_m)
        ctx.alloc_vgpr(1, "_tmp_lane_row")
        ctx.module.add(st.VAndB32(
            ctx.vgpr("_tmp_lane_row"), ctx.vgpr("v_lane_id"),
            st.Register(mfma.m - 1),
            comment=f"lane_row = lane_id % {mfma.m}",
        ))
        ctx.module.add(st.VAddU32(
            ctx.vgpr("_tmp_lr_row"), ctx.vgpr("_tmp_lr_row"),
            ctx.vgpr("_tmp_lane_row"),
            comment="+ lane_row",
        ))

        # lds_read_a = row * unroll_k + ki * mfma_k
        col_offset = ki * mfma.k
        ctx.module.add(st.VMulLOU32(
            ctx.vgpr("v_lds_read_a"), ctx.vgpr("_tmp_lr_row"),
            st.Register(self.tile.unroll_k),
            comment=f"row * {self.tile.unroll_k}",
        ))
        if col_offset > 0:
            ctx.module.add(st.VAddU32(
                ctx.vgpr("v_lds_read_a"), ctx.vgpr("v_lds_read_a"),
                st.Register(col_offset),
                comment=f"+ ki*mfma_k ({col_offset})",
            ))
        # To bytes
        if self._elem == 2:
            ctx.module.add(st.VLShiftLeftB32(
                ctx.vgpr("v_lds_read_a"), ctx.vgpr("v_lds_read_a"),
                st.Register(1), comment="* 2 (bytes)",
            ))

        # -- B: similar but using wave_n, ni, and B's LDS offset
        row_base_b = ni * mfma.n
        ctx.module.add(st.VMulLOU32(
            ctx.vgpr("_tmp_lr_row"), ctx.vgpr("v_wave_n"),
            st.Register(self.tile.n_per_wave),
            comment=f"wave_n * {self.tile.n_per_wave}",
        ))
        if row_base_b > 0:
            ctx.module.add(st.VAddU32(
                ctx.vgpr("_tmp_lr_row"), ctx.vgpr("_tmp_lr_row"),
                st.Register(row_base_b),
                comment=f"+ ni*mfma_n ({row_base_b})",
            ))
        # + lane_row for B (same: lane_id % mfma_n, and mfma_m == mfma_n for 16x16)
        ctx.module.add(st.VAddU32(
            ctx.vgpr("_tmp_lr_row"), ctx.vgpr("_tmp_lr_row"),
            ctx.vgpr("_tmp_lane_row"),
            comment="+ lane_row (B)",
        ))

        ctx.module.add(st.VMulLOU32(
            ctx.vgpr("v_lds_read_b"), ctx.vgpr("_tmp_lr_row"),
            st.Register(self.tile.unroll_k),
            comment=f"row * {self.tile.unroll_k}",
        ))
        if col_offset > 0:
            ctx.module.add(st.VAddU32(
                ctx.vgpr("v_lds_read_b"), ctx.vgpr("v_lds_read_b"),
                st.Register(col_offset),
                comment=f"+ ki*mfma_k ({col_offset})",
            ))
        if self._elem == 2:
            ctx.module.add(st.VLShiftLeftB32(
                ctx.vgpr("v_lds_read_b"), ctx.vgpr("v_lds_read_b"),
                st.Register(1), comment="* 2 (bytes)",
            ))
        # Add B's LDS base offset
        ctx.module.add(st.VAddU32(
            ctx.vgpr("v_lds_read_b"), ctx.vgpr("v_lds_read_b"),
            st.Register(self._lds_offset_b),
            comment=f"+ lds_offset_b ({self._lds_offset_b})",
        ))

    def emit_all_prologue(self, ctx: TileContext) -> None:
        """Emit all prologue address computations in one call.

        Convenience wrapper that computes global-load and LDS-write
        addresses.  Call this after thread indices are computed but
        before the first global load.
        """
        with ctx.scope("_addr_prologue"):
            self.emit_global_load_addr_a(ctx)
            self.emit_global_load_addr_b(ctx)
            self.emit_lds_write_addr(ctx)
