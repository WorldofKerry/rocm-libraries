# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Scale data layout descriptors for MX GEMM kernels.

Separates the *memory layout* of scale data from the *loading mechanism*
(VMEM vs LDS).  Each layout knows how to compute SRD base offsets,
per-thread voffsets, ds_read addresses, and K-advance strides for its
specific byte arrangement.

Layouts:
- ``LinearLayout``:  row-major ``scale[M, K/mx_block]``, 1 byte per element.
- ``E8M0ShuffleLayout``:  AITER pre-swizzled format used by TensileLite
  with ``--mx-scale-format 1``.  (Placeholder -- not yet implemented.)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..emit.context import AsmContext
from ..problem import TileConfig

__all__ = [
    "ScaleLayout",
    "LinearLayout",
    "E8M0ShuffleLayout",
]


class ScaleLayout(ABC):
    """How scale data is organized in global memory."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for logging/debug."""

    @property
    @abstractmethod
    def mx_scale_format(self) -> int:
        """TensileLite --mx-scale-format value (0=linear, 1=pre-swizzled)."""

    @abstractmethod
    def k_advance_bytes(self, tile: TileConfig) -> int:
        """Bytes to advance the scale SRD per K-loop iteration."""

    @abstractmethod
    def srd_base_offset_expr(self, ctx: AsmContext, tile: TileConfig,
                             wg_id_reg: str, matrix: str) -> None:
        """Emit code to compute SRD base offset into s_tmp0.

        After this call, ``s_tmp0`` holds the byte offset from
        ``ptr_scale_{matrix}`` for this workgroup's tile.

        Args:
            wg_id_reg: SGPR name holding the workgroup index
                       (``s_wg_id_x`` for A, ``s_wg_id_y`` for B).
            matrix: ``"a"`` or ``"b"``.
        """

    @abstractmethod
    def emit_vmem_voffset(self, ctx: AsmContext, tile: TileConfig,
                          matrix: str) -> None:
        """Emit per-lane VGPR voffset for VMEM scale loads.

        After this call, ``v_scale_voff_{matrix}`` contains the
        per-lane byte offset from the SRD base.
        """

    @abstractmethod
    def emit_vmem_soffsets(self, ctx: AsmContext, tile: TileConfig,
                           matrix: str) -> None:
        """Emit per-mi/ni SGPRs for VMEM buffer_load soffset.

        For matrix ``"a"``: creates ``s_soff_sa_{mi}`` for mi=1..mr-1.
        For matrix ``"b"``: creates ``s_soff_sb_{ni}`` for ni=1..nr-1.
        mi=0 / ni=0 use soffset=0 (no extra SGPR needed).
        """


class LinearLayout(ScaleLayout):
    """Row-major linear: ``scale[M, K/mx_block]``, stride = K/mx_block bytes.

    Per-lane voffset = ``(wave_m * mr * 16 + lane_id & 15) * stride``
    Per-mi soffset = ``mi * 16 * stride``
    SRD advance per K-iter = ``unroll_k / mx_block`` bytes
    SRD base = ``ptr + wg_id * (wg_tile / 32) * stride``
    """

    @property
    def name(self) -> str:
        return "linear"

    @property
    def mx_scale_format(self) -> int:
        return 0

    def k_advance_bytes(self, tile: TileConfig) -> int:
        return tile.unroll_k // tile.mfma.mx_block

    def srd_base_offset_expr(self, ctx: AsmContext, tile: TileConfig,
                             wg_id_reg: str, matrix: str) -> None:
        wg_tile = tile.wg_m if matrix == "a" else tile.wg_n
        stride_reg = f"s_stride_scale_{matrix}"
        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg(wg_id_reg),
                  str(wg_tile),
                  comment=f"{wg_id_reg} * {wg_tile} (tile M/N rows)")
        ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                 ctx.sreg(stride_reg),
                 comment=f"* stride_scale_{matrix} -> byte offset")

    def emit_vmem_voffset(self, ctx: AsmContext, tile: TileConfig,
                          matrix: str) -> None:
        mfma = tile.mfma
        mr = tile.mfma_m_repeat if matrix == "a" else tile.mfma_n_repeat
        wave_reg = "v_wave_m" if matrix == "a" else "v_wave_n"
        stride_reg = f"s_stride_scale_{matrix}"
        voff_name = f"v_scale_voff_{matrix}"

        if not ctx.has(voff_name):
            ctx.alloc_vgpr_permanent(1, voff_name)

        ctx.comment(f"Scale {matrix.upper()} per-lane voffset (linear)")
        ctx.v_mul(ctx.vreg("v_tmp0"),
                  str(mr * mfma.m), ctx.vreg(wave_reg),
                  comment=f"{wave_reg} * {mr * mfma.m}")
        ctx.inst("v_and_b32", ctx.vreg("v_tmp1"),
                 ctx.vreg("v_lane_id"), "15",
                 comment="lane_id & 15 (M-row within MFMA tile)")
        ctx.inst("v_add_u32", ctx.vreg("v_tmp0"),
                 ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
                 comment="M-row relative to wave's start")
        ctx.inst("v_mul_lo_u32", ctx.vreg(voff_name),
                 ctx.sreg(stride_reg), ctx.vreg("v_tmp0"),
                 comment=f"* {stride_reg} -> byte offset")
        ctx.raw("")

    def emit_vmem_soffsets(self, ctx: AsmContext, tile: TileConfig,
                           matrix: str) -> None:
        mfma = tile.mfma
        repeat = tile.mfma_m_repeat if matrix == "a" else tile.mfma_n_repeat
        stride_reg = f"s_stride_scale_{matrix}"
        prefix = "sa" if matrix == "a" else "sb"

        ctx.comment(f"Scale {matrix.upper()} per-mi soffsets (linear)")
        for idx in range(1, repeat):
            soff = f"s_soff_{prefix}_{idx}"
            ctx.alloc_sgpr_permanent(1, soff)
            ctx.inst("s_mul_i32", ctx.sreg(soff),
                     ctx.sreg(stride_reg),
                     str(idx * mfma.m),
                     comment=f"soff_{prefix}[{idx}] = stride * {idx * mfma.m}")
        ctx.raw("")


class E8M0ShuffleLayout(ScaleLayout):
    """AITER e8m0_shuffle pre-swizzled layout.

    view(M/32, 2, 16, K_scales/8, 2, 4).permute(0, 3, 5, 2, 4, 1)
    Each 32-M-row block occupies (K_scales/8) * 256 bytes.
    K-advance per iteration = 256 bytes (one d3 unit).

    VMEM addressing:
      Per-lane voffset = lane_id * 4 (each lane reads 4 scale bytes)
      Per-group soffset = (wave_m * n_groups + group) * 256
      Each 256-byte group covers 2 MFMA tiles (32 M-rows) x 2 K-scale cols
    """

    @property
    def name(self) -> str:
        return "e8m0_shuffle"

    @property
    def mx_scale_format(self) -> int:
        return 1

    def k_advance_bytes(self, tile: TileConfig) -> int:
        return 256

    def srd_base_offset_expr(self, ctx: AsmContext, tile: TileConfig,
                             wg_id_reg: str, matrix: str) -> None:
        # Each 32-M-row block = (K_scales / 8) * 256 bytes
        # = stride_scale * 32 bytes
        wg_tile = tile.wg_m if matrix == "a" else tile.wg_n
        stride_reg = f"s_stride_scale_{matrix}"
        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg(wg_id_reg),
                  str(wg_tile // 32),
                  comment=f"{wg_id_reg} * {wg_tile // 32} (MT/32)")
        ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                 ctx.sreg(stride_reg),
                 comment=f"* stride_scale_{matrix}")
        ctx.s_lshl(ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"), 5,
                   comment="* 32 (pre-swizzled block stride)")

    def emit_vmem_voffset(self, ctx: AsmContext, tile: TileConfig,
                          matrix: str) -> None:
        voff_name = f"v_dtl_off_scale_{matrix}"
        if not ctx.has(voff_name):
            ctx.alloc_vgpr_permanent(1, voff_name)

        ctx.comment(f"Scale {matrix.upper()} swizzled voffset: lane_id * 4")
        ctx.v_lshl(ctx.vreg(voff_name),
                   ctx.vreg("v_lane_id"), 2,
                   comment="lane_id * 4 -> swizzled scale voffset")
        if matrix == "a":
            # B reuses the same voffset value
            b_name = "v_dtl_off_scale_b"
            if not ctx.has(b_name):
                ctx.alloc_vgpr_permanent(1, b_name)
            ctx.inst("v_mov_b32", ctx.vreg(b_name),
                     ctx.vreg(voff_name),
                     comment="scaleB voffset = same")
        ctx.raw("")

    def emit_vmem_soffsets(self, ctx: AsmContext, tile: TileConfig,
                           matrix: str) -> None:
        mfma = tile.mfma
        mr = tile.mfma_m_repeat if matrix == "a" else tile.mfma_n_repeat
        n_groups = (mr + 1) // 2
        wave_reg = "v_wave_m" if matrix == "a" else "v_wave_n"
        prefix = f"s_scale_soff_{matrix}"

        # Allocate group soffset SGPRs
        for g in range(n_groups):
            soff = f"{prefix}{g}"
            if not ctx.has(soff):
                ctx.alloc_sgpr_permanent(1, soff)

        ctx.comment(f"Scale {matrix.upper()} group soffsets (e8m0_shuffle)")
        # group_g soffset = (wave * n_groups + g) * 256
        ctx.v_mul(ctx.vreg("v_tmp0"),
                  str(n_groups), ctx.vreg(wave_reg),
                  comment=f"{wave_reg} * {n_groups}")
        ctx.inst("v_readfirstlane_b32", ctx.sreg("s_tmp0"),
                 ctx.vreg("v_tmp0"),
                 comment=f"{wave_reg} * {n_groups} -> SGPR")
        ctx.s_lshl(ctx.sreg(f"{prefix}0"), ctx.sreg("s_tmp0"), 8,
                   comment=f"group0 soffset {matrix.upper()} = base * 256")
        for g in range(1, n_groups):
            ctx.inst("s_add_u32", ctx.sreg(f"{prefix}{g}"),
                     ctx.sreg(f"{prefix}0"), str(g * 256),
                     comment=f"group{g} soffset {matrix.upper()} = base + {g * 256}")
        ctx.raw("")
