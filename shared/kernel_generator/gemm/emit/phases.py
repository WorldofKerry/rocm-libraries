# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM kernel phase implementations.

Each phase is a named step in the tile tree that emits assembly for
one part of the kernel.  All address computation goes through
coordinate transforms via ``emit_affine()``.

Phase signature: ``(level: TileLevel, ctx: AsmContext) -> None``

Phases access kernel state through ``ctx._metadata``:
  - ``ctx._metadata["tile"]``:    TileConfig
  - ``ctx._metadata["problem"]``: GemmProblem
  - ``ctx._metadata["layouts"]``: GemmLayouts (transform descriptors)
  - ``ctx._metadata["kernel"]``:  GemmKernel (for tree access)
"""
from __future__ import annotations

import math

from .context import AsmContext
from .layouts import emit_affine, GemmLayouts
from ..problem import DataType, GemmProblem, MfmaConfig, TileConfig
from ..tile.tree import TileLevel, TilePhase
from ..tile.transforms import Embed, Dim

__all__ = [
    "phase_streamk_store",
    "WORKGROUP_EPILOGUE_PHASES",
    "default_mfma_visitor",
]


# ===================================================================
# Helpers: extract ctx._metadata
# ===================================================================

def _tile(ctx: AsmContext) -> TileConfig:
    return ctx._metadata["tile"]

def _problem(ctx: AsmContext) -> GemmProblem:
    return ctx._metadata["problem"]

def _layouts(ctx: AsmContext) -> GemmLayouts:
    return ctx._metadata["layouts"]


# ===================================================================
# Store epilogue (uses transforms for address computation)
# ===================================================================
def phase_store_d(level: TileLevel, ctx: AsmContext) -> None:
    """Store accumulators to D using buffer_store_short with a buffer SRD.

    Supports two layouts:
    - Row-major (standalone): D[m * N + n], soffset for M, imm for N
    - Column-major (TensileLite): D[m + n * M], soffset for N, imm for M

    For MXFP4 via TensileLite, output is BFloat16 (v_cvt_pk_bf16_f32).
    For standalone, output is FP16 (v_cvt_pk_f16_f32).
    """
    tile = _tile(ctx)
    problem = _problem(ctx)
    mfma = tile.mfma
    acc_per = mfma.acc_vgprs
    elem = 2
    elem_int = 2
    # TensileLite uses column-major BF16; standalone uses row-major FP16
    colmajor = (ctx._metadata.get("use_1d_grid", False) 
                or ctx._metadata.get("use_wave_abi", False)
                or ctx._metadata.get("colmajor_output", False))
    use_bf16 = (problem.dtype == DataType.BF16) or (colmajor and mfma.is_mx)

    ctx.comment("=== Store D via buffer SRD ===")

    # ---- 1. Build raw buffer SRD for D (4 SGPRs) ----
    ctx.alloc_sgpr_permanent(4, "s_srd_d")

    ctx.comment("SRD for D matrix (raw buffer mode)")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_d", 0, 1),
             ctx.sreg("s_ptr_D", 0, 1), comment="SRD_D base lo")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_d", 1, 1),
             ctx.sreg("s_ptr_D", 1, 1), comment="SRD_D base hi")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_d", 2, 1), "0xFFFFFFFF",
             comment="SRD_D size (unlimited)")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_d", 3, 1), "0x20000",
             comment="SRD_D flags: raw buffer")
    ctx.raw("")

    # ---- 2. Per-lane base byte offset for (mi=0, ni=0, ai=0) ----
    # MFMA output lane mapping (16x16):
    #   lane_n      = lane_id % mfma.n        (column within MFMA tile)
    #   lane_m_base = (lane_id / mfma.m) * 4  (base row within MFMA tile)
    ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.n - 1,
              comment=f"lane_n = lane_id % {mfma.n}")
    ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
               int(math.log2(mfma.m)), comment=f"lane_id / {mfma.m}")
    ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_tmp1"), 2,
               comment="* 4 -> lane_m_base")

    # global_row = wg_id_x * wg_m + wave_m * m_per_wave + lane_m_base
    ctx.v_mul(ctx.vreg("v_addr_d", 0, 1), str(tile.m_per_wave),
              ctx.vreg("v_wave_m"), comment=f"wave_m * {tile.m_per_wave}")
    ctx.v_add(ctx.vreg("v_addr_d", 0, 1),
              ctx.vreg("v_addr_d", 0, 1), ctx.vreg("v_tmp1"),
              comment="+ lane_m_base")
    ctx.s_mul(ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_x"),
              str(tile.wg_m), comment=f"wg_id_x * {tile.wg_m}")
    ctx.v_add(ctx.vreg("v_addr_d", 0, 1), ctx.sreg("s_tmp1"),
              ctx.vreg("v_addr_d", 0, 1),
              comment="+ wg_base_m -> global_row")

    # global_col = wg_id_y * wg_n + wave_n * n_per_wave + lane_n
    ctx.v_mul(ctx.vreg("v_addr_d", 1, 1), str(tile.n_per_wave),
              ctx.vreg("v_wave_n"), comment=f"wave_n * {tile.n_per_wave}")
    ctx.v_add(ctx.vreg("v_addr_d", 1, 1),
              ctx.vreg("v_addr_d", 1, 1), ctx.vreg("v_tmp0"),
              comment="+ lane_n")
    ctx.s_mul(ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_y"),
              str(tile.wg_n), comment=f"wg_id_y * {tile.wg_n}")
    ctx.v_add(ctx.vreg("v_addr_d", 1, 1), ctx.sreg("s_tmp1"),
              ctx.vreg("v_addr_d", 1, 1),
              comment="+ wg_base_n -> global_col")

    if colmajor:
        # Column-major: voffset = (global_row + global_col * M) * elem
        ctx.inst("v_mul_lo_u32", ctx.vreg("v_addr_d", 1, 1),
                 ctx.sreg("s_M"), ctx.vreg("v_addr_d", 1, 1),
                 comment="global_col * M")
        ctx.v_add(ctx.vreg("v_addr_d", 0, 1),
                  ctx.vreg("v_addr_d", 0, 1), ctx.vreg("v_addr_d", 1, 1),
                  comment="+ global_row -> col-major linear index")
    else:
        # Row-major: voffset = (global_row * N + global_col) * elem
        ctx.inst("v_mul_lo_u32", ctx.vreg("v_addr_d", 0, 1),
                 ctx.sreg("s_N"), ctx.vreg("v_addr_d", 0, 1),
                 comment="global_row * N")
        ctx.v_add(ctx.vreg("v_addr_d", 0, 1),
                  ctx.vreg("v_addr_d", 0, 1), ctx.vreg("v_addr_d", 1, 1),
                  comment="+ global_col -> row-major linear index")

    # Scale to bytes (elem_int == 2 for f16/bf16)
    ctx.v_lshl(ctx.vreg("v_addr_d", 0, 1),
                ctx.vreg("v_addr_d", 0, 1), 1,
                comment="* 2 -> byte offset")
    # v_addr_d[0] = per-lane base byte offset (voffset for buffer ops)
    ctx.raw("")

    total_elems = tile.mfma_m_repeat * tile.mfma_n_repeat * acc_per
    ctx.comment(f"Store {total_elems} elements"
                f" ({tile.mfma_m_repeat}x{tile.mfma_n_repeat}x{acc_per})"
                f" via buffer_store_short")

    if colmajor:
        # Column-major: soffset for N (col_stride = M * elem), imm for M
        ctx.s_lshl(ctx.sreg("s_tmp0"), ctx.sreg("s_M"), 1,
                    comment="col_stride = M * 2 bytes")
        ctx.raw("")
        _store_d_colmajor(ctx, tile, mfma, acc_per, elem_int, use_bf16)
    else:
        # Row-major: soffset for M (row_stride = N * elem), imm for N
        ctx.s_lshl(ctx.sreg("s_tmp0"), ctx.sreg("s_N"), 1,
                    comment="row_stride = N * 2 bytes")
        ctx.raw("")
        use_pk_cvt = (acc_per % 2 == 0)
        if use_pk_cvt:
            _store_d_packed(ctx, tile, mfma, acc_per, elem_int, use_bf16)
        else:
            _store_d_scalar(ctx, tile, mfma, acc_per, elem_int, use_bf16)

    ctx.s_waitcnt("vmcnt(0)", comment="wait for stores")
    ctx.raw("")



def _store_d_colmajor(ctx: AsmContext, tile: TileConfig, mfma: MfmaConfig, acc_per: int, elem_int: int, use_bf16: bool) -> None:
    """Column-major store for TensileLite ABI.

    Layout: D[m + n * M], stride along M = 1 element, stride along N = M.
    - soffset: ni * mfma.n * col_stride (N advancement, large, in SGPR)
    - immediate offset: (mi * mfma.m + ai) * elem (M advancement, small)

    Uses buffer_store_dwordx2 (GWVW=4) when acc_per is divisible by 4,
    or buffer_store_dword (GWVW=2) when divisible by 2, to store
    packed BF16/FP16 at consecutive M addresses.
    """
    cvt_inst = "v_cvt_pk_bf16_f32" if use_bf16 else "v_cvt_pk_f16_f32"
    use_gwvw4 = acc_per >= 4 and acc_per % 4 == 0

    for ni in range(tile.mfma_n_repeat):
        # soffset_ni = ni * mfma.n * col_stride
        if ni == 0:
            ctx.s_mov(ctx.sreg("s_tmp1"), "0",
                      comment=f"soffset = 0 (ni={ni})")
        else:
            ctx.s_mul(ctx.sreg("s_tmp1"), ctx.sreg("s_tmp0"),
                      str(ni * mfma.n),
                      comment=f"soffset = {ni * mfma.n} * col_stride"
                              f" (ni={ni})")

        for mi in range(tile.mfma_m_repeat):
            if use_gwvw4:
                # GWVW=4: pack 4 elements into buffer_store_dwordx2
                for ai_base in range(0, acc_per, 4):
                    imm = (mi * mfma.m + ai_base) * elem_int
                    # Pack pairs: (ai+0, ai+1) -> v_store_tmp, (ai+2, ai+3) -> v_store_tmp+1
                    for pair in range(2):
                        ai_lo = ai_base + pair * 2
                        ai_hi = ai_lo + 1
                        acc_lo = (mi * tile.mfma_n_repeat + ni) * acc_per + ai_lo
                        acc_hi = (mi * tile.mfma_n_repeat + ni) * acc_per + ai_hi
                        ctx.inst("v_accvgpr_read_b32", ctx.vreg("v_tmp0"),
                                 ctx.areg("acc_C", acc_lo, 1),
                                 comment=f"acc[{acc_lo}] a{ai_lo}")
                        ctx.inst("v_accvgpr_read_b32", ctx.vreg("v_tmp1"),
                                 ctx.areg("acc_C", acc_hi, 1),
                                 comment=f"acc[{acc_hi}] a{ai_hi}")
                        ctx.inst(cvt_inst, ctx.vreg("v_store_tmp", pair, 1),
                                 ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
                                 comment=f"pk cvt a{ai_lo},a{ai_hi}")
                    # Store 8 bytes (4 bf16/fp16 values)
                    _emit_buffer_store_dwordx2(ctx, "v_store_tmp",
                                               "s_tmp1", imm,
                                               f"store D m{mi}_n{ni} GWVW=4")
            elif acc_per % 2 == 0:
                for ai_base in range(0, acc_per, 2):
                    ai_lo = ai_base
                    ai_hi = ai_base + 1
                    imm_lo = (mi * mfma.m + ai_lo) * elem_int

                    acc_idx_lo = (mi * tile.mfma_n_repeat + ni) * acc_per + ai_lo
                    acc_idx_hi = (mi * tile.mfma_n_repeat + ni) * acc_per + ai_hi

                    ctx.inst("v_accvgpr_read_b32", ctx.vreg("v_tmp0"),
                             ctx.areg("acc_C", acc_idx_lo, 1),
                             comment=f"acc[{acc_idx_lo}] m{mi}_n{ni}_a{ai_lo}")
                    ctx.inst("v_accvgpr_read_b32", ctx.vreg("v_tmp1"),
                             ctx.areg("acc_C", acc_idx_hi, 1),
                             comment=f"acc[{acc_idx_hi}] m{mi}_n{ni}_a{ai_hi}")
                    ctx.inst(cvt_inst, ctx.vreg("v_store_tmp"),
                             ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
                             comment=f"pk cvt a{ai_lo},a{ai_hi}")

                    _emit_buffer_store_dword(ctx, "v_store_tmp",
                                             "s_tmp1", imm_lo,
                                             f"store D m{mi}_n{ni}_a{ai_lo}a{ai_hi}")
            else:
                cvt_scalar = "v_cvt_bf16_f32_e32" if use_bf16 else "v_cvt_f16_f32_e32"
                for ai in range(acc_per):
                    imm = (mi * mfma.m + ai) * elem_int
                    acc_idx = (mi * tile.mfma_n_repeat + ni) * acc_per + ai

                    ctx.inst("v_accvgpr_read_b32", ctx.vreg("v_store_tmp"),
                             ctx.areg("acc_C", acc_idx, 1),
                             comment=f"acc[{acc_idx}] m{mi}_n{ni}_a{ai}")
                    ctx.inst(cvt_scalar, ctx.vreg("v_store_tmp"),
                             ctx.vreg("v_store_tmp"),
                             comment="f32->bf16" if use_bf16 else "f32->f16")
                    _emit_buffer_store_short(ctx, "v_store_tmp",
                                             "s_tmp1", imm,
                                             f"store D m{mi}_n{ni}_a{ai}")


def _store_d_scalar(ctx: AsmContext, tile: TileConfig, mfma: MfmaConfig, acc_per: int, elem_int: int, use_bf16: bool = False) -> None:
    """Emit buffer_store_short with individual f32->f16/bf16 convert per element.

    Fallback for odd acc_per (uncommon).
    """
    cvt_scalar = "v_cvt_bf16_f32_e32" if use_bf16 else "v_cvt_f16_f32_e32"
    for mi in range(tile.mfma_m_repeat):
        for ai in range(acc_per):
            row_delta = mi * mfma.m + ai
            if row_delta == 0:
                ctx.s_mov(ctx.sreg("s_tmp1"), "0",
                          comment=f"soffset = 0 (mi={mi} ai={ai})")
            else:
                ctx.s_mul(ctx.sreg("s_tmp1"), ctx.sreg("s_tmp0"),
                          str(row_delta),
                          comment=f"soffset = {row_delta} * row_stride"
                                  f" (mi={mi} ai={ai})")

            for ni in range(tile.mfma_n_repeat):
                acc_idx = (mi * tile.mfma_n_repeat + ni) * acc_per + ai
                ni_imm = ni * mfma.n * elem_int

                ctx.inst("v_accvgpr_read_b32", ctx.vreg("v_store_tmp"),
                         ctx.areg("acc_C", acc_idx, 1),
                         comment=f"acc[{acc_idx}] m{mi}_n{ni}_a{ai}")
                ctx.inst(cvt_scalar, ctx.vreg("v_store_tmp"),
                         ctx.vreg("v_store_tmp"),
                         comment="f32->bf16" if use_bf16 else "f32->f16")
                _emit_buffer_store_short(ctx, "v_store_tmp",
                                         "s_tmp1", ni_imm,
                                         f"store D m{mi}_n{ni}_a{ai}")


def _store_d_packed(ctx: AsmContext, tile: TileConfig, mfma: MfmaConfig, acc_per: int, elem_int: int, use_bf16: bool = False) -> None:
    """Emit buffer_store_short with v_cvt_pk_f16/bf16_f32 for accumulator pairs.

    Processes ai in pairs (0,1), (2,3), etc.  For each pair:
      - Precompute soffset_lo and soffset_hi (= soffset_lo + row_stride)
        so the inner ni loop has no scalar add/sub overhead.
      - v_cvt_pk_f16_f32 packs both f32 values into {f16_hi, f16_lo}.
      - Store bits[15:0] with soffset_lo, extract bits[31:16] via
        v_lshrrev_b32 and store with soffset_hi.
    """
    # Extra SGPR for the hi-row soffset (avoids s_add/s_sub per ni)
    ctx.alloc_sgpr_permanent(1, "s_soffset_hi")

    for mi in range(tile.mfma_m_repeat):
        for ai_base in range(0, acc_per, 2):
            ai_lo = ai_base
            ai_hi = ai_base + 1

            # soffset_lo = (mi * mfma.m + ai_lo) * row_stride
            row_delta_lo = mi * mfma.m + ai_lo
            if row_delta_lo == 0:
                ctx.s_mov(ctx.sreg("s_tmp1"), "0",
                          comment=f"soffset_lo = 0 (mi={mi} ai={ai_lo})")
            else:
                ctx.s_mul(ctx.sreg("s_tmp1"), ctx.sreg("s_tmp0"),
                          str(row_delta_lo),
                          comment=f"soffset_lo = {row_delta_lo} * row_stride"
                                  f" (mi={mi} ai={ai_lo})")

            # soffset_hi = soffset_lo + row_stride
            ctx.inst("s_add_u32", ctx.sreg("s_soffset_hi"),
                     ctx.sreg("s_tmp1"), ctx.sreg("s_tmp0"),
                     comment=f"soffset_hi = soffset_lo + row_stride"
                             f" (ai={ai_hi})")

            for ni in range(tile.mfma_n_repeat):
                acc_idx_lo = (mi * tile.mfma_n_repeat + ni) * acc_per + ai_lo
                acc_idx_hi = (mi * tile.mfma_n_repeat + ni) * acc_per + ai_hi
                ni_imm = ni * mfma.n * elem_int

                # Read both accumulators and pack-convert
                ctx.inst("v_accvgpr_read_b32", ctx.vreg("v_tmp0"),
                         ctx.areg("acc_C", acc_idx_lo, 1),
                         comment=f"acc[{acc_idx_lo}] m{mi}_n{ni}_a{ai_lo}")
                ctx.inst("v_accvgpr_read_b32", ctx.vreg("v_tmp1"),
                         ctx.areg("acc_C", acc_idx_hi, 1),
                         comment=f"acc[{acc_idx_hi}] m{mi}_n{ni}_a{ai_hi}")
                ctx.inst("v_cvt_pk_bf16_f32" if use_bf16 else "v_cvt_pk_f16_f32", ctx.vreg("v_store_tmp"),
                         ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
                         comment=f"pk cvt a{ai_lo},a{ai_hi}")

                # Store lo half: bits[15:0] = f16(ai_lo)
                _emit_buffer_store_short(ctx, "v_store_tmp",
                                         "s_tmp1", ni_imm,
                                         f"store D m{mi}_n{ni}_a{ai_lo}")

                # Extract hi half and store at ai_hi row
                ctx.v_lshr(ctx.vreg("v_store_tmp"),
                           ctx.vreg("v_store_tmp"), 16,
                           comment="extract hi f16")
                _emit_buffer_store_short(ctx, "v_store_tmp",
                                         "s_soffset_hi", ni_imm,
                                         f"store D m{mi}_n{ni}_a{ai_hi}")



def _emit_buffer_store_dword(ctx: AsmContext, vdata_name: str, soffset_name: str, imm_offset: int,
                              comment):
    """Emit one buffer_store_dword via the D matrix SRD.

    Stores 32 bits (two packed BF16/FP16 values) at consecutive addresses.
    """
    vdata = ctx.vreg(vdata_name)
    voffset = ctx.vreg("v_addr_d", 0, 1)
    srd = ctx.sreg("s_srd_d", 0, 4)
    soffset = ctx.sreg(soffset_name)

    if imm_offset == 0:
        ctx.inst("buffer_store_dword", vdata, voffset, srd,
                 soffset, "offen nt", comment=comment)
    elif imm_offset < 4096:
        ctx.inst("buffer_store_dword", vdata, voffset, srd,
                 soffset, f"offen offset:{imm_offset} nt",
                 comment=comment)
    else:
        ctx.inst("s_add_u32", soffset, soffset, str(imm_offset),
                 comment=f"fold imm {imm_offset} into soffset")
        ctx.inst("buffer_store_dword", vdata, voffset, srd,
                 soffset, "offen nt", comment=comment)
        ctx.inst("s_sub_u32", soffset, soffset, str(imm_offset),
                 comment="restore soffset")


def _emit_buffer_store_dwordx2(ctx: AsmContext, vdata_name: str, soffset_name: str, imm_offset: int,
                               comment):
    """Emit one buffer_store_dwordx2 via the D matrix SRD (GWVW=4).

    Stores 64 bits (four packed BF16/FP16 values) at consecutive addresses.
    """
    vdata = ctx.vreg(vdata_name, 0, 2)
    voffset = ctx.vreg("v_addr_d", 0, 1)
    srd = ctx.sreg("s_srd_d", 0, 4)
    soffset = ctx.sreg(soffset_name)

    if imm_offset == 0:
        ctx.inst("buffer_store_dwordx2", vdata, voffset, srd,
                 soffset, "offen nt", comment=comment)
    elif imm_offset < 4096:
        ctx.inst("buffer_store_dwordx2", vdata, voffset, srd,
                 soffset, f"offen offset:{imm_offset} nt",
                 comment=comment)
    else:
        ctx.inst("s_add_u32", soffset, soffset, str(imm_offset),
                 comment=f"fold imm {imm_offset} into soffset")
        ctx.inst("buffer_store_dwordx2", vdata, voffset, srd,
                 soffset, "offen nt", comment=comment)
        ctx.inst("s_sub_u32", soffset, soffset, str(imm_offset),
                 comment="restore soffset")


def _emit_buffer_store_short(ctx: AsmContext, vdata_name: str, soffset_name: str, imm_offset: int,
                              comment):
    """Emit one buffer_store_short via the D matrix SRD.

    If imm_offset exceeds the 12-bit immediate field (4095), the excess
    is folded into soffset with a temporary s_add/s_sub pair.
    """
    vdata = ctx.vreg(vdata_name)
    voffset = ctx.vreg("v_addr_d", 0, 1)
    srd = ctx.sreg("s_srd_d", 0, 4)
    soffset = ctx.sreg(soffset_name)

    if imm_offset == 0:
        ctx.inst("buffer_store_short", vdata, voffset, srd,
                 soffset, "offen", comment=comment)
    elif imm_offset < 4096:
        ctx.inst("buffer_store_short", vdata, voffset, srd,
                 soffset, f"offen offset:{imm_offset}",
                 comment=comment)
    else:
        # Immediate too large; fold into soffset temporarily
        ctx.inst("s_add_u32", soffset, soffset, str(imm_offset),
                 comment=f"fold imm {imm_offset} into soffset")
        ctx.inst("buffer_store_short", vdata, voffset, srd,
                 soffset, "offen", comment=comment)
        ctx.inst("s_sub_u32", soffset, soffset, str(imm_offset),
                 comment="restore soffset")




# ===================================================================
# MFMA visitor
# ===================================================================

def default_mfma_visitor(level: TileLevel, ctx: AsmContext) -> None:
    """Tile-tree visitor: emit LDS reads + MFMAs at the mfma leaf level."""
    if level.name != "mfma":
        return  # only emit at leaf

    mi = ctx.indices.get("wave.mi", 0)
    ni = ctx.indices.get("wave.ni", 0)
    ki = ctx.indices.get("wave.ki", 0)

    tile = _tile(ctx)
    mfma = tile.mfma
    elem = _problem(ctx).element_bytes

    # LDS read A
    a_off = (mi * mfma.m * tile.unroll_k + ki * mfma.k) * elem
    if a_off > 0:
        ctx.v_add(ctx.vreg("v_tmp0"), str(a_off), ctx.vreg("v_lds_rd_a"),
                  comment=f"lds_rd_a + mi={mi} ki={ki}")
        a_addr = ctx.vreg("v_tmp0")
    else:
        a_addr = ctx.vreg("v_lds_rd_a")
    for r in range(mfma.a_vgprs):
        ctx.ds_read(ctx.vreg("v_a", r, 1), a_addr, offset=r * 4, width=1,
                    comment=f"LDS read A[{r}] mi={mi} ki={ki}")

    # LDS read B
    b_off = (ni * mfma.n * tile.unroll_k + ki * mfma.k) * elem
    if b_off > 0:
        ctx.v_add(ctx.vreg("v_tmp1"), str(b_off), ctx.vreg("v_lds_rd_b"),
                  comment=f"lds_rd_b + ni={ni} ki={ki}")
        b_addr = ctx.vreg("v_tmp1")
    else:
        b_addr = ctx.vreg("v_lds_rd_b")
    for r in range(mfma.b_vgprs):
        ctx.ds_read(ctx.vreg("v_b", r, 1), b_addr, offset=r * 4, width=1,
                    comment=f"LDS read B[{r}] ni={ni} ki={ki}")

    ctx.s_waitcnt("lgkmcnt(0)", comment="wait LDS reads")

    acc_per = mfma.acc_vgprs
    acc_off = (mi * tile.mfma_n_repeat + ni) * acc_per
    ctx.inst(mfma.instruction_name,
             ctx.areg("acc_C", acc_off, acc_per),
             ctx.vreg("v_a", 0, mfma.a_vgprs),
             ctx.vreg("v_b", 0, mfma.b_vgprs),
             ctx.areg("acc_C", acc_off, acc_per),
             comment=f"mfma m{mi}_n{ni} k{ki}")
    ctx.raw("")




def phase_streamk_store(level: TileLevel, ctx: AsmContext) -> None:
    """StreamK epilogue: store partial results to workspace or final D.

    Full-K workgroups (is_partial==0): store directly to D (same as
    phase_store_d).
    Partial-K workgroups (is_partial==1): store f32 accumulators to
    workspace buffer at per-WG slot.

    The workspace is laid out as:
        workspace[sk_wg_index * MT_M * MT_N] (f32 per element)

    A separate fixup kernel later sums all partials per output tile
    and writes the final f16/bf16 result to D.
    """
    tile = _tile(ctx)
    problem = _problem(ctx)

    # Check if StreamK is active
    if not ctx.has("s_is_partial"):
        phase_store_d(level, ctx)
        return

    ctx.comment("=== StreamK Epilogue ===")
    ctx.inst("s_cmp_eq_u32", ctx.sreg("s_is_partial"), "0",
             comment="is_partial == 0?")
    ctx.inst("s_cbranch_scc1", "sk_full_store",
             comment="full K -> store D directly")

    # Partial-K path: store f32 accumulators to workspace
    ctx.comment("Partial-K: store f32 accumulators to workspace")
    mfma = tile.mfma
    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    acc_per = mfma.acc_vgprs
    elem = problem.element_bytes

    # Workspace offset: wg_serial * MT_M * MT_N * sizeof(f32)
    # wg_serial is in s_wg_id_x (1D grid for StreamK)
    ws_tile_bytes = tile.wg_m * tile.wg_n * 4  # f32 per element
    ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"),
             str(ws_tile_bytes),
             comment=f"ws_offset = wg_serial * {ws_tile_bytes}")
    ctx.inst("s_add_u32", ctx.sreg("s_tmp0"),
             ctx.sreg("s_workspace_ptr", 0, 1), ctx.sreg("s_tmp0"),
             comment="ws_addr_lo = ws_ptr + offset")
    ctx.inst("s_addc_u32", ctx.sreg("s_tmp1"),
             ctx.sreg("s_workspace_ptr", 1, 1), "0",
             comment="ws_addr_hi")

    # Build SRD for workspace
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_a", 0, 1), ctx.sreg("s_tmp0"),
             comment="ws SRD lo")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_a", 1, 1), ctx.sreg("s_tmp1"),
             comment="ws SRD hi")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_a", 2, 1), "0xFFFFFFFF",
             comment="limit")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_a", 3, 1), "0x20000",
             comment="flags")

    # Store each accumulator as f32 (no conversion)
    # Layout: contiguous f32 in row-major order
    # acc_C[(mi * nr + ni) * acc_per + ai] -> workspace[row * MT_N + col]
    # row = wave_m * m_per_wave + mi * mfma_m + lane_row
    # col = wave_n * n_per_wave + ni * mfma_n + lane_col
    #
    # For simplicity: use flat_store with per-lane addressing
    # Each lane stores its own accumulator values

    # Per-lane base: (wave_m * m_per_wave + lane_row) * MT_N * 4
    ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
              comment=f"lane_row = lane_id % {mfma.m}")
    ctx.v_mul(ctx.vreg("v_tmp1"), str(tile.m_per_wave),
              ctx.vreg("v_wave_m"), comment=f"wave_m * {tile.m_per_wave}")
    ctx.v_add(ctx.vreg("v_tmp0"), ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
              comment="m_row = wave_m * m_per_wave + lane_row")

    for mi in range(mr):
        m_off = mi * mfma.m
        for ni in range(nr):
            acc_off = (mi * nr + ni) * acc_per
            n_base = ni * mfma.n
            # Compute voffset for this (mi, ni) block
            # row = m_row + mi * mfma_m
            # For each acc register, the lane maps to a specific (row, col)
            # mfma_f16_16x16x16: 4 accumulators per lane, each is one (row, col)
            # acc[0] -> (lane_row, lane_group*4 + 0)
            # etc.
            for ai in range(acc_per):
                # Compute byte offset: (m_row + mi*M) * MT_N * 4 + (n_col) * 4
                # This is complex per-lane -- use buffer_store_dword
                col = n_base + (ctx.vreg("v_lane_id") if ai == 0 else "...")
                # Simplified: store linearly and let fixup kernel handle layout
                linear_off = (acc_off + ai) * tile.wave_size  # rough
                ctx.inst("buffer_store_dword",
                         ctx.areg("acc_C", acc_off + ai, 1),
                         ctx.vreg("v_tmp0"),  # voffset placeholder
                         ctx.sreg("s_srd_a", 0, 4),
                         str(linear_off * 4), "offen",
                         comment=f"ws store acc m{mi}_n{ni}_a{ai}")

    ctx.inst("s_branch", "sk_epilogue_done",
             comment="skip full store")

    ctx.label("sk_full_store")
    # Full-K path: standard store to D
    phase_store_d(level, ctx)

    ctx.label("sk_epilogue_done")


# ===================================================================
# Phase lists
# ===================================================================

WORKGROUP_EPILOGUE_PHASES = [
    TilePhase("store_d", phase_store_d),
]


# ===================================================================
# StreamK epilogue: conditional store (direct or workspace)
# ===================================================================

def phase_store_streamk(level: TileLevel, ctx: AsmContext) -> None:
    """StreamK epilogue: 3-way branch based on WG's K-range ownership.

    Path 1 (SOLE OWNER): is_partial == 0
        Full K range computed by this WG. Direct store to D.

    Path 2 (PARTIAL NON-OWNER): is_partial == 1, iter_start != 0
        Partial K range, not the owner. Store f32 accumulators to
        workspace, set flag, done.

    Path 3 (PARTIAL OWNER): is_partial == 1, iter_start == 0
        Partial K range, owns the tile. Poll flags for other
        partitions, load and accumulate their partials, then
        convert and store to D.
    """
    import math
    tile = _tile(ctx)

    ctx.comment("=== StreamK Epilogue ===")

    # Path 1: sole owner (full K range)
    ctx.inst("s_cmp_eq_u32", ctx.sreg("s_is_partial"), "0",
             comment="sole owner (full K)?")
    ctx.inst("s_cbranch_scc1", "sk_store_direct",
             comment="yes -> direct store to D")

    # Path 2 vs 3: check if this WG is the owner (iter_start == 0)
    ctx.inst("s_cmp_eq_u32", ctx.sreg("s_iter_start"), "0",
             comment="tile owner (iter_start == 0)?")
    ctx.inst("s_cbranch_scc1", "sk_owner_reduce",
             comment="yes -> poll + accumulate + store")

    # ---- Path 2: non-owner partial ----
    ctx.comment("Non-owner: store partial to workspace + set flag")
    _compute_global_partition_idx(ctx, tile)
    _store_workspace(ctx, tile)
    _set_flag(ctx)
    ctx.inst("s_branch", "sk_store_done", comment="done (non-owner)")
    ctx.raw("")

    # ---- Path 3: owner with partials ----
    ctx.label("sk_owner_reduce")
    ctx.comment("Owner: poll flags, load partials, accumulate")
    _compute_global_partition_idx(ctx, tile)
    _owner_reduce(ctx, tile)
    # Fall through to sk_store_direct

    # ---- Path 1 & 3 endpoint: direct store to D ----
    ctx.label("sk_store_direct")
    phase_store_d(level, ctx)

    ctx.label("sk_store_done")



def _compute_global_partition_idx(ctx: 'AsmContext', tile: 'TileConfig') -> None:
    """Compute global workspace slot from WG IDs and partition index.

    global_slot = (wg_id_y * tiles_m + wg_id_x) * num_partitions + partition_idx
    Overwrites s_partition_idx with the global slot.
    """
    tiles_m = tile.wg_m  # Actually need M / wg_m, use s_M
    # tile_serial = wg_id_y * (M / wg_m) + wg_id_x
    ctx.inst("s_lshr_b32", ctx.sreg("s_tmp0"), ctx.sreg("s_M"),
             str(int(tile.wg_m).bit_length() - 1),
             comment=f"tiles_m = M / {tile.wg_m}")
    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_y"),
              ctx.sreg("s_tmp0"),
              comment="wg_id_y * tiles_m")
    ctx.inst("s_add_u32", ctx.sreg("s_tmp0"),
             ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"),
             comment="+ wg_id_x -> tile_serial")
    # global_slot = tile_serial * num_partitions + partition_idx
    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
              ctx.sreg("s_num_partitions"),
              comment="tile_serial * num_partitions")
    ctx.inst("s_add_u32", ctx.sreg("s_partition_idx"),
             ctx.sreg("s_tmp0"), ctx.sreg("s_partition_idx"),
             comment="+ partition_idx -> global_slot")


def _build_ws_srd(ctx: 'AsmContext') -> None:
    """Build workspace buffer SRD from s_workspace_ptr.

    Always emits the MOV instructions so each branch that needs
    the SRD gets its own initialization at runtime.
    """
    if not ctx.has("s_srd_ws"):
        ctx.alloc_sgpr_permanent(4, "s_srd_ws")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_ws", 0, 1),
             ctx.sreg("s_workspace_ptr", 0, 1), comment="WS SRD lo")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_ws", 1, 1),
             ctx.sreg("s_workspace_ptr", 1, 1), comment="WS SRD hi")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_ws", 2, 1), "0xFFFFFFFF",
             comment="WS SRD size")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_ws", 3, 1), "0x20000",
             comment="WS SRD flags")


def _ws_voffset(ctx: 'AsmContext', tile: 'TileConfig') -> None:
    """Compute per-lane voffset within a tile's workspace region.

    Result in v_tmp2. Layout: row-major within tile,
    element_offset = (base_m * wg_n + base_n) * 4 bytes.
    """
    import math
    mfma = tile.mfma
    ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.n - 1,
              comment=f"lane_n = lane_id % {mfma.n}")
    ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
               int(math.log2(mfma.m)), comment=f"lane_id / {mfma.m}")
    ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_tmp1"), 2,
               comment="* 4 -> lane_m_base")
    ctx.v_mul(ctx.vreg("v_tmp2"), str(tile.m_per_wave),
              ctx.vreg("v_wave_m"), comment=f"wave_m * {tile.m_per_wave}")
    ctx.v_add(ctx.vreg("v_tmp2"), ctx.vreg("v_tmp2"), ctx.vreg("v_tmp1"),
              comment="+ lane_m_base -> base_m")
    ctx.v_mul(ctx.vreg("v_tmp3"), str(tile.n_per_wave),
              ctx.vreg("v_wave_n"), comment=f"wave_n * {tile.n_per_wave}")
    ctx.v_add(ctx.vreg("v_tmp3"), ctx.vreg("v_tmp3"), ctx.vreg("v_tmp0"),
              comment="+ lane_n -> base_n")
    ctx.v_mul(ctx.vreg("v_tmp2"), str(tile.wg_n), ctx.vreg("v_tmp2"),
              comment=f"base_m * {tile.wg_n}")
    ctx.v_add(ctx.vreg("v_tmp2"), ctx.vreg("v_tmp2"), ctx.vreg("v_tmp3"),
              comment="+ base_n")
    ctx.v_lshl(ctx.vreg("v_tmp2"), ctx.vreg("v_tmp2"), 2,
               comment="* 4 -> byte offset within tile")


def _store_workspace(ctx: 'AsmContext', tile: 'TileConfig') -> None:
    """Store f32 accumulators to workspace[partition_idx * tile_area]."""
    mfma = tile.mfma
    acc_per = mfma.acc_vgprs
    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    tile_area = tile.wg_m * tile.wg_n

    _build_ws_srd(ctx)

    # Base offset: partition_idx * tile_area * 4
    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_partition_idx"),
              str(tile_area * 4), comment="partition base offset")

    _ws_voffset(ctx, tile)
    # Add partition base
    ctx.v_add(ctx.vreg("v_tmp2"), ctx.sreg("s_tmp0"), ctx.vreg("v_tmp2"),
              comment="+ partition base -> voffset")
    ctx.raw("")

    # mi_stride and ni_stride for soffset/imm addressing
    mi_stride = mfma.m * tile.wg_n * 4
    ni_stride = mfma.n * 4
    ai_stride = tile.wg_n * 4

    for mi in range(mr):
        soff = mi * mi_stride
        if soff > 0:
            ctx.s_mov(ctx.sreg("s_tmp1"), str(soff),
                      comment=f"soffset mi={mi}")
        for ni in range(nr):
            acc_base = (mi * nr + ni) * acc_per
            for ai in range(acc_per):
                imm = ni * ni_stride + ai * ai_stride
                acc_reg = ctx.areg("acc_C", acc_base + ai, 1)
                ctx.inst("v_accvgpr_read_b32",
                         ctx.vreg("v_tmp0"), acc_reg,
                         comment=f"acc[{acc_base + ai}]")
                soff_reg = ctx.sreg("s_tmp1") if soff > 0 else "0"
                ctx.inst("buffer_store_dword",
                         ctx.vreg("v_tmp0"),
                         ctx.vreg("v_tmp2"),
                         ctx.sreg("s_srd_ws", 0, 4), soff_reg,
                         f"offen offset:{imm}",
                         comment=f"ws[m{mi}_n{ni}_a{ai}]")

    ctx.s_waitcnt("vmcnt(0)", comment="wait workspace stores")
    ctx.raw("")


def _set_flag(ctx: 'AsmContext') -> None:
    """Set flags[partition_idx] = 1 to signal completion.

    All waves execute this -- s_buffer_store is a scalar op
    and all waves write the same value (idempotent).
    """
    ctx.comment("Set completion flag")
    # Build flags SRD if not already done
    if not ctx.has("s_srd_flags"):
        ctx.alloc_sgpr_permanent(4, "s_srd_flags")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_flags", 0, 1),
                 ctx.sreg("s_flags_ptr", 0, 1), comment="flags SRD lo")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_flags", 1, 1),
                 ctx.sreg("s_flags_ptr", 1, 1), comment="flags SRD hi")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_flags", 2, 1), "0xFFFFFFFF",
                 comment="flags SRD size")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_flags", 3, 1), "0x20000",
                 comment="flags SRD flags")

    # soffset = partition_idx * 4
    ctx.s_lshl(ctx.sreg("s_tmp0"), ctx.sreg("s_partition_idx"), 2,
               comment="flag offset = partition_idx * 4")
    ctx.s_mov(ctx.sreg("s_tmp1"), "1", comment="flag value = 1")

    # Barrier ensures all waves finished stores, then set flag
    ctx.s_barrier(comment="sync waves before flag")
    ctx.inst("s_buffer_store_dword", ctx.sreg("s_tmp1"),
             ctx.sreg("s_srd_flags", 0, 4), ctx.sreg("s_tmp0"),
             comment="flags[partition_idx] = 1")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait flag store")
    ctx.inst("s_dcache_wb", comment="flush scalar cache to L2 for visibility")
    ctx.raw("")

def _owner_reduce(ctx: 'AsmContext', tile: 'TileConfig') -> None:
    """Owner WG: poll flags, load partials from workspace, accumulate.

    For each partition p > 0 of this tile:
      1. Poll flags[first_partition + p] until non-zero
      2. Load f32 workspace[partition_idx_of_p * tile_area + lane_offset]
      3. v_add_f32 into accumulators
    """
    mfma = tile.mfma
    acc_per = mfma.acc_vgprs
    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    tile_area = tile.wg_m * tile.wg_n
    total_accs = mr * nr * acc_per

    # Build workspace + flags SRDs. Must emit initialization MOVs
    # unconditionally because the non-owner path (which also emits
    # SRD setup) is on a different branch and the owner never
    # executes it at runtime.
    _build_ws_srd(ctx)
    if not ctx.has("s_srd_flags"):
        ctx.alloc_sgpr_permanent(4, "s_srd_flags")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_flags", 0, 1),
             ctx.sreg("s_flags_ptr", 0, 1), comment="flags SRD lo")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_flags", 1, 1),
             ctx.sreg("s_flags_ptr", 1, 1), comment="flags SRD hi")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_flags", 2, 1), "0xFFFFFFFF",
             comment="flags SRD size")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_flags", 3, 1), "0x20000",
             comment="flags SRD flags")

    _ws_voffset(ctx, tile)
    # v_tmp2 = per-lane byte offset within tile (no partition base yet)

    # Loop over partitions: partition_idx+1 .. partition_idx+num_partitions-1
    # The owner is partition_idx (iter_start==0). Other partitions are
    # partition_idx+1, partition_idx+2, etc.
    # num_partitions_per_tile is passed via kernarg.
    ctx.s_mov(ctx.sreg("s_tmp0"), "1", comment="p = 1 (first non-owner)")

    ctx.label("sk_reduce_loop")
    # Check if p >= num_partitions_per_tile
    ctx.inst("s_cmp_ge_u32", ctx.sreg("s_tmp0"),
             ctx.sreg("s_num_partitions"),
             comment="p >= num_partitions?")
    ctx.inst("s_cbranch_scc1", "sk_reduce_done",
             comment="all partitions accumulated")

    # Flag index = partition_idx + p
    # Use s_iter_start as scratch (it's 0 for the owner and no longer needed)
    ctx.inst("s_add_u32", ctx.sreg("s_iter_start"),
             ctx.sreg("s_partition_idx"), ctx.sreg("s_tmp0"),
             comment="flag_idx = partition_idx + p")
    ctx.s_lshl(ctx.sreg("s_iter_start"), ctx.sreg("s_iter_start"), 2,
               comment="flag_offset = flag_idx * 4")

    # Poll flag -- s_iter_start holds the offset (preserved across iterations),
    # s_tmp1 receives the loaded value
    ctx.label("sk_poll_flag")
    ctx.inst("s_buffer_load_dword", ctx.sreg("s_tmp1"),
             ctx.sreg("s_srd_flags", 0, 4), ctx.sreg("s_iter_start"),
             "glc", comment="load flag (bypass scalar cache)")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait flag load")
    ctx.inst("s_cmp_eq_u32", ctx.sreg("s_tmp1"), "0",
             comment="flag still 0?")
    ctx.inst("s_cbranch_scc1", "sk_poll_flag",
             comment="spin until flag set")

    # Load partial from workspace and accumulate
    # ws_base = (partition_idx + p) * tile_area * 4
    ctx.inst("s_add_u32", ctx.sreg("s_tmp1"),
             ctx.sreg("s_partition_idx"), ctx.sreg("s_tmp0"),
             comment="ws_slot = partition_idx + p")
    ctx.s_mul(ctx.sreg("s_tmp1"), ctx.sreg("s_tmp1"),
              str(tile_area * 4), comment="ws_base = slot * tile_bytes")

    # Add per-lane offset (v_tmp2) to ws_base
    ctx.v_add(ctx.vreg("v_tmp3"), ctx.sreg("s_tmp1"), ctx.vreg("v_tmp2"),
              comment="ws_addr = ws_base + lane_offset")

    # Load each accumulator element, add to existing acc
    mi_stride = mfma.m * tile.wg_n * 4
    ni_stride = mfma.n * 4
    ai_stride = tile.wg_n * 4

    for mi in range(mr):
        soff = mi * mi_stride
        if soff > 0:
            ctx.s_mov(ctx.sreg("s_tmp1"), str(soff),
                      comment=f"soffset mi={mi}")
        for ni in range(nr):
            acc_base = (mi * nr + ni) * acc_per
            for ai in range(acc_per):
                imm = ni * ni_stride + ai * ai_stride
                soff_reg = ctx.sreg("s_tmp1") if soff > 0 else "0"
                ctx.inst("buffer_load_dword",
                         ctx.vreg("v_tmp0"),
                         ctx.vreg("v_tmp3"),
                         ctx.sreg("s_srd_ws", 0, 4), soff_reg,
                         f"offen offset:{imm}",
                         comment=f"load ws[m{mi}_n{ni}_a{ai}]")
                ctx.s_waitcnt("vmcnt(0)", comment="wait load")
                acc_reg = ctx.areg("acc_C", acc_base + ai, 1)
                # Read acc, add, write back
                ctx.inst("v_accvgpr_read_b32",
                         ctx.vreg("v_tmp4"), acc_reg,
                         comment=f"read acc[{acc_base + ai}]")
                ctx.inst("v_add_f32",
                         ctx.vreg("v_tmp4"),
                         ctx.vreg("v_tmp4"), ctx.vreg("v_tmp0"),
                         comment="acc += partial")
                ctx.inst("v_accvgpr_write_b32",
                         acc_reg, ctx.vreg("v_tmp4"),
                         comment=f"write acc[{acc_base + ai}]")

    ctx.s_waitcnt("vmcnt(0)", comment="wait all loads")

    # Next partition
    ctx.inst("s_add_u32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"), "1",
             comment="p++")
    ctx.inst("s_branch", "sk_reduce_loop", comment="next partition")

    ctx.label("sk_reduce_done")
    ctx.raw("")

def _store_workspace(ctx: AsmContext, tile: 'TileConfig') -> None:
    """Store raw f32 accumulators to workspace buffer.

    Layout: workspace[partition_idx * MT_M * MT_N + element_offset]
    Uses soffset for mi stepping (to stay within 16-bit imm limit)
    and immediate offset for ni/ai within each mi block.
    """
    import math
    mfma = tile.mfma
    acc_per = mfma.acc_vgprs
    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat

    ctx.comment("Store f32 accumulators to workspace")

    # Build workspace SRD from s_workspace_ptr
    ctx.alloc_sgpr_permanent(4, "s_srd_ws")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_ws", 0, 1),
             ctx.sreg("s_workspace_ptr", 0, 1), comment="WS SRD base lo")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_ws", 1, 1),
             ctx.sreg("s_workspace_ptr", 1, 1), comment="WS SRD base hi")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_ws", 2, 1), "0xFFFFFFFF",
             comment="WS SRD size")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_ws", 3, 1), "0x20000",
             comment="WS SRD flags: raw buffer")

    tile_area = tile.wg_m * tile.wg_n
    # Compute base offset: partition_idx * tile_area * 4 (f32)
    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_partition_idx"),
              str(tile_area * 4),
              comment=f"partition offset = idx * {tile_area} * 4")

    # Per-lane voffset: (wave_m * m_per_wave + lane_m_base) * wg_n + (wave_n * n_per_wave + lane_n)
    # Then scale by 4 (f32) and add partition base
    ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.n - 1,
              comment=f"lane_n = lane_id % {mfma.n}")
    ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
               int(math.log2(mfma.m)), comment=f"lane_id / {mfma.m}")
    ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_tmp1"), 2,
               comment="* 4 -> lane_m_base")

    ctx.v_mul(ctx.vreg("v_tmp2"), str(tile.m_per_wave),
              ctx.vreg("v_wave_m"), comment=f"wave_m * {tile.m_per_wave}")
    ctx.v_add(ctx.vreg("v_tmp2"), ctx.vreg("v_tmp2"), ctx.vreg("v_tmp1"),
              comment="+ lane_m_base -> base_m")

    ctx.v_mul(ctx.vreg("v_tmp3"), str(tile.n_per_wave),
              ctx.vreg("v_wave_n"), comment=f"wave_n * {tile.n_per_wave}")
    ctx.v_add(ctx.vreg("v_tmp3"), ctx.vreg("v_tmp3"), ctx.vreg("v_tmp0"),
              comment="+ lane_n -> base_n")

    # voffset = (base_m * wg_n + base_n) * 4 + partition_base
    ctx.v_mul(ctx.vreg("v_tmp2"), str(tile.wg_n), ctx.vreg("v_tmp2"),
              comment=f"base_m * {tile.wg_n}")
    ctx.v_add(ctx.vreg("v_tmp2"), ctx.vreg("v_tmp2"), ctx.vreg("v_tmp3"),
              comment="+ base_n")
    ctx.v_lshl(ctx.vreg("v_tmp2"), ctx.vreg("v_tmp2"), 2,
               comment="* 4 -> byte offset")
    ctx.v_add(ctx.vreg("v_tmp2"), ctx.sreg("s_tmp0"), ctx.vreg("v_tmp2"),
              comment="+ partition base -> voffset")
    ctx.raw("")

    # Store accumulators using soffset for mi stepping
    # mi_stride = mfma.m * wg_n * 4 bytes (row block step)
    mi_stride = mfma.m * tile.wg_n * 4
    # ni_stride = mfma.n * 4 bytes (col block step)
    ni_stride = mfma.n * 4
    # ai_stride = wg_n * 4 bytes (one row step within mi block)
    ai_stride = tile.wg_n * 4

    for mi in range(mr):
        soff = mi * mi_stride
        if soff > 0:
            ctx.s_mov(ctx.sreg("s_tmp1"), str(soff),
                      comment=f"soffset for mi={mi}")

        for ni in range(nr):
            acc_base = (mi * nr + ni) * acc_per
            for ai in range(acc_per):
                imm = ni * ni_stride + ai * ai_stride
                acc_reg = ctx.areg("acc_C", acc_base + ai, 1)
                ctx.inst("v_accvgpr_read_b32",
                         ctx.vreg("v_tmp0"), acc_reg,
                         comment=f"read acc[{acc_base + ai}]")
                soff_reg = ctx.sreg("s_tmp1") if soff > 0 else "0"
                ctx.inst("buffer_store_dword",
                         ctx.vreg("v_tmp0"),
                         ctx.vreg("v_tmp2"),
                         ctx.sreg("s_srd_ws", 0, 4), soff_reg,
                         f"offen offset:{imm}",
                         comment=f"ws[m{mi}_n{ni}_a{ai}]")

    ctx.s_waitcnt("vmcnt(0)", comment="wait workspace stores")
    ctx.raw("")

