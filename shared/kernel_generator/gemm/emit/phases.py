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
from typing import Callable

from .context import AsmContext
from .layouts import emit_affine, GemmLayouts
from ..problem import GemmProblem, MfmaConfig, TileConfig
from ..tile.tree import TileLevel, TilePhase
from ..tile.transforms import Embed, Dim

__all__ = [
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
# Prologue phases
# ===================================================================

def phase_load_kernargs(level: TileLevel, ctx: AsmContext) -> None:
    """Load kernel arguments from TensileLite kernarg segment + WG decomposition."""
    import math as _math
    tile = ctx._metadata["tile"]
    mfma = tile.mfma
    ctx.comment("Load kernel arguments (TensileLite layout)")
    karg = ctx.sreg("s_kernarg")
    ctx.inst("s_load_dword", ctx.sreg("s_M"), karg, "16", comment="M")
    ctx.inst("s_load_dword", ctx.sreg("s_N"), karg, "20", comment="N")
    ctx.inst("s_load_dword", ctx.sreg("s_K"), karg, "28", comment="K")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_D"), karg, "32", comment="D ptr")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_A"), karg, "48", comment="A ptr")
    b_offset = "64" if getattr(mfma, 'is_mx', False) else "56"
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_B"), karg, b_offset,
             comment="B ptr")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait for kernarg loads")
    ctx.raw("")

    # 1D WG decomposition (DTL setup)
    if ctx._metadata.get("use_1d_grid", False):
        # 1D WG decomposition using pure scalar integer math
        # numWG_m = ceil(M / MT_M), tile_n = serial / numWG_m, tile_m = serial % numWG_m
        import math as _math
        _log2_mt = int(_math.log2(tile.wg_m))
        ctx.s_mov(ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_x"),
                  comment="save wg_serial")
        # numWG_m = (M + MT_M - 1) >> log2(MT_M)
        ctx.inst("s_add_u32", ctx.sreg("s_tmp0"), ctx.sreg("s_M"),
                 str(tile.wg_m - 1), comment=f"M + {tile.wg_m - 1}")
        ctx.inst("s_lshr_b32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                 str(_log2_mt), comment=f"numWG_m = ceil(M/{tile.wg_m})")
        # Division: tile_n = serial / numWG_m, tile_m = serial % numWG_m
        # Use s_ff1 to detect if numWG_m is power-of-2 and use shift
        ctx.inst("s_sub_u32", ctx.sreg("s_wg_id_x"),
                 ctx.sreg("s_tmp0"), "1", comment="numWG_m - 1")
        ctx.inst("s_and_b32", ctx.sreg("s_wg_id_y"),
                 ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"),
                 comment="numWG_m & (numWG_m-1) == 0 if power-of-2")
        # For now assume power-of-2 (all our tiles/problems satisfy this)
        ctx.inst("s_ff1_i32_b32", ctx.sreg("s_wg_id_y"),
                 ctx.sreg("s_tmp0"), comment="log2(numWG_m)")
        ctx.inst("s_lshr_b32", ctx.sreg("s_wg_id_y"),
                 ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_y"),
                 comment="tile_n = serial >> log2(numWG_m)")  # s3 = tile_n  (s_wg_id_y)
        # tile_m = serial & (numWG_m - 1)
        ctx.inst("s_sub_u32", ctx.sreg("s_wg_id_x"),
                 ctx.sreg("s_tmp0"), "1", comment="numWG_m - 1")
        ctx.inst("s_and_b32", ctx.sreg("s_wg_id_x"),
                 ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_x"),
                 comment="tile_m = serial & (numWG_m - 1)")




def phase_thread_indexing(level: TileLevel, ctx: AsmContext) -> None:
    """Compute wave_id, lane_id, wave_m, wave_n from thread ID."""
    tile = _tile(ctx)
    ctx.comment("Thread indexing")
    log2_ws = int(math.log2(tile.wave_size))
    ctx.v_lshr(ctx.vreg("v_wave_id"), ctx.vreg("v_tid"), log2_ws,
               comment=f"wave_id = tid >> {log2_ws}")
    ctx.v_and(ctx.vreg("v_lane_id"), ctx.vreg("v_tid"), tile.wave_size - 1,
              comment=f"lane_id = tid & {tile.wave_size - 1}")
    if tile.waves_n > 1:
        log2_wn = int(math.log2(tile.waves_n))
        ctx.v_lshr(ctx.vreg("v_wave_m"), ctx.vreg("v_wave_id"), log2_wn,
                   comment=f"wave_m = wave_id >> {log2_wn}")
        ctx.v_and(ctx.vreg("v_wave_n"), ctx.vreg("v_wave_id"),
                  tile.waves_n - 1,
                  comment=f"wave_n = wave_id & {tile.waves_n - 1}")
    else:
        ctx.v_mov(ctx.vreg("v_wave_m"), ctx.vreg("v_wave_id"),
                  comment="wave_m = wave_id (waves_n=1)")
        ctx.v_mov(ctx.vreg("v_wave_n"), "0", comment="wave_n = 0")
    ctx.raw("")


def phase_load_cluster_setup(level: TileLevel, ctx: AsmContext) -> None:
    """Compute global-load thread cluster coordinates for A and B.

    For symmetric tiles (wg_m == wg_n), A and B use the same mapping.
    For asymmetric tiles, B gets its own row/col computed from wg_n.
    """
    tile = _tile(ctx)
    problem = _problem(ctx)
    elem = problem.element_bytes

    def _emit_cluster(wg_dim: int, row_reg: str, col_reg: str, label: str) -> None:
        """Emit thread cluster coords for a tile of size wg_dim x unroll_k."""
        elems_per_thread = (wg_dim * tile.unroll_k) // tile.block_size
        contiguous_k = min(elems_per_thread, tile.unroll_k)
        k_groups = max(1, tile.unroll_k // contiguous_k)
        rows = tile.block_size // k_groups

        ctx.comment(f"Load cluster {label}: {rows} rows x "
                     f"{k_groups} K-groups ({contiguous_k} elems each)")
        if k_groups == 1:
            ctx.v_mov(row_reg, ctx.vreg("v_tid"),
                      comment=f"{label} row = tid")
            ctx.v_mov(col_reg, "0", comment=f"{label} col = 0")
        else:
            log2_kg = int(math.log2(k_groups))
            ctx.v_lshr(row_reg, ctx.vreg("v_tid"), log2_kg,
                       comment=f"{label} row = tid >> {log2_kg}")
            ctx.v_and(col_reg, ctx.vreg("v_tid"), k_groups - 1,
                      comment=f"{label} tid % {k_groups}")
            if contiguous_k > 1:
                log2_ck = int(math.log2(contiguous_k))
                ctx.v_lshl(col_reg, col_reg,
                           log2_ck, comment=f"* {contiguous_k} -> k_start")

    # A cluster
    _emit_cluster(tile.wg_m, ctx.vreg("v_gload_row"), ctx.vreg("v_gload_col"), "A")
    ctx.raw("")

    # B cluster (may differ from A if wg_m != wg_n)
    if tile.wg_m == tile.wg_n:
        ctx.comment("B uses same cluster as A (symmetric tile)")
        ctx.v_mov(ctx.vreg("v_gload_row_b"), ctx.vreg("v_gload_row"),
                  comment="B row = A row")
        ctx.v_mov(ctx.vreg("v_gload_col_b"), ctx.vreg("v_gload_col"),
                  comment="B col = A col")
    else:
        _emit_cluster(tile.wg_n, ctx.vreg("v_gload_row_b"),
                      ctx.vreg("v_gload_col_b"), "B")
    ctx.raw("")


def phase_lds_addrs(level: TileLevel, ctx: AsmContext) -> None:
    """Compute LDS write AND read offsets using coordinate transforms."""
    tile = _tile(ctx)
    problem = _problem(ctx)
    layouts = _layouts(ctx)
    mfma = tile.mfma
    elem = problem.element_bytes

    # -- LDS write addresses (via transforms) --
    wr_bindings = {
        "row": ctx.vreg("v_gload_row"),
        "col": ctx.vreg("v_gload_col"),
    }
    ctx.comment(f"LDS write A: {layouts.lds_a}")
    emit_affine(ctx, layouts.lds_a, wr_bindings,
                result=ctx.vreg("v_lds_wr_a"),
                scale=elem, comment="lds_wr_a")
    ctx.raw("")

    wr_bindings_b = {
        "row": ctx.vreg("v_gload_row_b"),
        "col": ctx.vreg("v_gload_col_b"),
    }
    ctx.comment(f"LDS write B: {layouts.lds_b} + offset {layouts.lds_b_offset}")
    emit_affine(ctx, layouts.lds_b, wr_bindings_b,
                result=ctx.vreg("v_lds_wr_b"),
                scale=elem, base=str(layouts.lds_b_offset),
                comment="lds_wr_b")
    ctx.raw("")

    # -- LDS read addresses (via transforms) --
    # MFMA lane mapping: lane_row = lane_id % mfma_m, lane_k from lane groups
    k_per_group = mfma.k // (tile.wave_size // mfma.m)
    ctx.comment("MFMA lane mapping: lane_row, lane_k")
    ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
              comment=f"lane_row = lane_id % {mfma.m}")
    ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
               int(math.log2(mfma.m)), comment=f"lane_id / {mfma.m}")
    ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_tmp1"),
               int(math.log2(k_per_group)),
               comment=f"* {k_per_group} -> lane_k_offset")
    ctx.raw("")

    # LDS row stride matches the write layout (includes padding)
    pad_e = tile.lds_pad // elem if tile.lds_pad > 0 else 0
    lds_row_stride = tile.unroll_k + pad_e

    # LDS read A: a_row = wave_m * m_per_wave + lane_row
    lds_rd_a = Embed(
        [Dim("a_row", tile.wg_m), Dim("a_k", tile.unroll_k)],
        Dim("lds_rd_a", tile.wg_m * lds_row_stride),
        [lds_row_stride, 1],
    )
    ctx.comment(f"LDS read A: {lds_rd_a}")
    ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(tile.m_per_wave),
              ctx.vreg("v_wave_m"), comment=f"wave_m * {tile.m_per_wave}")
    ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
              ctx.vreg("v_tmp0"), comment="+ lane_row")
    emit_affine(ctx, lds_rd_a,
                bindings={"a_row": ctx.vreg("v_lds_rd_a"),
                          "a_k": ctx.vreg("v_tmp1")},
                result=ctx.vreg("v_lds_rd_a"), scale=elem,
                comment="lds_rd_a = (row * unroll_k + lane_k) * elem")
    ctx.raw("")

    # LDS read B: b_row = wave_n * n_per_wave + lane_row
    lds_rd_b = Embed(
        [Dim("b_row", tile.wg_n), Dim("b_k", tile.unroll_k)],
        Dim("lds_rd_b", tile.wg_n * lds_row_stride),
        [lds_row_stride, 1],
    )
    ctx.comment(f"LDS read B: {lds_rd_b} + lds_b_offset")
    ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(tile.n_per_wave),
              ctx.vreg("v_wave_n"), comment=f"wave_n * {tile.n_per_wave}")
    ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
              ctx.vreg("v_tmp0"), comment="+ lane_row")
    emit_affine(ctx, lds_rd_b,
                bindings={"b_row": ctx.vreg("v_lds_rd_b"),
                          "b_k": ctx.vreg("v_tmp1")},
                result=ctx.vreg("v_lds_rd_b"), scale=elem,
                base=str(layouts.lds_b_offset),
                comment="lds_rd_b = lds_b_off + (row * unroll_k + lane_k) * elem")
    ctx.raw("")


def phase_init_acc(level: TileLevel, ctx: AsmContext) -> None:
    """Zero-initialize accumulator registers."""
    tile = _tile(ctx)
    acc_total = tile.mfma_m_repeat * tile.mfma_n_repeat * tile.mfma.acc_vgprs
    ctx.comment(f"Init {acc_total} accumulators to zero")
    for i in range(acc_total):
        ctx.inst("v_accvgpr_write_b32", ctx.areg("acc_C", i, 1), "0")
    ctx.raw("")


def phase_global_addrs(level: TileLevel, ctx: AsmContext) -> None:
    """Compute 64-bit global addresses for A and B using transforms.

    Uses emit_affine() with dynamic_coefficients for the K dimension.
    """
    tile = _tile(ctx)
    problem = _problem(ctx)
    layouts = _layouts(ctx)

    for name, ptr_s, wg_id_s, wg_size, layout in [
        ("A", "s_ptr_A", "s_wg_id_x", tile.wg_m, layouts.global_a),
        ("B", "s_ptr_B", "s_wg_id_y", tile.wg_n, layouts.global_b),
    ]:
        addr_v = "v_addr_a" if name == "A" else "v_addr_b"
        dim_name = "m" if name == "A" else "n"
        # Use B-specific load cluster coords for B
        row_reg = "v_gload_row" if name == "A" else "v_gload_row_b"
        col_reg = "v_gload_col" if name == "A" else "v_gload_col_b"

        ctx.comment(f"Global address {name}: {layout} [{dim_name} coeff = s_K]")

        # global_row = wg_id * wg_size + thread_row
        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg(wg_id_s), str(wg_size),
                  comment=f"wg_id * {wg_size}")
        ctx.v_add(ctx.vreg("v_tmp0"), ctx.sreg("s_tmp0"),
                  ctx.vreg(row_reg), comment="+ thread_row -> global_row")

        # offset = global_row * K + col  (K is dynamic, via transform)
        emit_affine(ctx, layout,
                    bindings={dim_name: ctx.vreg("v_tmp0"),
                              "k": ctx.vreg(col_reg)},
                    result=ctx.vreg("v_tmp0"),
                    dynamic_coefficients={dim_name: ctx.sreg("s_K")},
                    scale=problem.element_bytes,
                    comment=f"{name} offset via transform")

        # 64-bit: ptr + byte_offset
        ctx.inst("v_add_co_u32", ctx.vreg(addr_v, 0, 1), "vcc",
                 ctx.sreg(ptr_s, 0, 1), ctx.vreg("v_tmp0"),
                 comment=f"addr_{name}_lo")
        ctx.v_mov(ctx.vreg("v_tmp1"), ctx.sreg(ptr_s, 1, 1),
                  comment=f"{name}_hi to VGPR (const bus)")
        ctx.inst("v_addc_co_u32", ctx.vreg(addr_v, 1, 1), "vcc",
                 ctx.vreg("v_tmp1"), "0", "vcc",
                 comment=f"addr_{name}_hi + carry")
        ctx.raw("")

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
    colmajor = ctx._metadata.get("use_1d_grid", False) or ctx._metadata.get("use_wave_abi", False)
    use_bf16 = colmajor and mfma.is_mx  # MXFP4 dest is BFloat16 in TL

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
            _store_d_packed(ctx, tile, mfma, acc_per, elem_int)
        else:
            _store_d_scalar(ctx, tile, mfma, acc_per, elem_int)

    ctx.s_waitcnt("vmcnt(0)", comment="wait for stores")
    ctx.raw("")



def _store_d_colmajor(ctx: AsmContext, tile: TileConfig, mfma: MfmaConfig, acc_per: int, elem_int: int, use_bf16: bool) -> None:
    """Column-major store for TensileLite ABI.

    Layout: D[m + n * M], stride along M = 1 element, stride along N = M.
    - soffset: ni * mfma.n * col_stride (N advancement, large, in SGPR)
    - immediate offset: (mi * mfma.m + ai) * elem (M advancement, small)

    Uses buffer_store_dword to store packed BF16/FP16 pairs (ai, ai+1)
    at consecutive M addresses, avoiding potential buffer_store_short
    format conversion issues on gfx950.
    """
    cvt_inst = "v_cvt_pk_bf16_f32" if use_bf16 else "v_cvt_pk_f16_f32"

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
            if acc_per % 2 == 0:
                for ai_base in range(0, acc_per, 2):
                    ai_lo = ai_base
                    ai_hi = ai_base + 1
                    # Column-major: ai_lo and ai_hi are at consecutive M addresses
                    # so we can store the packed dword (2x bf16) at ai_lo's address
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

                    # Store packed dword (both bf16 values at consecutive M rows)
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


def _store_d_scalar(ctx: AsmContext, tile: TileConfig, mfma: MfmaConfig, acc_per: int, elem_int: int) -> None:
    """Emit buffer_store_short with individual v_cvt_f16_f32_e32 per element.

    Fallback for odd acc_per (uncommon).
    """
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
                ctx.inst("v_cvt_f16_f32_e32", ctx.vreg("v_store_tmp"),
                         ctx.vreg("v_store_tmp"), comment="f32->f16")
                _emit_buffer_store_short(ctx, "v_store_tmp",
                                         "s_tmp1", ni_imm,
                                         f"store D m{mi}_n{ni}_a{ai}")


def _store_d_packed(ctx: AsmContext, tile: TileConfig, mfma: MfmaConfig, acc_per: int, elem_int: int) -> None:
    """Emit buffer_store_short with v_cvt_pk_f16_f32 for accumulator pairs.

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
                ctx.inst("v_cvt_pk_f16_f32", ctx.vreg("v_store_tmp"),
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


# ===================================================================
# Pipelined K-loop (single phase, handles entire loop)
# ===================================================================

def _emit_wave_compute(ctx: AsmContext, wave: TileLevel, visitor: Callable) -> None:
    """Walk the wave level with proper mi/ni/ki iteration.

    Used by pipelined/optimized K-loops to correctly iterate over
    all MFMA tiles within a K-tile.
    """
    if wave is None or wave.inner is None:
        return
    for mi in range(wave.repeats_m):
        for ni in range(wave.repeats_n):
            for ki in range(wave.repeats_k):
                ctx.set_index("wave", "mi", mi)
                ctx.set_index("wave", "ni", ni)
                ctx.set_index("wave", "ki", ki)
                visitor(wave.inner, ctx)

# ===================================================================
# Phase lists
# ===================================================================

WORKGROUP_EPILOGUE_PHASES = [
    TilePhase("store_d", phase_store_d),
]
