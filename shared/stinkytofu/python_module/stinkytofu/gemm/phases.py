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
from typing import Optional

from .asm_context import AsmContext
from .asm_transforms import emit_affine, GemmLayouts
from .problem import GemmProblem, TileConfig
from .tile import TileLevel, TilePhase, walk_tile_tree
from .transforms import Embed, Dim

__all__ = [
    "WORKGROUP_PROLOGUE_PHASES",
    "WORKGROUP_EPILOGUE_PHASES",
    "WAVE_PROLOGUE_PHASES",
    "WAVE_EPILOGUE_PHASES",
    "PIPELINED_PROLOGUE_PHASES",
    "default_mfma_visitor",
]


# ===================================================================
# Helpers: extract ctx._metadata
# ===================================================================

def _tile(ctx) -> TileConfig:
    return ctx._metadata["tile"]

def _problem(ctx) -> GemmProblem:
    return ctx._metadata["problem"]

def _layouts(ctx) -> GemmLayouts:
    return ctx._metadata["layouts"]


# ===================================================================
# Prologue phases
# ===================================================================

def phase_load_kernargs(level, ctx):
    """Load kernel arguments from the kernarg segment."""
    ctx.comment("Load kernel arguments")
    karg = ctx.sreg("s_kernarg")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_A"), karg, "0", comment="A ptr")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_B"), karg, "8", comment="B ptr")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_D"), karg, "16", comment="D ptr")
    ctx.inst("s_load_dword", ctx.sreg("s_M"), karg, "24", comment="M")
    ctx.inst("s_load_dword", ctx.sreg("s_N"), karg, "28", comment="N")
    ctx.inst("s_load_dword", ctx.sreg("s_K"), karg, "32", comment="K")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait for kernarg loads")
    ctx.raw("")


def phase_thread_indexing(level, ctx):
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


def phase_load_cluster_setup(level, ctx):
    """Compute global-load thread cluster coordinates (row, col)."""
    tile = _tile(ctx)
    problem = _problem(ctx)
    elem = problem.element_bytes
    elems_per_thread = (tile.wg_m * tile.unroll_k) // tile.block_size
    contiguous_k = min(elems_per_thread, tile.unroll_k)
    k_groups = max(1, tile.unroll_k // contiguous_k)

    ctx.comment(f"Global-load cluster: {tile.block_size // k_groups} rows x "
                f"{k_groups} K-groups ({contiguous_k} elems each)")
    if k_groups == 1:
        ctx.v_mov(ctx.vreg("v_gload_row"), ctx.vreg("v_tid"),
                  comment="row = tid (k_groups=1)")
        ctx.v_mov(ctx.vreg("v_gload_col"), "0", comment="col = 0")
    else:
        log2_kg = int(math.log2(k_groups))
        ctx.v_lshr(ctx.vreg("v_gload_row"), ctx.vreg("v_tid"), log2_kg,
                   comment=f"row = tid >> {log2_kg}")
        ctx.v_and(ctx.vreg("v_gload_col"), ctx.vreg("v_tid"), k_groups - 1,
                  comment=f"tid % {k_groups}")
        if contiguous_k > 1:
            log2_ck = int(math.log2(contiguous_k))
            ctx.v_lshl(ctx.vreg("v_gload_col"), ctx.vreg("v_gload_col"),
                       log2_ck, comment=f"* {contiguous_k} -> k_start")
    ctx.raw("")


def phase_lds_addrs(level, ctx):
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

    ctx.comment(f"LDS write B: {layouts.lds_b} + offset {layouts.lds_b_offset}")
    emit_affine(ctx, layouts.lds_b, wr_bindings,
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

    # LDS read A: a_row = wave_m * m_per_wave + lane_row
    lds_rd_a = Embed(
        [Dim("a_row", tile.wg_m), Dim("a_k", tile.unroll_k)],
        Dim("lds_rd_a", tile.wg_m * tile.unroll_k),
        [tile.unroll_k, 1],
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
        Dim("lds_rd_b", tile.wg_n * tile.unroll_k),
        [tile.unroll_k, 1],
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


def phase_init_acc(level, ctx):
    """Zero-initialize accumulator registers."""
    tile = _tile(ctx)
    acc_total = tile.mfma_m_repeat * tile.mfma_n_repeat * tile.mfma.acc_vgprs
    ctx.comment(f"Init {acc_total} accumulators to zero")
    for i in range(acc_total):
        ctx.inst("v_accvgpr_write_b32", ctx.areg("acc_C", i, 1), "0")
    ctx.raw("")


def phase_global_addrs(level, ctx):
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

        ctx.comment(f"Global address {name}: {layout} [{dim_name} coeff = s_K]")

        # global_row = wg_id * wg_size + thread_row
        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg(wg_id_s), str(wg_size),
                  comment=f"wg_id * {wg_size}")
        ctx.v_add(ctx.vreg("v_tmp0"), ctx.sreg("s_tmp0"),
                  ctx.vreg("v_gload_row"), comment="+ thread_row -> global_row")

        # offset = global_row * K + col  (K is dynamic, via transform)
        emit_affine(ctx, layout,
                    bindings={dim_name: ctx.vreg("v_tmp0"),
                              "k": ctx.vreg("v_gload_col")},
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
# K-loop phases
# ===================================================================

def phase_global_load(level, ctx):
    """Load A/B tiles from global memory into VGPRs."""
    problem = _problem(ctx)
    tile = _tile(ctx)
    _emit_global_load_impl(ctx, problem, tile)


def _emit_global_load_impl(ctx, problem, tile):
    """Shared implementation: emit global_load instructions for A and B."""
    for name, addr_name in [("A", "v_addr_a"), ("B", "v_addr_b")]:
        gload_name = f"v_gload_{name.lower()}"
        load = ctx.get(gload_name)
        addr = ctx.vreg(addr_name, 0, 2)
        ctx.comment(f"Global load {name} tile")
        for i in range(0, load.count, 4):
            cnt = min(4, load.count - i)
            width = {4: "dwordx4", 2: "dwordx2", 1: "dword"}[cnt]
            dst = ctx.vreg(gload_name, i, cnt)
            off = f"off offset:{i * 4}" if i > 0 else "off"
            ctx.inst(f"global_load_{width}", dst, addr, off,
                     comment=f"load {name}[{i}:{i+cnt}]")

    ctx.s_waitcnt("0", comment="wait for global loads")
    ctx.raw("")


def phase_lds_write(level, ctx):
    """Write loaded A/B data from VGPRs into LDS + barrier."""
    tile = _tile(ctx)
    for name in ["a", "b"]:
        load = ctx.get(f"v_gload_{name}")
        addr_reg = ctx.vreg(f"v_lds_wr_{name}")
        ctx.comment(f"LDS write {name.upper()}")
        for i in range(0, load.count, 4):
            cnt = min(4, load.count - i)
            src = ctx.vreg(f"v_gload_{name}", i, cnt)
            ctx.ds_write(addr_reg, src, offset=i * 4, width=cnt,
                         comment=f"LDS write {name.upper()}[{i}:{i+cnt}]")

    ctx.s_waitcnt("lgkmcnt(0)", comment="wait for LDS writes")
    ctx.s_barrier(comment="sync workgroup after LDS fill")
    ctx.raw("")


def phase_k_advance(level, ctx):
    """Advance A/B global pointers by unroll_k + barrier."""
    tile = _tile(ctx)
    k_stride = tile.unroll_k * _problem(ctx).element_bytes
    ctx.comment("Advance A, B pointers by unroll_k")
    for addr in ["v_addr_a", "v_addr_b"]:
        ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                 str(k_stride), ctx.vreg(addr, 0, 1),
                 comment=f"{addr} += {k_stride}")
        ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                 ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")
    ctx.raw("")
    ctx.s_barrier(comment="sync before next K-tile LDS write")


def phase_k_loop_control(level, ctx):
    """Decrement K-tile counter and branch."""
    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc1", "k_loop", comment="branch if k_tiles > 0")
    ctx.raw("")


def phase_k_loop_init(level, ctx):
    """Compute K-tile loop counter."""
    tile = _tile(ctx)
    log2_uk = int(math.log2(tile.unroll_k))
    ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
               comment=f"k_tiles = K / {tile.unroll_k}")
    ctx.raw("")


def phase_k_loop_label(level, ctx):
    """Emit the K-loop label."""
    ctx.label("k_loop")
    ctx.raw("")


# ===================================================================
# Store epilogue (uses transforms for address computation)
# ===================================================================

def phase_store_d(level, ctx):
    """Store accumulators to D using coordinate transforms.

    Uses emit_affine() with dynamic coefficient for the row stride
    (N is runtime).  Computes address once per (mi, ni) tile and
    increments by row stride for each accumulator element.
    """
    tile = _tile(ctx)
    problem = _problem(ctx)
    layouts = _layouts(ctx)
    mfma = tile.mfma
    acc_per = mfma.acc_vgprs
    elem = problem.element_bytes

    ctx.comment("Store D via transform")

    # MFMA 16x16x16 output: d_m = (lane_id/16)*4 + ai, d_n = lane_id % 16
    ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
              comment="lane_n = lane_id % 16")
    ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
               int(math.log2(mfma.m)), comment="lane_id / 16")
    ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_tmp1"), 2,
               comment="* 4 -> lane_m_base")

    # Row stride in bytes: N * elem (from transform coefficient)
    ctx.s_lshl(ctx.sreg("s_tmp0"), ctx.sreg("s_N"),
               int(math.log2(elem)),
               comment=f"row_stride = N * {elem} bytes")

    for mi in range(tile.mfma_m_repeat):
        for ni in range(tile.mfma_n_repeat):
            acc_base = (mi * tile.mfma_n_repeat + ni) * acc_per

            # Compute row = wg_m + wave_m*m_per_wave + mi*mfma_m + lane_m_base
            ctx.v_mul(ctx.vreg("v_addr_d", 0, 1), str(tile.m_per_wave),
                      ctx.vreg("v_wave_m"), comment=f"wave_m * {tile.m_per_wave}")
            ctx.v_add(ctx.vreg("v_addr_d", 0, 1),
                      ctx.vreg("v_addr_d", 0, 1), ctx.vreg("v_tmp1"),
                      comment="+ lane_m_base")
            row_imm = mi * mfma.m
            if row_imm:
                ctx.v_add(ctx.vreg("v_addr_d", 0, 1), str(row_imm),
                          ctx.vreg("v_addr_d", 0, 1),
                          comment=f"+ mi*mfma_m ({row_imm})")
            ctx.s_mul(ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_x"),
                      str(tile.wg_m), comment=f"wg_id_x * {tile.wg_m}")
            ctx.v_add(ctx.vreg("v_addr_d", 0, 1), ctx.sreg("s_tmp1"),
                      ctx.vreg("v_addr_d", 0, 1), comment="+ wg_base_m -> row")

            # Compute col = wg_n + wave_n*n_per_wave + ni*mfma_n + lane_n
            ctx.v_mul(ctx.vreg("v_addr_d", 1, 1), str(tile.n_per_wave),
                      ctx.vreg("v_wave_n"), comment=f"wave_n * {tile.n_per_wave}")
            ctx.v_add(ctx.vreg("v_addr_d", 1, 1),
                      ctx.vreg("v_addr_d", 1, 1), ctx.vreg("v_tmp0"),
                      comment="+ lane_n")
            col_imm = ni * mfma.n
            if col_imm:
                ctx.v_add(ctx.vreg("v_addr_d", 1, 1), str(col_imm),
                          ctx.vreg("v_addr_d", 1, 1),
                          comment=f"+ ni*mfma_n ({col_imm})")
            ctx.s_mul(ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_y"),
                      str(tile.wg_n), comment=f"wg_id_y * {tile.wg_n}")
            ctx.v_add(ctx.vreg("v_addr_d", 1, 1), ctx.sreg("s_tmp1"),
                      ctx.vreg("v_addr_d", 1, 1), comment="+ wg_base_n -> col")

            # offset = row * N + col  (via transform, N is dynamic)
            emit_affine(ctx, layouts.global_d,
                        bindings={"m": ctx.vreg("v_addr_d", 0, 1),
                                  "n": ctx.vreg("v_addr_d", 1, 1)},
                        result=ctx.vreg("v_addr_d", 0, 1),
                        dynamic_coefficients={"m": ctx.sreg("s_N")},
                        scale=elem,
                        comment=f"D offset via transform m{mi}_n{ni}")

            # 64-bit: D_ptr + byte_offset
            ctx.inst("v_add_co_u32", ctx.vreg("v_addr_d", 0, 1), "vcc",
                     ctx.sreg("s_ptr_D", 0, 1), ctx.vreg("v_addr_d", 0, 1),
                     comment="D_lo + offset")
            ctx.v_mov(ctx.vreg("v_addr_d", 1, 1),
                      ctx.sreg("s_ptr_D", 1, 1), comment="D_hi")
            ctx.inst("v_addc_co_u32", ctx.vreg("v_addr_d", 1, 1), "vcc",
                     ctx.vreg("v_addr_d", 1, 1), "0", "vcc",
                     comment="D_hi + carry")

            # Store 4 acc values, stride by row_stride_bytes
            for ai in range(acc_per):
                ctx.inst("v_accvgpr_read_b32", ctx.vreg("v_store_tmp"),
                         ctx.areg("acc_C", acc_base + ai, 1),
                         comment=f"acc[{acc_base + ai}]")
                ctx.inst("v_cvt_f16_f32_e32", ctx.vreg("v_store_tmp"),
                         ctx.vreg("v_store_tmp"), comment="f32->f16")
                ctx.inst("global_store_short",
                         ctx.vreg("v_addr_d", 0, 2),
                         ctx.vreg("v_store_tmp"), "off",
                         comment=f"store D m{mi}_n{ni}_a{ai}")
                if ai < acc_per - 1:
                    ctx.inst("v_add_co_u32", ctx.vreg("v_addr_d", 0, 1),
                             "vcc", ctx.sreg("s_tmp0"),
                             ctx.vreg("v_addr_d", 0, 1),
                             comment="next row (+N*elem)")
                    ctx.inst("v_addc_co_u32", ctx.vreg("v_addr_d", 1, 1),
                             "vcc", ctx.vreg("v_addr_d", 1, 1), "0", "vcc",
                             comment="carry")

    ctx.s_waitcnt("vmcnt(0)", comment="wait for stores")
    ctx.raw("")


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
    ctx.inst(f"v_mfma_f32_{mfma.m}x{mfma.n}x{mfma.k}_f16",
             ctx.areg("acc_C", acc_off, acc_per),
             ctx.vreg("v_a", 0, mfma.a_vgprs),
             ctx.vreg("v_b", 0, mfma.b_vgprs),
             ctx.areg("acc_C", acc_off, acc_per),
             comment=f"mfma m{mi}_n{ni} k{ki}")
    ctx.raw("")


# ===================================================================
# Pipelined K-loop (single phase, handles entire loop)
# ===================================================================

def _emit_wave_compute(ctx, wave, visitor):
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


def phase_pipelined_k_loop(level, ctx):
    """Entire K-loop with software pipelining.

    Overlaps global_load(n+1) with compute(n).  Skips prefetch on
    last iteration to avoid out-of-bounds loads.
    """
    tile = _tile(ctx)
    problem = _problem(ctx)
    kernel = ctx._metadata["kernel"]
    elem = problem.element_bytes
    k_stride = tile.unroll_k * elem

    # K-tile count
    ctx.comment("Pipelined K-loop setup")
    log2_uk = int(math.log2(tile.unroll_k))
    ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
               comment=f"k_tiles = K / {tile.unroll_k}")
    ctx.raw("")

    # Prefetch first tile
    ctx.comment("Prefetch first K-tile")
    _emit_global_load_impl(ctx, problem, tile)

    # Loop
    ctx.label("k_loop")
    ctx.raw("")

    # Write current VGPRs to LDS
    phase_lds_write(None, ctx)

    # Decrement + conditional prefetch
    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc0", "skip_prefetch",
             comment="skip prefetch on last iteration")
    ctx.raw("")

    # Advance pointers + async global load for next tile
    ctx.comment("Advance ptrs + prefetch next K-tile (async)")
    for addr in ["v_addr_a", "v_addr_b"]:
        ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                 str(k_stride), ctx.vreg(addr, 0, 1),
                 comment=f"{addr} += {k_stride}")
        ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                 ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")

    # Issue global loads (no waitcnt)
    for name, addr_name in [("A", "v_addr_a"), ("B", "v_addr_b")]:
        gload_name = f"v_gload_{name.lower()}"
        load = ctx.get(gload_name)
        addr = ctx.vreg(addr_name, 0, 2)
        for i in range(0, load.count, 4):
            cnt = min(4, load.count - i)
            width = {4: "dwordx4", 2: "dwordx2", 1: "dword"}[cnt]
            dst = ctx.vreg(gload_name, i, cnt)
            off = f"off offset:{i * 4}" if i > 0 else "off"
            ctx.inst(f"global_load_{width}", dst, addr, off,
                     comment=f"prefetch {name}[{i}:{i+cnt}]")
    ctx.raw("")

    ctx.label("skip_prefetch")
    ctx.raw("")

    # Compute: walk wave with full mi/ni/ki iteration
    ctx.comment("Compute: MFMA from LDS (overlaps with global_load)")
    wave = kernel.tile_tree.find("wave")
    _emit_wave_compute(ctx, wave, kernel.mfma_visitor)
    ctx.raw("")

    # Barriers + wait + loop control
    ctx.s_barrier(comment="sync before next K-tile LDS write")
    ctx.s_waitcnt("0", comment="wait for any pending global_load")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc1", "k_loop", comment="branch if k_tiles > 0")
    ctx.raw("")


# ===================================================================
# Phase lists (ready to use in TileLevel constructors)
# ===================================================================

WORKGROUP_PROLOGUE_PHASES = [
    TilePhase("load_kernargs", phase_load_kernargs),
    TilePhase("thread_indexing", phase_thread_indexing),
    TilePhase("load_cluster_setup", phase_load_cluster_setup),
    TilePhase("lds_addrs", phase_lds_addrs),
    TilePhase("init_acc", phase_init_acc),
    TilePhase("global_addrs", phase_global_addrs),
    TilePhase("k_loop_init", phase_k_loop_init),
    TilePhase("k_loop_label", phase_k_loop_label),
]

WORKGROUP_EPILOGUE_PHASES = [
    TilePhase("store_d", phase_store_d),
]

WAVE_PROLOGUE_PHASES = [
    TilePhase("global_load", phase_global_load),
    TilePhase("lds_write", phase_lds_write),
]

WAVE_EPILOGUE_PHASES = [
    TilePhase("k_advance", phase_k_advance),
    TilePhase("k_loop_control", phase_k_loop_control),
]

PIPELINED_PROLOGUE_PHASES = [
    TilePhase("load_kernargs", phase_load_kernargs),
    TilePhase("thread_indexing", phase_thread_indexing),
    TilePhase("load_cluster_setup", phase_load_cluster_setup),
    TilePhase("lds_addrs", phase_lds_addrs),
    TilePhase("init_acc", phase_init_acc),
    TilePhase("global_addrs", phase_global_addrs),
    TilePhase("pipelined_k_loop", phase_pipelined_k_loop),
]


# ===================================================================
# Optimized K-loop: double-buffered LDS + pipelining + interleaved
# MFMA/LR + fine-grained waitcnt
# ===================================================================

def phase_optimized_k_loop(level, ctx):
    """K-loop with fully interleaved instruction scheduling.

    All overhead (advance, global_load, LDS toggle, ds_write) is
    interleaved between MFMAs to execute during MFMA pipeline time.
    Double-buffered LDS. Subtile-scheduled MFMA/LR with ds_read_b64.
    """
    tile = _tile(ctx)
    problem = _problem(ctx)
    kernel = ctx._metadata["kernel"]
    elem = problem.element_bytes
    mfma = tile.mfma

    lds_half = (tile.wg_m + tile.wg_n) * tile.unroll_k * elem
    k_stride = tile.unroll_k * elem
    log2_uk = int(math.log2(tile.unroll_k))

    ctx.alloc_sgpr_permanent(1, "s_lds_db_step")

    # K-tile count
    ctx.comment("=== Fully interleaved K-loop ===")
    ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
               comment=f"k_tiles = K / {tile.unroll_k}")
    ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_half),
              comment=f"DB toggle step = {lds_half}")
    ctx.raw("")

    # Prefetch first tile + write to LDS
    ctx.comment("Prefetch first K-tile")
    _emit_global_load_impl(ctx, problem, tile)
    ctx.comment("Write first tile to LDS buf[0]")
    _emit_lds_write_impl(ctx, tile)

    # K-loop
    ctx.label("k_loop")
    ctx.raw("")

    # Decrement + conditional prefetch
    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc0", "skip_prefetch",
             comment="skip prefetch on last iteration")

    # Advance pointers
    for addr in ["v_addr_a", "v_addr_b"]:
        ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                 str(k_stride), ctx.vreg(addr, 0, 1),
                 comment=f"{addr} += {k_stride}")
        ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                 ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")

    # Issue global loads (async, no wait)
    _emit_global_load_no_wait(ctx, problem, tile)
    ctx.raw("")

    ctx.label("skip_prefetch")
    ctx.raw("")

    # Subtile compute with interleaved post-compute overhead
    ctx.comment("Subtile compute + interleaved overhead")
    _emit_scheduled_compute(ctx, tile, problem)

    # Post-compute: wait for global_load, toggle LDS, write, barrier
    ctx.s_waitcnt("vmcnt(0)", comment="wait for global_load")

    ctx.comment("Toggle LDS double-buffer offsets")
    for reg in ["v_lds_wr_a", "v_lds_wr_b", "v_lds_rd_a", "v_lds_rd_b"]:
        ctx.v_add(ctx.vreg(reg), ctx.sreg("s_lds_db_step"), ctx.vreg(reg),
                  comment=f"{reg} += db_step")
    ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
             ctx.sreg("s_lds_db_step"),
             comment="negate step for next toggle")
    ctx.raw("")

    ctx.comment("Write next tile to other LDS buffer")
    _emit_lds_write_impl(ctx, tile)

    # Loop control
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc1", "k_loop",
             comment="branch if k_tiles > 0")
    ctx.raw("")


def _emit_lds_write_impl(ctx, tile):
    """Emit LDS write + waitcnt + barrier (shared by all K-loop variants)."""
    for name in ["a", "b"]:
        load = ctx.get(f"v_gload_{name}")
        addr_reg = ctx.vreg(f"v_lds_wr_{name}")
        for i in range(0, load.count, 4):
            cnt = min(4, load.count - i)
            src = ctx.vreg(f"v_gload_{name}", i, cnt)
            ctx.ds_write(addr_reg, src, offset=i * 4, width=cnt,
                         comment=f"LDS write {name.upper()}[{i}:{i+cnt}]")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait LDS writes")
    ctx.s_barrier(comment="sync workgroup")
    ctx.raw("")


def _emit_global_load_no_wait(ctx, problem, tile):
    """Issue global loads for A and B without waitcnt (async)."""
    for name, addr_name in [("A", "v_addr_a"), ("B", "v_addr_b")]:
        gload_name = f"v_gload_{name.lower()}"
        load = ctx.get(gload_name)
        addr = ctx.vreg(addr_name, 0, 2)
        for i in range(0, load.count, 4):
            cnt = min(4, load.count - i)
            width = {4: "dwordx4", 2: "dwordx2", 1: "dword"}[cnt]
            dst = ctx.vreg(gload_name, i, cnt)
            off = f"off offset:{i * 4}" if i > 0 else "off"
            ctx.inst(f"global_load_{width}", dst, addr, off,
                     comment=f"prefetch {name}[{i}:{i+cnt}]")


def _emit_interleaved_compute(ctx, tile, problem):
    """Emit all MFMA tiles with double-buffered LDS read interleaving.

    Issues ds_read for MFMA[i+1] during MFMA[i] (32-cycle MFMA
    hides 20-cycle ds_read latency). Uses v_a/v_b and v_a2/v_b2
    as alternating operand buffers.
    """
    mfma = tile.mfma
    elem = problem.element_bytes

    a_bufs = ["v_a", "v_a2"]
    b_bufs = ["v_b", "v_b2"]

    # Flatten iteration order
    iters = []
    for mi in range(tile.mfma_m_repeat):
        for ni in range(tile.mfma_n_repeat):
            for ki in range(tile.k_iterations):
                iters.append((mi, ni, ki))

    def emit_lr(idx, buf):
        """Emit LDS reads for iteration idx into buffer buf."""
        mi, ni, ki = iters[idx]
        a_name, b_name = a_bufs[buf], b_bufs[buf]

        a_off = (mi * mfma.m * tile.unroll_k + ki * mfma.k) * elem
        if a_off > 0:
            ctx.v_add(ctx.vreg("v_tmp0"), str(a_off),
                      ctx.vreg("v_lds_rd_a"),
                      comment=f"LR A addr m{mi}k{ki}")
            a_addr = ctx.vreg("v_tmp0")
        else:
            a_addr = ctx.vreg("v_lds_rd_a")
        for r in range(mfma.a_vgprs):
            ctx.ds_read(ctx.vreg(a_name, r, 1), a_addr,
                        offset=r * 4, width=1,
                        comment=f"LR A[{r}] m{mi}k{ki}")

        b_off = (ni * mfma.n * tile.unroll_k + ki * mfma.k) * elem
        if b_off > 0:
            ctx.v_add(ctx.vreg("v_tmp1"), str(b_off),
                      ctx.vreg("v_lds_rd_b"),
                      comment=f"LR B addr n{ni}k{ki}")
            b_addr = ctx.vreg("v_tmp1")
        else:
            b_addr = ctx.vreg("v_lds_rd_b")
        for r in range(mfma.b_vgprs):
            ctx.ds_read(ctx.vreg(b_name, r, 1), b_addr,
                        offset=r * 4, width=1,
                        comment=f"LR B[{r}] n{ni}k{ki}")

    # Pre-load first operands
    emit_lr(0, buf=0)
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait first LDS reads")

    for i, (mi, ni, ki) in enumerate(iters):
        cur_buf = i % 2

        # Prefetch next operands (hidden by MFMA latency)
        if i < len(iters) - 1:
            emit_lr(i + 1, buf=1 - cur_buf)

        # MFMA
        acc_per = mfma.acc_vgprs
        acc_off = (mi * tile.mfma_n_repeat + ni) * acc_per
        ctx.inst(
            f"v_mfma_f32_{mfma.m}x{mfma.n}x{mfma.k}_f16",
            ctx.areg("acc_C", acc_off, acc_per),
            ctx.vreg(a_bufs[cur_buf], 0, mfma.a_vgprs),
            ctx.vreg(b_bufs[cur_buf], 0, mfma.b_vgprs),
            ctx.areg("acc_C", acc_off, acc_per),
            comment=f"mfma m{mi}_n{ni}_k{ki}")

        # Wait for prefetch (MFMA latency covers most of it)
        if i < len(iters) - 1:
            ctx.s_waitcnt("lgkmcnt(0)", comment="wait LR")


# Phase list for optimized K-loop
OPTIMIZED_PROLOGUE_PHASES = [
    TilePhase("load_kernargs", phase_load_kernargs),
    TilePhase("thread_indexing", phase_thread_indexing),
    TilePhase("load_cluster_setup", phase_load_cluster_setup),
    TilePhase("lds_addrs", phase_lds_addrs),
    TilePhase("init_acc", phase_init_acc),
    TilePhase("global_addrs", phase_global_addrs),
    TilePhase("optimized_k_loop", phase_optimized_k_loop),
]


# ===================================================================
# Scheduled K-loop: uses TileOp-based scheduling with interleaved
# global loads between MFMAs (see scheduled_codegen.py / DESIGN.md)
# ===================================================================

def phase_scheduled_k_loop(level, ctx):
    """K-loop using the three-layer scheduled codegen.

    Generates TileOps from the tile config, schedules them into MFMA
    slots with interleaved global loads and LDS writes, then emits
    assembly by walking the schedule.
    """
    from .scheduled_codegen import TilePlan, emit_scheduled_kernel
    from .schedule import SchedulingRules

    tile = _tile(ctx)
    problem = _problem(ctx)
    layouts = _layouts(ctx)
    elem = problem.element_bytes
    mfma = tile.mfma

    lds_half = (tile.wg_m + tile.wg_n) * tile.unroll_k * elem
    k_stride = tile.unroll_k * elem
    log2_uk = int(math.log2(tile.unroll_k))

    ctx.alloc_sgpr_permanent(1, "s_lds_db_step")

    # K-tile count
    ctx.comment("=== Scheduled K-loop ===")
    ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
               comment=f"k_tiles = K / {tile.unroll_k}")
    ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_half),
              comment=f"DB toggle step = {lds_half}")
    ctx.raw("")

    # Prefetch first tile + write to LDS
    ctx.comment("Prefetch first K-tile")
    _emit_global_load_impl(ctx, problem, tile)
    ctx.comment("Write first tile to LDS buf[0]")
    _emit_lds_write_impl(ctx, tile)

    # Build tile plan and schedule
    plan = TilePlan.build(tile, problem)
    schedule = plan.schedule(SchedulingRules())

    # K-loop
    ctx.label("k_loop")
    ctx.raw("")

    # Decrement + conditional prefetch
    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc0", "skip_prefetch",
             comment="skip prefetch on last iteration")

    # Advance pointers
    for addr in ["v_addr_a", "v_addr_b"]:
        ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                 str(k_stride), ctx.vreg(addr, 0, 1),
                 comment=f"{addr} += {k_stride}")
        ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                 ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")

    # Issue global loads (async, no wait) -- interleaved in schedule
    _emit_global_load_no_wait(ctx, problem, tile)
    ctx.raw("")

    ctx.label("skip_prefetch")
    ctx.raw("")

    # Emit scheduled compute (MFMAs with interleaved side ops)
    ctx.comment("Scheduled compute")
    emit_scheduled_kernel(schedule, ctx, tile, problem, layouts)

    # Post-compute: wait for global_load, toggle LDS, write, barrier
    ctx.s_waitcnt("vmcnt(0)", comment="wait for global_load")

    ctx.comment("Toggle LDS double-buffer offsets")
    for reg in ["v_lds_wr_a", "v_lds_wr_b", "v_lds_rd_a", "v_lds_rd_b"]:
        ctx.v_add(ctx.vreg(reg), ctx.sreg("s_lds_db_step"), ctx.vreg(reg),
                  comment=f"{reg} += db_step")
    ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
             ctx.sreg("s_lds_db_step"),
             comment="negate step for next toggle")
    ctx.raw("")

    ctx.comment("Write next tile to other LDS buffer")
    _emit_lds_write_impl(ctx, tile)

    # Loop control
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc1", "k_loop",
             comment="branch if k_tiles > 0")
    ctx.raw("")


SCHEDULED_PROLOGUE_PHASES = [
    TilePhase("load_kernargs", phase_load_kernargs),
    TilePhase("thread_indexing", phase_thread_indexing),
    TilePhase("load_cluster_setup", phase_load_cluster_setup),
    TilePhase("lds_addrs", phase_lds_addrs),
    TilePhase("init_acc", phase_init_acc),
    TilePhase("global_addrs", phase_global_addrs),
    TilePhase("scheduled_k_loop", phase_scheduled_k_loop),
]


# ===================================================================
# Subtile-scheduled compute: group 4 MFMAs between reads
# ===================================================================

def _emit_subtile_compute_legacy(ctx, tile, problem):
    """Subtile-scheduled MFMA with ds_read_b64 and merged ki.

    All B values for all (ni, ki) loaded upfront. A double-buffered
    with prefetch between mi groups. 8 MFMAs per group (4 ni * 2 ki)
    fully hide the A prefetch latency.

    ds_read uses the offset field (no VALU addr computation needed).
    """
    mfma = tile.mfma
    elem = problem.element_bytes
    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    ki_count = tile.k_iterations
    av = mfma.a_vgprs
    bv = mfma.b_vgprs

    # Allocate B buffers: one per (ni, ki)
    b_names = {}
    for ni in range(nr):
        for ki in range(ki_count):
            name = f"v_b_s{ni}k{ki}"
            if not ctx.has(name):
                ctx.alloc_vgpr_permanent(bv, name)
            b_names[(ni, ki)] = name

    # A double-buffer: one set per (buf, ki)
    a_names = {}
    for buf in range(2):
        for ki in range(ki_count):
            name = f"v_a_b{buf}k{ki}"
            if not ctx.has(name):
                ctx.alloc_vgpr_permanent(av, name)
            a_names[(buf, ki)] = name

    def a_off(mi, ki):
        return (mi * mfma.m * tile.unroll_k + ki * mfma.k) * elem

    def b_off(ni, ki):
        return (ni * mfma.n * tile.unroll_k + ki * mfma.k) * elem

    def read_a(mi, ki, buf):
        name = a_names[(buf, ki)]
        ctx.ds_read(ctx.vreg(name, 0, av), ctx.vreg("v_lds_rd_a"),
                    offset=a_off(mi, ki), width=av,
                    comment=f"LR A m{mi}k{ki} buf{buf}")

    def read_b(ni, ki):
        name = b_names[(ni, ki)]
        ctx.ds_read(ctx.vreg(name, 0, bv), ctx.vreg("v_lds_rd_b"),
                    offset=b_off(ni, ki), width=bv,
                    comment=f"LR B n{ni}k{ki}")

    def do_mfma(mi, ni, ki, a_buf):
        acc_per = mfma.acc_vgprs
        acc_off = (mi * nr + ni) * acc_per
        ctx.inst(
            f"v_mfma_f32_{mfma.m}x{mfma.n}x{mfma.k}_f16",
            ctx.areg("acc_C", acc_off, acc_per),
            ctx.vreg(a_names[(a_buf, ki)], 0, av),
            ctx.vreg(b_names[(ni, ki)], 0, bv),
            ctx.areg("acc_C", acc_off, acc_per),
            comment=f"mfma m{mi}_n{ni}_k{ki}")

    total_mfma = mr * nr * ki_count
    ctx.comment(f"Subtile: {mr}m x {nr}n x {ki_count}k = {total_mfma} MFMAs")

    # Load ALL B values for all (ni, ki) upfront
    for ki in range(ki_count):
        for ni in range(nr):
            read_b(ni, ki)

    # Load A[mi=0] for all ki
    cur_a = 0
    for ki in range(ki_count):
        read_a(0, ki, cur_a)

    total_initial_reads = nr * ki_count + ki_count
    ctx.s_waitcnt("lgkmcnt(0)",
                  comment=f"wait {total_initial_reads} initial reads")
    ctx.raw("")

    # For each mi: prefetch A[mi+1], execute nr*ki_count MFMAs
    for mi in range(mr):
        has_prefetch = mi < mr - 1
        if has_prefetch:
            next_a = 1 - cur_a
            for ki in range(ki_count):
                read_a(mi + 1, ki, next_a)

        # Execute all MFMAs for this mi
        for ki in range(ki_count):
            for ni in range(nr):
                do_mfma(mi, ni, ki, cur_a)

        # Wait for A prefetch (hidden by nr*ki_count MFMAs)
        if has_prefetch:
            ctx.s_waitcnt("lgkmcnt(0)",
                          comment=f"wait A[{mi+1}] ({nr*ki_count} MFMAs hid)")
            cur_a = next_a

        ctx.raw("")


# ===================================================================
# Automated subtile scheduler with preamble
# ===================================================================

def _emit_scheduled_compute(ctx, tile, problem):
    """Automated subtile scheduler with preamble + interleaved A prefetch.

    Structure:
      Preamble: all B reads + A[mi=0] reads -> waitcnt (one-time)
      Per mi group: prefetch A[mi+1] -> 8 MFMAs -> waitcnt
      MFMA order: (mi, ki, ni) to maximize A operand reuse

    The preamble loads all data needed for mi=0. Each subsequent mi
    group prefetches A for the next mi during the current group's
    8 MFMAs (32 cycles >> 20 cycle ds_read latency).
    """
    mfma = tile.mfma
    elem = problem.element_bytes
    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    ki_count = tile.k_iterations
    av = mfma.a_vgprs
    bv = mfma.b_vgprs

    # Ensure registers allocated
    b_names = {}
    for ni in range(nr):
        for ki in range(ki_count):
            name = f"v_b_s{ni}k{ki}"
            if not ctx.has(name):
                ctx.alloc_vgpr_permanent(bv, name)
            b_names[(ni, ki)] = name

    a_names = {}
    for buf in range(2):
        for ki in range(ki_count):
            name = f"v_a_b{buf}k{ki}"
            if not ctx.has(name):
                ctx.alloc_vgpr_permanent(av, name)
            a_names[(buf, ki)] = name

    def a_off(mi, ki):
        return (mi * mfma.m * tile.unroll_k + ki * mfma.k) * elem

    def b_off(ni, ki):
        return (ni * mfma.n * tile.unroll_k + ki * mfma.k) * elem

    total = mr * nr * ki_count
    ctx.comment(f"Scheduled: {total} MFMAs ({mr}m x {nr}n x {ki_count}k)")

    # Preamble: load all B + A[0]
    for ki in range(ki_count):
        for ni in range(nr):
            ctx.ds_read(ctx.vreg(b_names[(ni, ki)], 0, bv),
                        ctx.vreg("v_lds_rd_b"),
                        offset=b_off(ni, ki), width=bv,
                        comment=f"pre B n{ni}k{ki}")

    cur_a = 0
    for ki in range(ki_count):
        ctx.ds_read(ctx.vreg(a_names[(cur_a, ki)], 0, av),
                    ctx.vreg("v_lds_rd_a"),
                    offset=a_off(0, ki), width=av,
                    comment=f"pre A m0k{ki} b{cur_a}")

    ctx.s_waitcnt("lgkmcnt(0)", comment="wait preamble")
    ctx.raw("")

    # Per-mi group: prefetch A[mi+1] + nr*ki_count MFMAs
    for mi in range(mr):
        # Prefetch A for next mi (if not last)
        if mi < mr - 1:
            next_a = 1 - cur_a
            for ki in range(ki_count):
                ctx.ds_read(ctx.vreg(a_names[(next_a, ki)], 0, av),
                            ctx.vreg("v_lds_rd_a"),
                            offset=a_off(mi + 1, ki), width=av,
                            comment=f"LR A m{mi+1}k{ki} b{next_a}")

        # MFMAs: iterate ki then ni (group by A operand)
        for ki in range(ki_count):
            for ni in range(nr):
                acc_per = mfma.acc_vgprs
                acc_off = (mi * nr + ni) * acc_per
                ctx.inst(
                    f"v_mfma_f32_{mfma.m}x{mfma.n}x{mfma.k}_f16",
                    ctx.areg("acc_C", acc_off, acc_per),
                    ctx.vreg(a_names[(cur_a, ki)], 0, av),
                    ctx.vreg(b_names[(ni, ki)], 0, bv),
                    ctx.areg("acc_C", acc_off, acc_per),
                    comment=f"MFMA m{mi}_n{ni}_k{ki}")

        # Wait for A prefetch (hidden by nr*ki_count MFMAs)
        if mi < mr - 1:
            ctx.s_waitcnt("lgkmcnt(0)",
                          comment=f"wait A[{mi+1}] ({nr*ki_count} MFMAs hid)")
            cur_a = next_a

        ctx.raw("")
