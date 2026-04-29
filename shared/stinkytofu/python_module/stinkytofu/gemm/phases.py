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
    "INTERLEAVED_PROLOGUE_PHASES",
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
    """Compute global-load thread cluster coordinates for A and B.

    For symmetric tiles (wg_m == wg_n), A and B use the same mapping.
    For asymmetric tiles, B gets its own row/col computed from wg_n.
    """
    tile = _tile(ctx)
    problem = _problem(ctx)
    elem = problem.element_bytes

    def _emit_cluster(wg_dim, row_reg, col_reg, label):
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

    pad_e = tile.lds_pad // elem if tile.lds_pad > 0 else 0
    lds_half = (tile.wg_m + tile.wg_n) * (tile.unroll_k + pad_e) * elem
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

    pad_e = tile.lds_pad // elem if tile.lds_pad > 0 else 0
    lds_half = (tile.wg_m + tile.wg_n) * (tile.unroll_k + pad_e) * elem
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
    # For large tiles (128+ MFMAs), interleave global loads in compute.
    # For small tiles (16 MFMAs), keep loads in loop prefix (less disruption).
    use_interleaved = tile.total_mfma_per_wave >= 64
    schedule = plan.schedule(SchedulingRules(),
                             interleave_global_loads=use_interleaved)

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

    # Advance pointers + issue global loads
    for addr in ["v_addr_a", "v_addr_b"]:
        ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                 str(k_stride), ctx.vreg(addr, 0, 1),
                 comment=f"{addr} += {k_stride}")
        ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                 ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")

    # Always emit global loads in the loop prefix (async, no wait).
    # For tiles using partitioned compute, global loads are NOT interleaved
    # in the compute section -- they run here before compute starts.
    _emit_global_load_no_wait(ctx, problem, tile)
    ctx.raw("")

    ctx.label("skip_prefetch")
    ctx.raw("")

    # Emit scheduled compute (MFMAs with interleaved side ops)
    ctx.comment("Scheduled compute")
    # Preamble approach: load all B + A[0] upfront, then per-mi groups.
    # For 64 MFMAs (256x256x32), VGPR budget is ~92 + 256 acc = 348, fits fine.
    # Only fall back to partitioned compute for 256+ MFMAs where B regs exceed budget.
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
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (mi * mfma.m * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem

    def b_off(ni, ki):
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (ni * mfma.n * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem

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
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (mi * mfma.m * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem

    def b_off(ni, ki):
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (ni * mfma.n * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem

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


# ===================================================================
# Fully-interleaved K-loop: ALL overhead between MFMAs
# ===================================================================

def phase_fully_interleaved_k_loop(level, ctx):
    """K-loop with ALL overhead instructions interleaved between MFMAs.

    Unlike phase_scheduled_k_loop which places PREFIX/SUFFIX in
    sequential blocks, this version distributes every non-compute
    instruction (k_tiles--, branch, ptr_advance, global_load, vmcnt,
    toggle, ds_write, barrier) into the gaps between MFMA instructions.

    Structure for mr x nr MFMAs (e.g. 4x4 = 16 MFMAs):
      PREAMBLE: all B reads + A[0] reads -> waitcnt
      mi=0 group: prefetch A[1], interleaved PREFIX ops + MFMAs
      mi=1 group: prefetch A[2], interleaved global_loads + MFMAs
      mi=2 group: prefetch A[3], interleaved SUFFIX ops (vmcnt, toggle) + MFMAs
      mi=3 group: interleaved ds_writes + MFMAs
      POSTAMBLE: lgkmcnt(0), barrier, loop branch

    Supports any (mr, nr, ki_count) tile config. Overhead instructions
    are distributed round-robin across the nr*ki_count MFMAs per group.
    """
    tile = _tile(ctx)
    problem = _problem(ctx)
    elem = problem.element_bytes
    mfma = tile.mfma

    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    ki_count = tile.k_iterations
    av = mfma.a_vgprs
    bv = mfma.b_vgprs

    pad_e = tile.lds_pad // elem if tile.lds_pad > 0 else 0
    lds_half = (tile.wg_m + tile.wg_n) * (tile.unroll_k + pad_e) * elem
    k_stride = tile.unroll_k * elem
    log2_uk = int(math.log2(tile.unroll_k))

    ctx.alloc_sgpr_permanent(1, "s_lds_db_step")

    # === Setup ===
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

    # === Allocate operand registers ===
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
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (mi * mfma.m * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem

    def b_off(ni, ki):
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (ni * mfma.n * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem

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
            comment=f"MFMA m{mi}_n{ni}_k{ki}")

    # === Build overhead instruction lists ===
    # These are closures that emit one instruction each.

    # PREFIX ops: k_tiles--, conditional branch, ptr advance (4 VALU),
    # global loads
    prefix_ops = []
    prefix_ops.append(lambda: ctx.s_sub(
        ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
        comment="k_tiles--"))
    prefix_ops.append(lambda: ctx.inst(
        "s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
        comment="SCC = (k_tiles != 0)"))
    prefix_ops.append(lambda: ctx.inst(
        "s_cbranch_scc0", "skip_gload",
        comment="skip gload on last iteration"))

    # Pointer advance: 4 VALU instructions (add_lo, addc_hi for A and B)
    for addr in ["v_addr_a", "v_addr_b"]:
        _addr = addr  # capture for closure
        prefix_ops.append(lambda _a=_addr: ctx.inst(
            "v_add_co_u32", ctx.vreg(_a, 0, 1), "vcc",
            str(k_stride), ctx.vreg(_a, 0, 1),
            comment=f"{_a} += {k_stride}"))
        prefix_ops.append(lambda _a=_addr: ctx.inst(
            "v_addc_co_u32", ctx.vreg(_a, 1, 1), "vcc",
            ctx.vreg(_a, 1, 1), "0", "vcc", comment="carry"))

    # Global loads (conditional -- after skip_gload label they are skipped)
    gload_ops = []
    for name, addr_name in [("A", "v_addr_a"), ("B", "v_addr_b")]:
        gload_name = f"v_gload_{name.lower()}"
        load = ctx.get(gload_name)
        for i in range(0, load.count, 4):
            cnt = min(4, load.count - i)
            width = {4: "dwordx4", 2: "dwordx2", 1: "dword"}[cnt]
            _gn = gload_name
            _an = addr_name
            _i, _cnt, _w, _nm = i, cnt, width, name
            gload_ops.append(lambda _gn=_gn, _an=_an, _i=_i, _cnt=_cnt,
                             _w=_w, _nm=_nm: ctx.inst(
                f"global_load_{_w}",
                ctx.vreg(_gn, _i, _cnt),
                ctx.vreg(_an, 0, 2),
                f"off offset:{_i * 4}" if _i > 0 else "off",
                comment=f"gload {_nm}[{_i}:{_i+_cnt}]"))

    # SUFFIX ops: vmcnt(0), toggle (5 ops), ds_writes
    suffix_pre_write = []
    suffix_pre_write.append(lambda: ctx.s_waitcnt(
        "vmcnt(0)", comment="wait for global_load"))
    for reg in ["v_lds_wr_a", "v_lds_wr_b", "v_lds_rd_a", "v_lds_rd_b"]:
        _reg = reg
        suffix_pre_write.append(lambda _r=_reg: ctx.v_add(
            ctx.vreg(_r), ctx.sreg("s_lds_db_step"), ctx.vreg(_r),
            comment=f"{_r} += db_step"))
    suffix_pre_write.append(lambda: ctx.inst(
        "s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
        ctx.sreg("s_lds_db_step"),
        comment="negate step for next toggle"))

    ds_write_ops = []
    for name in ["a", "b"]:
        load = ctx.get(f"v_gload_{name}")
        for i in range(0, load.count, 4):
            cnt = min(4, load.count - i)
            _name, _i, _cnt = name, i, cnt
            ds_write_ops.append(lambda _n=_name, _i=_i, _cnt=_cnt: ctx.ds_write(
                ctx.vreg(f"v_lds_wr_{_n}"),
                ctx.vreg(f"v_gload_{_n}", _i, _cnt),
                offset=_i * 4, width=_cnt,
                comment=f"ds_write {_n.upper()}[{_i}:{_i+_cnt}]"))

    # === Distribute overhead across mi groups ===
    # Total overhead ops to interleave:
    #   prefix_ops (7) + gload_ops (varies) + suffix_pre_write (6)
    #   + ds_write_ops (varies)
    # We assign them to mi groups as follows:
    #   mi=0: prefix_ops (counter, branch, ptr advance)
    #   mi=1: gload_ops (conditional global loads, skip_gload label after)
    #   mi=2: suffix_pre_write (vmcnt, toggle/negate)
    #   mi=3+: ds_write_ops
    # For mr < 4, we compress: combine groups into fewer mi iterations.

    all_overhead = []
    # Group 0: prefix
    all_overhead.append(("prefix", prefix_ops))
    # Group 1: global loads (these are conditional)
    all_overhead.append(("gload", gload_ops))
    # Group 2: suffix pre-write (vmcnt + toggle)
    all_overhead.append(("suffix_pre_write", suffix_pre_write))
    # Group 3: ds writes
    all_overhead.append(("ds_write", ds_write_ops))

    # If mr < 4, merge groups to fit within available mi groups.
    # The merged list preserves ordering constraints.
    if mr < len(all_overhead):
        merged = [[] for _ in range(mr)]
        for idx, (tag, ops) in enumerate(all_overhead):
            target = min(idx, mr - 1)
            merged[target].extend(ops)
        overhead_per_mi = [(f"group{i}", merged[i]) for i in range(mr)]
    else:
        # Distribute: first 4 groups get assigned, remaining mi groups
        # get empty lists (just compute, no overhead)
        overhead_per_mi = list(all_overhead)
        for i in range(len(all_overhead), mr):
            overhead_per_mi.append((f"compute_only_{i}", []))

    # === K-loop ===
    ctx.label("k_loop")
    ctx.raw("")

    # Preamble: load all B + A[0]
    total = mr * nr * ki_count
    ctx.comment(f"Interleaved: {total} MFMAs ({mr}m x {nr}n x {ki_count}k)")

    for ki in range(ki_count):
        for ni in range(nr):
            read_b(ni, ki)

    cur_a = 0
    for ki in range(ki_count):
        read_a(0, ki, cur_a)

    ctx.s_waitcnt("lgkmcnt(0)", comment="wait preamble")
    ctx.raw("")

    # Per-mi group: A prefetch + interleaved overhead + MFMAs
    for mi in range(mr):
        group_tag, group_ops = overhead_per_mi[mi]
        mfmas_this_group = nr * ki_count
        has_prefetch = mi < mr - 1

        # Prefetch A for next mi
        if has_prefetch:
            next_a = 1 - cur_a
            for ki in range(ki_count):
                read_a(mi + 1, ki, next_a)

        # Emit skip_gload label before the gload group's MFMAs
        # so that conditional branch skips exactly the gload ops.
        if group_tag == "gload":
            # gload ops are conditional: emit them before MFMAs, after
            # the branch in the previous group already set up the skip.
            for op in group_ops:
                op()
            ctx.label("skip_gload")
            ctx.raw("")
            # MFMAs with no interleaved ops (gloads already emitted)
            for ki in range(ki_count):
                for ni in range(nr):
                    do_mfma(mi, ni, ki, cur_a)
        else:
            # Interleave overhead ops between MFMAs in this group
            op_idx = 0
            mfma_idx = 0
            for ki in range(ki_count):
                for ni in range(nr):
                    # Emit overhead op(s) before this MFMA
                    if op_idx < len(group_ops) and mfma_idx < mfmas_this_group:
                        # Distribute: emit ~ceil(remaining_ops / remaining_mfmas) ops
                        remaining_ops = len(group_ops) - op_idx
                        remaining_mfmas = mfmas_this_group - mfma_idx
                        ops_now = max(1, (remaining_ops + remaining_mfmas - 1) // remaining_mfmas)
                        for _ in range(ops_now):
                            if op_idx < len(group_ops):
                                group_ops[op_idx]()
                                op_idx += 1
                    do_mfma(mi, ni, ki, cur_a)
                    mfma_idx += 1

            # Emit any remaining ops after the last MFMA
            while op_idx < len(group_ops):
                group_ops[op_idx]()
                op_idx += 1

        # Wait for A prefetch
        if has_prefetch:
            ctx.s_waitcnt("lgkmcnt(0)",
                          comment=f"wait A[{mi+1}] ({mfmas_this_group} MFMAs hid)")
            cur_a = next_a

        ctx.raw("")

    # Postamble: wait for ds_writes, barrier, loop branch
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait for LDS writes")
    ctx.s_barrier(comment="sync workgroup after LDS fill")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc1", "k_loop",
             comment="branch if k_tiles > 0")
    ctx.raw("")


INTERLEAVED_PROLOGUE_PHASES = [
    TilePhase("load_kernargs", phase_load_kernargs),
    TilePhase("thread_indexing", phase_thread_indexing),
    TilePhase("load_cluster_setup", phase_load_cluster_setup),
    TilePhase("lds_addrs", phase_lds_addrs),
    TilePhase("init_acc", phase_init_acc),
    TilePhase("global_addrs", phase_global_addrs),
    TilePhase("fully_interleaved_k_loop", phase_fully_interleaved_k_loop),
]


# ===================================================================
# PGR=2 K-loop: double-buffered global loads for latency hiding
# ===================================================================

def phase_pgr2_k_loop(level, ctx):
    """K-loop with PGR=2: global loads issued 2 iterations ahead.

    Two sets of global load buffers alternate. Each iteration:
      1. Compute current K-tile from LDS
      2. Wait for global loads issued LAST iteration (vmcnt)
      3. Toggle LDS + write last iteration's data to LDS
      4. Issue global loads for tile N+2 into the other buffer
      5. Barrier

    This gives each global load ~2 full iterations of compute time
    to complete, eliminating the vmcnt stall that limits PGR=1.

    Uses a Main Loop + NLL (No-Load Loop) structure:
      Prologue: load tile 0, write to LDS, load tile 1
      Main loop: iterations 2..K/uk-1 (with global loads)
      NLL: last iteration (compute only, write tile K/uk-1)
    """
    tile = _tile(ctx)
    problem = _problem(ctx)
    elem = problem.element_bytes
    mfma = tile.mfma

    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    ki_count = tile.k_iterations
    av = mfma.a_vgprs
    bv = mfma.b_vgprs

    pad_e = tile.lds_pad // elem if tile.lds_pad > 0 else 0
    lds_half = (tile.wg_m + tile.wg_n) * (tile.unroll_k + pad_e) * elem
    k_stride = tile.unroll_k * elem
    log2_uk = int(math.log2(tile.unroll_k))

    ctx.alloc_sgpr_permanent(1, "s_lds_db_step")

    # Allocate TWO sets of global load buffers
    for name in ["a", "b"]:
        load = ctx.get(f"v_gload_{name}")
        # Second buffer: v_gload2_a, v_gload2_b
        if not ctx.has(f"v_gload2_{name}"):
            ctx.alloc_vgpr_permanent(load.count, f"v_gload2_{name}")

    ctx.comment("=== PGR=2 K-loop ===")
    ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
               comment=f"k_tiles = K / {tile.unroll_k}")
    ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_half),
              comment=f"DB toggle step = {lds_half}")
    ctx.raw("")

    # === Prologue: load tile 0 -> buf[0], write to LDS ===
    ctx.comment("Prologue: load tile 0 into buf[0]")
    _emit_global_load_impl(ctx, problem, tile)  # loads into v_gload_a/b, waits
    ctx.comment("Write tile 0 to LDS buf[0]")
    _emit_lds_write_impl(ctx, tile)  # writes + lgkmcnt + barrier

    # Advance pointers to tile 1
    for addr in ["v_addr_a", "v_addr_b"]:
        ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                 str(k_stride), ctx.vreg(addr, 0, 1),
                 comment=f"{addr} += {k_stride}")
        ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                 ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")

    # Load tile 1 -> buf[1] (async, no wait -- will be consumed 2 iterations later)
    ctx.comment("Prefetch tile 1 into buf[1] (async)")
    for name, addr_name in [("A", "v_addr_a"), ("B", "v_addr_b")]:
        gload2 = f"v_gload2_{name.lower()}"
        load = ctx.get(gload2)
        addr = ctx.vreg(addr_name, 0, 2)
        for i in range(0, load.count, 4):
            cnt = min(4, load.count - i)
            width = {4: "dwordx4", 2: "dwordx2", 1: "dword"}[cnt]
            dst = ctx.vreg(gload2, i, cnt)
            off = f"off offset:{i * 4}" if i > 0 else "off"
            ctx.inst(f"global_load_{width}", dst, addr, off,
                     comment=f"prefetch tile1 {name}[{i}:{i+cnt}]")

    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "2",
              comment="k_tiles -= 2 (prologue consumed 2)")
    ctx.raw("")

    # === Allocate operand registers ===
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
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (mi * mfma.m * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem

    def b_off(ni, ki):
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (ni * mfma.n * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem

    def emit_compute(ctx, mr, nr, ki_count, mfma, a_names, b_names, a_off, b_off, tile, elem, label=""):
        """Emit preamble + compute (16 MFMAs with A prefetch)."""
        av = mfma.a_vgprs
        bv = mfma.b_vgprs

        # Preamble: load all B + A[0]
        for ki in range(ki_count):
            for ni in range(nr):
                name = b_names[(ni, ki)]
                ctx.ds_read(ctx.vreg(name, 0, bv), ctx.vreg("v_lds_rd_b"),
                            offset=b_off(ni, ki), width=bv,
                            comment=f"LR B n{ni}k{ki}")

        cur_a = 0
        for ki in range(ki_count):
            name = a_names[(cur_a, ki)]
            ctx.ds_read(ctx.vreg(name, 0, av), ctx.vreg("v_lds_rd_a"),
                        offset=a_off(0, ki), width=av,
                        comment=f"LR A m0k{ki} b{cur_a}")

        ctx.s_waitcnt("lgkmcnt(0)", comment=f"wait preamble {label}")
        ctx.raw("")

        # Per-mi groups: A prefetch + MFMAs
        for mi in range(mr):
            has_prefetch = mi < mr - 1
            if has_prefetch:
                next_a = 1 - cur_a
                for ki in range(ki_count):
                    name = a_names[(next_a, ki)]
                    ctx.ds_read(ctx.vreg(name, 0, av), ctx.vreg("v_lds_rd_a"),
                                offset=a_off(mi + 1, ki), width=av,
                                comment=f"LR A m{mi+1}k{ki} b{next_a}")

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

            if has_prefetch:
                ctx.s_waitcnt("lgkmcnt(0)",
                              comment=f"wait A[{mi+1}]")
                cur_a = next_a
            ctx.raw("")

    def emit_gload_to_buf(ctx, problem, tile, buf_suffix):
        """Issue global loads into buffer buf_suffix ('', '2')."""
        for name, addr_name in [("A", "v_addr_a"), ("B", "v_addr_b")]:
            gload_name = f"v_gload{buf_suffix}_{name.lower()}"
            load = ctx.get(gload_name)
            addr = ctx.vreg(addr_name, 0, 2)
            for i in range(0, load.count, 4):
                cnt = min(4, load.count - i)
                width = {4: "dwordx4", 2: "dwordx2", 1: "dword"}[cnt]
                dst = ctx.vreg(gload_name, i, cnt)
                off = f"off offset:{i * 4}" if i > 0 else "off"
                ctx.inst(f"global_load_{width}", dst, addr, off,
                         comment=f"gload {name}[{i}:{i+cnt}]")

    def emit_lds_write_from_buf(ctx, tile, buf_suffix):
        """Write global load buffer to LDS."""
        for name in ["a", "b"]:
            gload_name = f"v_gload{buf_suffix}_{name}"
            load = ctx.get(gload_name)
            addr_reg = ctx.vreg(f"v_lds_wr_{name}")
            for i in range(0, load.count, 4):
                cnt = min(4, load.count - i)
                src = ctx.vreg(gload_name, i, cnt)
                ctx.ds_write(addr_reg, src, offset=i * 4, width=cnt,
                             comment=f"LDS write {name.upper()}[{i}:{i+cnt}]")

    # === Main loop ===
    # cur_gload_buf alternates: "2" means buf[1] has in-flight data,
    # "" means buf[0] has in-flight data. We start with "2" in flight.
    # Track which buffer suffix to WAIT for and which to LOAD into.
    # After prologue: buf[1] ("2") is in flight. We wait for it, then
    # load into buf[0] ("").

    ctx.label("k_loop")
    ctx.raw("")

    # --- Compute current K-tile from LDS ---
    ctx.comment("Compute current K-tile")
    emit_compute(ctx, mr, nr, ki_count, mfma, a_names, b_names,
                 a_off, b_off, tile, elem, label="main")

    # --- Wait for in-flight global loads (buf[1], issued last iter) ---
    ctx.s_waitcnt("vmcnt(0)", comment="wait for in-flight global loads")

    # Toggle LDS double buffer
    ctx.comment("Toggle LDS + write in-flight data")
    for reg in ["v_lds_wr_a", "v_lds_wr_b", "v_lds_rd_a", "v_lds_rd_b"]:
        ctx.v_add(ctx.vreg(reg), ctx.sreg("s_lds_db_step"), ctx.vreg(reg),
                  comment=f"{reg} += db_step")
    ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
             ctx.sreg("s_lds_db_step"),
             comment="negate step for next toggle")

    # Write in-flight buffer to LDS. We always write from buf[1] ("2")
    # in odd iterations and buf[0] ("") in even. To simplify, we swap
    # the buffer contents after each write.
    # For the first main-loop iteration, buf "2" was loaded in prologue.
    emit_lds_write_from_buf(ctx, tile, "2")

    # Swap buf contents: copy buf[0] <- buf[1], so next iteration
    # we can write buf[0]. (This costs VALU but avoids tracking state.)
    # Actually, simpler: just alternate which buffer we load into.
    # But we need an s_flag to track. Let me use a different approach:
    # Always load into buf[0] (""), always wait-and-write buf[1] ("2").
    # After writing buf[1], copy buf[0] to buf[1] and load into buf[0].
    # Wait -- that's expensive (copying all VGPRs).

    # Simplest approach: just always load into buf "" and write from buf "2".
    # After writing buf "2" to LDS:
    # 1. Copy buf "" -> buf "2" (the loads that just completed into buf "")
    # 2. Issue new loads into buf ""
    # This requires N copy instructions (where N = gload VGPRs).
    # For 16 VGPRs: 16 v_mov_b32 = 16 instructions. Too many.

    # Better approach: swap the REGISTER NAMES, not the data.
    # We can't easily swap register names in the emit framework.

    # Simplest correct approach: always use buf "" for new loads,
    # and in the write phase, write from WHICHEVER buffer was loaded
    # last iteration. Use an SGPR flag to track which buffer to write.

    # Actually, let me just unroll the main loop by 2 iterations:
    # Iter A: load into "", wait for "2", write "2" to LDS
    # Iter B: load into "2", wait for "", write "" to LDS
    # This avoids any buffer management overhead.

    ctx.s_waitcnt("lgkmcnt(0)", comment="wait LDS writes")
    ctx.s_barrier(comment="sync workgroup")

    # Advance pointers and issue new global loads into buf[0] ("")
    for addr in ["v_addr_a", "v_addr_b"]:
        ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                 str(k_stride), ctx.vreg(addr, 0, 1),
                 comment=f"{addr} += {k_stride}")
        ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                 ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")
    emit_gload_to_buf(ctx, problem, tile, "")

    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc0", "k_nll",
             comment="branch to NLL if last iteration")
    ctx.raw("")

    # === Second half of unrolled main loop (swap buffers) ===
    ctx.comment("--- Main loop iter B (load buf2, write buf0) ---")

    # Compute current K-tile from LDS
    emit_compute(ctx, mr, nr, ki_count, mfma, a_names, b_names,
                 a_off, b_off, tile, elem, label="mainB")

    # Wait for buf[0] ("") loads
    ctx.s_waitcnt("vmcnt(0)", comment="wait for buf[0] global loads")

    # Toggle LDS
    for reg in ["v_lds_wr_a", "v_lds_wr_b", "v_lds_rd_a", "v_lds_rd_b"]:
        ctx.v_add(ctx.vreg(reg), ctx.sreg("s_lds_db_step"), ctx.vreg(reg),
                  comment=f"{reg} += db_step")
    ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
             ctx.sreg("s_lds_db_step"),
             comment="negate step for next toggle")

    # Write buf[0] ("") to LDS
    emit_lds_write_from_buf(ctx, tile, "")

    ctx.s_waitcnt("lgkmcnt(0)", comment="wait LDS writes")
    ctx.s_barrier(comment="sync workgroup")

    # Advance pointers and load into buf[1] ("2")
    for addr in ["v_addr_a", "v_addr_b"]:
        ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                 str(k_stride), ctx.vreg(addr, 0, 1),
                 comment=f"{addr} += {k_stride}")
        ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                 ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")
    emit_gload_to_buf(ctx, problem, tile, "2")

    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc1", "k_loop",
             comment="branch to main loop if more iterations")
    ctx.raw("")

    # === Second NLL: last iteration wrote into buf "2" ===
    # We need to compute the last K-tile and write buf "2" to LDS
    ctx.comment("NLL (iter B): compute + write buf2")
    emit_compute(ctx, mr, nr, ki_count, mfma, a_names, b_names,
                 a_off, b_off, tile, elem, label="nll_b")
    ctx.s_waitcnt("vmcnt(0)", comment="wait buf2 final")
    for reg in ["v_lds_wr_a", "v_lds_wr_b", "v_lds_rd_a", "v_lds_rd_b"]:
        ctx.v_add(ctx.vreg(reg), ctx.sreg("s_lds_db_step"), ctx.vreg(reg),
                  comment=f"{reg} += db_step")
    ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
             ctx.sreg("s_lds_db_step"), comment="negate")
    emit_lds_write_from_buf(ctx, tile, "2")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait LDS writes")
    ctx.s_barrier(comment="sync")
    # Final compute from LDS
    emit_compute(ctx, mr, nr, ki_count, mfma, a_names, b_names,
                 a_off, b_off, tile, elem, label="final")
    ctx.inst("s_branch", "k_done", comment="skip k_nll")
    ctx.raw("")

    # === NLL for iter A exit: buf "" was loaded, buf "2" was last written ===
    ctx.label("k_nll")
    ctx.comment("NLL (iter A): compute + write buf0")
    emit_compute(ctx, mr, nr, ki_count, mfma, a_names, b_names,
                 a_off, b_off, tile, elem, label="nll_a")
    ctx.s_waitcnt("vmcnt(0)", comment="wait buf0 final")
    for reg in ["v_lds_wr_a", "v_lds_wr_b", "v_lds_rd_a", "v_lds_rd_b"]:
        ctx.v_add(ctx.vreg(reg), ctx.sreg("s_lds_db_step"), ctx.vreg(reg),
                  comment=f"{reg} += db_step")
    ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
             ctx.sreg("s_lds_db_step"), comment="negate")
    emit_lds_write_from_buf(ctx, tile, "")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait LDS writes")
    ctx.s_barrier(comment="sync")
    # Final compute from LDS
    emit_compute(ctx, mr, nr, ki_count, mfma, a_names, b_names,
                 a_off, b_off, tile, elem, label="final_a")

    ctx.label("k_done")
    ctx.raw("")


PGR2_PROLOGUE_PHASES = [
    TilePhase("load_kernargs", phase_load_kernargs),
    TilePhase("thread_indexing", phase_thread_indexing),
    TilePhase("load_cluster_setup", phase_load_cluster_setup),
    TilePhase("lds_addrs", phase_lds_addrs),
    TilePhase("init_acc", phase_init_acc),
    TilePhase("global_addrs", phase_global_addrs),
    TilePhase("pgr2_k_loop", phase_pgr2_k_loop),
]


# ===================================================================
# DirectToLDS K-loop: buffer_load ... ,lds eliminates ds_write
# ===================================================================

def phase_dtl_setup(level, ctx):
    """Set up SRDs, per-lane offsets, and LDS write bases for DTL."""
    tile = _tile(ctx)
    problem = _problem(ctx)
    elem = problem.element_bytes
    mfma = tile.mfma

    ctx.comment("=== DirectToLDS setup ===")

    # Load kernel arguments (same as phase_load_kernargs)
    karg = ctx.sreg("s_kernarg")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_A"), karg, "0", comment="A ptr")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_B"), karg, "8", comment="B ptr")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_D"), karg, "16", comment="D ptr")
    ctx.inst("s_load_dword", ctx.sreg("s_M"), karg, "24", comment="M")
    ctx.inst("s_load_dword", ctx.sreg("s_N"), karg, "28", comment="N")
    ctx.inst("s_load_dword", ctx.sreg("s_K"), karg, "32", comment="K")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait for kernarg loads")
    ctx.raw("")

    # Thread indexing
    log2_ws = int(math.log2(tile.wave_size))
    ctx.comment("Thread indexing")
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
                  comment="wave_m = wave_id")
        ctx.v_mov(ctx.vreg("v_wave_n"), "0", comment="wave_n = 0")
    ctx.raw("")

    # DTL per-lane offset computation
    # Contiguous mapping: thread t loads 8 fp16 starting at linear position t*8
    # row = t / threads_per_row, col = (t % threads_per_row) * 8
    # threads_per_row = unroll_k / 8 = 4 for unroll_k=32
    threads_per_row = tile.unroll_k // 8  # 8 elements per dwordx4 load (fp16)
    log2_tpr = int(math.log2(threads_per_row))

    ctx.comment(f"DTL per-lane offset: {threads_per_row} threads/row, 8 elems/thread")
    # thread_row = tid >> log2_tpr
    ctx.v_lshr(ctx.vreg("v_tmp0"), ctx.vreg("v_tid"), log2_tpr,
               comment=f"thread_row = tid >> {log2_tpr}")
    # thread_col_byte = (tid & (tpr-1)) * 16  (8 fp16 = 16 bytes)
    ctx.v_and(ctx.vreg("v_tmp1"), ctx.vreg("v_tid"), threads_per_row - 1,
              comment=f"tid & {threads_per_row - 1}")
    ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_tmp1"), 4,
               comment="* 16 -> col_bytes")

    # A offset = thread_row * K * elem + col_bytes
    # K * elem = s_K * 2 for fp16
    ctx.s_lshl(ctx.sreg("s_k_stride"), ctx.sreg("s_K"), int(math.log2(elem)),
               comment=f"s_k_stride = K * {elem}")
    ctx.inst("v_mul_lo_u32", ctx.vreg("v_dtl_off_a"),
             ctx.sreg("s_k_stride"), ctx.vreg("v_tmp0"),
             comment="row * K * elem")
    ctx.v_add(ctx.vreg("v_dtl_off_a"), ctx.vreg("v_dtl_off_a"),
              ctx.vreg("v_tmp1"), comment="+ col_bytes -> A offset")
    # B uses same mapping (B is N x K, row-major with stride K)
    ctx.v_mov(ctx.vreg("v_dtl_off_b"), ctx.vreg("v_dtl_off_a"),
              comment="B offset = same mapping")
    ctx.raw("")

    # SRD setup for A
    # SRD base = ptr_A + wg_id_x * wg_m * K * elem
    ctx.comment("SRD setup for A")
    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"), str(tile.wg_m),
              comment=f"wg_id_x * {tile.wg_m}")
    ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
             ctx.sreg("s_k_stride"),
             comment="* K * elem -> wg tile offset A")
    # 64-bit add: srd_a = ptr_A + wg_offset
    ctx.inst("s_add_u32", ctx.sreg("s_srd_a", 0, 1),
             ctx.sreg("s_ptr_A", 0, 1), ctx.sreg("s_tmp0"),
             comment="SRD_A base lo")
    ctx.inst("s_addc_u32", ctx.sreg("s_srd_a", 1, 1),
             ctx.sreg("s_ptr_A", 1, 1), "0",
             comment="SRD_A base hi")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_a", 2, 1), "0xFFFFFFFF",
             comment="SRD_A limit (no OOB check)")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_a", 3, 1), "0x20000",
             comment="SRD_A flags: data_format=4")
    ctx.raw("")

    # SRD setup for B
    ctx.comment("SRD setup for B")
    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_y"), str(tile.wg_n),
              comment=f"wg_id_y * {tile.wg_n}")
    ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
             ctx.sreg("s_k_stride"),
             comment="* K * elem -> wg tile offset B")
    ctx.inst("s_add_u32", ctx.sreg("s_srd_b", 0, 1),
             ctx.sreg("s_ptr_B", 0, 1), ctx.sreg("s_tmp0"),
             comment="SRD_B base lo")
    ctx.inst("s_addc_u32", ctx.sreg("s_srd_b", 1, 1),
             ctx.sreg("s_ptr_B", 1, 1), "0",
             comment="SRD_B base hi")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_b", 2, 1), "0xFFFFFFFF",
             comment="SRD_B limit")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_b", 3, 1), "0x20000",
             comment="SRD_B flags")
    ctx.raw("")

    # Scalar offsets for 2nd load line (rows 64-127)
    # soffset = rows_per_load * K * elem
    rows_per_load = tile.block_size // threads_per_row  # 256/4 = 64
    ctx.comment(f"Scalar offset for 2nd load line ({rows_per_load} rows)")
    ctx.s_mul(ctx.sreg("s_soffset_a"), ctx.sreg("s_k_stride"),
              str(rows_per_load),
              comment=f"soffset_a = {rows_per_load} * K * elem")
    ctx.s_mov(ctx.sreg("s_soffset_b"), ctx.sreg("s_soffset_a"),
              comment="soffset_b = same")
    ctx.raw("")

    # LDS write bases (SGPR, loaded from VGPR via v_readfirstlane)
    # Each wave needs its own m0 value.
    # LDS base for wave w: w * rows_per_wave * unroll_k * elem
    rows_per_wave = tile.wave_size // threads_per_row  # 64/4 = 16
    lds_stride_per_wave = rows_per_wave * tile.unroll_k * elem  # 16*32*2 = 1024

    ctx.comment("LDS write bases (per-wave via v_readfirstlane)")
    # Compute per-thread LDS write address (same formula as row-major layout)
    # lds_wr = thread_row * unroll_k * elem + col_bytes
    # But we already have thread_row in v_tmp0 and col_bytes in v_tmp1
    ctx.v_mul(ctx.vreg("v_tmp0"), str(tile.unroll_k * elem),
              ctx.vreg("v_tmp0"), comment=f"row * {tile.unroll_k * elem}")
    ctx.v_add(ctx.vreg("v_tmp0"), ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
              comment="+ col_bytes -> lds_wr_per_thread")
    # Extract wave's lane-0 value as the SGPR base
    ctx.inst("v_readfirstlane_b32", ctx.sreg("s_lds_wr_a_sg"),
             ctx.vreg("v_tmp0"),
             comment="lds_wr_a base for this wave")

    # B LDS write base: offset by lds_b_offset
    layouts = _layouts(ctx)
    ctx.inst("s_add_u32", ctx.sreg("s_lds_wr_b_sg"),
             ctx.sreg("s_lds_wr_a_sg"), str(layouts.lds_b_offset),
             comment=f"lds_wr_b = lds_wr_a + {layouts.lds_b_offset}")
    ctx.raw("")

    # LDS read addresses (same as non-DTL)
    k_per_group = mfma.k // (tile.wave_size // mfma.m)
    ctx.comment("MFMA lane mapping for LDS reads")
    ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
              comment=f"lane_row = lane_id % {mfma.m}")
    ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
               int(math.log2(mfma.m)), comment=f"lane_id / {mfma.m}")
    ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_tmp1"),
               int(math.log2(k_per_group)),
               comment=f"* {k_per_group} -> lane_k_offset")

    # LDS read A: addr = (row * unroll_k + lane_k) * elem
    ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(tile.m_per_wave),
              ctx.vreg("v_wave_m"), comment=f"wave_m * {tile.m_per_wave}")
    ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
              ctx.vreg("v_tmp0"), comment="+ lane_row -> row")
    ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(tile.unroll_k),
              ctx.vreg("v_lds_rd_a"), comment=f"* {tile.unroll_k}")
    ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
              ctx.vreg("v_tmp1"), comment="+ lane_k_offset")
    ctx.v_lshl(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
               int(math.log2(elem)), comment=f"* {elem} -> bytes")
    ctx.raw("")

    # LDS read B: addr = lds_b_offset + (row * unroll_k + lane_k) * elem
    ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(tile.n_per_wave),
              ctx.vreg("v_wave_n"), comment=f"wave_n * {tile.n_per_wave}")
    ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
              ctx.vreg("v_tmp0"), comment="+ lane_row -> row")
    ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(tile.unroll_k),
              ctx.vreg("v_lds_rd_b"), comment=f"* {tile.unroll_k}")
    ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
              ctx.vreg("v_tmp1"), comment="+ lane_k_offset")
    ctx.v_lshl(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
               int(math.log2(elem)), comment=f"* {elem} -> bytes")
    ctx.v_add(ctx.vreg("v_lds_rd_b"), str(layouts.lds_b_offset),
              ctx.vreg("v_lds_rd_b"), comment=f"+ lds_b_offset")
    ctx.raw("")

    # Init accumulators
    acc_total = tile.mfma_m_repeat * tile.mfma_n_repeat * tile.mfma.acc_vgprs
    ctx.comment(f"Init {acc_total} accumulators")
    for i in range(acc_total):
        ctx.inst("v_accvgpr_write_b32", ctx.areg("acc_C", i, 1), "0")
    ctx.raw("")


def _emit_dtl_loads(ctx, tile, problem, label=""):
    """Issue buffer_load_dwordx4 with ,lds for both A and B."""
    elem = problem.element_bytes
    threads_per_row = tile.unroll_k // 8
    rows_per_load = tile.block_size // threads_per_row
    lds_stride_per_load = rows_per_load * tile.unroll_k * elem
    layouts = _layouts(ctx)
    num_loads = tile.wg_m // rows_per_load

    for name, srd, soffset, lds_wr_sg, dtl_off, lds_base_offset in [
        ("A", "s_srd_a", "s_soffset_a", "s_lds_wr_a_sg", "v_dtl_off_a", 0),
        ("B", "s_srd_b", "s_soffset_b", "s_lds_wr_b_sg", "v_dtl_off_b", layouts.lds_b_offset),
    ]:
        ctx.comment(f"DTL load {name} {label}")
        ctx.inst("s_mov_b32", "m0", ctx.sreg(lds_wr_sg),
                 comment=f"m0 = LDS write base {name}")

        # Use cumulative soffset for multi-line DTL loads
        ctx.s_mov(ctx.sreg("s_tmp0"), "0", comment="cumulative soffset")
        for load_idx in range(num_loads):
            ctx.inst("buffer_load_dwordx4",
                     ctx.vreg(dtl_off), ctx.sreg(srd, 0, 4),
                     ctx.sreg("s_tmp0"), "offen offset:0, lds",
                     comment=f"DTL {name} line {load_idx}")
            if load_idx < num_loads - 1:
                ctx.inst("s_add_u32", "m0", "m0", str(lds_stride_per_load),
                         comment=f"m0 += {lds_stride_per_load}")
                ctx.inst("s_add_u32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                         ctx.sreg(soffset),
                         comment=f"soffset += {name}_stride")
    ctx.raw("")


def phase_dtl_k_loop(level, ctx):
    """K-loop with DirectToLDS: buffer_load_dwordx4 ... ,lds."""
    tile = _tile(ctx)
    problem = _problem(ctx)
    elem = problem.element_bytes
    mfma = tile.mfma
    layouts = _layouts(ctx)

    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    ki_count = tile.k_iterations
    av = mfma.a_vgprs
    bv = mfma.b_vgprs

    pad_e = tile.lds_pad // elem if tile.lds_pad > 0 else 0
    lds_half = (tile.wg_m + tile.wg_n) * (tile.unroll_k + pad_e) * elem
    log2_uk = int(math.log2(tile.unroll_k))

    ctx.alloc_sgpr_permanent(1, "s_lds_db_step")

    ctx.comment("=== DTL K-loop ===")
    ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
               comment=f"k_tiles = K / {tile.unroll_k}")
    ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_half),
              comment=f"DB toggle step = {lds_half}")
    ctx.raw("")

    # Prologue: DTL load tile 0 + wait + barrier
    ctx.comment("Prologue: DTL load tile 0")
    _emit_dtl_loads(ctx, tile, problem, "tile0")
    ctx.s_waitcnt("vmcnt(0)", comment="wait for DTL loads")
    ctx.s_barrier(comment="sync after DTL fill")
    ctx.raw("")

    # Allocate operand registers
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
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (mi * mfma.m * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem

    def b_off(ni, ki):
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (ni * mfma.n * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem

    # K-loop
    ctx.label("k_loop")
    ctx.raw("")

    # Decrement + conditional DTL load
    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc0", "dtl_skip_load",
             comment="skip DTL load on last iteration")

    # Advance SRD bases by k_stride
    k_stride = tile.unroll_k * elem
    for srd in ["s_srd_a", "s_srd_b"]:
        ctx.inst("s_add_u32", ctx.sreg(srd, 0, 1),
                 ctx.sreg(srd, 0, 1), str(k_stride),
                 comment=f"{srd} += {k_stride}")
        ctx.inst("s_addc_u32", ctx.sreg(srd, 1, 1),
                 ctx.sreg(srd, 1, 1), "0", comment="carry")

    # Toggle LDS write bases (point to other buffer)
    ctx.inst("s_add_u32", ctx.sreg("s_lds_wr_a_sg"),
             ctx.sreg("s_lds_wr_a_sg"), ctx.sreg("s_lds_db_step"),
             comment="lds_wr_a += db_step")
    ctx.inst("s_add_u32", ctx.sreg("s_lds_wr_b_sg"),
             ctx.sreg("s_lds_wr_b_sg"), ctx.sreg("s_lds_db_step"),
             comment="lds_wr_b += db_step")

    # Issue DTL loads into the OTHER buffer
    _emit_dtl_loads(ctx, tile, problem, "next")
    ctx.raw("")

    ctx.label("dtl_skip_load")
    ctx.raw("")

    # Compute from current LDS buffer
    ctx.comment(f"Compute: {mr}x{nr}x{ki_count} = {mr*nr*ki_count} MFMAs")

    # Preamble: load all B + A[0]
    for ki in range(ki_count):
        for ni in range(nr):
            name = b_names[(ni, ki)]
            ctx.ds_read(ctx.vreg(name, 0, bv), ctx.vreg("v_lds_rd_b"),
                        offset=b_off(ni, ki), width=bv,
                        comment=f"LR B n{ni}k{ki}")

    cur_a = 0
    for ki in range(ki_count):
        name = a_names[(cur_a, ki)]
        ctx.ds_read(ctx.vreg(name, 0, av), ctx.vreg("v_lds_rd_a"),
                    offset=a_off(0, ki), width=av,
                    comment=f"LR A m0k{ki} b{cur_a}")

    ctx.s_waitcnt("lgkmcnt(0)", comment="wait preamble")
    ctx.raw("")

    # Per-mi groups: A prefetch + MFMAs
    for mi in range(mr):
        has_prefetch = mi < mr - 1
        if has_prefetch:
            next_a = 1 - cur_a
            for ki in range(ki_count):
                name = a_names[(next_a, ki)]
                ctx.ds_read(ctx.vreg(name, 0, av), ctx.vreg("v_lds_rd_a"),
                            offset=a_off(mi + 1, ki), width=av,
                            comment=f"LR A m{mi+1}k{ki} b{next_a}")

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

        if has_prefetch:
            ctx.s_waitcnt("lgkmcnt(0)",
                          comment=f"wait A[{mi+1}]")
            cur_a = next_a
        ctx.raw("")

    # Post-compute: wait for DTL loads + toggle read addrs + barrier
    ctx.s_waitcnt("vmcnt(0)", comment="wait for DTL loads")

    # Toggle LDS read addresses
    for reg in ["v_lds_rd_a", "v_lds_rd_b"]:
        ctx.v_add(ctx.vreg(reg), ctx.sreg("s_lds_db_step"), ctx.vreg(reg),
                  comment=f"{reg} += db_step")
    ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
             ctx.sreg("s_lds_db_step"),
             comment="negate step for next toggle")
    ctx.raw("")

    ctx.s_barrier(comment="sync workgroup")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc1", "k_loop",
             comment="branch if k_tiles > 0")
    ctx.raw("")


DTL_PROLOGUE_PHASES = [
    TilePhase("dtl_setup", phase_dtl_setup),
    TilePhase("dtl_k_loop", phase_dtl_k_loop),
]

# ===================================================================
# Fully-interleaved K-loop for large tiles (128+ MFMAs)
# Moves suffix ops (vmcnt, toggle, ds_writes) into late MFMAs
# and prefix ops (ptr advance, global loads) into early MFMAs.
# ===================================================================

def phase_interleaved_large_k_loop(level, ctx):
    """K-loop with suffix/prefix interleaved into MFMA gaps.

    Per ki phase (64 MFMAs): preamble (9 reads + wait), then 8 mi groups.
    Prefix (global loads) interleaved with ki=0 early mi groups.
    Suffix (vmcnt, toggle, ds_writes) interleaved with last ki late mi groups.
    """
    tile = _tile(ctx)
    problem = _problem(ctx)
    elem = problem.element_bytes
    mfma = tile.mfma
    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    ki_count = tile.k_iterations
    av = mfma.a_vgprs
    bv = mfma.b_vgprs
    pad_e = tile.lds_pad // elem if tile.lds_pad > 0 else 0
    lds_half = (tile.wg_m + tile.wg_n) * (tile.unroll_k + pad_e) * elem
    k_stride = tile.unroll_k * elem
    log2_uk = int(math.log2(tile.unroll_k))

    ctx.alloc_sgpr_permanent(1, "s_lds_db_step")
    ctx.comment("=== Interleaved large K-loop ===")
    ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
               comment=f"k_tiles = K / {tile.unroll_k}")
    ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_half),
              comment=f"DB toggle step = {lds_half}")
    ctx.raw("")

    # Prefetch first tile
    ctx.comment("Prefetch first K-tile")
    _emit_global_load_impl(ctx, problem, tile)
    ctx.comment("Write first tile to LDS buf[0]")
    _emit_lds_write_impl(ctx, tile)

    # Per-(ni,ki) B and per-(buf,ki) A registers
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
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (mi * mfma.m * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem

    def b_off(ni, ki):
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (ni * mfma.n * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem

    def do_mfma(mi, ni, ki, a_buf):
        acc_per = mfma.acc_vgprs
        acc_off = (mi * nr + ni) * acc_per
        ctx.inst(f"v_mfma_f32_{mfma.m}x{mfma.n}x{mfma.k}_f16",
                 ctx.areg("acc_C", acc_off, acc_per),
                 ctx.vreg(a_names[(a_buf, ki)], 0, av),
                 ctx.vreg(b_names[(ni, ki)], 0, bv),
                 ctx.areg("acc_C", acc_off, acc_per),
                 comment=f"MFMA m{mi}_n{ni}_k{ki}")

    # K-loop
    ctx.label("k_loop")
    ctx.raw("")
    ctx.comment(f"Interleaved: {mr}x{nr}x{ki_count} = {mr*nr*ki_count} MFMAs")

    for ki in range(ki_count):
        is_first_ki = (ki == 0)
        is_last_ki = (ki == ki_count - 1)

        # Preamble: B reads + A[0]
        for ni in range(nr):
            ctx.ds_read(ctx.vreg(b_names[(ni, ki)], 0, bv),
                        ctx.vreg("v_lds_rd_b"),
                        offset=b_off(ni, ki), width=bv,
                        comment=f"LR B n{ni}k{ki}")
        cur_a = 0
        ctx.ds_read(ctx.vreg(a_names[(cur_a, ki)], 0, av),
                    ctx.vreg("v_lds_rd_a"),
                    offset=a_off(0, ki), width=av,
                    comment=f"LR A m0k{ki} b{cur_a}")
        ctx.s_waitcnt("lgkmcnt(0)", comment=f"wait preamble k{ki}")
        ctx.raw("")

        for mi in range(mr):
            has_a_pf = (mi < mr - 1)

            # A prefetch before MFMAs
            if has_a_pf:
                next_a = 1 - cur_a
                ctx.ds_read(ctx.vreg(a_names[(next_a, ki)], 0, av),
                            ctx.vreg("v_lds_rd_a"),
                            offset=a_off(mi + 1, ki), width=av,
                            comment=f"LR A m{mi+1}k{ki} b{next_a}")

            # PREFIX: conditional global loads in ki=0 early mi groups
            if is_first_ki and mi == 0:
                # k_tiles-- and conditional skip of global loads
                ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
                          comment="k_tiles--")
                ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
                         comment="SCC = (k_tiles != 0)")
                ctx.inst("s_cbranch_scc0", "skip_gload",
                         comment="skip gload on last iter")
                for addr in ["v_addr_a", "v_addr_b"]:
                    ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                             str(k_stride), ctx.vreg(addr, 0, 1),
                             comment=f"{addr} += {k_stride}")
                    ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                             ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")
                # Global loads
                for name, aname in [("A", "v_addr_a"), ("B", "v_addr_b")]:
                    gn = f"v_gload_{name.lower()}"
                    load = ctx.get(gn)
                    for i in range(0, load.count, 4):
                        cnt = min(4, load.count - i)
                        w = {4: "dwordx4", 2: "dwordx2", 1: "dword"}[cnt]
                        ctx.inst(f"global_load_{w}", ctx.vreg(gn, i, cnt),
                                 ctx.vreg(aname, 0, 2),
                                 f"off offset:{i*4}" if i > 0 else "off",
                                 comment=f"gload {name}[{i}:{i+cnt}]")
                ctx.label("skip_gload")
                # All 8 MFMAs for mi=0
                for ni in range(nr):
                    do_mfma(mi, ni, ki, cur_a)

            elif is_first_ki and mi == 1:
                # Regular MFMAs (global loads already issued above)
                for ni in range(nr):
                    do_mfma(mi, ni, ki, cur_a)

            # SUFFIX: interleave with last ki, mi=mr-2 (vmcnt+toggle) and mi=mr-1 (ds_writes)
            elif is_last_ki and mi == mr - 2:
                # vmcnt + toggle interleaved with MFMAs
                ctx.s_waitcnt("vmcnt(0)", comment="wait gload")
                do_mfma(mi, 0, ki, cur_a)
                ctx.v_add(ctx.vreg("v_lds_wr_a"), ctx.sreg("s_lds_db_step"),
                          ctx.vreg("v_lds_wr_a"), comment="wr_a += db")
                do_mfma(mi, 1, ki, cur_a)
                ctx.v_add(ctx.vreg("v_lds_wr_b"), ctx.sreg("s_lds_db_step"),
                          ctx.vreg("v_lds_wr_b"), comment="wr_b += db")
                do_mfma(mi, 2, ki, cur_a)
                ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.sreg("s_lds_db_step"),
                          ctx.vreg("v_lds_rd_a"), comment="rd_a += db")
                do_mfma(mi, 3, ki, cur_a)
                ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.sreg("s_lds_db_step"),
                          ctx.vreg("v_lds_rd_b"), comment="rd_b += db")
                do_mfma(mi, 4, ki, cur_a)
                ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
                         ctx.sreg("s_lds_db_step"), comment="negate")
                for ni in range(5, nr):
                    do_mfma(mi, ni, ki, cur_a)

            elif is_last_ki and mi == mr - 1:
                # ds_writes interleaved with last mi group MFMAs
                writes = []
                for name in ["a", "b"]:
                    load = ctx.get(f"v_gload_{name}")
                    for i in range(0, load.count, 4):
                        cnt = min(4, load.count - i)
                        writes.append((name, i, cnt))
                w_idx = 0
                for ni in range(nr):
                    if w_idx < len(writes):
                        n, i, c = writes[w_idx]
                        ctx.ds_write(ctx.vreg(f"v_lds_wr_{n}"),
                                     ctx.vreg(f"v_gload_{n}", i, c),
                                     offset=i * 4, width=c,
                                     comment=f"ds_wr {n.upper()}[{i}:{i+c}]")
                        w_idx += 1
                    do_mfma(mi, ni, ki, cur_a)
                while w_idx < len(writes):
                    n, i, c = writes[w_idx]
                    ctx.ds_write(ctx.vreg(f"v_lds_wr_{n}"),
                                 ctx.vreg(f"v_gload_{n}", i, c),
                                 offset=i * 4, width=c,
                                 comment=f"ds_wr {n.upper()}[{i}:{i+c}]")
                    w_idx += 1
            else:
                # Regular mi group: just MFMAs
                for ni in range(nr):
                    do_mfma(mi, ni, ki, cur_a)

            # Wait for A prefetch
            if has_a_pf:
                ctx.s_waitcnt("lgkmcnt(0)", comment=f"wait A[{mi+1}]k{ki}")
                cur_a = next_a

        ctx.raw("")

    # Postamble: wait for ds_writes, barrier, loop control
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait ds_writes")
    ctx.s_barrier(comment="sync workgroup")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc1", "k_loop", comment="next K-tile")
    ctx.raw("")





INTERLEAVED_LARGE_PROLOGUE_PHASES = [
    TilePhase("load_kernargs", phase_load_kernargs),
    TilePhase("thread_indexing", phase_thread_indexing),
    TilePhase("load_cluster_setup", phase_load_cluster_setup),
    TilePhase("lds_addrs", phase_lds_addrs),
    TilePhase("init_acc", phase_init_acc),
    TilePhase("global_addrs", phase_global_addrs),
    TilePhase("interleaved_large_k_loop", phase_interleaved_large_k_loop),
]
