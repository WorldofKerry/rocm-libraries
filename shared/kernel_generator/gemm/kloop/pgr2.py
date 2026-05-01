"""PGR=2 interleaved K-loop: gloads before compute, writes in late MFMAs.

Key optimizations over phase_pgr2_k_loop:
1. Gloads issued BEFORE compute (overlap ds_read latency)
2. vmcnt + toggle + ds_writes interleaved with late MFMAs
3. No s_nops -- back-to-back MFMAs are fine on gfx950
4. Barrier after last MFMA (not in sequential suffix)
5. Correct PGR=2: check k_tiles before decrementing for gloads
"""
from __future__ import annotations

import math

from ..emit.context import AsmContext
from ..emit.layouts import GemmLayouts
from ..problem import GemmProblem, TileConfig
from ..tile.tree import TilePhase
from ..emit.phases import (phase_load_kernargs, phase_thread_indexing,
                     phase_load_cluster_setup, phase_lds_addrs,
                     phase_init_acc, phase_global_addrs,
                     _emit_global_load_impl, _emit_lds_write_impl)

__all__ = ["phase_pgr2_interleaved_k_loop", "PGR2_INTERLEAVED_PROLOGUE_PHASES"]


def _tile(ctx): return ctx._metadata["tile"]
def _problem(ctx): return ctx._metadata["problem"]
def _layouts(ctx): return ctx._metadata["layouts"]


def _a_off(mi, ki, tile, mfma, elem):
    pad_e = tile.lds_pad // elem if tile.lds_pad > 0 else 0
    return (mi * mfma.m * (tile.unroll_k + pad_e) + ki * mfma.k) * elem


def _b_off(ni, ki, tile, mfma, elem):
    pad_e = tile.lds_pad // elem if tile.lds_pad > 0 else 0
    return (ni * mfma.n * (tile.unroll_k + pad_e) + ki * mfma.k) * elem


def _emit_gloads(ctx, problem, tile, buf_suffix):
    """Issue all global loads into buf_suffix."""
    for name, addr_name in [("A", "v_addr_a"), ("B", "v_addr_b")]:
        gn = f"v_gload{buf_suffix}_{name.lower()}"
        load = ctx.get(gn)
        addr = ctx.vreg(addr_name, 0, 2)
        for i in range(0, load.count, 4):
            cnt = min(4, load.count - i)
            w = {4: "dwordx4", 2: "dwordx2", 1: "dword"}[cnt]
            dst = ctx.vreg(gn, i, cnt)
            off = f"off offset:{i * 4}" if i > 0 else "off"
            ctx.inst(f"global_load_{w}", dst, addr, off,
                     comment=f"gload {name}[{i}:{i+cnt}]")


def _emit_ds_writes(ctx, tile, buf_suffix):
    """Write global load buffer to LDS."""
    for name in ["a", "b"]:
        gn = f"v_gload{buf_suffix}_{name}"
        load = ctx.get(gn)
        addr_reg = ctx.vreg(f"v_lds_wr_{name}")
        for i in range(0, load.count, 4):
            cnt = min(4, load.count - i)
            src = ctx.vreg(gn, i, cnt)
            ctx.ds_write(addr_reg, src, offset=i * 4, width=cnt,
                         comment=f"ds_wr {name.upper()}[{i}:{i+cnt}]")


def _emit_ds_writes_individual(ctx, tile, buf_suffix):
    """Return list of callables, each issuing one ds_write."""
    ops = []
    for name in ["a", "b"]:
        gn = f"v_gload{buf_suffix}_{name}"
        load = ctx.get(gn)
        for i in range(0, load.count, 4):
            cnt = min(4, load.count - i)
            _i, _cnt, _gn, _name = i, cnt, gn, name
            def emit(_i=_i, _cnt=_cnt, _gn=_gn, _name=_name):
                ctx.ds_write(ctx.vreg(f"v_lds_wr_{_name}"),
                             ctx.vreg(_gn, _i, _cnt),
                             offset=_i * 4, width=_cnt,
                             comment=f"ds_wr {_name.upper()}[{_i}:{_i+_cnt}]")
            ops.append(emit)
    return ops


def _emit_compute(ctx, tile, mfma, mr, nr, ki_count, a_names, b_names,
                  elem, label, write_ops=None, do_vmcnt=False, do_toggle=False):
    """Emit compute phase: preamble + MFMAs with A-prefetch.
    write_ops/do_vmcnt/do_toggle are handled in the caller now.
    """
    av = mfma.a_vgprs
    bv = mfma.b_vgprs

    # Preamble: read all B + A[0]
    for ki in range(ki_count):
        for ni in range(nr):
            name = b_names[(ni, ki)]
            ctx.ds_read(ctx.vreg(name, 0, bv), ctx.vreg("v_lds_rd_b"),
                        offset=_b_off(ni, ki, tile, mfma, elem), width=bv,
                        comment=f"LR B n{ni}k{ki}")

    cur_a = 0
    for ki in range(ki_count):
        name = a_names[(cur_a, ki)]
        ctx.ds_read(ctx.vreg(name, 0, av), ctx.vreg("v_lds_rd_a"),
                    offset=_a_off(0, ki, tile, mfma, elem), width=av,
                    comment=f"LR A m0k{ki} b{cur_a}")

    ctx.s_waitcnt("lgkmcnt(0)", comment=f"wait preamble {label}")
    ctx.raw("")

    pass  # write ops handled by caller

    for mi in range(mr):
        has_prefetch = mi < mr - 1

        # A prefetch
        if has_prefetch:
            next_a = 1 - cur_a
            for ki in range(ki_count):
                name = a_names[(next_a, ki)]
                ctx.ds_read(ctx.vreg(name, 0, av), ctx.vreg("v_lds_rd_a"),
                            offset=_a_off(mi + 1, ki, tile, mfma, elem),
                            width=av,
                            comment=f"LR A m{mi+1}k{ki} b{next_a}")


        # MFMAs for this mi group
        for ki in range(ki_count):
            for ni in range(nr):
                acc_per = mfma.acc_vgprs
                acc_off = (mi * nr + ni) * acc_per
                ctx.inst(
                    mfma.instruction_name,
                    ctx.areg("acc_C", acc_off, acc_per),
                    ctx.vreg(a_names[(cur_a, ki)], 0, av),
                    ctx.vreg(b_names[(ni, ki)], 0, bv),
                    ctx.areg("acc_C", acc_off, acc_per),
                    comment=f"MFMA m{mi}_n{ni}_k{ki}")

        if has_prefetch:
            ctx.s_waitcnt("lgkmcnt(0)", comment=f"wait A[{mi+1}]")
            cur_a = next_a
        ctx.raw("")




def phase_pgr2_interleaved_k_loop(level, ctx):
    """K-loop with PGR=2: gloads before compute, writes in late MFMAs."""
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

    # Second global load buffer
    for name in ["a", "b"]:
        load = ctx.get(f"v_gload_{name}")
        if not ctx.has(f"v_gload2_{name}"):
            ctx.alloc_vgpr_permanent(load.count, f"v_gload2_{name}")

    ctx.comment("=== PGR=2 Interleaved K-loop ===")
    ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
               comment=f"k_tiles = K / {tile.unroll_k}")
    ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_half),
              comment=f"DB toggle step = {lds_half}")
    ctx.raw("")

    # Prologue: load tile 0 -> LDS buf 0, load tile 1 -> buf2 (async)
    ctx.comment("Prologue: load tile 0")
    _emit_global_load_impl(ctx, problem, tile)
    ctx.comment("Write tile 0 to LDS")
    _emit_lds_write_impl(ctx, tile)

    # Advance + prefetch tile 1 into buf2
    for addr in ["v_addr_a", "v_addr_b"]:
        ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                 str(k_stride), ctx.vreg(addr, 0, 1),
                 comment=f"{addr} += {k_stride}")
        ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                 ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")

    ctx.comment("Prefetch tile 1 into buf2")
    _emit_gloads(ctx, problem, tile, "2")
    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "2",
              comment="k_tiles -= 2")
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

    # === Main loop (2x unrolled) ===
    ctx.label("k_loop")
    ctx.raw("")

    # --- Iter A: preamble -> gloads (fill latency) -> compute -> suffix ---
    ctx.comment("--- Iter A ---")

    # Preamble: issue ds_reads for current tile
    for ki in range(ki_count):
        for ni in range(nr):
            name = b_names[(ni, ki)]
            ctx.ds_read(ctx.vreg(name, 0, bv), ctx.vreg("v_lds_rd_b"),
                        offset=_b_off(ni, ki, tile, mfma, elem), width=bv,
                        comment=f"LR B n{ni}k{ki}")
    cur_a_A = 0
    for ki in range(ki_count):
        name = a_names[(cur_a_A, ki)]
        ctx.ds_read(ctx.vreg(name, 0, av), ctx.vreg("v_lds_rd_a"),
                    offset=_a_off(0, ki, tile, mfma, elem), width=av,
                    comment=f"LR A m0k{ki} b{cur_a_A}")

    # Fill ds_read latency gap with gloads for next+2 tile
    ctx.inst("s_cmp_eq_u32", ctx.sreg("s_k_tiles"), "0",
             comment="any tiles left?")
    ctx.inst("s_cbranch_scc1", "skip_gload_A", comment="skip if none")
    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    for addr in ["v_addr_a", "v_addr_b"]:
        ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                 str(k_stride), ctx.vreg(addr, 0, 1),
                 comment=f"{addr} += {k_stride}")
        ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                 ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")
    _emit_gloads(ctx, problem, tile, "")
    ctx.label("skip_gload_A")

    # Now wait for preamble ds_reads (should be done after ~20 cycles of gloads)
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait preamble")
    ctx.raw("")

    # Compute: clean MFMAs with A-prefetch only
    for mi in range(mr):
        has_pf = mi < mr - 1
        if has_pf:
            next_a = 1 - cur_a_A
            for ki in range(ki_count):
                name = a_names[(next_a, ki)]
                ctx.ds_read(ctx.vreg(name, 0, av), ctx.vreg("v_lds_rd_a"),
                            offset=_a_off(mi + 1, ki, tile, mfma, elem),
                            width=av, comment=f"LR A m{mi+1}k{ki} b{next_a}")
        for ki in range(ki_count):
            for ni in range(nr):
                acc_per = mfma.acc_vgprs
                acc_off = (mi * nr + ni) * acc_per
                ctx.inst(mfma.instruction_name,
                         ctx.areg("acc_C", acc_off, acc_per),
                         ctx.vreg(a_names[(cur_a_A, ki)], 0, av),
                         ctx.vreg(b_names[(ni, ki)], 0, bv),
                         ctx.areg("acc_C", acc_off, acc_per),
                         comment=f"MFMA m{mi}_n{ni}_k{ki}")
        if has_pf:
            ctx.s_waitcnt("lgkmcnt(0)", comment=f"wait A[{mi+1}]")
            cur_a_A = next_a
        ctx.raw("")

    # Suffix: vmcnt + toggle + ds_writes + barrier
    ctx.s_waitcnt("vmcnt(0)", comment="wait prev gloads")
    for reg in ["v_lds_wr_a", "v_lds_wr_b", "v_lds_rd_a", "v_lds_rd_b"]:
        ctx.v_add(ctx.vreg(reg), ctx.sreg("s_lds_db_step"),
                  ctx.vreg(reg), comment=f"{reg} += db")
    ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
             ctx.sreg("s_lds_db_step"), comment="negate")
    _emit_ds_writes(ctx, tile, "2")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait writes")
    ctx.s_barrier(comment="sync")
    ctx.raw("")

    # Check loop exit
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc0", "k_nll",
             comment="NLL if last")
    ctx.raw("")

    # --- Iter B: same strategy ---
    ctx.comment("--- Iter B ---")

    # Preamble
    for ki in range(ki_count):
        for ni in range(nr):
            name = b_names[(ni, ki)]
            ctx.ds_read(ctx.vreg(name, 0, bv), ctx.vreg("v_lds_rd_b"),
                        offset=_b_off(ni, ki, tile, mfma, elem), width=bv,
                        comment=f"LR B n{ni}k{ki}")
    cur_a_B = 0
    for ki in range(ki_count):
        name = a_names[(cur_a_B, ki)]
        ctx.ds_read(ctx.vreg(name, 0, av), ctx.vreg("v_lds_rd_a"),
                    offset=_a_off(0, ki, tile, mfma, elem), width=av,
                    comment=f"LR A m0k{ki} b{cur_a_B}")

    # Fill latency with gloads
    ctx.inst("s_cmp_eq_u32", ctx.sreg("s_k_tiles"), "0",
             comment="any tiles left?")
    ctx.inst("s_cbranch_scc1", "skip_gload_B", comment="skip if none")
    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    for addr in ["v_addr_a", "v_addr_b"]:
        ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                 str(k_stride), ctx.vreg(addr, 0, 1),
                 comment=f"{addr} += {k_stride}")
        ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                 ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")
    _emit_gloads(ctx, problem, tile, "2")
    ctx.label("skip_gload_B")

    ctx.s_waitcnt("lgkmcnt(0)", comment="wait preamble")
    ctx.raw("")

    # Compute
    for mi in range(mr):
        has_pf = mi < mr - 1
        if has_pf:
            next_a = 1 - cur_a_B
            for ki in range(ki_count):
                name = a_names[(next_a, ki)]
                ctx.ds_read(ctx.vreg(name, 0, av), ctx.vreg("v_lds_rd_a"),
                            offset=_a_off(mi + 1, ki, tile, mfma, elem),
                            width=av, comment=f"LR A m{mi+1}k{ki} b{next_a}")
        for ki in range(ki_count):
            for ni in range(nr):
                acc_per = mfma.acc_vgprs
                acc_off = (mi * nr + ni) * acc_per
                ctx.inst(mfma.instruction_name,
                         ctx.areg("acc_C", acc_off, acc_per),
                         ctx.vreg(a_names[(cur_a_B, ki)], 0, av),
                         ctx.vreg(b_names[(ni, ki)], 0, bv),
                         ctx.areg("acc_C", acc_off, acc_per),
                         comment=f"MFMA m{mi}_n{ni}_k{ki}")
        if has_pf:
            ctx.s_waitcnt("lgkmcnt(0)", comment=f"wait A[{mi+1}]")
            cur_a_B = next_a
        ctx.raw("")

    # Suffix
    ctx.s_waitcnt("vmcnt(0)", comment="wait prev gloads")
    for reg in ["v_lds_wr_a", "v_lds_wr_b", "v_lds_rd_a", "v_lds_rd_b"]:
        ctx.v_add(ctx.vreg(reg), ctx.sreg("s_lds_db_step"),
                  ctx.vreg(reg), comment=f"{reg} += db")
    ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
             ctx.sreg("s_lds_db_step"), comment="negate")
    _emit_ds_writes(ctx, tile, "")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait writes")
    ctx.s_barrier(comment="sync")
    ctx.raw("")

    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc1", "k_loop",
             comment="loop if more")
    ctx.raw("")

    # === NLL for iter B exit ===
    ctx.comment("NLL (iter B): compute + write buf2")
    _emit_compute(ctx, tile, mfma, mr, nr, ki_count,
                  a_names, b_names, elem, "nll_b1")
    ctx.s_waitcnt("vmcnt(0)", comment="wait final gloads")
    for reg in ["v_lds_wr_a", "v_lds_wr_b", "v_lds_rd_a", "v_lds_rd_b"]:
        ctx.v_add(ctx.vreg(reg), ctx.sreg("s_lds_db_step"),
                  ctx.vreg(reg), comment=f"{reg} += db")
    ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
             ctx.sreg("s_lds_db_step"), comment="negate")
    _emit_ds_writes(ctx, tile, "2")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait writes")
    ctx.s_barrier(comment="sync")
    _emit_compute(ctx, tile, mfma, mr, nr, ki_count,
                  a_names, b_names, elem, "nll_b2")
    ctx.inst("s_branch", "k_done", comment="skip k_nll")
    ctx.raw("")

    # === NLL for iter A exit ===
    ctx.label("k_nll")
    ctx.comment("NLL (iter A): compute + write buf")
    _emit_compute(ctx, tile, mfma, mr, nr, ki_count,
                  a_names, b_names, elem, "nll_a1")
    ctx.s_waitcnt("vmcnt(0)", comment="wait final gloads")
    for reg in ["v_lds_wr_a", "v_lds_wr_b", "v_lds_rd_a", "v_lds_rd_b"]:
        ctx.v_add(ctx.vreg(reg), ctx.sreg("s_lds_db_step"),
                  ctx.vreg(reg), comment=f"{reg} += db")
    ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
             ctx.sreg("s_lds_db_step"), comment="negate")
    _emit_ds_writes(ctx, tile, "")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait writes")
    ctx.s_barrier(comment="sync")
    _emit_compute(ctx, tile, mfma, mr, nr, ki_count,
                  a_names, b_names, elem, "nll_a2")

    ctx.label("k_done")
    ctx.raw("")


PGR2_INTERLEAVED_PROLOGUE_PHASES = [
    TilePhase("load_kernargs", phase_load_kernargs),
    TilePhase("thread_indexing", phase_thread_indexing),
    TilePhase("load_cluster_setup", phase_load_cluster_setup),
    TilePhase("lds_addrs", phase_lds_addrs),
    TilePhase("init_acc", phase_init_acc),
    TilePhase("global_addrs", phase_global_addrs),
    TilePhase("pgr2_interleaved_k_loop", phase_pgr2_interleaved_k_loop),
]
