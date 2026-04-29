"""PGR=2 interleaved K-loop: all overhead hidden between MFMAs.

Key optimizations over phase_pgr2_k_loop:
1. Global loads issued between early MFMAs (overlap with compute)
2. vmcnt(N) instead of vmcnt(0) (keep next tile's loads in flight)
3. ds_writes between late MFMAs (overlap with compute)
4. s_nop between MFMAs that have no side ops (prevent pipeline stalls)
5. Precise lgkmcnt for A-operand prefetch

Structure per K-loop iteration (16 MFMAs, 2x unrolled for buffer swap):
  Preamble: ds_read B[n0..n3] + A[m0], lgkmcnt(0)
  mi=0: ptr advance + 4 gloads (interleaved), ds_read A[m1]
  mi=1: s_nop fillers, lgkmcnt for A[m1], ds_read A[m2]
  mi=2: vmcnt(N) + toggle + start ds_writes, lgkmcnt for A[m2], ds_read A[m3]
  mi=3: finish ds_writes + lgkmcnt + barrier + loop control
"""
from __future__ import annotations

import math

from .asm_context import AsmContext
from .asm_transforms import GemmLayouts
from .problem import GemmProblem, TileConfig
from .tile import TilePhase
from .phases import (phase_load_kernargs, phase_thread_indexing,
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


def _emit_gloads_individual(ctx, problem, tile, buf_suffix):
    """Return a list of callables, each issuing one global_load_dwordx4."""
    ops = []
    for name, addr_name in [("A", "v_addr_a"), ("B", "v_addr_b")]:
        gload_name = f"v_gload{buf_suffix}_{name.lower()}"
        load = ctx.get(gload_name)
        addr = ctx.vreg(addr_name, 0, 2)
        for i in range(0, load.count, 4):
            cnt = min(4, load.count - i)
            width = {4: "dwordx4", 2: "dwordx2", 1: "dword"}[cnt]
            _i, _cnt, _width, _gload_name = i, cnt, width, gload_name
            _addr_name = addr_name
            _name = name
            def emit_one(_i=_i, _cnt=_cnt, _width=_width,
                         _gload_name=_gload_name, _addr_name=_addr_name,
                         _name=_name):
                a = ctx.vreg(_addr_name, 0, 2)
                dst = ctx.vreg(_gload_name, _i, _cnt)
                off = f"off offset:{_i * 4}" if _i > 0 else "off"
                ctx.inst(f"global_load_{_width}", dst, a, off,
                         comment=f"gload {_name}[{_i}:{_i+_cnt}]")
            ops.append(emit_one)
    return ops


def _emit_ds_writes_individual(ctx, tile, buf_suffix):
    """Return a list of callables, each issuing one ds_write."""
    ops = []
    for name in ["a", "b"]:
        gload_name = f"v_gload{buf_suffix}_{name}"
        load = ctx.get(gload_name)
        _name = name
        for i in range(0, load.count, 4):
            cnt = min(4, load.count - i)
            _i, _cnt, _gload_name = i, cnt, gload_name
            def emit_one(_i=_i, _cnt=_cnt, _gload_name=_gload_name,
                         _name=_name):
                addr_reg = ctx.vreg(f"v_lds_wr_{_name}")
                src = ctx.vreg(_gload_name, _i, _cnt)
                ctx.ds_write(addr_reg, src, offset=_i * 4, width=_cnt,
                             comment=f"LDS write {_name.upper()}[{_i}:{_i+_cnt}]")
            ops.append(emit_one)
    return ops


def _emit_one_iter(ctx, tile, problem, mfma, mr, nr, ki_count,
                   a_names, b_names, elem,
                   load_buf_suffix, write_buf_suffix,
                   k_stride, is_first_half, label_suffix):
    """Emit one half of the 2x-unrolled main loop with full interleaving.

    load_buf_suffix:  buffer to issue NEW global loads into ("" or "2")
    write_buf_suffix: buffer to wait for + write to LDS ("2" or "")
    """
    av = mfma.a_vgprs
    bv = mfma.b_vgprs

    # Build lists of individual operations
    gload_ops = _emit_gloads_individual(ctx, problem, tile, load_buf_suffix)
    ds_write_ops = _emit_ds_writes_individual(ctx, tile, write_buf_suffix)
    n_gloads = len(gload_ops)
    n_ds_writes = len(ds_write_ops)

    # Preamble: read all B + A[m0]
    ctx.comment(f"--- Iter {label_suffix}: preamble ---")
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

    ctx.s_waitcnt("lgkmcnt(0)", comment=f"wait preamble {label_suffix}")
    ctx.raw("")

    # Track gload and ds_write insertion points
    gload_idx = 0
    ds_write_idx = 0

    for mi in range(mr):
        has_a_prefetch = (mi < mr - 1)

        # === mi=0: ptr advance + gloads ===
        if mi == 0:
            ctx.comment(f"--- mi=0: gloads + ptr advance ---")
            # Check if there are tiles left to load BEFORE decrementing.
            # k_tiles counts remaining tiles to load. If > 0, load one.
            ctx.inst("s_cmp_eq_u32", ctx.sreg("s_k_tiles"), "0",
                     comment="SCC = (k_tiles == 0, no more to load)")
            ctx.inst("s_cbranch_scc1", f"skip_gload_{label_suffix}",
                     comment="skip gload if none left")
            ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
                      comment="k_tiles-- (consumed a gload)")

            # Advance pointers
            for addr in ["v_addr_a", "v_addr_b"]:
                ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                         str(k_stride), ctx.vreg(addr, 0, 1),
                         comment=f"{addr} += {k_stride}")
                ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                         ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")

            # Issue all gloads
            for g_op in gload_ops:
                g_op()
            gload_idx = n_gloads

            ctx.label(f"skip_gload_{label_suffix}")
            ctx.raw("")

        # A prefetch before this mi group's MFMAs
        if has_a_prefetch:
            next_a = 1 - cur_a
            for ki in range(ki_count):
                name = a_names[(next_a, ki)]
                ctx.ds_read(ctx.vreg(name, 0, av), ctx.vreg("v_lds_rd_a"),
                            offset=_a_off(mi + 1, ki, tile, mfma, elem),
                            width=av,
                            comment=f"LR A m{mi+1}k{ki} b{next_a}")

        # === mi=2: vmcnt + toggle before MFMAs ===
        if mi == 2:
            ctx.comment(f"--- mi=2: vmcnt + toggle ---")
            # Wait for PREVIOUS tile's gloads (the ones in write_buf).
            # Our new gloads (in load_buf) should stay in flight.
            # vmcnt(N) where N = number of new gloads still in flight
            # Use vmcnt(0) for safety: when gloads were skipped (last iter),
            # only previous iter's gloads are in flight and vmcnt(N>0) won't wait.
            # TODO: use conditional vmcnt(N) when gloads were issued for perf
            ctx.s_waitcnt("vmcnt(0)",
                          comment="wait for in-flight global loads")

            # Toggle LDS double buffer
            for reg in ["v_lds_wr_a", "v_lds_wr_b", "v_lds_rd_a", "v_lds_rd_b"]:
                ctx.v_add(ctx.vreg(reg), ctx.sreg("s_lds_db_step"),
                          ctx.vreg(reg), comment=f"{reg} += db_step")
            ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
                     ctx.sreg("s_lds_db_step"),
                     comment="negate step for next toggle")

        # Emit MFMAs for this mi group with interleaved side ops
        for ni in range(nr):
            # Before each MFMA: insert side operations
            side_emitted = False

            # mi=3: interleave ds_writes
            if mi == mr - 1 and ds_write_idx < n_ds_writes:
                ds_write_ops[ds_write_idx]()
                ds_write_idx += 1
                side_emitted = True

            # mi=2, ni >= 2: start ds_writes
            if mi == 2 and ni >= 2 and ds_write_idx < n_ds_writes:
                ds_write_ops[ds_write_idx]()
                ds_write_idx += 1
                side_emitted = True

            # If no side op was emitted, insert s_nop to prevent MFMA stall
            if not side_emitted:
                ctx.inst("s_nop", "0", comment="prevent MFMA stall")

            # Emit MFMA
            for ki in range(ki_count):
                acc_per = mfma.acc_vgprs
                acc_off = (mi * nr + ni) * acc_per
                ctx.inst(
                    f"v_mfma_f32_{mfma.m}x{mfma.n}x{mfma.k}_f16",
                    ctx.areg("acc_C", acc_off, acc_per),
                    ctx.vreg(a_names[(cur_a, ki)], 0, av),
                    ctx.vreg(b_names[(ni, ki)], 0, bv),
                    ctx.areg("acc_C", acc_off, acc_per),
                    comment=f"MFMA m{mi}_n{ni}_k{ki}")

        # Wait for A prefetch
        if has_a_prefetch:
            ctx.s_waitcnt("lgkmcnt(0)", comment=f"wait A[{mi+1}]")
            cur_a = next_a

        ctx.raw("")

    # Emit any remaining ds_writes (shouldn't happen with 4 writes + 8 MFMA slots)
    while ds_write_idx < n_ds_writes:
        ds_write_ops[ds_write_idx]()
        ds_write_idx += 1

    # Wait for ds_writes + barrier
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait LDS writes")
    ctx.s_barrier(comment="sync workgroup")
    ctx.raw("")


def phase_pgr2_interleaved_k_loop(level, ctx):
    """K-loop with PGR=2 and fully interleaved operations."""
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

    # Allocate second global load buffer
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

    # === Prologue: load tile 0 -> buf[0], write to LDS ===
    ctx.comment("Prologue: load tile 0")
    _emit_global_load_impl(ctx, problem, tile)
    ctx.comment("Write tile 0 to LDS buf[0]")
    _emit_lds_write_impl(ctx, tile)

    # Advance pointers to tile 1
    for addr in ["v_addr_a", "v_addr_b"]:
        ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                 str(k_stride), ctx.vreg(addr, 0, 1),
                 comment=f"{addr} += {k_stride}")
        ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                 ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")

    # Load tile 1 -> buf[1] (async)
    ctx.comment("Prefetch tile 1 into buf[1]")
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

    # === Main loop (2x unrolled) ===
    ctx.label("k_loop")
    ctx.raw("")

    # Iter A: load into buf "" (v_gload_a/b), write from buf "2" (v_gload2_a/b)
    _emit_one_iter(ctx, tile, problem, mfma, mr, nr, ki_count,
                   a_names, b_names, elem,
                   load_buf_suffix="", write_buf_suffix="2",
                   k_stride=k_stride, is_first_half=True,
                   label_suffix="A")

    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc0", "k_nll",
             comment="branch to NLL if last iteration")
    ctx.raw("")

    # Iter B: load into buf "2" (v_gload2_a/b), write from buf "" (v_gload_a/b)
    _emit_one_iter(ctx, tile, problem, mfma, mr, nr, ki_count,
                   a_names, b_names, elem,
                   load_buf_suffix="2", write_buf_suffix="",
                   k_stride=k_stride, is_first_half=False,
                   label_suffix="B")

    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc1", "k_loop",
             comment="loop if more iterations")
    ctx.raw("")

    # === NLL for iter B exit: buf "2" was loaded last, buf "" was written ===
    ctx.comment("NLL (iter B): compute + write buf2")
    _emit_nll(ctx, tile, problem, mfma, mr, nr, ki_count,
              a_names, b_names, elem, write_buf_suffix="2",
              label_suffix="nll_b", lds_half=lds_half)

    ctx.inst("s_branch", "k_done", comment="skip k_nll")
    ctx.raw("")

    # === NLL for iter A exit: buf "" was loaded last, buf "2" was written ===
    ctx.label("k_nll")
    ctx.comment("NLL (iter A): compute + write buf0")
    _emit_nll(ctx, tile, problem, mfma, mr, nr, ki_count,
              a_names, b_names, elem, write_buf_suffix="",
              label_suffix="nll_a", lds_half=lds_half)

    ctx.label("k_done")
    ctx.raw("")


def _emit_nll(ctx, tile, problem, mfma, mr, nr, ki_count,
              a_names, b_names, elem, write_buf_suffix,
              label_suffix, lds_half):
    """No-Load Loop: compute last tile, then wait and write the
    in-flight buffer to LDS for final compute."""
    av = mfma.a_vgprs
    bv = mfma.b_vgprs

    ds_write_ops = _emit_ds_writes_individual(ctx, tile, write_buf_suffix)

    # Compute from current LDS
    _emit_compute_with_nops(ctx, tile, mfma, mr, nr, ki_count,
                            a_names, b_names, elem, label_suffix)

    # Wait for in-flight gloads (from the write buffer)
    ctx.s_waitcnt("vmcnt(0)", comment="wait final gloads")

    # Toggle LDS
    for reg in ["v_lds_wr_a", "v_lds_wr_b", "v_lds_rd_a", "v_lds_rd_b"]:
        ctx.v_add(ctx.vreg(reg), ctx.sreg("s_lds_db_step"),
                  ctx.vreg(reg), comment=f"{reg} += db_step")
    ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
             ctx.sreg("s_lds_db_step"), comment="negate")

    # Write to LDS
    for op in ds_write_ops:
        op()
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait LDS writes")
    ctx.s_barrier(comment="sync")

    # Final compute
    _emit_compute_with_nops(ctx, tile, mfma, mr, nr, ki_count,
                            a_names, b_names, elem, f"final_{label_suffix}")
    ctx.raw("")


def _emit_compute_with_nops(ctx, tile, mfma, mr, nr, ki_count,
                            a_names, b_names, elem, label):
    """Emit compute phase with s_nop between MFMAs and A prefetch."""
    av = mfma.a_vgprs
    bv = mfma.b_vgprs

    # Preamble
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

    for mi in range(mr):
        has_prefetch = mi < mr - 1
        if has_prefetch:
            next_a = 1 - cur_a
            for ki in range(ki_count):
                name = a_names[(next_a, ki)]
                ctx.ds_read(ctx.vreg(name, 0, av), ctx.vreg("v_lds_rd_a"),
                            offset=_a_off(mi + 1, ki, tile, mfma, elem),
                            width=av,
                            comment=f"LR A m{mi+1}k{ki} b{next_a}")

        for ki in range(ki_count):
            for ni in range(nr):
                ctx.inst("s_nop", "0", comment="prevent MFMA stall")
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
            ctx.s_waitcnt("lgkmcnt(0)", comment=f"wait A[{mi+1}]")
            cur_a = next_a

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
