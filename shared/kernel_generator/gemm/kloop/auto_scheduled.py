"""Auto-scheduled K-loop: compute phase auto-scheduled, prefix/suffix manual.

Uses ScheduleGraph ONLY for the compute phase (ds_reads + MFMAs with
A-operand prefetch). The K-loop prefix (global loads) and suffix
(vmcnt + toggle + ds_writes + barrier) remain as sequential blocks.

This avoids the problem of mixing vmcnt (global loads) with lgkmcnt
(ds_reads) in the same schedule, which causes mid-compute stalls.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

from ..schedule.graph import ScheduleGraph, OpType, SchedOp
from ..emit.context import AsmContext
from ..problem import GemmProblem, TileConfig
from ..tile.tree import TilePhase

__all__ = ["phase_auto_scheduled_k_loop"]


def _tile(ctx): return ctx._metadata["tile"]
def _problem(ctx): return ctx._metadata["problem"]
def _layouts(ctx): return ctx._metadata["layouts"]


def phase_auto_scheduled_k_loop(level, ctx):
    """K-loop with auto-scheduled compute phase."""
    from ..emit.phases import (_emit_global_load_impl, _emit_lds_write_impl,
                         _emit_global_load_no_wait)

    tile = _tile(ctx)
    problem = _problem(ctx)
    mfma = tile.mfma
    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    ki_count = tile.k_iterations
    av = mfma.a_vgprs
    bv = mfma.b_vgprs
    elem = problem.element_bytes
    lds_half = (tile.wg_m + tile.wg_n) * tile.unroll_k * elem
    k_stride = tile.unroll_k * elem
    log2_uk = int(math.log2(tile.unroll_k))

    ctx.alloc_sgpr_permanent(1, "s_lds_db_step")
    ctx.comment("=== Auto-scheduled K-loop ===")
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

    # Allocate operand registers
    b_reg = {}
    for ni in range(nr):
        for ki in range(ki_count):
            name = f"v_b_s{ni}k{ki}"
            if not ctx.has(name):
                ctx.alloc_vgpr_permanent(bv, name)
            b_reg[(ni, ki)] = name

    a_reg = {}
    for buf in range(2):
        for ki in range(ki_count):
            name = f"v_a_b{buf}k{ki}"
            if not ctx.has(name):
                ctx.alloc_vgpr_permanent(av, name)
            a_reg[(buf, ki)] = name

    def a_off(mi, ki):
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (mi * mfma.m * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem
    def b_off(ni, ki):
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (ni * mfma.n * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem

    # Build compute-only graph (ds_reads + MFMAs, no global loads/ds_writes)
    g = ScheduleGraph()
    b_ids = {}
    a_ids = {}

    for ki in range(ki_count):
        for ni in range(nr):
            _ni, _ki = ni, ki
            b_ids[(ni, ki)] = g.add_ds_read(
                "B", ni=ni, ki=ki,
                emit_fn=lambda _ni=_ni, _ki=_ki: ctx.ds_read(
                    ctx.vreg(b_reg[(_ni, _ki)], 0, bv),
                    ctx.vreg("v_lds_rd_b"),
                    offset=b_off(_ni, _ki), width=bv,
                    comment=f"LR B n{_ni}k{_ki}"))

        cur_a = 0
        for mi in range(mr):
            _mi, _ki, _buf = mi, ki, cur_a
            a_ids[(mi, ki)] = g.add_ds_read(
                "A", mi=mi, ki=ki, buf=cur_a,
                emit_fn=lambda _mi=_mi, _ki=_ki, _buf=_buf: ctx.ds_read(
                    ctx.vreg(a_reg[(_buf, _ki)], 0, av),
                    ctx.vreg("v_lds_rd_a"),
                    offset=a_off(_mi, _ki), width=av,
                    comment=f"LR A m{_mi}k{_ki} b{_buf}"))

            for ni in range(nr):
                deps = [b_ids[(ni, ki)], a_ids[(mi, ki)]]
                _mi2, _ni, _ki2, _buf2 = mi, ni, ki, cur_a
                acc_per = mfma.acc_vgprs
                _ao = (mi * nr + ni) * acc_per

                g.add_mfma(mi=mi, ni=ni, ki=ki, deps=deps,
                    emit_fn=lambda _mi=_mi2, _ni=_ni, _ki=_ki2, _buf=_buf2, _ao=_ao:
                        ctx.inst(mfma.instruction_name,
                            ctx.areg("acc_C", _ao, acc_per),
                            ctx.vreg(a_reg[(_buf, _ki)], 0, av),
                            ctx.vreg(b_reg[(_ni, _ki)], 0, bv),
                            ctx.areg("acc_C", _ao, acc_per),
                            comment=f"MFMA m{_mi}_n{_ni}_k{_ki}"))

            if mi < mr - 1:
                cur_a = 1 - cur_a

    schedule = g.schedule(max_side_per_slot=2, min_side_per_slot=1)
    ctx.comment(f"Auto-scheduled compute: {schedule.total_mfma} MFMAs, "
                f"{schedule.total_side_ops} side ops")

    # K-loop
    ctx.label("k_loop")
    ctx.raw("")

    # PREFIX: conditional global loads for next tile
    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc0", "auto_skip_gload",
             comment="skip gload on last iter")
    for addr in ["v_addr_a", "v_addr_b"]:
        ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                 str(k_stride), ctx.vreg(addr, 0, 1),
                 comment=f"{addr} += {k_stride}")
        ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                 ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")
    _emit_global_load_no_wait(ctx, problem, tile)
    ctx.label("auto_skip_gload")
    ctx.raw("")

    # COMPUTE: auto-scheduled ds_reads + MFMAs
    # Emit prologue (reads that couldn't fit into MFMA slots)
    for op in schedule.prologue:
        if op.emit_fn:
            op.emit_fn()

    # Emit interleaved schedule with auto-waitcnt
    lgkm_inflight = len(schedule.prologue)  # prologue reads are in flight
    pending_read_ids = set(op.id for op in schedule.prologue
                          if op.op_type == OpType.DS_READ)

    for slot in schedule.slots:
        # Emit side ops
        for op in slot.before_mfma:
            if op.op_type == OpType.NOP:
                ctx.inst("s_nop", "0", comment="pipeline fill")
            elif op.op_type == OpType.DS_READ:
                if op.emit_fn:
                    op.emit_fn()
                lgkm_inflight += 1
                pending_read_ids.add(op.id)
            elif op.emit_fn:
                op.emit_fn()

        # Before MFMA: insert waitcnt if needed
        if slot.mfma:
            need_wait = any(d in pending_read_ids for d in slot.mfma.deps)
            if need_wait and lgkm_inflight > 0:
                ctx.s_waitcnt(f"lgkmcnt(0)",
                              comment=f"wait {lgkm_inflight} LDS reads")
                lgkm_inflight = 0
                pending_read_ids.clear()
            if slot.mfma.emit_fn:
                slot.mfma.emit_fn()
    ctx.raw("")

    # SUFFIX: vmcnt + toggle + ds_writes + barrier
    ctx.s_waitcnt("vmcnt(0)", comment="wait for global loads")
    for reg in ["v_lds_wr_a", "v_lds_wr_b", "v_lds_rd_a", "v_lds_rd_b"]:
        ctx.v_add(ctx.vreg(reg), ctx.sreg("s_lds_db_step"), ctx.vreg(reg),
                  comment=f"{reg} += db_step")
    ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
             ctx.sreg("s_lds_db_step"), comment="negate")
    _emit_lds_write_impl(ctx, tile)

    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc1", "k_loop", comment="next K-tile")
    ctx.raw("")


def _get_prologue_phases():
    from ..emit.phases import (phase_load_kernargs, phase_thread_indexing,
                         phase_load_cluster_setup, phase_lds_addrs,
                         phase_init_acc, phase_global_addrs)
    return [
        TilePhase("load_kernargs", phase_load_kernargs),
        TilePhase("thread_indexing", phase_thread_indexing),
        TilePhase("load_cluster_setup", phase_load_cluster_setup),
        TilePhase("lds_addrs", phase_lds_addrs),
        TilePhase("init_acc", phase_init_acc),
        TilePhase("global_addrs", phase_global_addrs),
        TilePhase("auto_scheduled_k_loop", phase_auto_scheduled_k_loop),
    ]
