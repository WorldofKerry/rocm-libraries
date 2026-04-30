"""Auto-scheduled DTL K-loop using ScheduleGraph SPREAD algorithm.

Uses the same prologue (phase_dtl_interleaved_setup) and register allocation
as dtl_interleaved.  The K-loop body builds a ScheduleGraph with all 32
ds_reads and 128 MFMAs, uses SPREAD to interleave them, and fills every
remaining empty MFMA slot with s_nop to prevent pipeline stalls.

Key improvement over dtl_interleaved:
  - Every MFMA pair has at least one instruction between them (s_nop or real op)
  - This targets the 42% MFMA pipeline stall from back-to-back MFMAs
  - ds_reads are spread more evenly for better latency hiding
  - lgkmcnt values are auto-computed at emit time from dependency tracking
  - DTL loads remain at loop top (sequential m0/soffset setup)
"""
from __future__ import annotations

import math

from .asm_context import AsmContext
from .asm_transforms import GemmLayouts
from .problem import GemmProblem, TileConfig
from .tile import TilePhase
from .auto_scheduler import ScheduleGraph, OpType, Latencies
from .dtl_interleaved import (
    phase_dtl_interleaved_setup,
    _emit_dtl_loads_a, _emit_dtl_loads_b,
    _a_off, _b_off,
)

__all__ = ["phase_dtl_scheduled_k_loop", "DTL_SCHEDULED_PROLOGUE_PHASES"]


# ---------------------------------------------------------------------------
# lgkmcnt tracker -- auto-computes precise wait counts at emit time
# ---------------------------------------------------------------------------
class _LgkmTracker:
    """Track in-flight lgkm operations for auto-waitcnt insertion."""

    def __init__(self):
        self.inflight: list[int] = []  # op_ids in issue order (oldest first)

    def issue(self, op_id: int):
        self.inflight.append(op_id)

    def wait_for(self, op_ids: list[int]):
        """Compute lgkmcnt(N) needed to ensure all op_ids are complete.

        Returns None if all deps are already retired.
        """
        positions = []
        for oid in op_ids:
            try:
                positions.append(self.inflight.index(oid))
            except ValueError:
                pass  # already retired by a previous wait
        if not positions:
            return None
        # The latest (most-recently-issued) dep determines the wait value.
        # lgkmcnt(N) retires all but the newest N requests.
        max_pos = max(positions)
        return len(self.inflight) - max_pos - 1

    def apply_wait(self, wait_val):
        """Retire reads after a lgkmcnt(wait_val)."""
        if wait_val is None:
            return
        completed = len(self.inflight) - wait_val
        if completed > 0:
            self.inflight = self.inflight[completed:]


# ---------------------------------------------------------------------------
# Scheduled K-loop phase
# ---------------------------------------------------------------------------
def phase_dtl_scheduled_k_loop(level, ctx):
    """DTL K-loop with auto-scheduled ds_read interleaving."""
    tile = ctx._metadata["tile"]
    problem = ctx._metadata["problem"]
    layouts = ctx._metadata["layouts"]
    elem = problem.element_bytes
    mfma = tile.mfma

    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    ki_count = tile.k_iterations
    av = mfma.a_vgprs
    bv = mfma.b_vgprs

    # LDS half size for double-buffering
    if tile.lds_pad > 0:
        tpr = tile.unroll_k // 8
        rpl = tile.block_size // tpr
        nla = tile.wg_m // rpl
        nlb = tile.wg_n // rpl
        lds_half = (tile.wg_m * tile.unroll_k * elem + nla * tile.lds_pad
                    + tile.wg_n * tile.unroll_k * elem + nlb * tile.lds_pad)
    else:
        lds_half = (tile.wg_m + tile.wg_n) * tile.unroll_k * elem

    k_stride = tile.unroll_k * elem
    log2_uk = int(math.log2(tile.unroll_k))
    threads_per_row = tile.unroll_k // 8
    rows_per_load = tile.block_size // threads_per_row
    num_loads_a = tile.wg_m // rows_per_load
    num_loads_b = tile.wg_n // rows_per_load

    # -- Allocate registers (same as dtl_interleaved) --
    ctx.alloc_sgpr_permanent(1, "s_lds_db_step")

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

    # -- K-loop setup --
    ctx.comment("=== DTL Scheduled K-loop ===")
    ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
               comment=f"k_tiles = K / {tile.unroll_k}")
    ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_half),
              comment=f"DB step = {lds_half}")
    ctx.raw("")

    # Prologue: DTL load first tile
    ctx.comment("Prologue: DTL tile 0")
    _emit_dtl_loads_a(ctx, tile, problem, num_loads_a)
    _emit_dtl_loads_b(ctx, tile, problem, num_loads_b)
    ctx.s_waitcnt("vmcnt(0)", comment="wait DTL")
    ctx.s_barrier(comment="sync")
    ctx.raw("")

    # ================================================================
    # Build ScheduleGraph with all ds_reads + MFMAs
    # ================================================================
    g = ScheduleGraph()

    # -- B ds_reads (consumed by all mi groups) --
    b_read_ids = {}
    for ki in range(ki_count):
        for ni in range(nr):
            def _mk_b(ni_=ni, ki_=ki):
                def emit():
                    ctx.ds_read(ctx.vreg(b_names[(ni_, ki_)], 0, bv),
                                ctx.vreg("v_lds_rd_b"),
                                offset=_b_off(ni_, ki_, tile, mfma, elem),
                                width=bv, comment=f"LR B n{ni_}k{ki_}")
                return emit
            bid = g.add_ds_read("B", emit_fn=_mk_b(),
                                comment=f"LR B n{ni}k{ki}",
                                ni=ni, ki=ki)
            b_read_ids[(ni, ki)] = bid

    # -- A ds_reads + MFMAs, built per-mi for correct anti-dep wiring --
    a_read_ids = {}
    mfma_ids = {}

    for mi in range(mr):
        buf = mi % 2

        # A reads for this mi
        for ki in range(ki_count):
            a_deps = []
            # Anti-dep: can't overwrite buf until the previous user is done
            if mi >= 2:
                prev_mi = mi - 2
                a_deps.append(mfma_ids[(prev_mi, nr - 1, ki_count - 1)])

            def _mk_a(mi_=mi, ki_=ki, buf_=buf):
                def emit():
                    ctx.ds_read(ctx.vreg(a_names[(buf_, ki_)], 0, av),
                                ctx.vreg("v_lds_rd_a"),
                                offset=_a_off(mi_, ki_, tile, mfma, elem),
                                width=av,
                                comment=f"LR A m{mi_}k{ki_} b{buf_}")
                return emit

            aid = g.add_ds_read("A", deps=a_deps, emit_fn=_mk_a(),
                                comment=f"LR A m{mi}k{ki}",
                                mi=mi, ki=ki)
            a_read_ids[(mi, ki)] = aid

        # MFMAs for this mi (maintains correct spine order)
        for ki in range(ki_count):
            for ni in range(nr):
                m_deps = [b_read_ids[(ni, ki)], a_read_ids[(mi, ki)]]
                if ki > 0:
                    m_deps.append(mfma_ids[(mi, ni, ki - 1)])

                acc_per = mfma.acc_vgprs
                acc_off = (mi * nr + ni) * acc_per

                def _mk_mfma(mi_=mi, ni_=ni, ki_=ki, buf_=buf,
                             acc_off_=acc_off, acc_per_=acc_per):
                    def emit():
                        ctx.inst(
                            mfma.instruction_name,
                            ctx.areg("acc_C", acc_off_, acc_per_),
                            ctx.vreg(a_names[(buf_, ki_)], 0, av),
                            ctx.vreg(b_names[(ni_, ki_)], 0, bv),
                            ctx.areg("acc_C", acc_off_, acc_per_),
                            comment=f"MFMA m{mi_}_n{ni_}_k{ki_}")
                    return emit

                mid = g.add_mfma(mi, ni, ki, deps=m_deps,
                                 emit_fn=_mk_mfma())
                mfma_ids[(mi, ni, ki)] = mid

    # ================================================================
    # Schedule: SPREAD interleaving with s_nop gap fill
    # ================================================================
    sched = g.schedule(
        latencies=Latencies(ds_read=20, mfma=16),
        max_side_per_slot=2,
        min_side_per_slot=0,
    )

    # ================================================================
    # K-loop emission
    # ================================================================
    ctx.label("k_loop")
    ctx.raw("")

    # Manual prefix: loop control + conditional DTL loads
    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="more tiles?")
    ctx.inst("s_cbranch_scc0", "dtl_skip_all",
             comment="skip DTL on last iter")

    for srd in ["s_srd_a", "s_srd_b"]:
        ctx.inst("s_add_u32", ctx.sreg(srd, 0, 1),
                 ctx.sreg(srd, 0, 1), str(k_stride),
                 comment=f"{srd} += {k_stride}")
        ctx.inst("s_addc_u32", ctx.sreg(srd, 1, 1),
                 ctx.sreg(srd, 1, 1), "0", comment="carry")

    ctx.inst("s_add_u32", ctx.sreg("s_lds_wr_a_sg"),
             ctx.sreg("s_lds_wr_a_sg"), ctx.sreg("s_lds_db_step"),
             comment="wr_a += db")
    ctx.inst("s_add_u32", ctx.sreg("s_lds_wr_b_sg"),
             ctx.sreg("s_lds_wr_b_sg"), ctx.sreg("s_lds_db_step"),
             comment="wr_b += db")

    _emit_dtl_loads_a(ctx, tile, problem, num_loads_a)
    _emit_dtl_loads_b(ctx, tile, problem, num_loads_b)
    ctx.raw("")

    ctx.label("dtl_skip_all")
    ctx.raw("")

    # -- Emit scheduled block with auto-lgkmcnt --
    tracker = _LgkmTracker()

    # Prologue ops (reads that couldn't fit in the MFMA spine)
    for op in sched.prologue:
        if op.op_type == OpType.DS_READ:
            if op.emit_fn:
                op.emit_fn()
            tracker.issue(op.id)
        elif op.op_type == OpType.NOP:
            ctx.inst("s_nop", "0")
        elif op.emit_fn:
            op.emit_fn()

    # MFMA slots with interleaved side ops
    for slot in sched.slots:
        for op in slot.before_mfma:
            if op.op_type == OpType.DS_READ:
                if op.emit_fn:
                    op.emit_fn()
                tracker.issue(op.id)
            elif op.op_type == OpType.NOP:
                ctx.inst("s_nop", "0")
            elif op.emit_fn:
                op.emit_fn()

        # Auto-lgkmcnt before MFMA if it depends on outstanding ds_reads
        if slot.mfma:
            mfma_op = slot.mfma
            dep_reads = [d for d in mfma_op.deps
                         if d in g._ops
                         and g._ops[d].op_type == OpType.DS_READ]
            wait_val = tracker.wait_for(dep_reads)
            if wait_val is not None:
                ctx.s_waitcnt(f"lgkmcnt({wait_val})",
                              comment=f"wait for {mfma_op.comment}")
                tracker.apply_wait(wait_val)

            if mfma_op.emit_fn:
                mfma_op.emit_fn()

    # Postamble ops (from scheduler, if any)
    for op in sched.postamble:
        if op.emit_fn:
            op.emit_fn()

    # Manual postamble: vmcnt, toggle, negate, barrier, branch
    ctx.s_waitcnt("vmcnt(0)", comment="wait DTL")
    ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.sreg("s_lds_db_step"),
              ctx.vreg("v_lds_rd_a"), comment="rd_a += db")
    ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.sreg("s_lds_db_step"),
              ctx.vreg("v_lds_rd_b"), comment="rd_b += db")
    ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
             ctx.sreg("s_lds_db_step"), comment="negate db")
    ctx.s_barrier(comment="sync")

    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="more?")
    ctx.inst("s_cbranch_scc1", "k_loop", comment="loop")
    ctx.raw("")


DTL_SCHEDULED_PROLOGUE_PHASES = [
    TilePhase("dtl_interleaved_setup", phase_dtl_interleaved_setup),
    TilePhase("dtl_scheduled_k_loop", phase_dtl_scheduled_k_loop),
]
