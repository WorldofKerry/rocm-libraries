"""Partition-based DTL K-loop using MainloopScheduler + SlotPlacer.

The macrotile is divided into partitions. Within each partition, mi values
are processed with ping-pong A buffers. The SlotPlacer decides where to
interleave A-prefetch ds_reads between MFMAs.

Flow:
  1. PartitionPlan.from_tiling() -- derive partition structure
  2. MainloopScheduler.build_modules() -- create MFMA/LR emit closures
  3. SlotPlacer -- interleave LR ops between MFMAs
  4. Emit: preamble (manual) + scheduled body + suffix (manual)
"""
from __future__ import annotations

import math

from .asm_context import AsmContext
from .problem import GemmProblem, TileConfig
from .tile import TilePhase
from .partition_plan import PartitionPlan, Partition
from .mainloop_scheduler import MainloopScheduler, ScheduleModule, ModuleKind
from .slot_placer import SlotPlacer, PlacedOp, Path, SchedulingRules
from .dtl_interleaved import (
    phase_dtl_interleaved_setup,
    _emit_dtl_loads_a, _emit_dtl_loads_b,
    _a_off, _b_off,
)

__all__ = ["phase_dtl_partitioned_k_loop", "DTL_PARTITIONED_PROLOGUE_PHASES"]


def phase_dtl_partitioned_k_loop(level, ctx):
    """DTL K-loop with partition-based scheduling."""
    tile = ctx._metadata["tile"]
    problem = ctx._metadata["problem"]
    elem = problem.element_bytes
    mfma = tile.mfma

    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    ki_count = tile.k_iterations
    av = mfma.a_vgprs
    bv = mfma.b_vgprs

    lds_half = (tile.wg_m + tile.wg_n) * tile.unroll_k * elem
    k_stride = tile.unroll_k * elem
    log2_uk = int(math.log2(tile.unroll_k))
    threads_per_row = tile.unroll_k // 8
    rows_per_load = tile.block_size // threads_per_row
    num_loads_a = tile.wg_m // rows_per_load
    num_loads_b = tile.wg_n // rows_per_load

    partition_m = 2
    mfmas_per_mi = nr * ki_count  # 16

    # ---- Registers ----
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

    # ---- K-loop setup ----
    ctx.comment("=== DTL Partitioned K-loop ===")
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
    # Build MFMA + LR schedule via SlotPlacer
    # ================================================================
    all_mfma_ops = []
    for mi in range(mr):
        cur_buf = mi % 2
        for ki in range(ki_count):
            for ni in range(nr):
                acc_per = mfma.acc_vgprs
                acc_off = (mi * nr + ni) * acc_per

                def _mk_mfma(mi_=mi, ni_=ni, ki_=ki, buf_=cur_buf,
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

                all_mfma_ops.append(PlacedOp(
                    emit_fn=_mk_mfma(), op_type="mfma",
                    comment=f"m{mi}_n{ni}_k{ki}"))

    # LR paths: A-prefetch for mi+1, placed within mi's MFMA range
    lr_paths = []
    for mi in range(mr - 1):
        next_buf = (mi + 1) % 2
        path_ops = []
        for ki in range(ki_count):
            def _mk_lr(mi_=mi + 1, ki_=ki, buf_=next_buf):
                def emit():
                    ctx.ds_read(ctx.vreg(a_names[(buf_, ki_)], 0, av),
                                ctx.vreg("v_lds_rd_a"),
                                offset=_a_off(mi_, ki_, tile, mfma, elem),
                                width=av,
                                comment=f"LR A m{mi_}k{ki_} b{buf_}")
                return emit
            path_ops.append(PlacedOp(
                emit_fn=_mk_lr(), op_type="ds_read",
                comment=f"A m{mi+1}k{ki}"))
        lr_paths.append(Path(ops=path_ops, reverse=False, module_id=mi))

    # Suffix ops for the last mi group (placed backward)
    suffix_ops = [
        PlacedOp(emit_fn=lambda: ctx.s_waitcnt("vmcnt(0)", comment="wait DTL"),
                 op_type="wait", comment="vmcnt"),
        PlacedOp(emit_fn=lambda: ctx.v_add(ctx.vreg("v_lds_rd_a"),
                 ctx.sreg("s_lds_db_step"), ctx.vreg("v_lds_rd_a"),
                 comment="rd_a += db"), op_type="salu", comment="toggle_a"),
        PlacedOp(emit_fn=lambda: ctx.v_add(ctx.vreg("v_lds_rd_b"),
                 ctx.sreg("s_lds_db_step"), ctx.vreg("v_lds_rd_b"),
                 comment="rd_b += db"), op_type="salu", comment="toggle_b"),
        PlacedOp(emit_fn=lambda: ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"),
                 "0", ctx.sreg("s_lds_db_step"), comment="negate db"),
                 op_type="salu", comment="negate"),
    ]
    suffix_path = Path(ops=suffix_ops, reverse=True, module_id=99)

    # SlotPlacer: constrained LR placement + backward suffix
    rules = SchedulingRules(
        total_slots=(len(all_mfma_ops) - 1) * 2,
        min_ds_read_gap=4)

    placer = SlotPlacer(
        mfmas=all_mfma_ops,
        validators=[rules.one_ds_read_per_interval],
        on_place=rules.track_placement)

    # Place each LR path within its mi's MFMA interval range
    for mi, path in enumerate(lr_paths):
        mi_start_slot = mi * mfmas_per_mi * 2
        mi_end_slot = (mi + 1) * mfmas_per_mi * 2
        slot_a = mi_start_slot + 4
        slot_b = mi_start_slot + 20
        for i, op in enumerate(path.ops):
            target = slot_a if i == 0 else slot_b
            placed = False
            for s in range(target, mi_end_slot):
                if placer._can_place(s, op):
                    placer._slots[s].append(op)
                    if placer._on_place:
                        placer._on_place(placer, s, op)
                    placed = True
                    break
            if not placed:
                placer.leftovers.append(op)

    # Place suffix backward
    placer.place_path(suffix_path)

    schedule = placer.build()

    # ================================================================
    # Emit K-loop
    # ================================================================
    ctx.label("k_loop")
    ctx.raw("")

    # DTL prefix
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

    # Preamble: B + A[m0] with split-ki lgkmcnt(9)
    ctx.comment("Preamble: B + A[m0]")
    for ni in range(nr):
        ctx.ds_read(ctx.vreg(b_names[(ni, 0)], 0, bv),
                    ctx.vreg("v_lds_rd_b"),
                    offset=_b_off(ni, 0, tile, mfma, elem),
                    width=bv, comment=f"LR B n{ni}k0")
    ctx.ds_read(ctx.vreg(a_names[(0, 0)], 0, av),
                ctx.vreg("v_lds_rd_a"),
                offset=_a_off(0, 0, tile, mfma, elem),
                width=av, comment="LR A m0k0 b0")
    for ni in range(nr):
        ctx.ds_read(ctx.vreg(b_names[(ni, 1)], 0, bv),
                    ctx.vreg("v_lds_rd_b"),
                    offset=_b_off(ni, 1, tile, mfma, elem),
                    width=bv, comment=f"LR B n{ni}k1")
    ctx.ds_read(ctx.vreg(a_names[(0, 1)], 0, av),
                ctx.vreg("v_lds_rd_a"),
                offset=_a_off(0, 1, tile, mfma, elem),
                width=av, comment="LR A m0k1 b0")
    ctx.s_waitcnt("lgkmcnt(9)", comment="wait B[ki=0] + A[m0,k0]")
    ctx.raw("")

    # ---- Emit scheduled body ----
    inflight_lgkm = 9
    mfma_count = 0

    for side_ops, mfma_op in schedule.intervals:
        # Wait for B[ki=1] before mi=0 ki=1
        if mfma_count == nr and inflight_lgkm > 0:
            ctx.s_waitcnt("lgkmcnt(0)", comment="wait B[ki=1] + A[m0,k1]")
            inflight_lgkm = 0

        # Wait for A prefetch at each mi boundary
        if mfma_count > 0 and mfma_count % mfmas_per_mi == 0 and inflight_lgkm > 0:
            ctx.s_waitcnt("lgkmcnt(0)",
                          comment=f"wait A[m{mfma_count // mfmas_per_mi}]")
            inflight_lgkm = 0

        # Partition boundary comments
        if mfma_count % (partition_m * mfmas_per_mi) == 0:
            ctx.comment(f"--- Partition {mfma_count // (partition_m * mfmas_per_mi)} ---")

        for op in side_ops:
            if op.emit_fn:
                op.emit_fn()
            if op.op_type == "ds_read":
                inflight_lgkm += 1

        if mfma_op and mfma_op.emit_fn:
            mfma_op.emit_fn()
            mfma_count += 1

    # Emit leftovers (suffix ops that couldn't be placed)
    for op in schedule.leftovers:
        if op.emit_fn:
            op.emit_fn()

    # Postamble
    ctx.s_barrier(comment="sync")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0", comment="more?")
    ctx.inst("s_cbranch_scc1", "k_loop", comment="loop")
    ctx.raw("")


DTL_PARTITIONED_PROLOGUE_PHASES = [
    TilePhase("dtl_interleaved_setup", phase_dtl_interleaved_setup),
    TilePhase("dtl_partitioned_k_loop", phase_dtl_partitioned_k_loop),
]
