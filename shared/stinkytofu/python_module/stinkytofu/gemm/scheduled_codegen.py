"""Scheduled codegen: TilePlan -> Schedule -> Assembly.

Implements Layers 1-3 of the architecture (DESIGN.md):
  Layer 1: TilePlan.build() generates TileOps from tile config
  Layer 2: SlotPlacer interleaves non-MFMA ops between MFMAs
  Layer 3: emit_schedule() walks Schedule and emits assembly

Usage:
    plan = TilePlan.build(tile, problem)
    schedule = plan.schedule(rules=SchedulingRules())
    emit_scheduled_kernel(schedule, ctx, tile, problem, layouts)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .schedule import (
    OpKind, Value, TileOp, Slot, Schedule,
    SchedulingRules, VGPRPool, SlotPlacer, _new_op_id,
)
from .problem import GemmProblem, TileConfig
from .asm_context import AsmContext
from .asm_transforms import GemmLayouts, emit_affine

__all__ = ["TilePlan", "emit_scheduled_kernel"]


# ===================================================================
# Layer 1: TilePlan -- generate TileOps from tile config
# ===================================================================

@dataclass
class TilePlan:
    """All TileOps for one K-tile iteration of a GEMM kernel."""
    mfma_ops: List[TileOp]       # ordered MFMA ops
    lds_read_ops: List[TileOp]   # LDS reads (A and B)
    global_load_ops: List[TileOp]  # global loads (A and B)
    lds_write_ops: List[TileOp]  # LDS writes
    overhead_ops: List[TileOp]   # barriers, waits, ptr advance, etc.
    values: List[Value]          # all values
    tile: TileConfig
    problem: GemmProblem

    @staticmethod
    def build(tile: TileConfig, problem: GemmProblem) -> TilePlan:
        """Generate TileOps for one K-tile iteration.

        Structure matches _emit_scheduled_compute: preamble loads all B
        + A[mi=0], then per-mi groups with A prefetch + MFMAs.

        The scheduling separates WHAT (this method) from WHEN (SlotPlacer).
        """
        mfma = tile.mfma
        elem = problem.element_bytes
        mr = tile.mfma_m_repeat
        nr = tile.mfma_n_repeat
        ki_count = tile.k_iterations
        av = mfma.a_vgprs
        bv = mfma.b_vgprs
        acc_per = mfma.acc_vgprs

        values: List[Value] = []
        mfma_ops: List[TileOp] = []
        lds_read_ops: List[TileOp] = []
        global_load_ops: List[TileOp] = []
        lds_write_ops: List[TileOp] = []
        overhead_ops: List[TileOp] = []

        # -- Create B values (all loaded in preamble) --
        b_vals: Dict[Tuple[int,int], Value] = {}
        for ni in range(nr):
            for ki in range(ki_count):
                v = Value(f"b_n{ni}_k{ki}", "vgpr", bv, "k_tile")
                b_vals[(ni, ki)] = v
                values.append(v)
                b_off = (ni * mfma.n * tile.unroll_k + ki * mfma.k) * elem
                op = TileOp(
                    kind=OpKind.LDS_READ, op_id=_new_op_id(),
                    tile_coords={"ni": ni, "ki": ki},
                    outputs=[v], static_offset=b_off,
                    matrix="B", comment=f"preamble B n{ni}k{ki}",
                )
                lds_read_ops.append(op)

        # -- Create A values (double-buffered across mi groups) --
        a_vals: Dict[Tuple[int,int], Value] = {}  # (buf, ki) -> Value
        for buf in range(2):
            for ki in range(ki_count):
                v = Value(f"a_b{buf}_k{ki}", "vgpr", av, "prefetch")
                a_vals[(buf, ki)] = v
                values.append(v)

        # -- Preamble: A[mi=0] reads --
        cur_a = 0
        for ki in range(ki_count):
            a_off = (0 * mfma.m * tile.unroll_k + ki * mfma.k) * elem
            op = TileOp(
                kind=OpKind.LDS_READ, op_id=_new_op_id(),
                tile_coords={"mi": 0, "ki": ki, "buf": cur_a},
                outputs=[a_vals[(cur_a, ki)]], static_offset=a_off,
                matrix="A", comment=f"preamble A m0k{ki} b{cur_a}",
            )
            lds_read_ops.append(op)

        # -- Preamble wait --
        wait_preamble = TileOp(
            kind=OpKind.WAIT_LDS, op_id=_new_op_id(),
            tile_coords={}, comment="wait preamble",
        )
        overhead_ops.append(wait_preamble)

        # -- Accumulator values (permanent) --
        acc_vals: Dict[Tuple[int,int], Value] = {}
        for mi in range(mr):
            for ni in range(nr):
                v = Value(f"acc_m{mi}_n{ni}", "acc", acc_per, "permanent")
                acc_vals[(mi, ni)] = v
                values.append(v)

        # -- Per-mi group: A prefetch + MFMAs --
        for mi in range(mr):
            # Prefetch A for next mi (if not last)
            if mi < mr - 1:
                next_a = 1 - cur_a
                for ki in range(ki_count):
                    a_off = ((mi + 1) * mfma.m * tile.unroll_k + ki * mfma.k) * elem
                    op = TileOp(
                        kind=OpKind.LDS_READ, op_id=_new_op_id(),
                        tile_coords={"mi": mi + 1, "ki": ki, "buf": next_a},
                        outputs=[a_vals[(next_a, ki)]], static_offset=a_off,
                        matrix="A",
                        comment=f"LR A m{mi+1}k{ki} b{next_a}",
                    )
                    lds_read_ops.append(op)

            # MFMAs for this mi: iterate ki then ni (group by A operand)
            for ki in range(ki_count):
                for ni in range(nr):
                    op = TileOp(
                        kind=OpKind.MFMA, op_id=_new_op_id(),
                        tile_coords={"mi": mi, "ni": ni, "ki": ki},
                        inputs=[a_vals[(cur_a, ki)], b_vals[(ni, ki)],
                                acc_vals[(mi, ni)]],
                        outputs=[acc_vals[(mi, ni)]],
                        comment=f"MFMA m{mi}_n{ni}_k{ki}",
                    )
                    mfma_ops.append(op)

            # Wait for A prefetch (if issued)
            if mi < mr - 1:
                wait_a = TileOp(
                    kind=OpKind.WAIT_LDS, op_id=_new_op_id(),
                    tile_coords={"mi": mi + 1},
                    comment=f"wait A[{mi+1}]",
                )
                overhead_ops.append(wait_a)
                cur_a = next_a

        # -- Global load ops (for next K-tile, async) --
        for name in ["A", "B"]:
            gload_elems = (tile.wg_m if name == "A" else tile.wg_n) * tile.unroll_k // tile.block_size
            gload_vgprs = max(1, (gload_elems * elem + 3) // 4)
            for i in range(0, gload_vgprs, 4):
                cnt = min(4, gload_vgprs - i)
                op = TileOp(
                    kind=OpKind.GLOBAL_LOAD, op_id=_new_op_id(),
                    tile_coords={"chunk": i // 4},
                    iteration=1,  # loading for NEXT K-tile
                    static_offset=i * 4,
                    matrix=name,
                    comment=f"prefetch {name}[{i}:{i+cnt}]",
                )
                global_load_ops.append(op)

        # -- LDS write ops (write loaded data to LDS) --
        for name in ["A", "B"]:
            gload_elems = (tile.wg_m if name == "A" else tile.wg_n) * tile.unroll_k // tile.block_size
            gload_vgprs = max(1, (gload_elems * elem + 3) // 4)
            for i in range(0, gload_vgprs, 4):
                cnt = min(4, gload_vgprs - i)
                op = TileOp(
                    kind=OpKind.LDS_WRITE, op_id=_new_op_id(),
                    tile_coords={"chunk": i // 4},
                    static_offset=i * 4,
                    matrix=name,
                    comment=f"LDS write {name}[{i}:{i+cnt}]",
                )
                lds_write_ops.append(op)

        return TilePlan(
            mfma_ops=mfma_ops,
            lds_read_ops=lds_read_ops,
            global_load_ops=global_load_ops,
            lds_write_ops=lds_write_ops,
            overhead_ops=overhead_ops,
            values=values,
            tile=tile,
            problem=problem,
        )

    def schedule(self, rules: Optional[SchedulingRules] = None) -> Schedule:
        """Schedule this plan into MFMA slots + loop structure.

        Places global loads and LDS writes between MFMAs (forward),
        and wait ops (backward) to maximize latency hiding.
        """
        if rules is None:
            rules = SchedulingRules()

        placer = SlotPlacer(self.mfma_ops, rules)

        # Separate LDS reads into preamble vs prefetch groups
        preamble_reads = [op for op in self.lds_read_ops
                          if "preamble" in op.comment]
        prefetch_reads = [op for op in self.lds_read_ops
                          if "preamble" not in op.comment]

        # Group prefetch reads by target mi
        mi_groups: Dict[int, List[TileOp]] = {}
        for op in prefetch_reads:
            mi = op.tile_coords.get("mi", 0)
            mi_groups.setdefault(mi, []).append(op)

        # Place prefetch A reads forward, just before their mi-group's MFMAs
        mr = self.tile.mfma_m_repeat
        nr = self.tile.mfma_n_repeat
        ki_count = self.tile.k_iterations
        mfmas_per_group = nr * ki_count

        for mi in sorted(mi_groups.keys()):
            # Place reads starting at the slot just before this mi's MFMAs
            group_start = max(0, (mi - 1) * mfmas_per_group)
            placer.place_forward(mi_groups[mi], start_slot=group_start)

        # Global loads and LDS writes are handled by the K-loop phase
        # (phase_scheduled_k_loop), not by the scheduled compute section.
        # This avoids double-emission and keeps the conditional prefetch
        # logic in one place.

        return Schedule(
            slots=placer.slots,
            prologue_ops=[],  # filled by emit
            epilogue_ops=[],  # filled by emit
            loop_prefix=[],
            loop_suffix=[],
            values=self.values,
        )


# ===================================================================
# Layer 3: Emit assembly from a Schedule
# ===================================================================

def emit_scheduled_kernel(
    schedule: Schedule,
    ctx: AsmContext,
    tile: TileConfig,
    problem: GemmProblem,
    layouts: GemmLayouts,
) -> None:
    """Emit the K-loop body from a Schedule.

    Replaces the old phase_optimized_k_loop + _emit_scheduled_compute.
    Walks the schedule's MFMA slots and emits each op's assembly.
    """
    mfma = tile.mfma
    elem = problem.element_bytes

    # Allocate VGPR pool for operand values
    pool = VGPRPool(base=ctx._next["v"])

    # Allocate registers for all values
    for v in schedule.values:
        if v.scope == "permanent":
            continue  # accumulators already allocated
        pool.alloc(v)

    # Update ctx's next VGPR to account for pool allocations
    ctx._next["v"] = pool._next

    # --- Emit preamble: B reads + A[mi=0] reads ---
    preamble_reads = []
    for slot in schedule.slots:
        for op in slot.side_ops:
            if op.kind == OpKind.LDS_READ and "preamble" in op.comment:
                preamble_reads.append(op)

    # Actually, preamble reads aren't in slots -- they're before the MFMA spine.
    # Let me emit them directly from the plan data.
    # The Schedule structure needs refinement, but for now emit inline.

    ctx.comment(f"Scheduled: {len(schedule.slots)} MFMAs "
                f"({tile.mfma_m_repeat}m x {tile.mfma_n_repeat}n x {tile.k_iterations}k)")

    # Emit preamble B reads + A[0] reads
    b_reg_map: Dict[Tuple[int,int], str] = {}
    a_reg_map: Dict[Tuple[int,int], str] = {}

    nr = tile.mfma_n_repeat
    ki_count = tile.k_iterations
    mr = tile.mfma_m_repeat
    bv = mfma.b_vgprs
    av = mfma.a_vgprs
    acc_per = mfma.acc_vgprs

    def b_off(ni, ki):
        return (ni * mfma.n * tile.unroll_k + ki * mfma.k) * elem

    def a_off(mi, ki):
        return (mi * mfma.m * tile.unroll_k + ki * mfma.k) * elem

    # Allocate B register names (permanent for K-tile)
    for ni in range(nr):
        for ki in range(ki_count):
            name = f"v_b_s{ni}k{ki}"
            if not ctx.has(name):
                ctx.alloc_vgpr_permanent(bv, name)
            b_reg_map[(ni, ki)] = name

    # Allocate A double-buffer register names
    for buf in range(2):
        for ki in range(ki_count):
            name = f"v_a_b{buf}k{ki}"
            if not ctx.has(name):
                ctx.alloc_vgpr_permanent(av, name)
            a_reg_map[(buf, ki)] = name

    # Emit preamble reads
    for ki in range(ki_count):
        for ni in range(nr):
            ctx.ds_read(ctx.vreg(b_reg_map[(ni, ki)], 0, bv),
                        ctx.vreg("v_lds_rd_b"),
                        offset=b_off(ni, ki), width=bv,
                        comment=f"pre B n{ni}k{ki}")

    cur_a = 0
    for ki in range(ki_count):
        ctx.ds_read(ctx.vreg(a_reg_map[(cur_a, ki)], 0, av),
                    ctx.vreg("v_lds_rd_a"),
                    offset=a_off(0, ki), width=av,
                    comment=f"pre A m0k{ki} b{cur_a}")

    ctx.s_waitcnt("lgkmcnt(0)", comment="wait preamble")
    ctx.raw("")

    # --- Emit per-mi groups with interleaved side ops ---
    mfma_idx = 0
    for mi in range(mr):
        # Prefetch A for next mi
        has_prefetch = mi < mr - 1
        if has_prefetch:
            next_a = 1 - cur_a
            for ki in range(ki_count):
                ctx.ds_read(ctx.vreg(a_reg_map[(next_a, ki)], 0, av),
                            ctx.vreg("v_lds_rd_a"),
                            offset=a_off(mi + 1, ki), width=av,
                            comment=f"LR A m{mi+1}k{ki} b{next_a}")

        # Emit MFMAs for this mi-group, with interleaved side ops from slots
        for ki in range(ki_count):
            for ni in range(nr):
                slot = schedule.slots[mfma_idx]

                # Emit side ops placed BEFORE this MFMA
                for side_op in slot.side_ops:
                    _emit_side_op(side_op, ctx, tile, problem)

                # Emit MFMA
                acc_off = (mi * nr + ni) * acc_per
                ctx.inst(
                    f"v_mfma_f32_{mfma.m}x{mfma.n}x{mfma.k}_f16",
                    ctx.areg("acc_C", acc_off, acc_per),
                    ctx.vreg(a_reg_map[(cur_a, ki)], 0, av),
                    ctx.vreg(b_reg_map[(ni, ki)], 0, bv),
                    ctx.areg("acc_C", acc_off, acc_per),
                    comment=f"MFMA m{mi}_n{ni}_k{ki}")

                mfma_idx += 1

        # Wait for A prefetch
        if has_prefetch:
            ctx.s_waitcnt("lgkmcnt(0)",
                          comment=f"wait A[{mi+1}] ({nr*ki_count} MFMAs hid)")
            cur_a = next_a

        ctx.raw("")


def _emit_side_op(op: TileOp, ctx: AsmContext,
                  tile: TileConfig, problem: GemmProblem) -> None:
    """Emit assembly for a single side op placed between MFMAs."""
    if op.kind == OpKind.GLOBAL_LOAD:
        # Emit global_load_dwordx4
        name = op.matrix.lower()
        addr = ctx.vreg(f"v_addr_{name}", 0, 2)
        gload_name = f"v_gload_{name}"
        load = ctx.get(gload_name)
        i = op.tile_coords.get("chunk", 0) * 4
        cnt = min(4, load.count - i)
        width = {4: "dwordx4", 2: "dwordx2", 1: "dword"}[cnt]
        dst = ctx.vreg(gload_name, i, cnt)
        off = f"off offset:{i * 4}" if i > 0 else "off"
        ctx.inst(f"global_load_{width}", dst, addr, off,
                 comment=op.comment)
    elif op.kind == OpKind.LDS_WRITE:
        name = op.matrix.lower()
        gload_name = f"v_gload_{name}"
        addr_reg = ctx.vreg(f"v_lds_wr_{name}")
        load = ctx.get(gload_name)
        i = op.tile_coords.get("chunk", 0) * 4
        cnt = min(4, load.count - i)
        src = ctx.vreg(gload_name, i, cnt)
        ctx.ds_write(addr_reg, src, offset=i * 4, width=cnt,
                     comment=op.comment)
    elif op.kind == OpKind.WAIT_VMEM:
        ctx.s_waitcnt("vmcnt(0)", comment=op.comment)
    elif op.kind == OpKind.WAIT_LDS:
        ctx.s_waitcnt("lgkmcnt(0)", comment=op.comment)
    elif op.kind == OpKind.BARRIER:
        ctx.s_barrier(comment=op.comment)
    elif op.kind == OpKind.PTR_ADVANCE:
        k_stride = tile.unroll_k * problem.element_bytes
        for addr in ["v_addr_a", "v_addr_b"]:
            ctx.inst("v_add_co_u32", ctx.vreg(addr, 0, 1), "vcc",
                     str(k_stride), ctx.vreg(addr, 0, 1),
                     comment=f"{addr} += {k_stride}")
            ctx.inst("v_addc_co_u32", ctx.vreg(addr, 1, 1), "vcc",
                     ctx.vreg(addr, 1, 1), "0", "vcc", comment="carry")
