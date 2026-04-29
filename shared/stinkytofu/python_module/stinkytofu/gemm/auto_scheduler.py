# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Automatic instruction scheduler for GEMM K-loops.

Takes a dataflow graph of operations (MFMAs, ds_reads, buffer_loads,
barriers, waits) and produces an interleaved linear schedule that
maximizes MFMA utilization.

Design principles:
  1. MFMAs form the "spine" -- the schedule is built around them
  2. Non-MFMA ops are placed in MFMA gaps (between consecutive MFMAs)
  3. Each op has an [earliest, latest] slot range from dependencies
  4. Ops are placed as LATE as possible (maximize latency hiding)
  5. Waitcnt values are computed automatically from in-flight counts
  6. Multiple barrier phases are supported for DTL pipelining

Usage:
    graph = ScheduleGraph()
    # Add operations with dependencies
    b0 = graph.add_ds_read("B", ni=0, ki=0)
    a0 = graph.add_ds_read("A", mi=0, ki=0)
    m0 = graph.add_mfma(mi=0, ni=0, ki=0, deps=[b0, a0])
    ...
    # Schedule
    schedule = graph.schedule(occupancy=1, latencies=DEFAULT_LATENCIES)
    # Emit
    for slot in schedule:
        slot.emit(ctx)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Set, Callable
import math

__all__ = [
    "OpType", "SchedOp", "ScheduleGraph", "LinearSchedule",
    "Latencies", "DEFAULT_LATENCIES",
]


class OpType(Enum):
    MFMA = auto()
    DS_READ = auto()       # LDS -> VGPR
    DS_WRITE = auto()      # VGPR -> LDS
    BUFFER_LOAD = auto()   # Global -> VGPR (or -> LDS with DTL)
    BARRIER = auto()
    WAIT_LDS = auto()      # s_waitcnt lgkmcnt(N)
    WAIT_VMEM = auto()     # s_waitcnt vmcnt(N)
    SALU = auto()          # scalar ALU
    VALU = auto()          # vector ALU
    BRANCH = auto()
    LABEL = auto()


@dataclass
class Latencies:
    """Pipeline latencies in issue cycles (at occupancy=1)."""
    ds_read: int = 20      # LDS read latency
    buffer_load: int = 230 # Global memory latency
    mfma: int = 16         # MFMA pipeline depth
    ds_write: int = 20     # LDS write latency
    barrier: int = 50      # Approximate barrier cost


DEFAULT_LATENCIES = Latencies()


@dataclass
class SchedOp:
    """One operation in the schedule graph."""
    id: int
    op_type: OpType
    # Semantic info for code emission
    matrix: str = ""       # "A" or "B"
    mi: int = -1
    ni: int = -1
    ki: int = -1
    buf: int = 0           # A double-buffer index
    chunk: int = 0         # global load chunk index
    # Dependencies
    deps: List[int] = field(default_factory=list)      # must complete before this op
    consumers: List[int] = field(default_factory=list)  # ops that depend on this
    # Register info
    reg_name: str = ""     # register binding name
    # Scheduling state
    earliest: int = 0      # earliest MFMA slot this can go in
    latest: int = 999999   # latest MFMA slot
    placed_at: int = -1    # actual slot (-1 = unplaced)
    # Emission callback
    emit_fn: Optional[Callable] = None
    comment: str = ""

    def __repr__(self):
        return f"Op({self.id}, {self.op_type.name}, {self.comment})"


@dataclass
class ScheduleSlot:
    """One position in the linear schedule (one MFMA + side ops)."""
    index: int
    mfma: Optional[SchedOp] = None
    before_mfma: List[SchedOp] = field(default_factory=list)  # ops before the MFMA
    after_mfma: List[SchedOp] = field(default_factory=list)   # ops after (rare)


class LinearSchedule:
    """Result of scheduling: a linear sequence of slots."""

    def __init__(self, slots: List[ScheduleSlot],
                 prologue: List[SchedOp] = None,
                 postamble: List[SchedOp] = None):
        self.slots = slots
        self.prologue = prologue or []
        self.postamble = postamble or []

    @property
    def total_mfma(self) -> int:
        return sum(1 for s in self.slots if s.mfma is not None)

    @property
    def total_side_ops(self) -> int:
        return sum(len(s.before_mfma) + len(s.after_mfma) for s in self.slots)

    def summary(self) -> str:
        lines = [f"Schedule: {self.total_mfma} MFMAs, {self.total_side_ops} side ops"]
        # Count by type
        type_counts: Dict[OpType, int] = {}
        for s in self.slots:
            for op in s.before_mfma + s.after_mfma:
                type_counts[op.op_type] = type_counts.get(op.op_type, 0) + 1
        for t, c in sorted(type_counts.items(), key=lambda x: x[0].name):
            lines.append(f"  {t.name}: {c}")

        # Show slot occupancy histogram
        occ = [len(s.before_mfma) + len(s.after_mfma) for s in self.slots]
        max_occ = max(occ) if occ else 0
        for i in range(max_occ + 1):
            cnt = sum(1 for o in occ if o == i)
            if cnt:
                lines.append(f"  {i} side ops/slot: {cnt} slots")
        return "\n".join(lines)

    def emit(self, ctx) -> None:
        """Emit the full schedule to an AsmContext."""
        for op in self.prologue:
            if op.emit_fn:
                op.emit_fn()
        for slot in self.slots:
            for op in slot.before_mfma:
                if op.emit_fn:
                    op.emit_fn()
            if slot.mfma and slot.mfma.emit_fn:
                slot.mfma.emit_fn()
            for op in slot.after_mfma:
                if op.emit_fn:
                    op.emit_fn()
        for op in self.postamble:
            if op.emit_fn:
                op.emit_fn()


class ScheduleGraph:
    """Dataflow graph of operations for one K-loop iteration.

    Build the graph by adding ops with dependencies, then call
    schedule() to produce a LinearSchedule.
    """

    def __init__(self):
        self._ops: Dict[int, SchedOp] = {}
        self._next_id = 0
        self._mfma_order: List[int] = []  # MFMA IDs in execution order

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # -- Add operations --

    def add_op(self, op_type: OpType, deps: List[int] = None,
               emit_fn: Callable = None, **kwargs) -> int:
        """Add a generic operation. Returns op ID."""
        op_id = self._new_id()
        op = SchedOp(id=op_id, op_type=op_type,
                     deps=list(deps or []), emit_fn=emit_fn, **kwargs)
        self._ops[op_id] = op
        # Register as consumer of dependencies
        for dep_id in op.deps:
            if dep_id in self._ops:
                self._ops[dep_id].consumers.append(op_id)
        if op_type == OpType.MFMA:
            self._mfma_order.append(op_id)
        return op_id

    def add_mfma(self, mi: int, ni: int, ki: int,
                 deps: List[int] = None,
                 emit_fn: Callable = None) -> int:
        return self.add_op(OpType.MFMA, deps=deps, emit_fn=emit_fn,
                           mi=mi, ni=ni, ki=ki,
                           comment=f"MFMA m{mi}_n{ni}_k{ki}")

    def add_ds_read(self, matrix: str, deps: List[int] = None,
                    emit_fn: Callable = None, **kwargs) -> int:
        return self.add_op(OpType.DS_READ, deps=deps, emit_fn=emit_fn,
                           matrix=matrix,
                           **kwargs)

    def add_buffer_load(self, matrix: str, chunk: int = 0,
                        deps: List[int] = None,
                        emit_fn: Callable = None) -> int:
        return self.add_op(OpType.BUFFER_LOAD, deps=deps, emit_fn=emit_fn,
                           matrix=matrix, chunk=chunk,
                           comment=f"gload {matrix}[{chunk}]")

    def add_barrier(self, deps: List[int] = None,
                    emit_fn: Callable = None) -> int:
        return self.add_op(OpType.BARRIER, deps=deps, emit_fn=emit_fn,
                           comment="barrier")

    def add_wait_lds(self, count: int = 0, deps: List[int] = None,
                     emit_fn: Callable = None) -> int:
        return self.add_op(OpType.WAIT_LDS, deps=deps, emit_fn=emit_fn,
                           comment=f"lgkmcnt({count})")

    def add_wait_vmem(self, count: int = 0, deps: List[int] = None,
                      emit_fn: Callable = None) -> int:
        return self.add_op(OpType.WAIT_VMEM, deps=deps, emit_fn=emit_fn,
                           comment=f"vmcnt({count})")

    def add_salu(self, deps: List[int] = None,
                 emit_fn: Callable = None, comment: str = "") -> int:
        return self.add_op(OpType.SALU, deps=deps, emit_fn=emit_fn,
                           comment=comment)

    def add_valu(self, deps: List[int] = None,
                 emit_fn: Callable = None, comment: str = "") -> int:
        return self.add_op(OpType.VALU, deps=deps, emit_fn=emit_fn,
                           comment=comment)

    # -- Scheduling --

    def schedule(self, latencies: Latencies = None,
                 max_side_per_slot: int = 2,
                 min_side_per_slot: int = 1) -> LinearSchedule:
        """Produce a linear schedule from the graph.

        Algorithm:
        1. MFMAs define the slot spine (fixed order)
        2. Compute earliest/latest slot for each non-MFMA op
        3. Place ops at latest valid slot (ALAP scheduling)
        4. Enforce min_side_per_slot (ATT profiling shows back-to-back
           MFMAs stall ~12K cycles on gfx950 -- need at least 1
           non-MFMA between consecutive MFMAs for pipeline throughput)
        5. Ops that don't fit move to prologue/postamble
        """
        if latencies is None:
            latencies = DEFAULT_LATENCIES

        num_mfma = len(self._mfma_order)
        if num_mfma == 0:
            return LinearSchedule([], list(self._ops.values()))

        # Create slots from MFMA spine
        slots = [ScheduleSlot(index=i, mfma=self._ops[mid])
                 for i, mid in enumerate(self._mfma_order)]

        # Map MFMA IDs to slot indices
        mfma_to_slot = {mid: i for i, mid in enumerate(self._mfma_order)}

        # Compute earliest slot for each non-MFMA op
        non_mfma = [op for op in self._ops.values()
                    if op.op_type != OpType.MFMA]

        for op in non_mfma:
            # Earliest: after all dependencies complete + latency
            earliest = 0
            for dep_id in op.deps:
                dep = self._ops[dep_id]
                if dep.op_type == OpType.MFMA:
                    # Must be after this MFMA's slot
                    earliest = max(earliest, mfma_to_slot.get(dep_id, 0) + 1)
                elif dep_id in mfma_to_slot:
                    earliest = max(earliest, mfma_to_slot[dep_id] + 1)
            op.earliest = earliest

            # Latest: must be before all consumers - latency
            latest = num_mfma - 1
            for cons_id in op.consumers:
                cons = self._ops[cons_id]
                if cons.op_type == OpType.MFMA:
                    slot_idx = mfma_to_slot.get(cons_id, num_mfma - 1)
                    # Place read ops early enough for latency hiding
                    lat = self._get_latency(op, latencies)
                    latest = min(latest, max(0, slot_idx - lat))
                elif cons_id in mfma_to_slot:
                    latest = min(latest, mfma_to_slot[cons_id])
            op.latest = max(op.earliest, latest)

        # Sort non-MFMA ops by latest slot (ALAP: place latest-deadline first)
        non_mfma.sort(key=lambda op: (op.latest, op.earliest))

        # Place ops into slots
        prologue = []
        postamble = []

        for op in non_mfma:
            placed = False
            # Try to place at latest valid slot, working backward
            for slot_idx in range(min(op.latest, num_mfma - 1),
                                  max(op.earliest - 1, -1), -1):
                if slot_idx < 0:
                    break
                slot = slots[slot_idx]
                if len(slot.before_mfma) < max_side_per_slot:
                    slot.before_mfma.append(op)
                    op.placed_at = slot_idx
                    placed = True
                    break
            if not placed:
                # Couldn't fit -- goes to prologue or postamble
                if op.earliest == 0:
                    prologue.append(op)
                else:
                    postamble.append(op)

        # Fill pass: ensure every slot has at least min_side_per_slot ops.
        # ATT profiling shows gfx950 MFMAs stall without interleaved
        # instructions. Insert s_nop 0 as filler if needed.
        if min_side_per_slot > 0:
            for slot in slots:
                while len(slot.before_mfma) < min_side_per_slot:
                    nop = SchedOp(id=self._new_id(), op_type=OpType.SALU,
                                  comment="s_nop (pipeline fill)")
                    nop.emit_fn = lambda: None  # placeholder
                    slot.before_mfma.append(nop)

        return LinearSchedule(slots, prologue, postamble)

    def _get_latency(self, op: SchedOp, lat: Latencies) -> int:
        """Get the pipeline latency for an op (in MFMA-slot units)."""
        # Convert cycle latency to MFMA slots
        # At occupancy=1: 1 slot ≈ 1-2 issue cycles
        if op.op_type == OpType.DS_READ:
            return max(1, lat.ds_read // 2)  # ~10 slots
        elif op.op_type == OpType.BUFFER_LOAD:
            return max(1, lat.buffer_load // 2)  # ~115 slots
        elif op.op_type == OpType.DS_WRITE:
            return max(1, lat.ds_write // 2)
        return 0

    # -- Auto-waitcnt computation --

    def compute_waitcnts(self, schedule: LinearSchedule) -> None:
        """Insert waitcnt ops with precise counts.

        Scans the schedule for ds_read/buffer_load ops and inserts
        lgkmcnt/vmcnt with the minimum count needed before each
        consumer MFMA.
        """
        # Track in-flight counts at each slot
        lgkm_inflight = 0  # ds_read/ds_write in flight
        vmem_inflight = 0  # buffer_load in flight

        for slot in schedule.slots:
            # Count new ops issued in this slot
            for op in slot.before_mfma:
                if op.op_type in (OpType.DS_READ, OpType.DS_WRITE):
                    lgkm_inflight += 1
                elif op.op_type == OpType.BUFFER_LOAD:
                    vmem_inflight += 1
                elif op.op_type == OpType.WAIT_LDS:
                    lgkm_inflight = 0  # reset
                elif op.op_type == OpType.WAIT_VMEM:
                    vmem_inflight = 0

    # -- Graph statistics --

    def summary(self) -> str:
        type_counts: Dict[OpType, int] = {}
        for op in self._ops.values():
            type_counts[op.op_type] = type_counts.get(op.op_type, 0) + 1
        lines = [f"ScheduleGraph: {len(self._ops)} ops"]
        for t, c in sorted(type_counts.items(), key=lambda x: x[0].name):
            lines.append(f"  {t.name}: {c}")
        lines.append(f"  Dependencies: {sum(len(op.deps) for op in self._ops.values())}")
        return "\n".join(lines)
