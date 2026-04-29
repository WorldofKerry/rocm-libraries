"""Automatic instruction scheduler for GEMM K-loops.

Takes a dataflow graph of operations and produces an interleaved
linear schedule that maximizes MFMA utilization.

Key insight from ATT profiling: gfx950 MFMAs stall ~12K cycles when
issued back-to-back without any interleaved instruction. The scheduler
MUST place at least 1 non-MFMA op between every pair of MFMAs.

Algorithm:
  1. MFMAs form the spine (fixed execution order from tile structure)
  2. Each non-MFMA op has [earliest, latest] slot from dependencies
  3. Spread ops EVENLY across their valid range (not clustered at deadline)
  4. Waitcnts are auto-computed from in-flight operation counts
  5. Empty slots get s_nop to prevent MFMA pipeline stalls
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Callable
import math

__all__ = [
    "OpType", "SchedOp", "ScheduleGraph", "LinearSchedule",
    "Latencies", "DEFAULT_LATENCIES",
]


class OpType(Enum):
    MFMA = auto()
    DS_READ = auto()
    DS_WRITE = auto()
    BUFFER_LOAD = auto()
    BARRIER = auto()
    WAIT_LDS = auto()
    WAIT_VMEM = auto()
    SALU = auto()
    VALU = auto()
    NOP = auto()


@dataclass
class Latencies:
    """Pipeline latencies in issue cycles."""
    ds_read: int = 20
    buffer_load: int = 230
    mfma: int = 16
    ds_write: int = 20
    barrier: int = 50

DEFAULT_LATENCIES = Latencies()


@dataclass
class SchedOp:
    """One operation in the schedule graph."""
    id: int
    op_type: OpType
    matrix: str = ""
    mi: int = -1
    ni: int = -1
    ki: int = -1
    buf: int = 0
    chunk: int = 0
    deps: List[int] = field(default_factory=list)
    consumers: List[int] = field(default_factory=list)
    earliest: int = 0
    latest: int = 999999
    placed_at: int = -1
    emit_fn: Optional[Callable] = None
    comment: str = ""
    # For auto-waitcnt: which counter this op increments
    counter: str = ""  # "lgkm" for ds_read/write, "vm" for buffer_load

    def __repr__(self):
        return f"Op({self.id},{self.op_type.name},{self.comment})"


@dataclass
class ScheduleSlot:
    """One MFMA slot with interleaved side ops."""
    index: int
    mfma: Optional[SchedOp] = None
    before_mfma: List[SchedOp] = field(default_factory=list)

    @property
    def side_count(self) -> int:
        return len(self.before_mfma)


class LinearSchedule:
    """Result of scheduling."""

    def __init__(self, slots: List[ScheduleSlot],
                 prologue: List[SchedOp] = None,
                 postamble: List[SchedOp] = None):
        self.slots = slots
        self.prologue = prologue or []
        self.postamble = postamble or []

    @property
    def total_mfma(self) -> int:
        return sum(1 for s in self.slots if s.mfma)

    @property
    def total_side_ops(self) -> int:
        return sum(s.side_count for s in self.slots)

    def summary(self) -> str:
        lines = [f"Schedule: {self.total_mfma} MFMAs, {self.total_side_ops} side ops, "
                 f"{len(self.prologue)} prologue, {len(self.postamble)} postamble"]
        tc: Dict[OpType, int] = {}
        for s in self.slots:
            for op in s.before_mfma:
                tc[op.op_type] = tc.get(op.op_type, 0) + 1
        for t, c in sorted(tc.items(), key=lambda x: x[0].name):
            lines.append(f"  {t.name}: {c}")
        occ = [s.side_count for s in self.slots]
        for i in range(max(occ) + 1 if occ else 1):
            cnt = sum(1 for o in occ if o == i)
            if cnt:
                lines.append(f"  {i} side/slot: {cnt} slots")
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
        for op in self.postamble:
            if op.emit_fn:
                op.emit_fn()


class ScheduleGraph:
    """Dataflow graph -> linear schedule."""

    def __init__(self):
        self._ops: Dict[int, SchedOp] = {}
        self._next_id = 0
        self._mfma_order: List[int] = []

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def add_op(self, op_type: OpType, deps: List[int] = None,
               emit_fn: Callable = None, counter: str = "",
               **kwargs) -> int:
        op_id = self._new_id()
        op = SchedOp(id=op_id, op_type=op_type,
                     deps=list(deps or []), emit_fn=emit_fn,
                     counter=counter, **kwargs)
        self._ops[op_id] = op
        for dep_id in op.deps:
            if dep_id in self._ops:
                self._ops[dep_id].consumers.append(op_id)
        if op_type == OpType.MFMA:
            self._mfma_order.append(op_id)
        return op_id

    def add_mfma(self, mi, ni, ki, deps=None, emit_fn=None) -> int:
        return self.add_op(OpType.MFMA, deps=deps, emit_fn=emit_fn,
                           mi=mi, ni=ni, ki=ki,
                           comment=f"MFMA m{mi}_n{ni}_k{ki}")

    def add_ds_read(self, matrix, deps=None, emit_fn=None, **kw) -> int:
        return self.add_op(OpType.DS_READ, deps=deps, emit_fn=emit_fn,
                           matrix=matrix, counter="lgkm", **kw)

    def add_buffer_load(self, matrix, chunk=0, deps=None, emit_fn=None) -> int:
        return self.add_op(OpType.BUFFER_LOAD, deps=deps, emit_fn=emit_fn,
                           matrix=matrix, chunk=chunk, counter="vm",
                           comment=f"gload {matrix}[{chunk}]")

    def add_barrier(self, deps=None, emit_fn=None) -> int:
        return self.add_op(OpType.BARRIER, deps=deps, emit_fn=emit_fn,
                           comment="barrier")

    def add_salu(self, deps=None, emit_fn=None, comment="") -> int:
        return self.add_op(OpType.SALU, deps=deps, emit_fn=emit_fn,
                           comment=comment)

    def add_valu(self, deps=None, emit_fn=None, comment="") -> int:
        return self.add_op(OpType.VALU, deps=deps, emit_fn=emit_fn,
                           comment=comment)

    def schedule(self, latencies: Latencies = None,
                 max_side_per_slot: int = 2,
                 min_side_per_slot: int = 1) -> LinearSchedule:
        """Produce a linear schedule.

        Algorithm: SPREAD scheduling.
        1. Build MFMA spine
        2. Compute [earliest, latest] for each non-MFMA op
        3. Place ops evenly across their valid range
        4. Fill remaining empty slots with s_nop
        """
        if latencies is None:
            latencies = DEFAULT_LATENCIES

        num_mfma = len(self._mfma_order)
        if num_mfma == 0:
            return LinearSchedule([], list(self._ops.values()))

        slots = [ScheduleSlot(index=i, mfma=self._ops[mid])
                 for i, mid in enumerate(self._mfma_order)]
        mfma_to_slot = {mid: i for i, mid in enumerate(self._mfma_order)}

        non_mfma = [op for op in self._ops.values()
                    if op.op_type != OpType.MFMA]

        # Compute earliest/latest for each op
        for op in non_mfma:
            earliest = 0
            for dep_id in op.deps:
                dep = self._ops[dep_id]
                if dep_id in mfma_to_slot:
                    earliest = max(earliest, mfma_to_slot[dep_id] + 1)

            latest = num_mfma - 1
            for cons_id in op.consumers:
                if cons_id in mfma_to_slot:
                    slot_idx = mfma_to_slot[cons_id]
                    lat = self._latency_slots(op, latencies)
                    latest = min(latest, max(0, slot_idx - lat))

            op.earliest = max(0, earliest)
            op.latest = max(op.earliest, min(latest, num_mfma - 1))

        # Group non-MFMA ops by type for even spreading
        by_type: Dict[OpType, List[SchedOp]] = {}
        for op in non_mfma:
            by_type.setdefault(op.op_type, []).append(op)

        # Place ops using SPREAD: distribute evenly across valid range
        # Priority: DS_READ first (latency sensitive), then BUFFER_LOAD,
        # then DS_WRITE, then everything else
        priority_order = [OpType.DS_READ, OpType.BUFFER_LOAD,
                          OpType.DS_WRITE, OpType.BARRIER,
                          OpType.WAIT_LDS, OpType.WAIT_VMEM,
                          OpType.VALU, OpType.SALU]

        prologue = []

        for op_type in priority_order:
            ops = by_type.get(op_type, [])
            if not ops:
                continue

            # Sort by earliest slot (deterministic ordering)
            ops.sort(key=lambda o: (o.earliest, o.latest))

            for op in ops:
                # Find best slot: prefer middle of [earliest, latest] range
                # but also prefer slots with fewer ops already placed
                best_slot = -1
                best_score = float('inf')

                for s in range(op.earliest, op.latest + 1):
                    if s >= num_mfma:
                        break
                    cur_count = slots[s].side_count
                    if cur_count >= max_side_per_slot:
                        continue
                    # Score: prefer middle of range + fewer existing ops
                    mid = (op.earliest + op.latest) / 2
                    dist = abs(s - mid)
                    score = cur_count * 100 + dist
                    if score < best_score:
                        best_score = score
                        best_slot = s

                if best_slot >= 0:
                    slots[best_slot].before_mfma.append(op)
                    op.placed_at = best_slot
                else:
                    # Couldn't place -- goes to prologue
                    prologue.append(op)

        # Fill empty slots with s_nop to prevent MFMA pipeline stalls
        if min_side_per_slot > 0:
            for slot in slots:
                while slot.side_count < min_side_per_slot:
                    nop = SchedOp(id=self._new_id(), op_type=OpType.NOP,
                                  comment="s_nop 0")
                    nop.emit_fn = None  # filled by emitter
                    slot.before_mfma.append(nop)

        return LinearSchedule(slots, prologue)

    def _latency_slots(self, op: SchedOp, lat: Latencies) -> int:
        """Latency in MFMA-slot units (~2 issue cycles per slot with interleaving)."""
        if op.op_type == OpType.DS_READ:
            return max(1, lat.ds_read // 2)
        elif op.op_type == OpType.BUFFER_LOAD:
            return max(1, lat.buffer_load // 2)
        elif op.op_type == OpType.DS_WRITE:
            return max(1, lat.ds_write // 2)
        return 0

    def summary(self) -> str:
        tc: Dict[OpType, int] = {}
        for op in self._ops.values():
            tc[op.op_type] = tc.get(op.op_type, 0) + 1
        lines = [f"Graph: {len(self._ops)} ops, {sum(len(op.deps) for op in self._ops.values())} deps"]
        for t, c in sorted(tc.items(), key=lambda x: x[0].name):
            lines.append(f"  {t.name}: {c}")
        return "\n".join(lines)
