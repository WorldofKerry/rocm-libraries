"""Scheduling IR: TileOp, Value, Schedule, VGPRPool, SchedulingRules.

Layer 2 of the three-layer architecture (see DESIGN.md).
Separates WHAT to compute (TileOps with Value edges) from
WHEN to execute (slot-based placement) and HOW to emit (emitters).
"""
from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable

__all__ = [
    "OpKind",
    "Value",
    "TileOp",
    "Slot",
    "Schedule",
    "SchedulingRules",
    "VGPRPool",
    "SlotPlacer",
]

# ---------------------------------------------------------------------------
# Module-level unique op-id counter
# ---------------------------------------------------------------------------
_next_op_id = 0


def _new_op_id() -> int:
    global _next_op_id
    _next_op_id += 1
    return _next_op_id


# ---------------------------------------------------------------------------
# OpKind
# ---------------------------------------------------------------------------
class OpKind(Enum):
    MFMA = auto()
    LDS_READ = auto()
    LDS_WRITE = auto()
    GLOBAL_LOAD = auto()
    DIRECT_TO_LDS = auto()
    BARRIER = auto()
    WAIT_VMEM = auto()
    WAIT_LDS = auto()
    PTR_ADVANCE = auto()
    LDS_BUFFER_SWAP = auto()
    LOOP_CONTROL = auto()
    CUSTOM = auto()


# ---------------------------------------------------------------------------
# Value -- logical register group flowing between ops
# ---------------------------------------------------------------------------
@dataclass
class Value:
    name: str  # e.g. "a_tile_mi2_ki0"
    reg_class: str  # "vgpr", "sgpr", "acc"
    count: int  # number of registers
    scope: str  # "permanent", "partition", "k_tile", "prefetch"
    partition_id: int = 0  # which partition owns this value
    physical_reg: Optional[int] = None  # assigned by allocator


# ---------------------------------------------------------------------------
# TileOp -- one operation in the GEMM dataflow
# ---------------------------------------------------------------------------
@dataclass
class TileOp:
    kind: OpKind
    op_id: int
    tile_coords: Dict[str, int]  # e.g. {"mi": 2, "ni": 3, "ki": 0}
    iteration: int = 0  # pipeline stage (0=current, 1=next, 2=two-ahead)
    partition_id: int = 0
    inputs: List[Value] = field(default_factory=list)
    outputs: List[Value] = field(default_factory=list)
    static_offset: int = 0  # LDS byte offset for reads/writes
    matrix: str = ""  # "A" or "B" for load/read ops
    comment: str = ""
    payload: Optional[dict] = None  # custom data


# ---------------------------------------------------------------------------
# Slot -- one MFMA interval (16 cycles)
# ---------------------------------------------------------------------------
@dataclass
class Slot:
    index: int
    mfma: TileOp  # the MFMA defining this slot
    side_ops: List[TileOp] = field(default_factory=list)
    capacity: int = 2  # max side ops


# ---------------------------------------------------------------------------
# Schedule -- complete scheduled program
# ---------------------------------------------------------------------------
@dataclass
class Schedule:
    slots: List[Slot]
    prologue_ops: List[TileOp]  # pre-loop setup ops
    epilogue_ops: List[TileOp]  # post-loop ops
    loop_prefix: List[TileOp]  # ops before compute in K-loop
    loop_suffix: List[TileOp]  # ops after compute in K-loop
    values: List[Value]


# ---------------------------------------------------------------------------
# SchedulingRules -- pluggable constraints
# ---------------------------------------------------------------------------
@dataclass
class SchedulingRules:
    max_ds_read_per_interval: int = 1
    min_gap_lr_to_wait: int = 4  # MFMA intervals between ds_read and waitcnt
    spread_global_loads: bool = True
    no_m0_with_buffer_load: bool = True
    slots_per_interval: int = 2


# ---------------------------------------------------------------------------
# VGPRPool -- partition-scoped VGPR allocator with free-list reuse
# ---------------------------------------------------------------------------
class VGPRPool:
    """Partition-scoped VGPR allocator with free-list reuse."""

    def __init__(self, base: int = 0):
        """*base* is the starting VGPR index (after permanent allocs)."""
        self._base = base
        self._next = base  # bump pointer
        self._peak = base
        self._live: Dict[str, Tuple[int, int]] = {}  # name -> (start, count)
        self._free: List[Tuple[int, int]] = []  # sorted by start
        self._partition_map: Dict[str, int] = {}  # name -> partition_id

    def alloc(self, value: Value, alignment: int = 1) -> int:
        """Allocate physical regs for *value*, return start index.

        Auto-aligns: 4+ regs to 4-aligned, 2 regs to even.
        """
        count = value.count
        if count >= 4:
            alignment = max(alignment, 4)
        elif count >= 2:
            alignment = max(alignment, 2)

        # Try to reuse a free range
        start = self._try_free_list(count, alignment)
        if start is None:
            # Bump-allocate from the end
            start = self._align_up(self._next, alignment)
            self._next = start + count
            if self._next > self._peak:
                self._peak = self._next

        self._live[value.name] = (start, count)
        self._partition_map[value.name] = value.partition_id
        value.physical_reg = start
        return start

    def free_value(self, value: Value) -> None:
        """Return *value*'s regs to the free list."""
        entry = self._live.pop(value.name, None)
        if entry is None:
            return
        self._partition_map.pop(value.name, None)
        self._add_free(entry[0], entry[1])

    def free_partition(self, partition_id: int) -> None:
        """Free all live values scoped to the given *partition_id*."""
        names = [n for n, pid in self._partition_map.items() if pid == partition_id]
        for name in names:
            entry = self._live.pop(name, None)
            if entry is not None:
                self._add_free(entry[0], entry[1])
            self._partition_map.pop(name, None)

    @property
    def peak(self) -> int:
        """High-water mark (total VGPRs ever needed)."""
        return self._peak - self._base

    @property
    def current(self) -> int:
        """Currently live VGPR count."""
        return sum(c for _, c in self._live.values())

    # -- internals ----------------------------------------------------------

    def _try_free_list(self, count: int, alignment: int) -> Optional[int]:
        """Find a free range that fits *count* regs with *alignment*."""
        for i, (fstart, fcount) in enumerate(self._free):
            aligned = self._align_up(fstart, alignment)
            waste_before = aligned - fstart
            if fcount - waste_before >= count:
                self._free.pop(i)
                if waste_before > 0:
                    self._add_free(fstart, waste_before)
                leftover = fcount - waste_before - count
                if leftover > 0:
                    self._add_free(aligned + count, leftover)
                return aligned
        return None

    def _add_free(self, start: int, count: int) -> None:
        """Insert a range into the free list, merging adjacent ranges."""
        self._free.append((start, count))
        self._free.sort(key=lambda r: r[0])
        merged: List[Tuple[int, int]] = []
        for s, c in self._free:
            if merged and merged[-1][0] + merged[-1][1] == s:
                merged[-1] = (merged[-1][0], merged[-1][1] + c)
            else:
                merged.append((s, c))
        self._free = merged

    @staticmethod
    def _align_up(v: int, alignment: int) -> int:
        return (v + alignment - 1) // alignment * alignment


# ---------------------------------------------------------------------------
# SlotPlacer -- scheduler that places non-MFMA ops between MFMAs
# ---------------------------------------------------------------------------
class SlotPlacer:
    """Place non-MFMA TileOps into MFMA interval slots."""

    def __init__(self, mfma_ops: List[TileOp], rules: SchedulingRules):
        """Create slots from MFMA ops."""
        self.rules = rules
        self.slots = [Slot(index=i, mfma=op) for i, op in enumerate(mfma_ops)]

    def place_forward(self, ops: List[TileOp], start_slot: int = 0) -> None:
        """Place ops forward from *start_slot* (for GR paths -- start early)."""
        slot_idx = start_slot
        for op in ops:
            while slot_idx < len(self.slots):
                if self._can_place(self.slots[slot_idx], op):
                    self.place(slot_idx, op)
                    break
                slot_idx += 1

    def place_backward(self, ops: List[TileOp], end_slot: Optional[int] = None) -> None:
        """Place ops backward from *end_slot* (for wait paths -- delay waits)."""
        if end_slot is None:
            end_slot = len(self.slots) - 1
        slot_idx = end_slot
        for op in reversed(ops):
            while slot_idx >= 0:
                if self._can_place(self.slots[slot_idx], op):
                    self.place(slot_idx, op)
                    break
                slot_idx -= 1

    def _can_place(self, slot: Slot, op: TileOp) -> bool:
        """Check if *op* can be placed in *slot* respecting rules."""
        if len(slot.side_ops) >= slot.capacity:
            return False
        if op.kind == OpKind.LDS_READ:
            ds_reads = sum(1 for o in slot.side_ops if o.kind == OpKind.LDS_READ)
            if ds_reads >= self.rules.max_ds_read_per_interval:
                return False
        return True

    def place(self, slot_idx: int, op: TileOp) -> None:
        """Place an op in a specific slot."""
        self.slots[slot_idx].side_ops.append(op)
