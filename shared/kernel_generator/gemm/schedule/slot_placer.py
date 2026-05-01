"""Slot-based instruction interleaver for GEMM K-loops.

Interleaves non-MFMA instructions between MFMAs using a slot-based
placement engine. Each interval (pair of adjacent MFMAs) has 2 slots.

Two placement modes:
  - Forward: place early (for global loads -- hide latency)
  - Backward: place late (for waits -- delay as long as possible)

Rules are injected as validator/adjuster callbacks on the placer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

__all__ = ["SlotPlacer", "PlacedSchedule", "SchedulingRules"]

SLOTS_PER_INTERVAL = 2


@dataclass
class PlacedOp:
    """One placed operation with its emit function."""
    emit_fn: Callable
    op_type: str = ""     # "mfma", "ds_read", "buffer_load", "wait", "barrier", "salu", "nop"
    module_id: int = -1   # which module this op belongs to
    comment: str = ""
    reads_regs: tuple = ()  # register names this op reads (for lifetime analysis)


@dataclass
class PlacedSchedule:
    """Result of slot placement: MFMAs with interleaved side ops."""
    intervals: List[Tuple[List[PlacedOp], PlacedOp]]  # (side_ops, mfma) per interval
    prologue: List[PlacedOp] = field(default_factory=list)   # ops before first MFMA
    epilogue: List[PlacedOp] = field(default_factory=list)   # ops after last MFMA
    leftovers: List[PlacedOp] = field(default_factory=list)  # couldn't be placed

    def emit_all(self):
        """Call all emit_fns in scheduled order."""
        for op in self.prologue:
            if op.emit_fn:
                op.emit_fn()
        for side_ops, mfma_op in self.intervals:
            for op in side_ops:
                if op.emit_fn:
                    op.emit_fn()
            if mfma_op and mfma_op.emit_fn:
                mfma_op.emit_fn()
        for op in self.epilogue:
            if op.emit_fn:
                op.emit_fn()


@dataclass
class Path:
    """An ordered sequence of ops that must maintain relative order."""
    ops: List[PlacedOp]
    reverse: bool = False  # True = place backward (waits), False = place forward (loads)
    module_id: int = -1


class SlotPlacer:
    """Place side-op paths into MFMA interval slots.

    Each interval between adjacent MFMAs has SLOTS_PER_INTERVAL (2) slots.
    Paths are placed in order, respecting inter-path ordering constraints
    and per-slot validation rules.
    """

    def __init__(self, mfmas: List[PlacedOp],
                 validators: List[Callable] = None,
                 adjusters: List[Callable] = None,
                 on_place: Optional[Callable] = None):
        """
        Args:
            mfmas: MFMA ops in execution order (the spine).
            validators: (placer, slot_idx, op) -> bool. Reject invalid slots.
            adjusters: (placer, limit, op) -> limit. Shift search bounds.
            on_place: (placer, slot_idx, op) -> None. Called after successful placement.
        """
        self.mfmas = mfmas
        self.num_intervals = max(len(mfmas) - 1, 0)
        self.total_slots = self.num_intervals * SLOTS_PER_INTERVAL

        # Per-slot placed ops
        self._slots: List[List[PlacedOp]] = [[] for _ in range(self.total_slots)]
        self._validators = validators or []
        self._adjusters = adjusters or []
        self._on_place = on_place
        self.leftovers: List[PlacedOp] = []

        # Track path ordering: for each module_id, the last placed slot
        self._last_placed: Dict[int, int] = {}

    def _can_place(self, slot_idx: int, op: PlacedOp) -> bool:
        if slot_idx < 0 or slot_idx >= self.total_slots:
            return False
        if len(self._slots[slot_idx]) >= SLOTS_PER_INTERVAL:
            return False
        return all(v(self, slot_idx, op) for v in self._validators)

    def _adjust_limit(self, limit: int, op: PlacedOp) -> int:
        for adj in self._adjusters:
            limit = adj(self, limit, op)
        return limit

    def place_path(self, path: Path) -> None:
        """Place all ops from a path into slots.

        Forward paths scan from low slots to high.
        Backward paths scan from high slots to low.
        Ops within a path maintain their relative order.
        """
        if not path.ops:
            return

        if path.reverse:
            self._place_backward(path)
        else:
            self._place_forward(path)

    def _place_forward(self, path: Path) -> None:
        """Place ops starting from the earliest available slot."""
        cursor = 0
        # Respect ordering: start after the last op from any prior path
        # that this path depends on
        if path.module_id >= 0 and path.module_id in self._last_placed:
            cursor = self._last_placed[path.module_id] + 1

        for op in path.ops:
            limit = self._adjust_limit(cursor, op)
            placed = False
            for s in range(limit, self.total_slots):
                if self._can_place(s, op):
                    self._slots[s].append(op)
                    cursor = s + 1
                    if path.module_id >= 0:
                        self._last_placed[path.module_id] = s
                    if self._on_place:
                        self._on_place(self, s, op)
                    placed = True
                    break
            if not placed:
                self.leftovers.append(op)

    def _place_backward(self, path: Path) -> None:
        """Place ops starting from the latest available slot, working backward."""
        cursor = self.total_slots - 1

        for op in reversed(path.ops):
            limit = self._adjust_limit(cursor, op)
            placed = False
            for s in range(min(limit, self.total_slots - 1), -1, -1):
                if self._can_place(s, op):
                    self._slots[s].insert(0, op)  # prepend to maintain order
                    cursor = s - 1
                    if path.module_id >= 0:
                        self._last_placed[path.module_id] = s
                    if self._on_place:
                        self._on_place(self, s, op)
                    placed = True
                    break
            if not placed:
                self.leftovers.append(op)

    def build(self) -> PlacedSchedule:
        """Produce the final interleaved schedule."""
        intervals = []
        for i, mfma in enumerate(self.mfmas):
            if i < self.num_intervals:
                slot_base = i * SLOTS_PER_INTERVAL
                side = []
                for s in range(SLOTS_PER_INTERVAL):
                    side.extend(self._slots[slot_base + s])
                intervals.append((side, mfma))
            else:
                # Last MFMA has no interval after it
                intervals.append(([], mfma))

        return PlacedSchedule(
            intervals=intervals,
            leftovers=self.leftovers,
        )

    def slot_occupancy(self, slot_idx: int) -> int:
        """Number of ops already placed in this slot."""
        if 0 <= slot_idx < self.total_slots:
            return len(self._slots[slot_idx])
        return 0

    def interval_of(self, slot_idx: int) -> int:
        """Which interval (MFMA pair) this slot belongs to."""
        return slot_idx // SLOTS_PER_INTERVAL


class SchedulingRules:
    """Pluggable rule set for the SlotPlacer.

    Tracks placement state to enforce constraints across the full schedule.
    """

    def __init__(self, total_slots: int, min_ds_read_gap: int = 8):
        self._ds_read_intervals: set = set()  # intervals with a ds_read
        self._m0_intervals: set = set()        # intervals with m0 write
        self._buf_load_count: int = 0
        self._buf_load_target_spacing: int = 1
        self._last_buf_load_slot: int = -1
        self.min_ds_read_gap = min_ds_read_gap  # min intervals between ds_read and wait
        self._ds_read_slots: List[int] = []     # track ds_read positions for gap rule

    def one_ds_read_per_interval(self, placer: SlotPlacer, slot_idx: int, op: PlacedOp) -> bool:
        """At most one ds_read per interval."""
        if op.op_type != "ds_read":
            return True
        interval = placer.interval_of(slot_idx)
        return interval not in self._ds_read_intervals

    def min_gap_ds_read_to_wait(self, placer: SlotPlacer, slot_idx: int, op: PlacedOp) -> bool:
        """Wait instructions must be at least N intervals after their ds_reads."""
        if op.op_type != "wait":
            return True
        interval = placer.interval_of(slot_idx)
        # Check that no ds_read from the same module is too close
        for dr_slot in self._ds_read_slots:
            dr_interval = placer.interval_of(dr_slot)
            if abs(interval - dr_interval) < self.min_ds_read_gap:
                return False
        return True

    def no_m0_with_buffer_load(self, placer: SlotPlacer, slot_idx: int, op: PlacedOp) -> bool:
        """Don't place buffer_load in same interval as m0 write (HW hazard)."""
        interval = placer.interval_of(slot_idx)
        if op.op_type == "buffer_load" and interval in self._m0_intervals:
            return False
        if op.op_type == "m0_write" and interval in self._m0_intervals:
            # Check if buffer_load already placed here
            for existing in placer._slots[slot_idx]:
                if existing.op_type == "buffer_load":
                    return False
        return True

    def spread_buffer_loads(self, placer: SlotPlacer, limit: int, op: PlacedOp) -> int:
        """Push buffer_loads apart by target spacing."""
        if op.op_type != "buffer_load":
            return limit
        if self._last_buf_load_slot >= 0:
            return max(limit, self._last_buf_load_slot + self._buf_load_target_spacing)
        return limit

    def track_placement(self, placer: SlotPlacer, slot_idx: int, op: PlacedOp) -> None:
        """Update rule state after a successful placement.

        Called by SlotPlacer after placing an op (wire this as on_place callback).
        """
        interval = placer.interval_of(slot_idx)
        if op.op_type == "ds_read":
            self._ds_read_intervals.add(interval)
            self._ds_read_slots.append(slot_idx)
        elif op.op_type == "m0_write":
            self._m0_intervals.add(interval)
        elif op.op_type == "buffer_load":
            self._buf_load_count += 1
            self._last_buf_load_slot = slot_idx

    def setup_buffer_load_spreading(self, total_slots: int, num_buffer_loads: int):
        """Compute target spacing for buffer_load spreading."""
        if num_buffer_loads > 0:
            self._buf_load_target_spacing = max(1, total_slots // num_buffer_loads)
