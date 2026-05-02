"""Dependency-driven K-loop scheduler.

Takes a KLoopGraph (ops + deps) and produces a ScheduledKLoop with:
- MFMA backbone in dependency-respecting order
- ds_reads placed between MFMAs via list scheduling
- Suffix ops (waits, toggle) placed backward from the end
- Auto-inserted waitcnts based on dependency distances

See DESIGN_SCHEDULER.md for the full design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .kloop_graph import KLoopGraph, KLoopOp, OpKind, DepKind
from .slot_placer import PlacedOp, PlacedSchedule, SLOTS_PER_INTERVAL

__all__ = ["KLoopScheduler", "ScheduledKLoop"]

# Approximate MFMA latency in cycles (gfx950 v_mfma_f32_16x16x32_f16)
MFMA_CYCLES = 4
# ds_read latency in cycles
DS_READ_LATENCY = 20
# How many MFMAs a ds_read needs to be issued ahead of its consumer
DS_READ_LEAD_MFMAS = DS_READ_LATENCY // MFMA_CYCLES  # ~5


@dataclass
class ScheduledKLoop:
    """Result of scheduling a KLoopGraph."""
    # MFMA backbone in execution order
    mfma_order: List[KLoopOp] = field(default_factory=list)
    # For each MFMA position i, the side ops placed before MFMA[i]
    side_ops: List[List[KLoopOp]] = field(default_factory=list)
    # Ops placed after the last MFMA
    epilogue_ops: List[KLoopOp] = field(default_factory=list)
    # Pre-body ops (B reads for ki=0 that overlap with loads)
    pre_body_ops: List[KLoopOp] = field(default_factory=list)
    # Structural ops (barrier)
    barrier_op: Optional[KLoopOp] = None
    # Next-iter prefetch ops (advance, toggle, load)
    prefetch_ops: List[KLoopOp] = field(default_factory=list)
    # Preamble reads (A[m0] + B[ki=1] before scheduled body)
    preamble_ops: List[KLoopOp] = field(default_factory=list)
    # Auto-inserted waits: position -> waitcnt string
    waits: Dict[int, str] = field(default_factory=dict)

    def to_placed_schedule(self) -> PlacedSchedule:
        """Convert to a PlacedSchedule for compatibility with existing code."""
        intervals = []
        for i, mfma in enumerate(self.mfma_order):
            side = [PlacedOp(emit_fn=op.emit, op_type=op.kind.value,
                             comment=op.comment)
                    for op in self.side_ops[i]]
            mfma_placed = PlacedOp(emit_fn=mfma.emit, op_type="mfma",
                                   comment=mfma.comment)
            intervals.append((side, mfma_placed))
        epilogue = [PlacedOp(emit_fn=op.emit, op_type=op.kind.value,
                             comment=op.comment)
                    for op in self.epilogue_ops]
        return PlacedSchedule(intervals=intervals, epilogue=epilogue)


class KLoopScheduler:
    """Schedule a KLoopGraph into a concrete K-loop.

    Algorithm:
    1. Order MFMAs respecting WAR deps (mi order forced by A ping-pong)
    2. Classify side ops by placement strategy
    3. Place ds_reads forward (early, for latency hiding)
    4. Place suffix ops backward (late, to maximize overlap)
    5. Compute auto-waits from dependency distances
    """

    def __init__(self, graph: KLoopGraph) -> None:
        self.graph = graph

    def schedule(self) -> ScheduledKLoop:
        g = self.graph
        tile = g.tile
        mr = tile.mfma_m_repeat
        nr = tile.mfma_n_repeat
        ki_count = tile.k_iterations
        mfmas_per_mi = nr * ki_count

        # Phase 1: MFMA backbone order (mi, ki, ni -- matches manual)
        mfma_order = []
        for mi in range(mr):
            for ki in range(ki_count):
                for ni in range(nr):
                    name = f"mfma_m{mi}_n{ni}_k{ki}"
                    mfma_order.append(g.ops[name])
        mfma_positions = {op.name: i for i, op in enumerate(mfma_order)}
        n_mfma = len(mfma_order)

        # Phase 2: Classify side ops
        barrier_op = g.ops.get("barrier")
        prefetch_ops = []  # iteration=1
        b_reads_ki0 = []   # early B reads
        a_reads = {}       # (mi, ki) -> op
        b_reads = {}       # (ni, ki) -> op
        suffix_ops = []    # waits, toggle, negate

        for name, op in g.ops.items():
            if op.kind == OpKind.MFMA or op.kind == OpKind.BARRIER:
                continue
            if op.iteration == 1:
                prefetch_ops.append(op)
            elif name.startswith("read_b_n") and name.endswith("_k0"):
                b_reads_ki0.append(op)
                parts = name.replace("read_b_n", "").replace("_k", " ").split()
                b_reads[(int(parts[0]), 0)] = op
            elif name.startswith("read_b_n"):
                parts = name.replace("read_b_n", "").replace("_k", " ").split()
                b_reads[(int(parts[0]), int(parts[1]))] = op
            elif name.startswith("read_a_m"):
                parts = (name.replace("read_a_m", "")
                         .replace("_k", " ").replace("_buf", " ").split())
                a_reads[(int(parts[0]), int(parts[1]))] = op

        # Phase 3: Build the schedule
        # Pre-body: B reads for ki=0 (overlap with loads arriving)
        pre_body = list(b_reads_ki0)

        # Preamble: reads needed before first MFMA
        # A[m0, k0], B[all ni, ki=1], A[m0, k1]
        preamble = []
        if (0, 0) in a_reads:
            preamble.append(a_reads[(0, 0)])
        if ki_count > 1:
            for ni in range(nr):
                if (ni, 1) in b_reads:
                    preamble.append(b_reads[(ni, 1)])
            if (0, 1) in a_reads:
                preamble.append(a_reads[(0, 1)])

        # Track which reads have been placed in pre-body or preamble
        placed_reads = set()
        for op in pre_body + preamble:
            placed_reads.add(op.name)

        # Remaining ds_reads: A-prefetch for mi=1..mr-1
        # These get placed between MFMAs (forward from their WAR dep)
        remaining_reads = []
        for name, op in g.ops.items():
            if op.kind == OpKind.DS_READ and name not in placed_reads:
                remaining_reads.append(op)

        # For each remaining read, compute earliest MFMA position
        # based on WAR deps (must come after prior mi's last MFMA)
        read_earliest = {}
        for op in remaining_reads:
            war_deps = [d for d in g.deps
                        if d.consumer == op.name and d.kind == DepKind.WAR]
            earliest = 0
            for dep in war_deps:
                if dep.producer in mfma_positions:
                    earliest = max(earliest, mfma_positions[dep.producer] + 1)
            read_earliest[op.name] = earliest

        # For each remaining read, compute latest MFMA position
        # (must be placed before consuming MFMA minus latency lead)
        read_latest = {}
        for op in remaining_reads:
            consumers = [d.consumer for d in g.deps
                         if d.producer == op.name and d.kind == DepKind.RAW]
            latest = n_mfma - 1
            for c in consumers:
                if c in mfma_positions:
                    lead = DS_READ_LEAD_MFMAS
                    latest = min(latest, mfma_positions[c] - lead)
            read_latest[op.name] = max(latest, 0)

        # Build suffix: vmcnt wait + toggle + negate (reader suffix ops)
        # These will be placed backward from the end of the MFMA body
        # For now, suffix ops are declared by the reader (not in the graph yet)
        # The graph-based scheduler will add them in a future iteration.

        # Phase 4: Place remaining reads into MFMA intervals
        # Each MFMA position has a list of side ops placed before it
        side_ops: List[List[KLoopOp]] = [[] for _ in range(n_mfma)]
        reads_per_interval = {}  # interval -> count (max 1 ds_read)

        # Sort remaining reads by earliest position (greedy forward placement)
        remaining_reads.sort(key=lambda op: read_earliest[op.name])

        for op in remaining_reads:
            earliest = read_earliest[op.name]
            latest = read_latest.get(op.name, n_mfma - 1)
            target = max(earliest, 0)

            placed = False
            for pos in range(target, min(latest + 1, n_mfma)):
                interval = pos
                if reads_per_interval.get(interval, 0) < 1:
                    side_ops[pos].append(op)
                    reads_per_interval[interval] = (
                        reads_per_interval.get(interval, 0) + 1)
                    placed = True
                    break
            if not placed:
                # Fallback: place at latest even if constraint violated
                for pos in range(min(latest, n_mfma - 1), -1, -1):
                    side_ops[pos].append(op)
                    break

        # Phase 5: Auto-wait insertion
        waits = self._compute_waits(
            mfma_order, side_ops, pre_body, preamble, mfma_positions)

        return ScheduledKLoop(
            mfma_order=mfma_order,
            side_ops=side_ops,
            pre_body_ops=pre_body,
            barrier_op=barrier_op,
            prefetch_ops=prefetch_ops,
            preamble_ops=preamble,
            waits=waits,
        )

    def _compute_waits(self, mfma_order: List[KLoopOp],
                       side_ops: List[List[KLoopOp]],
                       pre_body: List[KLoopOp],
                       preamble: List[KLoopOp],
                       mfma_positions: Dict[str, int]) -> Dict[int, str]:
        """Compute where s_waitcnt lgkmcnt(N) needs to be inserted.

        Tracks in-flight lgkmcnt as ds_reads are issued, counts down
        as instructions execute. Inserts wait before each MFMA whose
        operand reads haven't completed yet.
        """
        g = self.graph
        waits: Dict[int, str] = {}

        # Map each MFMA to the ds_reads it depends on (RAW)
        mfma_read_deps: Dict[str, List[str]] = {}
        for dep in g.deps:
            if dep.kind == DepKind.RAW and dep.consumer in mfma_positions:
                prod_op = g.ops.get(dep.producer)
                if prod_op and prod_op.kind == OpKind.DS_READ:
                    mfma_read_deps.setdefault(dep.consumer, []).append(
                        dep.producer)

        # Build a timeline: position of each ds_read in the total
        # instruction sequence (pre_body, preamble, then interleaved body)
        read_positions: Dict[str, int] = {}
        pos = 0
        for op in pre_body:
            if op.kind == OpKind.DS_READ:
                read_positions[op.name] = pos
            pos += 1
        for op in preamble:
            if op.kind == OpKind.DS_READ:
                read_positions[op.name] = pos
            pos += 1
        # Barrier + sync adds some instruction distance
        pos += 2  # approximate barrier + vmcnt

        for i in range(len(mfma_order)):
            for op in side_ops[i]:
                if op.kind == OpKind.DS_READ:
                    read_positions[op.name] = pos
                pos += 1
            pos += 1  # MFMA itself

        # For each MFMA, check if all its operand reads have enough
        # distance. If not, note that a wait is needed.
        mfma_inst_pos = {}
        pos = len(pre_body) + len(preamble) + 2  # after pre_body + preamble + barrier
        for i in range(len(mfma_order)):
            pos += len(side_ops[i])
            mfma_inst_pos[mfma_order[i].name] = pos
            pos += 1

        for mfma_name, reads in mfma_read_deps.items():
            if mfma_name not in mfma_inst_pos:
                continue
            mfma_pos = mfma_inst_pos[mfma_name]
            for read_name in reads:
                if read_name not in read_positions:
                    continue
                read_pos = read_positions[read_name]
                distance = mfma_pos - read_pos
                if distance < DS_READ_LEAD_MFMAS:
                    # Need a wait before this MFMA
                    mfma_idx = mfma_positions[mfma_name]
                    waits[mfma_idx] = "lgkmcnt(0)"
                    break  # one wait per MFMA is enough

        return waits

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def mfma_order_names(self) -> List[str]:
        """Return MFMA op names in scheduled order (for testing)."""
        g = self.graph
        tile = g.tile
        names = []
        for mi in range(tile.mfma_m_repeat):
            for ki in range(tile.k_iterations):
                for ni in range(tile.mfma_n_repeat):
                    names.append(f"mfma_m{mi}_n{ni}_k{ki}")
        return names

    def read_schedule_summary(self) -> Dict[str, int]:
        """Return {read_name: mfma_position_placed_before} for testing."""
        result = self.schedule()
        summary = {}
        for i, ops in enumerate(result.side_ops):
            for op in ops:
                if op.kind == OpKind.DS_READ:
                    summary[op.name] = i
        return summary
