"""Pipeline scheduler: derives K-loop structure from dependency graph.

Takes a KLoopGraph (from ``graph_builder.build_kloop_graph``) and produces
a ``ScheduledPipeline`` containing ramp-up, steady-state body, drain, and
auto-computed waitcnts.

Key properties:
  - Loop order (produce-first vs consume-first) derived from op distances.
  - Waitcnts computed uniformly from RAW deps + issue timeline.
  - Ramp-up / drain depth derived from ``max(op.iteration)``.
  - No ``isinstance`` checks or PGR-specific branching.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from .kloop_graph import KLoopGraph, KLoopOp, OpKind, DepKind

__all__ = ["PipelineScheduler", "ScheduledPipeline"]


# ── Output ────────────────────────────────────────────────────────────

@dataclass
class ScheduledPipeline:
    """Complete scheduled K-loop ready for emission."""

    ramp_up: List[List[KLoopOp]]
    """Per-stage lists of ops for the prologue (len == pgr)."""

    body: List[KLoopOp]
    """Ordered ops for one steady-state iteration."""

    drain: List[List[KLoopOp]]
    """Per-stage drain iterations (consumers only, no producers)."""

    waitcnts: Dict[int, str]
    """body position → waitcnt string (e.g. ``{5: "lgkmcnt(3)"}``).

    Multiple counters at the same position are merged, e.g.
    ``"vmcnt(0) lgkmcnt(2)"``.
    """

    body_barrier_pos: int
    """Index of the barrier op inside *body*."""

    # Metadata
    pgr: int
    num_buffers: int
    is_consume_first: bool

    pre_body_count: int = 0
    """Number of ops at the start of *body* that go before the skip check."""

    ki_phased: bool = False
    """If True, consumers are sorted ki-first with a mid-copy barrier."""

    ki_phase_split: int = 0
    """Body position of the first ki=1 MFMA (divides ki=0 from ki=1)."""

    # ── helpers ────────────────────────────────────────────────────

    @property
    def producer_ops(self) -> List[KLoopOp]:
        """Body ops that belong to the producer phase."""
        return [op for op in self.body if op.iteration > 0]

    @property
    def consumer_ops(self) -> List[KLoopOp]:
        """Body ops that belong to the consumer phase."""
        return [op for op in self.body if op.iteration == 0
                and op.kind != OpKind.BARRIER]


# ── Scheduler ─────────────────────────────────────────────────────────

class PipelineScheduler:
    """Derive ramp-up / body / drain from a ``KLoopGraph``.

    Usage::

        graph = build_kloop_graph(streams, tile, pgr=1)
        pipeline = PipelineScheduler(graph).schedule()
    """

    def __init__(self, graph: KLoopGraph, *, ki_phased: bool = False) -> None:
        self.graph = graph
        self.ki_phased = ki_phased

    # ── public ────────────────────────────────────────────────────

    def schedule(self) -> ScheduledPipeline:
        g = self.graph

        # Derive PGR from max distance annotation.
        pgr = max((op.iteration for op in g.ops.values()), default=0)

        # Partition ops.
        producers = [op for op in g.ops.values() if op.iteration > 0]
        consumers = [op for op in g.ops.values()
                     if op.iteration == 0 and op.kind != OpKind.BARRIER]
        barrier_op = g.ops.get("barrier")

        # Determine body order from distances.
        is_consume_first = pgr >= 2

        ki_count = g.tile.k_iterations

        # Sort producers following ORDER/RAW chains.
        sorted_producers = self._topo_sort(producers, g)

        # Sort consumers: ki-phased or standard interleaving.
        if self.ki_phased and ki_count > 1:
            sorted_consumers = self._sort_consumers_ki_phased(consumers, g)
        else:
            sorted_consumers = self._sort_consumers(consumers, g)

        # ── Pre-body split (produce-first only) ─────────────────
        # B data reads for ki=0 with no WAR deps go before the
        # barrier to overlap with the barrier stall.  They read
        # from the stable read buffer (written + synced in the
        # previous iteration), so they're safe before the barrier.
        pre_body_count = 0
        if not is_consume_first:
            war_reads = {d.consumer for d in g.deps
                         if d.kind == DepKind.WAR}
            pre_body: List[KLoopOp] = []
            rest_consumers: List[KLoopOp] = []
            for op in sorted_consumers:
                if (op.kind in (OpKind.DS_READ, OpKind.SCALE_LOAD)
                        and "data_b" in op.name
                        and "_k0" in op.name
                        and op.name not in war_reads):
                    pre_body.append(op)
                else:
                    rest_consumers.append(op)
            pre_body_count = len(pre_body)

        # Assemble body.
        if is_consume_first:
            body: List[KLoopOp] = []
            if barrier_op:
                body.append(barrier_op)
            body.extend(sorted_consumers)
            body.extend(sorted_producers)
        else:
            # produce-first: pre_body → producers → barrier → rest
            body = list(pre_body) + list(sorted_producers)
            if barrier_op:
                body.append(barrier_op)
            body.extend(rest_consumers)

        barrier_pos = next(
            (i for i, op in enumerate(body) if op.kind == OpKind.BARRIER),
            -1)

        # ── Ramp-up ──────────────────────────────────────────────
        ramp_up = self._build_ramp_up(
            sorted_producers, barrier_op, pgr, is_consume_first)

        # ── Drain ────────────────────────────────────────────────
        drain = self._build_drain(
            sorted_consumers, barrier_op, pgr, is_consume_first)

        # ── Waitcnts ─────────────────────────────────────────────
        waitcnts = self._compute_waitcnts(body, g)

        # Infer num_buffers from WAR dep distances (default 2).
        war_dists = [d.min_cycles for d in g.deps if d.kind == DepKind.WAR]
        # In graph_builder the WAR distance is not yet stored in
        # min_cycles; for now fall back to 2.
        num_buffers = max(war_dists) if war_dists else 2

        # Ki-phase split: index of first ki=1 MFMA in body
        ki_split = 0
        if self.ki_phased and ki_count > 1:
            for i, op in enumerate(body):
                if op.kind == OpKind.MFMA and '_k1' in op.name:
                    ki_split = i
                    break

        return ScheduledPipeline(
            ramp_up=ramp_up,
            body=body,
            drain=drain,
            waitcnts=waitcnts,
            body_barrier_pos=barrier_pos,
            pgr=pgr,
            num_buffers=num_buffers,
            is_consume_first=is_consume_first,
            pre_body_count=pre_body_count,
            ki_phased=self.ki_phased and ki_count > 1,
            ki_phase_split=ki_split,
        )

    # ── private helpers ───────────────────────────────────────────

    def _topo_sort(self, ops: List[KLoopOp],
                   g: KLoopGraph) -> List[KLoopOp]:
        """Topological sort of *ops* following RAW / ORDER edges."""
        name_set = {op.name for op in ops}
        # Build adjacency within the subset.
        adj: Dict[str, List[str]] = defaultdict(list)
        in_deg: Dict[str, int] = {op.name: 0 for op in ops}
        for dep in g.deps:
            if dep.producer in name_set and dep.consumer in name_set:
                adj[dep.producer].append(dep.consumer)
                in_deg[dep.consumer] += 1

        queue = [n for n, d in in_deg.items() if d == 0]
        result: List[str] = []
        while queue:
            # Stable: sort by insertion order in graph.
            queue.sort(key=lambda n: list(g.ops.keys()).index(n))
            n = queue.pop(0)
            result.append(n)
            for succ in adj[n]:
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    queue.append(succ)

        name_to_op = {op.name: op for op in ops}
        return [name_to_op[n] for n in result]

    def _sort_consumers(self, consumers: List[KLoopOp],
                        g: KLoopGraph) -> List[KLoopOp]:
        """Sort consumer ops: interleave reads among MFMAs.

        Places each ds_read DS_READ_LEAD MFMAs before its earliest
        consumer MFMA, hiding LDS latency under MFMA execution.
        WAR-constrained reads (A ping-pong) are placed right after
        the WAR-dep MFMA.  Suffix ops (toggle_rd) go last.
        """

        reads: List[KLoopOp] = []
        mfmas: List[KLoopOp] = []
        suffix: List[KLoopOp] = []

        for op in consumers:
            if op.kind in (OpKind.DS_READ, OpKind.SCALE_LOAD):
                reads.append(op)
            elif op.kind == OpKind.MFMA:
                mfmas.append(op)
            else:
                suffix.append(op)

        # MFMAs: canonical order (m, ki, ni).
        tile = g.tile
        mr = tile.mfma_m_repeat
        nr = tile.mfma_n_repeat
        ki_count = tile.k_iterations

        mfma_order: List[KLoopOp] = []
        mfma_by_name = {op.name: op for op in mfmas}
        for mi in range(mr):
            for ki in range(ki_count):
                for ni in range(nr):
                    name = f"mfma_m{mi}_n{ni}_k{ki}"
                    if name in mfma_by_name:
                        mfma_order.append(mfma_by_name[name])

        # Build read -> earliest consuming MFMA position map.
        mfma_positions = {op.name: i for i, op in enumerate(mfma_order)}
        read_earliest: Dict[str, int] = {}
        for dep in g.deps:
            if dep.kind == DepKind.RAW and dep.producer in {r.name for r in reads}:
                if dep.consumer in mfma_positions:
                    pos = mfma_positions[dep.consumer]
                    cur = read_earliest.get(dep.producer, len(mfma_order))
                    read_earliest[dep.producer] = min(cur, pos)

        # WAR deps (ping-pong): read must come after WAR-dep MFMA.
        read_war_after: Dict[str, int] = {}
        for dep in g.deps:
            if dep.kind == DepKind.WAR and dep.consumer in {r.name for r in reads}:
                if dep.producer in mfma_positions:
                    pos = mfma_positions[dep.producer]
                    cur = read_war_after.get(dep.consumer, -1)
                    read_war_after[dep.consumer] = max(cur, pos)

        # Place each read DS_READ_LEAD MFMAs before its consumer,
        # or right after its WAR-dep MFMA for ping-pong reads.
        DS_READ_LEAD = 4
        reads_at_pos: Dict[int, List[KLoopOp]] = {}
        for r in reads:
            war_pos = read_war_after.get(r.name, -1)
            target = read_earliest.get(r.name, 0)
            if war_pos >= 0:
                # WAR-constrained: place right after WAR MFMA
                place = war_pos + 1
            else:
                # Place DS_READ_LEAD MFMAs before consumer
                place = max(target - DS_READ_LEAD, 0)
            reads_at_pos.setdefault(place, []).append(r)

        # Sort reads within each position by urgency
        for pos in reads_at_pos:
            reads_at_pos[pos].sort(
                key=lambda r: (read_earliest.get(r.name, 999), r.name))

        # Build result: reads interleaved among MFMAs
        result: List[KLoopOp] = []
        # Reads at position 0 go before first MFMA
        for r in reads_at_pos.get(0, []):
            result.append(r)
        for i, mfma_op in enumerate(mfma_order):
            result.append(mfma_op)
            # Reads placed after MFMA at position i (targeting i+1)
            for r in reads_at_pos.get(i + 1, []):
                result.append(r)
        result.extend(suffix)
        return result


    def _sort_consumers_ki_phased(
        self, consumers: List[KLoopOp], g: KLoopGraph,
    ) -> List[KLoopOp]:
        """Sort consumers with ki-phased ordering.

        All ki=0 MFMAs first (with ki=1 reads interleaved), then
        all ki=1 MFMAs.  A single lgkmcnt(0) at the ki boundary
        drains all ki=1 reads, eliminating per-MFMA waitcnts.

        ki=0 reads + scale reads go at position 0 (before any MFMA).
        ki=1 reads are interleaved among ki=0 MFMAs.
        """
        tile = g.tile
        mr = tile.mfma_m_repeat
        nr = tile.mfma_n_repeat
        ki_count = tile.k_iterations

        reads: List[KLoopOp] = []
        mfmas: List[KLoopOp] = []
        suffix: List[KLoopOp] = []

        for op in consumers:
            if op.kind in (OpKind.DS_READ, OpKind.SCALE_LOAD):
                reads.append(op)
            elif op.kind == OpKind.MFMA:
                mfmas.append(op)
            else:
                suffix.append(op)

        mfma_by_name = {op.name: op for op in mfmas}

        # Build ki-phased MFMA order: all ki=0 first, then all ki=1
        ki0_mfmas: List[KLoopOp] = []
        ki1_mfmas: List[KLoopOp] = []
        for ki in range(ki_count):
            for mi in range(mr):
                for ni in range(nr):
                    name = f"mfma_m{mi}_n{ni}_k{ki}"
                    if name in mfma_by_name:
                        if ki == 0:
                            ki0_mfmas.append(mfma_by_name[name])
                        else:
                            ki1_mfmas.append(mfma_by_name[name])

        # Partition reads by ki value
        ki0_reads: List[KLoopOp] = []
        ki1_reads: List[KLoopOp] = []
        for r in reads:
            if '_k1' in r.name:
                ki1_reads.append(r)
            else:
                ki0_reads.append(r)

        # Result: ki=0 reads, ki=0 MFMAs with ki=1 reads interleaved,
        # ki=1 MFMAs, suffix ops
        result: List[KLoopOp] = list(ki0_reads)

        # Interleave ki=1 reads among ki=0 MFMAs
        n_ki0 = len(ki0_mfmas)
        n_ki1_reads = len(ki1_reads)
        if n_ki1_reads > 0 and n_ki0 > 0:
            interval = max(1, n_ki0 // (n_ki1_reads + 1))
            read_idx = 0
            for i, mfma_op in enumerate(ki0_mfmas):
                result.append(mfma_op)
                if (i + 1) % interval == 0 and read_idx < n_ki1_reads:
                    result.append(ki1_reads[read_idx])
                    read_idx += 1
            while read_idx < n_ki1_reads:
                result.append(ki1_reads[read_idx])
                read_idx += 1
        else:
            result.extend(ki0_mfmas)

        result.extend(ki1_mfmas)
        result.extend(suffix)
        return result

    # ── ramp-up ───────────────────────────────────────────────────

    def _build_ramp_up(
        self,
        sorted_producers: List[KLoopOp],
        barrier_op: Optional[KLoopOp],
        pgr: int,
        is_consume_first: bool,
    ) -> List[List[KLoopOp]]:
        """Build ramp-up stages (one per PGR level)."""
        if pgr == 0:
            return []

        stages: List[List[KLoopOp]] = []

        for s in range(pgr):
            stage: List[KLoopOp] = []

            if s == 0:
                # Stage 0: first tile load + barrier.
                # Only emit load ops (no advance/toggle needed for tile 0).
                load_ops = [op for op in sorted_producers
                            if op.kind in (OpKind.GLOBAL_LOAD, OpKind.DS_WRITE)]
                stage.extend(load_ops)
                if barrier_op:
                    stage.append(barrier_op)
            else:
                # Stage s > 0: advance + toggle + load (prefetch next tile).
                stage.extend(sorted_producers)
                # For PGR >= 2, the last ramp-up stage has no barrier
                # (falls through to the body which starts with barrier).
                if not is_consume_first or s < pgr - 1:
                    if barrier_op:
                        stage.append(barrier_op)

            stages.append(stage)

        return stages

    # ── drain ─────────────────────────────────────────────────────

    def _build_drain(
        self,
        sorted_consumers: List[KLoopOp],
        barrier_op: Optional[KLoopOp],
        pgr: int,
        is_consume_first: bool,
    ) -> List[List[KLoopOp]]:
        """Build drain stages (consumer-only iterations at the end).

        For PGR=0: no drain needed (every iteration has its own load).
        For PGR>=1: pgr-1 extra drain iterations after the last
        producer skip in the body.  (The body's skip-check already
        handles the transition; drain stages capture the remaining
        consume-only iterations.)
        """
        if pgr <= 1:
            # PGR 0 or 1: the body loop handles everything via skip-check.
            return []

        stages: List[List[KLoopOp]] = []
        for _ in range(pgr - 1):
            stage: List[KLoopOp] = []
            if barrier_op:
                stage.append(barrier_op)
            stage.extend(sorted_consumers)
            stages.append(stage)
        return stages

    # ── waitcnts ──────────────────────────────────────────────────

    def _compute_waitcnts(
        self,
        body: List[KLoopOp],
        g: KLoopGraph,
    ) -> Dict[int, str]:
        """Derive all waitcnts from issue order + RAW deps.

        For each consumer op that depends (RAW) on a counter-tracked
        producer, compute the tightest wait value:

            wait = ops_of_same_counter_issued_after_dep

        Multiple counters at the same position are merged.
        """
        # Build name→position map.
        pos_of: Dict[str, int] = {op.name: i for i, op in enumerate(body)}

        # Reverse map: for each consumer, collect its RAW producers
        # that are tracked by a hw counter.
        consumer_deps: Dict[str, List[str]] = defaultdict(list)
        for dep in g.deps:
            if dep.kind != DepKind.RAW:
                continue
            prod_op = g.ops.get(dep.producer)
            if prod_op and prod_op.hw_counter and dep.consumer in pos_of:
                consumer_deps[dep.consumer].append(dep.producer)

        # Per-counter issue timeline: position → list of counter types.
        counter_at: Dict[int, str] = {}
        for i, op in enumerate(body):
            if op.hw_counter:
                counter_at[i] = op.hw_counter

        # Cumulative count of each counter type at each position.
        vmcnt_cum: List[int] = []
        lgkm_cum: List[int] = []
        vc = 0
        lc = 0
        for i in range(len(body)):
            ct = counter_at.get(i)
            if ct == "vmcnt":
                vc += 1
            elif ct == "lgkmcnt":
                lc += 1
            vmcnt_cum.append(vc)
            lgkm_cum.append(lc)

        def _cum(counter: str, pos: int) -> int:
            if counter == "vmcnt":
                return vmcnt_cum[pos]
            return lgkm_cum[pos]

        # For each consumer, compute the required wait per counter.
        raw_waits: Dict[int, Dict[str, int]] = defaultdict(dict)
        # key = body position, value = {counter: wait_value}

        for cons_name, prod_names in consumer_deps.items():
            cons_pos = pos_of[cons_name]

            # Group producers by counter.
            by_counter: Dict[str, int] = {}  # counter → latest prod pos
            for pn in prod_names:
                prod_op = g.ops.get(pn)
                if not prod_op or not prod_op.hw_counter:
                    continue
                ct = prod_op.hw_counter
                pp = pos_of.get(pn)
                if pp is None:
                    # Producer not in body (e.g. ramp-up only).
                    # Need vmcnt(0) at the start of the body.
                    by_counter[ct] = -1
                else:
                    by_counter[ct] = max(by_counter.get(ct, -1), pp)

            for ct, latest_pos in by_counter.items():
                if latest_pos < 0:
                    # Producer was in ramp-up → need full drain.
                    wait_val = 0
                else:
                    # Count same-counter ops issued AFTER the dep
                    # but BEFORE the consumer.
                    issued_after = _cum(ct, cons_pos - 1) - _cum(ct, latest_pos)
                    wait_val = max(issued_after, 0)

                hw_max = 15 if ct == "lgkmcnt" else 63
                wait_val = min(wait_val, hw_max)

                cur = raw_waits[cons_pos].get(ct)
                if cur is None or wait_val < cur:
                    raw_waits[cons_pos][ct] = wait_val

        # Elide redundant waits: track inflight count per counter.
        # Only emit a waitcnt when the required value is tighter
        # (lower) than the current guaranteed state.
        inflight_vm = 0
        inflight_lgkm = 0
        elided_waits: Dict[int, Dict[str, int]] = {}

        for i in range(len(body)):
            # Update inflight from ops issued at this position
            ct = counter_at.get(i)
            if ct == "vmcnt":
                inflight_vm += 1
            elif ct == "lgkmcnt":
                inflight_lgkm += 1

            if i in raw_waits:
                for counter, wait_val in raw_waits[i].items():
                    current = inflight_vm if counter == "vmcnt" else inflight_lgkm
                    # Only emit if the wait is tighter than current state
                    if wait_val < current:
                        elided_waits.setdefault(i, {})[counter] = wait_val
                        # Update inflight to reflect the wait
                        if counter == "vmcnt":
                            inflight_vm = wait_val
                        else:
                            inflight_lgkm = wait_val

        # Merge per-position waits into strings.
        waitcnts: Dict[int, str] = {}
        for pos, counters in sorted(elided_waits.items()):
            parts = []
            for ct in ("vmcnt", "lgkmcnt"):
                if ct in counters:
                    parts.append(f"{ct}({counters[ct]})")
            if parts:
                waitcnts[pos] = " ".join(parts)

        return waitcnts
