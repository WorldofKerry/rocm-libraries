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
    # Scale load ops for prologue (first subtile A + all B)
    prologue_scale_ops: List[KLoopOp] = field(default_factory=list)
    # Scale advance op for next iteration (iteration=1)
    scale_advance_op: Optional[KLoopOp] = None
    # Auto-inserted waits: position -> waitcnt string
    waits: Dict[int, str] = field(default_factory=dict)


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
        suffix_ops = []    # waits, toggle
        scale_a_ops = {}   # (mi, ki) -> op
        scale_b_ops = {}   # (ni, ki) -> op
        scale_advance_op = None

        for name, op in g.ops.items():
            if op.kind == OpKind.MFMA or op.kind == OpKind.BARRIER:
                continue
            if name == "advance_scale":
                scale_advance_op = op
                continue
            if op.kind == OpKind.SCALE_LOAD or (
                    op.kind == OpKind.DS_READ
                    and name.startswith(("scale_a_m", "scale_b_n"))):
                if name.startswith("scale_a_m"):
                    parts = name.replace("scale_a_m", "").replace("_k", " ").split()
                    scale_a_ops[(int(parts[0]), int(parts[1]))] = op
                elif name.startswith("scale_b_n"):
                    parts = name.replace("scale_b_n", "").replace("_k", " ").split()
                    scale_b_ops[(int(parts[0]), int(parts[1]))] = op
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
        # NOTE: prologue_scale_ops are added to placed_reads later
        # (after they're built in Phase 4a) to prevent double-placement.

        # Phase 4: Place reads into MFMA intervals
        side_ops: List[List[KLoopOp]] = [[] for _ in range(n_mfma)]
        reads_per_interval = {}

        # Phase 4a: Place scale load ops
        # Group scale ops into subtiles: first subtile goes to prologue,
        # remaining subtiles are placed at subtile boundaries in side_ops.
        prologue_scale_ops: List[KLoopOp] = []
        partition_m = min(4, mr)

        if scale_a_ops or scale_b_ops:
            # Prologue: A scales for first subtile + all B scales
            for mi in range(partition_m):
                for ki in range(ki_count):
                    if (mi, ki) in scale_a_ops:
                        prologue_scale_ops.append(scale_a_ops[(mi, ki)])
            for ni in range(nr):
                for ki in range(ki_count):
                    if (ni, ki) in scale_b_ops:
                        prologue_scale_ops.append(scale_b_ops[(ni, ki)])

            # Remaining subtiles: place A scale loads at subtile boundaries.
            # Each subtile boundary is at the first MFMA of the next
            # partition_m group.
            n_subtiles = (mr + partition_m - 1) // partition_m
            for st in range(1, n_subtiles):
                st_start_mi = st * partition_m
                st_end_mi = min(st_start_mi + partition_m, mr)
                # Find the MFMA position of the first MFMA in this subtile
                first_mfma_name = f"mfma_m{st_start_mi}_n0_k0"
                if first_mfma_name in mfma_positions:
                    target_pos = mfma_positions[first_mfma_name]
                    for mi in range(st_start_mi, st_end_mi):
                        for ki in range(ki_count):
                            if (mi, ki) in scale_a_ops:
                                op = scale_a_ops[(mi, ki)]
                                side_ops[target_pos].insert(0, op)

        # Phase 4b: Build remaining_reads now that prologue_scale_ops is known
        for op in prologue_scale_ops:
            placed_reads.add(op.name)
        # Also mark subtile scale ops as placed
        for mfma_pos_ops in side_ops:
            for op in mfma_pos_ops:
                if op.name.startswith(("scale_a_m", "scale_b_n")):
                    placed_reads.add(op.name)
        remaining_reads = []
        for name, op in g.ops.items():
            if op.kind == OpKind.DS_READ and name not in placed_reads:
                remaining_reads.append(op)

        # For each remaining read, compute earliest MFMA position
        read_earliest = {}
        for op in remaining_reads:
            war_deps = [d for d in g.deps
                        if d.consumer == op.name and d.kind == DepKind.WAR]
            earliest = 0
            for dep in war_deps:
                if dep.producer in mfma_positions:
                    earliest = max(earliest, mfma_positions[dep.producer] + 1)
            read_earliest[op.name] = earliest

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

        remaining_reads.sort(key=lambda op: read_earliest[op.name])
        for op in remaining_reads:
            earliest = read_earliest[op.name]
            latest = read_latest.get(op.name, n_mfma - 1)
            target = max(earliest, 0)
            placed = False
            for pos in range(target, min(latest + 1, n_mfma)):
                if reads_per_interval.get(pos, 0) < 1:
                    side_ops[pos].append(op)
                    reads_per_interval[pos] = (
                        reads_per_interval.get(pos, 0) + 1)
                    placed = True
                    break
            if not placed:
                for pos in range(min(latest, n_mfma - 1), -1, -1):
                    side_ops[pos].append(op)
                    break

        # Phase 4c: Place suffix ops backward from end
        suffix_names = ["suffix_vmcnt", "suffix_toggle"]
        suffix_ops_list = [g.ops[n] for n in suffix_names if n in g.ops]
        epilogue_ops: List[KLoopOp] = []
        if suffix_ops_list:
            # Place backward: last suffix at last MFMA, etc.
            cursor = n_mfma - 1
            for op in reversed(suffix_ops_list):
                while cursor >= 0 and reads_per_interval.get(cursor, 0) >= 1:
                    cursor -= 1
                if cursor >= 0:
                    side_ops[cursor].insert(0, op)
                    cursor -= 1
                else:
                    epilogue_ops.append(op)

        # Phase 5: Auto-wait insertion
        waits = self._compute_waits(
            mfma_order, side_ops, pre_body, preamble,
            prologue_scale_ops, mfma_positions)

        return ScheduledKLoop(
            mfma_order=mfma_order,
            side_ops=side_ops,
            epilogue_ops=epilogue_ops,
            pre_body_ops=pre_body,
            barrier_op=barrier_op,
            prefetch_ops=prefetch_ops,
            preamble_ops=preamble,
            prologue_scale_ops=prologue_scale_ops,
            scale_advance_op=scale_advance_op,
            waits=waits,
        )

    def _compute_waits(self, mfma_order: List[KLoopOp],
                       side_ops: List[List[KLoopOp]],
                       pre_body: List[KLoopOp],
                       preamble: List[KLoopOp],
                       prologue_scale_ops: List[KLoopOp],
                       mfma_positions: Dict[str, int]) -> Dict[int, str]:
        """Compute where s_waitcnt lgkmcnt(N) needs to be inserted.

        Simulates lgkmcnt tracking: each ds_read increments the count,
        and we insert waits before MFMAs whose operands haven't been
        guaranteed available yet.
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

        # Build timeline of all ds_reads in issue order
        read_issue_order: List[str] = []
        for op in pre_body:
            if op.kind == OpKind.DS_READ:
                read_issue_order.append(op.name)
        for op in preamble:
            if op.kind == OpKind.DS_READ:
                read_issue_order.append(op.name)
        for op in prologue_scale_ops:
            if op.kind == OpKind.DS_READ:
                read_issue_order.append(op.name)
        for i in range(len(mfma_order)):
            for op in side_ops[i]:
                if op.kind == OpKind.DS_READ:
                    read_issue_order.append(op.name)

        # Assign each read a position in the issue order (0 = first issued)
        read_issue_idx = {name: idx for idx, name in enumerate(read_issue_order)}
        total_reads = len(read_issue_order)

        # Count reads issued before each MFMA position
        reads_before_mfma: Dict[int, int] = {}
        n_preamble_reads = sum(1 for op in pre_body + preamble + prologue_scale_ops
                               if op.kind == OpKind.DS_READ)
        count = n_preamble_reads
        for i in range(len(mfma_order)):
            reads_before_mfma[i] = count
            for op in side_ops[i]:
                if op.kind == OpKind.DS_READ:
                    count += 1

        # For each MFMA, find the latest ds_read it depends on
        for mfma_name, reads in mfma_read_deps.items():
            if mfma_name not in mfma_positions:
                continue
            mfma_idx = mfma_positions[mfma_name]

            # Find latest read this MFMA depends on
            latest_dep_idx = -1
            for read_name in reads:
                if read_name in read_issue_idx:
                    latest_dep_idx = max(latest_dep_idx, read_issue_idx[read_name])

            if latest_dep_idx < 0:
                continue

            # lgkmcnt at this MFMA = total_issued - completed
            # We need latest_dep_idx to have completed
            # lgkmcnt represents reads still in-flight:
            #   inflight = total_issued - completed_count
            # We need inflight <= (total_issued - latest_dep_idx - 1)
            # i.e. lgkmcnt(total_reads_before_this_mfma - latest_dep_idx - 1)
            issued_before = reads_before_mfma.get(mfma_idx, 0)
            wait_for = issued_before - latest_dep_idx - 1
            if wait_for < 0:
                wait_for = 0

            if mfma_idx in waits:
                # Take the tighter (lower) wait
                existing = int(waits[mfma_idx].split("(")[1].rstrip(")"))
                wait_for = min(wait_for, existing)

            wait_for = min(wait_for, 15)  # hardware max lgkmcnt
            waits[mfma_idx] = f"lgkmcnt({wait_for})"

        return waits

# ===================================================================
# Phase function: emit a scheduled K-loop as assembly
# ===================================================================

def scheduled_kloop_phase(level, ctx) -> None:
    """Phase function: dependency-driven scheduled K-loop.

    Drop-in replacement for composable_kloop_phase. Builds a KLoopGraph,
    schedules it, and emits the result as assembly.
    """
    import math
    from ..problem import TileConfig, GemmProblem
    from ..memory.global_loader import GlobalLoader, DTLLoader, BufferLoader
    from ..memory.lds_reader import LDSReader
    from .kloop_graph import (
        KLoopGraph, MFMABlock, DSReadBlock, GlobalLoadBlock, ScaleBlock,
        SuffixBlock,
    )

    tile = ctx._metadata["tile"]
    problem = ctx._metadata["problem"]
    use_dtl = ctx._metadata.get("use_dtl", True)

    # Build loader/reader (same as composable_kloop_phase)
    loader_cls = ctx._metadata.get("loader_cls",
                                   DTLLoader if use_dtl else BufferLoader)
    loader = loader_cls(ctx, tile, problem)
    swizzle = ctx._metadata.get("swizzle", None)
    reader = LDSReader(ctx, tile, problem, swizzle=swizzle)

    # Store reader in metadata for MFMABlock emit closures
    ctx._metadata["_reader"] = reader

    scale_loader = None
    use_real_scales = ctx._metadata.get("use_real_scales", False)
    if use_real_scales and tile.mfma.is_mx:
        from ..memory.scale_loader import VMEMScaleLoader
        swizzled = ctx._metadata.get("swizzled_scales", False)
        scale_loader = VMEMScaleLoader(ctx, tile, swizzled=swizzled)

    # Build dependency graph
    graph = KLoopGraph(tile, problem)
    GlobalLoadBlock(loader).register(graph)
    DSReadBlock(reader).register(graph)
    if scale_loader:
        ScaleBlock(scale_loader, tile).register(graph)
    MFMABlock(ctx, tile, scale_loader).register(graph)
    SuffixBlock(reader, scale_loader, loader).register(graph)
    graph.validate()

    # Schedule
    scheduler = KLoopScheduler(graph)
    result = scheduler.schedule()

    # === Emit assembly ===
    elem = problem.element_bytes
    lds_data_half = int((tile.wg_m + tile.wg_n) * tile.unroll_k * elem)
    log2_uk = int(math.log2(tile.unroll_k))

    # DB step register
    ctx.alloc_sgpr_permanent(1, "s_lds_db_step")

    ctx.comment("=== Scheduled K-loop ===")
    ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
               comment=f"k_tiles = K / {tile.unroll_k}")
    ctx.alloc_sgpr_permanent(1, "s_rd_db")
    ctx.s_mov(ctx.sreg("s_rd_db"), "0", comment="rd_db = 0 (read from buffer 0)")
    ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_data_half),
              comment=f"DB step = {lds_data_half}")
    ctx.raw("")

    # Precompute offsets
    loader.precompute_soffsets()
    if scale_loader:
        scale_loader.precompute_soffsets()

    # PGR=2: prefetch 2 tiles in prologue for better latency hiding
    pgr2 = ctx._metadata.get("pgr2", False)

    # Prologue: load first tile
    ctx.comment("Prologue: load tile 0")
    loader.emit_loads()
    if result.prologue_scale_ops:
        for op in result.prologue_scale_ops:
            if op.emit:
                op.emit()
        extra = len(result.prologue_scale_ops)
        ctx.s_waitcnt(f"vmcnt({extra})",
                      comment=f"wait DTL (leave {extra} scale loads)")
    else:
        ctx.s_waitcnt("vmcnt(0)", comment="wait DTL loads")
    ctx.s_barrier(comment="sync first tile")
    ctx.raw("")

    if pgr2:
        # Prefetch tile 1 into the other LDS buffer (skip if K = unroll_k)
        ctx.comment("PGR2: prefetch tile 1")
        ctx.inst("s_cmp_le_u32", ctx.sreg("s_k_tiles"), "1",
                 comment="skip prefetch if only 1 tile")
        ctx.inst("s_cbranch_scc1", "pgr2_skip",
                 comment="skip if k_tiles <= 1")
        loader.advance()
        loader.toggle_write()
        loader.emit_loads()
        ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
                  comment="k_tiles-- (prefetched tile 1)")
        ctx.label("pgr2_skip")
        ctx.raw("")

    # ============== K-loop body ==============
    ctx.label("k_loop")
    ctx.raw("")

    # Pre-body: early B reads (overlap with arriving loads)
    ctx.comment("Early B reads (overlap with loads)")
    for op in result.pre_body_ops:
        if op.emit:
            op.emit()

    # Conditional next-tile load
    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="more tiles?")
    ctx.inst("s_cbranch_scc0", "load_skip_all",
             comment="skip loads on last iter")

    for op in result.prefetch_ops:
        if op.emit:
            op.emit()

    if result.scale_advance_op and result.scale_advance_op.emit:
        result.scale_advance_op.emit()
    # Swizzled mode: re-emit scale loads for next iteration
    if scale_loader and not scale_loader.has_cross_iter_prefetch:
        for op in result.prologue_scale_ops:
            if op.emit:
                op.emit()

    ctx.raw("")
    ctx.label("load_skip_all")

    loader.emit_sync()
    ctx.raw("")

    # Preamble reads
    ctx.comment("Preamble: A[m0] + B ki=1")
    reader = ctx._metadata.get("_reader")
    _needs_recompute = reader and not getattr(reader, '_precomputed_swizzle', False)
    for op in result.preamble_ops:
        # B ki>0 reads in the preamble need per-ni recompute because
        # all B ki=0 reads (pre_body) ran first, leaving the base
        # register set to the last ni's address.
        if (_needs_recompute and op.name.startswith("read_b_") and "_k0" not in op.name):
            import re as _re
            _m = _re.search(r"read_b_n(\d+)", op.name)
            if _m:
                reader.emit_recompute_b_for_ni(int(_m.group(1)))
        if op.emit:
            op.emit()

    # Compute preamble inflight count
    nr = tile.mfma_n_repeat
    ki_count = tile.k_iterations
    preamble_inflight = nr + 1  # B[ki=0] + A[m0,k0]
    if ki_count > 1:
        preamble_inflight += nr + 1

    first_batch = nr + 1
    remaining = preamble_inflight - first_batch
    wait_cnt = min(remaining, 15)
    ctx.s_waitcnt(f"lgkmcnt({wait_cnt})",
                  comment="wait B[ki=0] + A[m0,k0]")

    if result.prologue_scale_ops:
        num_dtl = loader.num_inflight if hasattr(loader, "num_inflight") else 0
        ctx.s_waitcnt(f"vmcnt({num_dtl})",
                      comment=f"wait scales (leave {num_dtl} DTL in-flight)")
    ctx.raw("")

    # Recompute per-ki LDS read bases
    reader.emit_recompute_ki_bases()
    ctx.raw("")

    # Emit scheduled MFMA body
    mr = tile.mfma_m_repeat
    mfmas_per_mi = nr * ki_count
    partition_m = 4
    inflight_lgkm = preamble_inflight
    mfma_count = 0

    for i, mfma_op in enumerate(result.mfma_order):
        # Auto-wait from scheduler
        if i in result.waits:
            ctx.s_waitcnt(result.waits[i],
                          comment=f"auto-wait before MFMA[{i}]")
            inflight_lgkm = 0

        # Wait for preamble B[ki=1] before mi=0 ki=1
        if mfma_count == nr and inflight_lgkm > 0:
            ctx.s_waitcnt("lgkmcnt(0)",
                          comment="wait B[ki=1] + A[m0,k1]")
            inflight_lgkm = 0

        # Wait at mi boundaries for A prefetch
        if (mfma_count > 0 and mfma_count % mfmas_per_mi == 0
                and inflight_lgkm > 0):
            ctx.s_waitcnt("lgkmcnt(0)",
                          comment=f"wait A[m{mfma_count // mfmas_per_mi}]")
            inflight_lgkm = 0

        # Scale subtile boundary vmcnt wait (auto-placed by scheduler)
        if result.prologue_scale_ops:
            mps = partition_m * mfmas_per_mi
            n_st = mr // partition_m
            if mfma_count > 0 and mfma_count % mps == 0:
                st_idx = mfma_count // mps
                if st_idx < n_st:
                    num_dtl = loader.num_inflight if hasattr(loader, "num_inflight") else 0
                    ctx.s_waitcnt(f"vmcnt({num_dtl})",
                                  comment=f"wait scale_a subtile {st_idx} (leave DTL)")

        # Partition comment
        if mfma_count % (partition_m * mfmas_per_mi) == 0:
            ctx.comment(
                f"--- Partition {mfma_count // (partition_m * mfmas_per_mi)} ---")

        # Side ops (reads, suffix)
        for op in result.side_ops[i]:
            if op.emit:
                op.emit()
            if op.kind == OpKind.DS_READ:
                inflight_lgkm += 1

        # MFMA
        if mfma_op.emit:
            mfma_op.emit()
        mfma_count += 1

    # Epilogue ops
    for op in result.epilogue_ops:
        if op.emit:
            op.emit()

    # Postamble
    ctx.s_barrier(comment="sync")
    has_pf = (scale_loader
              and scale_loader.has_cross_iter_prefetch)
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="more?")
    if has_pf:
        ctx.inst("s_cbranch_scc0", "k_loop_end",
                 comment="exit if last")
        ctx.inst("s_branch", "k_loop", comment="loop back")
        ctx.label("k_loop_end")
    else:
        ctx.inst("s_cbranch_scc1", "k_loop", comment="loop")
    ctx.raw("")
