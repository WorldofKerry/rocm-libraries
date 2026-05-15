"""Build a KLoopGraph from a list of LDSStream objects.

Replaces the separate GlobalLoadBlock, DSReadBlock, ScaleBlock,
MFMABlock, and SuffixBlock with a single ``build_kloop_graph``
function that derives the full dependency graph from stream metadata.
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from .kloop_graph import KLoopGraph, KLoopOp, OpKind, DepKind

if TYPE_CHECKING:
    from ..memory.lds_stream import LDSStream
    from ..problem import TileConfig, GemmProblem

__all__ = ["build_kloop_graph"]


def build_kloop_graph(
    streams: List['LDSStream'],
    tile: 'TileConfig',
    pgr: int = 1,
    num_buffers: int = 2,
    mfma_emitter: Optional[object] = None,
    problem: Optional['GemmProblem'] = None,
) -> KLoopGraph:
    """Construct a KLoopGraph from a list of LDS streams.

    The graph captures all producer (global-load) and consumer
    (ds-read, MFMA, suffix) operations with their dependency edges.
    Emit callbacks are placeholder ``None`` -- the graph structure
    (ops + deps) is the deliverable for this phase.

    Args:
        streams: LDSStream instances (data and/or scale).
        tile: Tile configuration for MFMA geometry.
        pgr: Prefetch global-read depth.  Producer ops get
             ``iteration=pgr``; 0 means no prefetch.
        num_buffers: Number of LDS ping-pong buffers (controls WAR
                     dep distance for A-matrix reads).
        mfma_emitter: Reserved for future emit wiring (unused).
        problem: Optional GemmProblem (stored on the graph).

    Returns:
        A fully-wired KLoopGraph ready for scheduling.
    """
    graph = KLoopGraph(tile, problem)

    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    ki_count = tile.k_iterations

    # Classify streams by role based on name convention.
    data_streams: List['LDSStream'] = []
    scale_streams: List['LDSStream'] = []
    for s in streams:
        if s.name.startswith("scale_"):
            scale_streams.append(s)
        elif s.name.startswith("data_"):
            data_streams.append(s)
        # NullStreams and unknowns: still get producer/suffix ops
        # but no consumer reads.

    # ------------------------------------------------------------------
    # 1. Producer ops  (iteration = pgr)
    # ------------------------------------------------------------------
    # Only create producer ops (advance/toggle/load) for streams that
    # actually load from global memory.  LDSInputStream has
    # num_global_loads=0 and skips the entire producer chain.
    terminal_producers: List[str] = []  # last op per stream's chain

    for s in streams:
        if s.num_global_loads == 0:
            # No producer ops -- data already in LDS (e.g. LDSInputStream)
            continue

        adv = f"advance_{s.name}"
        tog = f"toggle_wr_{s.name}"
        ld  = f"load_{s.name}"

        graph.add_op(KLoopOp(adv, OpKind.SCALAR, emit=None,
                             iteration=pgr,
                             comment=f"advance {s.name} SRD"))
        graph.add_op(KLoopOp(tog, OpKind.SCALAR, emit=None,
                             iteration=pgr,
                             comment=f"toggle write ptr {s.name}"))
        graph.add_op(KLoopOp(ld, OpKind.GLOBAL_LOAD, emit=None,
                             iteration=pgr,
                             comment=f"global load {s.name}"))

        # ORDER: advance -> toggle_wr -> load
        graph.add_dep(adv, tog, DepKind.RAW)
        graph.add_dep(tog, ld, DepKind.RAW)

        last = ld  # default terminal is the load

        if s.needs_lds_write:
            wr = f"write_{s.name}"
            graph.add_op(KLoopOp(wr, OpKind.DS_WRITE, emit=None,
                                 iteration=pgr,
                                 comment=f"ds_write {s.name}"))
            graph.add_dep(ld, wr, DepKind.RAW)
            last = wr

        terminal_producers.append(last)

    # ------------------------------------------------------------------
    # 2. Barrier  (iteration = 0)
    # ------------------------------------------------------------------
    graph.add_op(KLoopOp("barrier", OpKind.BARRIER, emit=None,
                         comment="sync workgroup"))
    for tp in terminal_producers:
        graph.add_dep(tp, "barrier", DepKind.SYNC)

    # ------------------------------------------------------------------
    # 3. Consumer ops -- data stream reads  (iteration = 0)
    # ------------------------------------------------------------------
    for s in data_streams:
        matrix = s.name.split("_", 1)[1]  # "a" or "b"
        is_a = (matrix == "a")

        if is_a:
            # A-matrix: indexed by (mi, ki), ping-pong buffer = mi % num_buffers
            for mi in range(mr):
                for ki in range(ki_count):
                    rname = f"read_{s.name}_m{mi}_k{ki}"
                    graph.add_op(KLoopOp(
                        rname, OpKind.DS_READ, emit=None,
                        comment=f"ds_read {s.name} m{mi} k{ki}"))

                    # SYNC: barrier -> read
                    graph.add_dep("barrier", rname, DepKind.SYNC)

                    # WAR: ping-pong -- read at mi reuses buffer
                    # from mi - num_buffers
                    if mi >= num_buffers:
                        prior_mi = mi - num_buffers
                        last_mfma = (
                            f"mfma_m{prior_mi}_n{nr - 1}_k{ki_count - 1}"
                        )
                        graph.add_dep(last_mfma, rname, DepKind.WAR)
        else:
            # B-matrix: indexed by (ni, ki), no ping-pong
            for ni in range(nr):
                for ki in range(ki_count):
                    rname = f"read_{s.name}_n{ni}_k{ki}"
                    graph.add_op(KLoopOp(
                        rname, OpKind.DS_READ, emit=None,
                        comment=f"ds_read {s.name} n{ni} k{ki}"))

                    # SYNC: barrier -> read
                    graph.add_dep("barrier", rname, DepKind.SYNC)

    # ------------------------------------------------------------------
    # 4. Consumer ops -- scale stream reads  (iteration = 0)
    #    One read per 2-mi (or 2-ni) group for LDS-based scales.
    # ------------------------------------------------------------------
    mx_block = 2  # LDS scales group pairs of mi/ni values

    for s in scale_streams:
        matrix = s.name.split("_", 1)[1]  # "a" or "b"
        repeat = mr if matrix == "a" else nr
        num_groups = (repeat + mx_block - 1) // mx_block

        # VMEM scale streams have region_size=0 (no LDS).
        # Use SCALE_LOAD (vmcnt) for VMEM, DS_READ (lgkmcnt) for LDS.
        scale_op_kind = OpKind.SCALE_LOAD if s.region_size == 0 else OpKind.DS_READ

        for g in range(num_groups):
            rname = f"read_{s.name}_g{g}"
            graph.add_op(KLoopOp(
                rname, scale_op_kind, emit=None,
                comment=f"scale_read {s.name} group {g}"))

            # SYNC: barrier -> read
            graph.add_dep("barrier", rname, DepKind.SYNC)

            # RAW: read -> all MFMAs in this group
            mi_lo = g * mx_block
            mi_hi = min(mi_lo + mx_block, repeat)
            if matrix == "a":
                for mi2 in range(mi_lo, mi_hi):
                    for ki2 in range(ki_count):
                        for ni in range(nr):
                            graph.add_dep(
                                rname,
                                f"mfma_m{mi2}_n{ni}_k{ki2}",
                                DepKind.RAW)
            else:  # "b"
                for ni2 in range(mi_lo, mi_hi):
                    for ki2 in range(ki_count):
                        for mi in range(mr):
                            graph.add_dep(
                                rname,
                                f"mfma_m{mi}_n{ni2}_k{ki2}",
                                DepKind.RAW)

    # ------------------------------------------------------------------
    # 5. MFMA ops  (iteration = 0)
    # ------------------------------------------------------------------
    for mi in range(mr):
        for ki in range(ki_count):
            for ni in range(nr):
                mname = f"mfma_m{mi}_n{ni}_k{ki}"
                # TODO: emit callback will be wired in Phase 4.
                graph.add_op(KLoopOp(
                    mname, OpKind.MFMA, emit=None,
                    comment=f"m{mi}_n{ni}_k{ki}"))

                # RAW deps from data reads
                for s in data_streams:
                    mat = s.name.split("_", 1)[1]
                    if mat == "a":
                        graph.add_dep(
                            f"read_{s.name}_m{mi}_k{ki}",
                            mname, DepKind.RAW, min_cycles=20)
                    else:
                        graph.add_dep(
                            f"read_{s.name}_n{ni}_k{ki}",
                            mname, DepKind.RAW, min_cycles=20)

    # ------------------------------------------------------------------
    # 6. Suffix ops  (iteration = 0)
    # ------------------------------------------------------------------
    last_mfma = f"mfma_m{mr - 1}_n{nr - 1}_k{ki_count - 1}"

    for s in streams:
        if s.num_global_loads == 0:
            # LDS-input streams don't toggle (no double-buffer needed)
            continue

        tname = f"toggle_rd_{s.name}"
        graph.add_op(KLoopOp(
            tname, OpKind.SCALAR, emit=None,
            comment=f"toggle read ptr {s.name}"))

        # ORDER: all MFMAs (represented by the very last MFMA) -> toggle_rd
        # Using the last MFMA is sufficient since MFMA order is fixed.
        graph.add_dep(last_mfma, tname, DepKind.RAW)

    return graph
