"""Dependency graph for K-loop operations.

Building blocks register operations and dependencies into a KLoopGraph.
The KLoopScheduler then consumes the graph to produce an instruction
sequence with auto-placed waits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

from ..emit.context import AsmContext
from ..problem import TileConfig, GemmProblem, MfmaConfig

__all__ = [
    "OpKind", "DepKind", "KLoopOp", "Dep", "KLoopGraph",
    "BuildingBlock", "MFMABlock", "DSReadBlock", "GlobalLoadBlock", "SuffixBlock",
]


class OpKind(Enum):
    MFMA = "mfma"
    DS_READ = "ds_read"
    GLOBAL_LOAD = "global_load"
    BARRIER = "barrier"
    SCALAR = "scalar"
    SCALE_LOAD = "scale_load"
    WAIT = "wait"


class DepKind(Enum):
    RAW = "raw"    # Read-After-Write: consumer reads producer's output
    WAR = "war"    # Write-After-Read: consumer overwrites producer's input
    SYNC = "sync"  # Ordering via barrier/waitcnt


@dataclass
class KLoopOp:
    """A single operation in the K-loop."""
    name: str
    kind: OpKind
    emit: Optional[Callable[[], None]] = None
    iteration: int = 0  # 0 = current iter, 1 = next iter (prefetch)
    comment: str = ""

    # Hardware counter this op affects (for auto-wait insertion)
    # "lgkmcnt" for ds_read/ds_write, "vmcnt" for global_load/scale_load
    hw_counter: Optional[str] = None

    def __post_init__(self):
        if self.hw_counter is None:
            if self.kind == OpKind.DS_READ:
                self.hw_counter = "lgkmcnt"
            elif self.kind in (OpKind.GLOBAL_LOAD, OpKind.SCALE_LOAD):
                self.hw_counter = "vmcnt"


@dataclass
class Dep:
    """Dependency edge between two operations."""
    producer: str
    consumer: str
    kind: DepKind
    min_cycles: int = 0  # minimum cycles between producer issue and consumer issue


class KLoopGraph:
    """Dependency DAG for one K-loop iteration.

    Ops from iteration=0 (current) and iteration=1 (next-iter prefetch)
    coexist in the graph. The scheduler uses iteration tags to determine
    cross-iteration overlap and conditional skip on last iteration.
    """

    def __init__(self, tile: TileConfig, problem: GemmProblem) -> None:
        self.tile = tile
        self.problem = problem
        self.ops: Dict[str, KLoopOp] = {}
        self.deps: List[Dep] = []

    def add_op(self, op: KLoopOp) -> None:
        if op.name in self.ops:
            raise ValueError(f"Duplicate op name: {op.name}")
        self.ops[op.name] = op

    def add_dep(self, producer: str, consumer: str,
                kind: DepKind = DepKind.RAW,
                min_cycles: int = 0) -> None:
        self.deps.append(Dep(producer, consumer, kind, min_cycles))

    def predecessors(self, op_name: str) -> List[str]:
        return [d.producer for d in self.deps if d.consumer == op_name]

    def successors(self, op_name: str) -> List[str]:
        return [d.consumer for d in self.deps if d.producer == op_name]

    def mfma_ops(self) -> List[KLoopOp]:
        """All MFMA ops in insertion order."""
        return [op for op in self.ops.values() if op.kind == OpKind.MFMA]

    def ds_read_ops(self) -> List[KLoopOp]:
        """All ds_read ops."""
        return [op for op in self.ops.values() if op.kind == OpKind.DS_READ]

    def side_ops(self) -> List[KLoopOp]:
        """All non-MFMA ops."""
        return [op for op in self.ops.values() if op.kind != OpKind.MFMA]

    def validate(self) -> None:
        """Check that all dep references exist."""
        for dep in self.deps:
            if dep.producer not in self.ops:
                raise ValueError(
                    f"Dep references unknown producer: {dep.producer}")
            if dep.consumer not in self.ops:
                raise ValueError(
                    f"Dep references unknown consumer: {dep.consumer}")


# ===================================================================
# Building block protocol
# ===================================================================

class BuildingBlock:
    """Interface for K-loop components that declare ops and deps."""

    def register(self, graph: KLoopGraph) -> None:
        raise NotImplementedError


# ===================================================================
# Concrete building blocks
# ===================================================================

class MFMABlock(BuildingBlock):
    """Declares MFMA operations with RAW deps on A/B operands.

    Also declares WAR deps for A ping-pong: mi=N+2 can't overwrite
    buf until mi=N's last MFMA finishes reading it.
    """

    def __init__(self, ctx: AsmContext, tile: TileConfig,
                 scale_loader: object = None) -> None:
        self.ctx = ctx
        self.tile = tile
        self.mfma = tile.mfma
        self.scale_loader = scale_loader

    def register(self, graph: KLoopGraph) -> None:
        mr = self.tile.mfma_m_repeat
        nr = self.tile.mfma_n_repeat
        ki_count = self.tile.k_iterations
        mfma = self.mfma
        ctx = self.ctx
        sl = self.scale_loader

        for mi in range(mr):
            buf = mi % 2
            for ki in range(ki_count):
                for ni in range(nr):
                    name = f"mfma_m{mi}_n{ni}_k{ki}"
                    acc_per = mfma.acc_vgprs
                    acc_off = (mi * nr + ni) * acc_per

                    def _mk_emit(mi_=mi, ni_=ni, ki_=ki,
                                 aoff=acc_off, aper=acc_per,
                                 buf_=buf):
                        def emit():
                            reader = ctx._metadata.get("_reader")
                            a_reg = ctx.vreg(
                                reader.a_names[(buf_, ki_)], 0, reader.av)
                            b_reg = ctx.vreg(
                                reader.b_names[(ni_, ki_)], 0, reader.bv)
                            acc = ctx.areg("acc_C", aoff, aper)

                            if sl and sl.has_scales:
                                sl.emit_mfma(ctx, mfma, acc, a_reg,
                                             b_reg, mi_, ni_, ki_)
                            elif mfma.is_mx:
                                ctx.inst(
                                    mfma.instruction_name, acc,
                                    a_reg, b_reg, acc,
                                    ctx.vreg("v_mxscale"),
                                    ctx.vreg("v_mxscale"),
                                    f"cbsz:{mfma.cbsz} blgp:{mfma.blgp}",
                                    comment=f"MFMA m{mi_}_n{ni_}_k{ki_}")
                            else:
                                ctx.inst(
                                    mfma.instruction_name, acc,
                                    a_reg, b_reg, acc,
                                    comment=f"MFMA m{mi_}_n{ni_}_k{ki_}")
                        return emit

                    graph.add_op(KLoopOp(
                        name=name, kind=OpKind.MFMA,
                        emit=_mk_emit(),
                        comment=f"m{mi}_n{ni}_k{ki}"))

                    # RAW: MFMA needs its A and B operands
                    graph.add_dep(
                        f"read_a_m{mi}_k{ki}_buf{buf}", name,
                        DepKind.RAW, min_cycles=20)
                    graph.add_dep(
                        f"read_b_n{ni}_k{ki}", name,
                        DepKind.RAW, min_cycles=20)

                    # Scale deps (if applicable)
                    if sl and sl.has_scales:
                        sa = f"scale_a_m{mi}_k{ki}"
                        sb = f"scale_b_n{ni}_k{ki}"
                        if sa in graph.ops:
                            graph.add_dep(sa, name, DepKind.RAW)
                        if sb in graph.ops:
                            graph.add_dep(sb, name, DepKind.RAW)


class DSReadBlock(BuildingBlock):
    """Declares ds_read operations with SYNC deps on barrier
    and WAR deps for A ping-pong register reuse.

    The WAR deps automatically create the partition structure:
    read_a(mi=2) can't issue until mfma(mi=0) finishes, so the
    scheduler naturally groups MFMAs by mi with A-prefetch interleaving.
    """

    def __init__(self, reader: object) -> None:
        self.reader = reader

    def register(self, graph: KLoopGraph) -> None:
        tile = graph.tile
        mr = tile.mfma_m_repeat
        nr = tile.mfma_n_repeat
        ki_count = tile.k_iterations
        reader = self.reader

        # A reads with ping-pong WAR deps
        for mi in range(mr):
            buf = mi % 2
            for ki in range(ki_count):
                name = f"read_a_m{mi}_k{ki}_buf{buf}"

                def _mk(mi_=mi, ki_=ki, buf_=buf):
                    def emit():
                        reader.emit_read_a(mi_, ki_, buf_)
                    return emit

                graph.add_op(KLoopOp(
                    name=name, kind=OpKind.DS_READ,
                    emit=_mk(),
                    comment=f"ds_read A m{mi} k{ki} buf{buf}"))

                # SYNC: must wait for barrier (LDS filled)
                graph.add_dep("barrier", name, DepKind.SYNC)

                # WAR: A ping-pong. mi reuses buffer from mi-2.
                # Can't overwrite buf until mi-2's last MFMA is done.
                if mi >= 2:
                    prior_mi = mi - 2
                    last_mfma = f"mfma_m{prior_mi}_n{nr-1}_k{ki_count-1}"
                    graph.add_dep(last_mfma, name, DepKind.WAR)

        # B reads (no ping-pong, single buffer)
        for ni in range(nr):
            for ki in range(ki_count):
                name = f"read_b_n{ni}_k{ki}"

                def _mk(ni_=ni, ki_=ki):
                    def emit():
                        reader.emit_read_b(ni_, ki_)
                    return emit

                graph.add_op(KLoopOp(
                    name=name, kind=OpKind.DS_READ,
                    emit=_mk(),
                    comment=f"ds_read B n{ni} k{ki}"))

                graph.add_dep("barrier", name, DepKind.SYNC)


class GlobalLoadBlock(BuildingBlock):
    """Declares next-iteration global loads, advance, toggle, and barrier.

    iteration=1 ops are automatically overlapped with current-iter compute
    by the scheduler. On the last K-loop iteration, iteration=1 ops are
    skipped via a conditional branch the emitter generates.
    """

    def __init__(self, loader: object) -> None:
        self.loader = loader

    def register(self, graph: KLoopGraph) -> None:
        loader = self.loader

        graph.add_op(KLoopOp(
            "advance", OpKind.SCALAR, loader.advance, iteration=1,
            comment="advance A/B ptrs"))

        graph.add_op(KLoopOp(
            "toggle", OpKind.SCALAR, loader.toggle_write, iteration=1,
            comment="toggle LDS write buffer"))

        graph.add_op(KLoopOp(
            "global_load_next", OpKind.GLOBAL_LOAD,
            loader.emit_loads, iteration=1,
            comment="DTL/global load next tile"))

        graph.add_op(KLoopOp(
            "barrier", OpKind.BARRIER,
            lambda: loader.emit_sync(),
            comment="sync workgroup"))

        # advance and toggle before load
        graph.add_dep("advance", "global_load_next", DepKind.RAW)
        graph.add_dep("toggle", "global_load_next", DepKind.RAW)

        # barrier after load completes
        graph.add_dep("global_load_next", "barrier", DepKind.SYNC)


class SuffixBlock(BuildingBlock):
    """Declares suffix ops: vmcnt wait and LDS read toggle.

    With XOR-based double-buffer toggling, no negate step is needed.
    These are placed backward from the end of the MFMA body to
    maximize overlap between current-iteration compute and
    next-iteration loads.
    """

    def __init__(self, reader: object, scale_loader: object = None,
                 loader: object = None) -> None:
        self.reader = reader
        self.scale_loader = scale_loader
        self.loader = loader

    def register(self, graph: KLoopGraph) -> None:
        reader = self.reader
        sl = self.scale_loader
        loader = self.loader

        pf_inflight = 0
        if sl and hasattr(sl, 'has_cross_iter_prefetch') and sl.has_cross_iter_prefetch:
            pf_inflight = sl.cross_iter_inflight(4, graph.tile.mfma_n_repeat)

        def _emit_vmcnt():
            reader.ctx.s_waitcnt(
                f"vmcnt({pf_inflight})",
                comment=f"wait loads (leave {pf_inflight} prefetch in-flight)")

        def _emit_toggle():
            reader.toggle_read()

        graph.add_op(KLoopOp(
            "suffix_vmcnt", OpKind.WAIT, _emit_vmcnt,
            comment="vmcnt wait for DTL loads"))

        graph.add_op(KLoopOp(
            "suffix_toggle", OpKind.SCALAR, _emit_toggle,
            comment="toggle LDS read buffer"))

        # Suffix deps: vmcnt before toggle
        graph.add_dep("suffix_vmcnt", "suffix_toggle", DepKind.RAW)

        # Suffix must happen after all MFMAs of the last partition that
        # reads from the current buffer. In practice, placed backward
        # from the end by the scheduler.
