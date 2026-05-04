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
]


class OpKind(Enum):
    MFMA = "mfma"
    DS_READ = "ds_read"
    DS_WRITE = "ds_write"
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
            if self.kind in (OpKind.DS_READ, OpKind.DS_WRITE):
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

