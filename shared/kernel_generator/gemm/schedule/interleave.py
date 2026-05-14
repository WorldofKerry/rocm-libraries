"""Interleaving helpers for K-loop emission.

Extracts the repeated pattern of spreading N load ops evenly among
M MFMA ops.  Used by both ``_emit_copy_interleaved`` and
``_emit_copy_ki_phased`` in ``pipeline_emitter.py``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from .kloop_graph import KLoopOp, OpKind

if TYPE_CHECKING:
    from ..emit.context import AsmContext
    from ..memory.global_loader import GlobalLoader

__all__ = [
    "classify_body_ops",
    "build_dtl_sequence",
    "emit_mfmas_with_dtl_interleaved",
    "emit_mfmas_with_reads_and_dtl",
    "emit_mfmas_with_reads_interleaved",
]


def classify_body_ops(
    body: List[KLoopOp],
    producer_start: int,
) -> dict:
    """Classify body ops into named groups.

    Returns dict with keys:
        barrier, ki0_reads, ki1_reads, ki0_mfmas, ki1_mfmas,
        suffix, scalar_prods, load_prods, all_consumers, all_producers
    """
    consumers = body[:producer_start]
    producers = body[producer_start:]

    result = {
        "barrier": None,
        "ki0_reads": [],
        "ki1_reads": [],
        "ki0_mfmas": [],
        "ki1_mfmas": [],
        "suffix": [],
        "scalar_prods": [],
        "load_prods": [],
    }

    for op in consumers:
        if op.kind == OpKind.BARRIER:
            result["barrier"] = op
        elif op.kind in (OpKind.DS_READ, OpKind.SCALE_LOAD):
            if "_k1" in op.name:
                result["ki1_reads"].append(op)
            else:
                result["ki0_reads"].append(op)
        elif op.kind == OpKind.MFMA:
            if "_k1" in op.name:
                result["ki1_mfmas"].append(op)
            else:
                result["ki0_mfmas"].append(op)
        else:
            result["suffix"].append(op)

    for op in producers:
        if op.kind == OpKind.SCALAR:
            result["scalar_prods"].append(op)
        elif op.kind == OpKind.GLOBAL_LOAD:
            result["load_prods"].append(op)
        elif op.kind == OpKind.DS_WRITE:
            # Scale LDS writes: must happen after the corresponding
            # global load completes.  Grouped with load_prods so
            # they are emitted together in the producer phase.
            result["load_prods"].append(op)

    return result


def build_dtl_sequence(
    load_prods: List[KLoopOp],
    loader: 'GlobalLoader',
) -> List[Tuple[str, object]]:
    """Build an ordered sequence of individual DTL load operations.

    Returns list of (kind, payload) tuples:
        ('scale_op', KLoopOp)  -- scale global load op
        ('dtl_a', int)         -- A matrix DTL load index
        ('dtl_b', int)         -- B matrix DTL load index

    Scale ops come first (they have longer latency), then A/B
    loads are alternated.
    """
    seq: List[Tuple[str, object]] = []

    # Scale loads first
    for op in load_prods:
        if "scale" in op.name:
            seq.append(("scale_op", op))

    # Alternate A and B DTL loads
    n_a = loader.num_loads_a
    n_b = loader.num_loads_b
    for j in range(max(n_a, n_b)):
        if j < n_a:
            seq.append(("dtl_a", j))
        if j < n_b:
            seq.append(("dtl_b", j))

    return seq


def emit_dtl_load(
    kind: str,
    payload: object,
    loader: 'GlobalLoader',
    emit_op_fn: object,
    m0_state: dict,
) -> None:
    """Emit a single DTL load operation.

    Args:
        kind: 'scale_op', 'dtl_a', or 'dtl_b'
        payload: KLoopOp (for scale) or int index (for dtl_a/b)
        loader: GlobalLoader with emit_dtl_* methods
        emit_op_fn: callable to emit a KLoopOp
        m0_state: dict tracking {'a': bool, 'b': bool} for m0 setup
    """
    if kind == "scale_op":
        emit_op_fn(payload)
    elif kind == "dtl_a":
        if not m0_state.get("a"):
            loader.emit_dtl_m0_a()
            m0_state["a"] = True
        loader.emit_dtl_load_a_single(payload)
    elif kind == "dtl_b":
        if not m0_state.get("b"):
            loader.emit_dtl_m0_b()
            m0_state["b"] = True
        loader.emit_dtl_load_b_single(payload)


def emit_mfmas_with_dtl_interleaved(
    ctx: 'AsmContext',
    mfma_ops: List[KLoopOp],
    dtl_seq: List[Tuple[str, object]],
    loader: 'GlobalLoader',
    emit_op_fn: object,
    comment: str = "",
) -> None:
    """Emit MFMA ops with DTL loads spread evenly among them.

    Args:
        ctx: Assembly context (for comments).
        mfma_ops: MFMA ops to emit.
        dtl_seq: DTL load sequence from ``build_dtl_sequence``.
        loader: GlobalLoader for per-line DTL emission.
        emit_op_fn: callable(KLoopOp) to emit one op.
        comment: Optional section comment.
    """
    n_mfma = len(mfma_ops)
    total_loads = len(dtl_seq)

    if comment:
        ctx.comment(f"{comment} ({n_mfma} MFMAs + {total_loads} DTL)")

    interval = max(1, n_mfma // (total_loads + 1)) if total_loads > 0 else n_mfma + 1
    load_idx = 0
    m0_state: dict = {}

    for i, mfma_op in enumerate(mfma_ops):
        emit_op_fn(mfma_op)
        if (i + 1) % interval == 0 and load_idx < total_loads:
            kind, payload = dtl_seq[load_idx]
            emit_dtl_load(kind, payload, loader, emit_op_fn, m0_state)
            load_idx += 1

    # Remaining loads after all MFMAs
    while load_idx < total_loads:
        kind, payload = dtl_seq[load_idx]
        emit_dtl_load(kind, payload, loader, emit_op_fn, m0_state)
        load_idx += 1


def emit_mfmas_with_reads_and_dtl(
    ctx: 'AsmContext',
    mfma_ops: List[KLoopOp],
    read_ops: List[KLoopOp],
    dtl_seq: List[Tuple[str, object]],
    loader: 'GlobalLoader',
    emit_op_fn: object,
    comment: str = "",
) -> None:
    """Emit MFMAs with both ds_reads and DTL loads spread evenly.

    Reads and DTL loads are independently spaced among the MFMAs
    using separate intervals (they use different HW counters:
    lgkmcnt for reads, vmcnt for DTL loads).

    Args:
        ctx: Assembly context (for comments).
        mfma_ops: MFMA ops to emit.
        read_ops: ds_read ops to interleave (lgkmcnt).
        dtl_seq: DTL load sequence from ``build_dtl_sequence`` (vmcnt).
        loader: GlobalLoader for per-line DTL emission.
        emit_op_fn: callable(KLoopOp) to emit one op.
        comment: Optional section comment.
    """
    n_mfma = len(mfma_ops)
    n_reads = len(read_ops)
    n_dtl = len(dtl_seq)

    if comment:
        ctx.comment(f"{comment} ({n_mfma} MFMAs + {n_reads} reads + {n_dtl} DTL)")

    if n_mfma == 0:
        for r in read_ops:
            emit_op_fn(r)
        return

    # Build merged schedule: each event has a fractional position
    # indicating when it should fire relative to the MFMA sequence.
    # Reads and DTL loads are independently spaced.
    events: List[Tuple[float, str, int]] = []  # (position, kind, index)

    if n_reads > 0:
        step = n_mfma / (n_reads + 1)
        for j in range(n_reads):
            events.append((step * (j + 1), 'read', j))

    if n_dtl > 0:
        step = n_mfma / (n_dtl + 1)
        for j in range(n_dtl):
            events.append((step * (j + 1), 'dtl', j))

    events.sort(key=lambda e: e[0])

    m0_state: dict = {}
    event_idx = 0

    for i, mfma_op in enumerate(mfma_ops):
        emit_op_fn(mfma_op)
        # Emit all events whose target position is <= current MFMA index+1
        while event_idx < len(events) and events[event_idx][0] <= i + 1:
            _, kind, idx = events[event_idx]
            if kind == 'read':
                emit_op_fn(read_ops[idx])
            else:
                k, payload = dtl_seq[idx]
                emit_dtl_load(k, payload, loader, emit_op_fn, m0_state)
            event_idx += 1

    # Remaining events after all MFMAs
    while event_idx < len(events):
        _, kind, idx = events[event_idx]
        if kind == 'read':
            emit_op_fn(read_ops[idx])
        else:
            k, payload = dtl_seq[idx]
            emit_dtl_load(k, payload, loader, emit_op_fn, m0_state)
        event_idx += 1


def emit_mfmas_with_reads_interleaved(
    mfma_ops: List[KLoopOp],
    read_ops: List[KLoopOp],
    emit_op_fn: object,
    comment: str = "",
    ctx: Optional['AsmContext'] = None,
) -> None:
    """Emit MFMA ops with ds_read ops spread evenly among them.

    Used for interleaving ki=1 reads among ki=0 MFMAs, and
    next-copy ki=0 reads among ki=1 post-barrier MFMAs.
    """
    n_mfma = len(mfma_ops)
    n_reads = len(read_ops)

    if comment and ctx:
        ctx.comment(f"{comment} ({n_mfma} MFMAs + {n_reads} reads)")

    interval = max(1, n_mfma // (n_reads + 1)) if n_reads > 0 else n_mfma + 1
    read_idx = 0

    for i, mfma_op in enumerate(mfma_ops):
        emit_op_fn(mfma_op)
        if (i + 1) % interval == 0 and read_idx < n_reads:
            emit_op_fn(read_ops[read_idx])
            read_idx += 1

    while read_idx < n_reads:
        emit_op_fn(read_ops[read_idx])
        read_idx += 1
