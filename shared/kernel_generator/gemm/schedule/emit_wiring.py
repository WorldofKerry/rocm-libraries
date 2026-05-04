"""Wire emit callbacks onto KLoopGraph ops.

Takes a KLoopGraph (from graph_builder) and connects each op's emit
callback to the actual codegen functions on LDSStreams, LDSReader,
GlobalLoader, and MFMAEmitter.

This is the bridge between the new stream-based graph and the existing
assembly emission code. It allows the new PipelineScheduler/Emitter to
produce real assembly using the existing tested codegen primitives.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..memory.lds_stream import LDSStream, LDSBufferManager
    from ..memory.lds_reader import LDSReader
    from ..memory.global_loader import GlobalLoader
    from ..memory.mfma_emitter import MFMAEmitter
    from ..emit.context import AsmContext
    from .kloop_graph import KLoopGraph

__all__ = ["wire_emit_callbacks"]


def wire_emit_callbacks(
    graph: 'KLoopGraph',
    streams: list,
    buffer_mgr: 'LDSBufferManager',
    loader: 'GlobalLoader',
    reader: 'LDSReader',
    mfma_emitter: 'MFMAEmitter',
    ctx: 'AsmContext',
    scale_loader: Optional[object] = None,
) -> None:
    """Connect graph op emit callbacks to actual codegen functions.

    Mutates the graph in-place, setting ``op.emit`` on each op.

    Args:
        graph: KLoopGraph with ops that have ``emit=None``.
        streams: List of LDSStream instances.
        buffer_mgr: LDSBufferManager for barrier/negate.
        loader: GlobalLoader (DTLLoader or BufferLoader).
        reader: LDSReader for data reads.
        mfma_emitter: MFMAEmitter for MFMA instructions.
        ctx: AsmContext for register resolution.
        scale_loader: Optional LDSScaleLoader for scale reads.
    """
    tile = ctx._metadata["tile"]
    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    ki_count = tile.k_iterations

    stream_map = {s.name: s for s in streams}

    for op_name, op in graph.ops.items():
        # -- Producer ops --
        if op_name.startswith("advance_"):
            sname = op_name[len("advance_"):]
            if sname in stream_map:
                s = stream_map[sname]
                op.emit = lambda s=s: s.advance(ctx)

        elif op_name.startswith("toggle_wr_"):
            sname = op_name[len("toggle_wr_"):]
            if sname in stream_map:
                s = stream_map[sname]
                op.emit = lambda s=s: s.toggle_write(ctx)

        elif op_name.startswith("load_"):
            sname = op_name[len("load_"):]
            if sname in stream_map:
                s = stream_map[sname]
                op.emit = lambda s=s: s.emit_global_loads(ctx)

        elif op_name.startswith("write_"):
            sname = op_name[len("write_"):]
            if sname in stream_map:
                s = stream_map[sname]
                op.emit = lambda s=s: s.emit_lds_writes(ctx)

        # -- Barrier --
        elif op_name == "barrier":
            op.emit = lambda: buffer_mgr.emit_barrier(ctx)

        # -- Data reads --
        elif op_name.startswith("read_data_a_"):
            # read_data_a_m{mi}_k{ki}
            parts = op_name.replace("read_data_a_m", "").replace("_k", " ").split()
            mi, ki = int(parts[0]), int(parts[1])
            buf = mi % 2  # ping-pong buffer index
            op.emit = lambda mi=mi, ki=ki, buf=buf: reader.emit_read_a(mi, ki, buf)

        elif op_name.startswith("read_data_b_"):
            # read_data_b_n{ni}_k{ki}
            parts = op_name.replace("read_data_b_n", "").replace("_k", " ").split()
            ni, ki = int(parts[0]), int(parts[1])
            op.emit = lambda ni=ni, ki=ki: reader.emit_read_b(ni, ki)

        # -- Scale reads --
        elif op_name.startswith("read_scale_a_"):
            # read_scale_a_g{group}
            group = int(op_name.replace("read_scale_a_g", ""))
            mi = group * 2  # primary mi for this group
            if scale_loader is not None:
                op.emit = lambda mi=mi: scale_loader.emit_read_a(mi, 0)

        elif op_name.startswith("read_scale_b_"):
            group = int(op_name.replace("read_scale_b_g", ""))
            ni = group * 2
            if scale_loader is not None:
                op.emit = lambda ni=ni: scale_loader.emit_read_b(ni, 0)

        # -- MFMAs --
        elif op_name.startswith("mfma_"):
            # mfma_m{mi}_n{ni}_k{ki}
            parts = op_name.replace("mfma_m", "").replace("_n", " ").replace("_k", " ").split()
            mi, ni, ki = int(parts[0]), int(parts[1]), int(parts[2])
            buf = mi % 2
            a_names = reader.a_names
            b_names = reader.b_names
            a_reg = ctx.vreg(a_names[(buf, ki)])
            b_reg = ctx.vreg(b_names[(ni, ki)])
            acc_start = (mi * nr + ni) * tile.mfma.acc_vgprs
            acc = ctx.areg("acc_C", acc_start, tile.mfma.acc_vgprs)

            op.emit = lambda acc=acc, a_reg=a_reg, b_reg=b_reg, mi=mi, ni=ni, ki=ki: \
                mfma_emitter.emit(ctx, acc, a_reg, b_reg, mi, ni, ki)

        # -- Suffix toggles --
        elif op_name.startswith("toggle_rd_"):
            sname = op_name[len("toggle_rd_"):]
            if sname in stream_map:
                s = stream_map[sname]
                op.emit = lambda s=s: s.toggle_read(ctx)
