"""Kernel pipeline: TilePartitioner + ComputePipeline + Epilogue.

Separates work distribution, compute, and output into independent
components following the CK/TensileLite/Triton pattern.

Usage::

    pipeline = KernelPipeline(
        partitioner=GridPartitioner(),
        compute=ScheduledCompute(loader, reader, scale_loader),
        epilogue=DirectEpilogue(),
    )
    pipeline.emit(ctx)
"""
from __future__ import annotations

import math
from typing import Optional

from ..emit.context import AsmContext
from ..problem import TileConfig, GemmProblem, MfmaConfig
from ..memory.global_loader import GlobalLoader, DTLLoader, BufferLoader
from ..memory.lds_reader import LDSReader
from .kloop_graph import (
    KLoopGraph, KLoopOp, OpKind, MFMABlock, DSReadBlock,
    GlobalLoadBlock, SuffixBlock,
)
from .kloop_scheduler import KLoopScheduler, ScheduledKLoop

__all__ = [
    "TilePartitioner", "GridPartitioner",
    "ComputePipeline", "ScheduledCompute",
    "Epilogue", "DirectEpilogue",
    "KernelPipeline",
]


# ===================================================================
# TilePartitioner: work distribution
# ===================================================================

class TilePartitioner:
    """Base class: determines what each workgroup computes.

    Must set in ctx._metadata:
        s_k_tiles: number of K-tile iterations for this WG
    Global addresses must be adjusted if k_start != 0.
    """

    def emit(self, ctx: AsmContext) -> None:
        raise NotImplementedError

    def grid_dims(self, problem: GemmProblem,
                  tile: TileConfig) -> tuple:
        raise NotImplementedError


class GridPartitioner(TilePartitioner):
    """Simple 2D grid. Each WG computes one (m_tile, n_tile), full K.

    This is a no-op partitioner -- the setup phase already handles
    WG ID decomposition and address computation for the full K range.
    The K-tile count is computed from s_K.
    """

    def emit(self, ctx: AsmContext) -> None:
        tile = ctx._metadata["tile"]
        log2_uk = int(math.log2(tile.unroll_k))
        ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
                   comment=f"k_tiles = K / {tile.unroll_k}")
        ctx.raw("")

    def grid_dims(self, problem, tile):
        return (problem.m // tile.wg_m, problem.n // tile.wg_n, 1)


# ===================================================================
# ComputePipeline: the K-loop body
# ===================================================================

class ComputePipeline:
    """Base class: K-loop that loads data and computes MFMAs.

    Reads s_k_tiles from registers (set by TilePartitioner).
    Accumulates into acc_C registers.
    Does NOT store results -- that's the Epilogue's job.
    """

    def emit(self, ctx: AsmContext) -> None:
        raise NotImplementedError


class ScheduledCompute(ComputePipeline):
    """K-loop using KLoopGraph + KLoopScheduler.

    Builds a dependency graph of MFMA/ds_read/load ops, schedules
    them, and emits the interleaved instruction sequence.
    """

    def __init__(self, loader: GlobalLoader, reader: LDSReader,
                 scale_loader: object = None) -> None:
        self.loader = loader
        self.reader = reader
        self.scale_loader = scale_loader

    def emit(self, ctx: AsmContext) -> None:
        tile = ctx._metadata["tile"]
        problem = ctx._metadata["problem"]
        loader = self.loader
        reader = self.reader
        scale_loader = self.scale_loader
        elem = problem.element_bytes
        lds_data_half = int((tile.wg_m + tile.wg_n) * tile.unroll_k * elem)

        # Store reader in metadata for MFMABlock emit closures
        ctx._metadata["_reader"] = reader

        # DB step register
        ctx.alloc_sgpr_permanent(1, "s_lds_db_step")
        ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_data_half),
                  comment=f"DB step = {lds_data_half}")
        ctx.raw("")

        # Precompute offsets
        loader.precompute_soffsets()
        if scale_loader:
            scale_loader.precompute_soffsets()

        # Build dependency graph
        graph = KLoopGraph(tile, problem)
        GlobalLoadBlock(loader).register(graph)
        DSReadBlock(reader).register(graph)
        MFMABlock(ctx, tile, scale_loader).register(graph)
        SuffixBlock(reader, scale_loader, loader).register(graph)
        graph.validate()

        # Schedule
        schedule = KLoopScheduler(graph).schedule()

        # Emit
        self._emit_prologue(ctx, loader, scale_loader)
        self._emit_loop(ctx, schedule, loader, reader, scale_loader)

    def _emit_prologue(self, ctx, loader, scale_loader):
        """Load first K-tile into LDS."""
        ctx.comment("Prologue: load tile 0")
        loader.emit_loads()
        if scale_loader:
            scale_loader.emit_initial_loads(4)
            extra = scale_loader.num_initial_inflight(4)
            ctx.s_waitcnt(f"vmcnt({extra})",
                          comment=f"wait DTL (leave {extra} scale loads)")
        else:
            ctx.s_waitcnt("vmcnt(0)", comment="wait DTL loads")
        ctx.s_barrier(comment="sync first tile")
        ctx.raw("")

    def _emit_loop(self, ctx, schedule, loader, reader, scale_loader):
        """Emit the K-loop: iterate over K-tiles."""
        tile = ctx._metadata["tile"]
        nr = tile.mfma_n_repeat
        mr = tile.mfma_m_repeat
        ki_count = tile.k_iterations
        mfmas_per_mi = nr * ki_count
        partition_m = 4

        ctx.label("k_loop")
        ctx.raw("")

        # Pre-body: early B reads (overlap with arriving loads)
        ctx.comment("Early B reads (overlap with loads)")
        for op in schedule.pre_body_ops:
            if op.emit:
                op.emit()

        # Conditional next-tile load
        ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
                  comment="k_tiles--")
        ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
                 comment="more tiles?")
        ctx.inst("s_cbranch_scc0", "load_skip_all",
                 comment="skip loads on last iter")

        for op in schedule.prefetch_ops:
            if op.emit:
                op.emit()

        if scale_loader:
            if not scale_loader.has_cross_iter_prefetch:
                scale_loader.advance()
            scale_loader.emit_loop_loads()

        ctx.raw("")
        ctx.label("load_skip_all")

        loader.emit_sync()
        ctx.raw("")

        # Preamble reads
        ctx.comment("Preamble: A[m0] + B ki=1")
        for op in schedule.preamble_ops:
            if op.emit:
                op.emit()

        # Preamble inflight tracking
        preamble_inflight = nr + 1
        if ki_count > 1:
            preamble_inflight += nr + 1
        first_batch = nr + 1
        remaining = preamble_inflight - first_batch
        wait_cnt = min(remaining, 15)
        ctx.s_waitcnt(f"lgkmcnt({wait_cnt})",
                      comment="wait B[ki=0] + A[m0,k0]")

        if scale_loader:
            scale_loader.emit_scale_wait(loader)
        ctx.raw("")

        reader.emit_recompute_ki_bases()
        ctx.raw("")

        # Emit scheduled MFMA body
        inflight_lgkm = preamble_inflight
        mfma_count = 0

        for i, mfma_op in enumerate(schedule.mfma_order):
            if i in schedule.waits:
                ctx.s_waitcnt(schedule.waits[i],
                              comment=f"auto-wait before MFMA[{i}]")
                inflight_lgkm = 0

            if mfma_count == nr and inflight_lgkm > 0:
                ctx.s_waitcnt("lgkmcnt(0)",
                              comment="wait B[ki=1] + A[m0,k1]")
                inflight_lgkm = 0

            if (mfma_count > 0 and mfma_count % mfmas_per_mi == 0
                    and inflight_lgkm > 0):
                ctx.s_waitcnt("lgkmcnt(0)",
                              comment=f"wait A[m{mfma_count // mfmas_per_mi}]")
                inflight_lgkm = 0

            if scale_loader:
                mps = partition_m * mfmas_per_mi
                n_st = mr // partition_m
                if mfma_count > 0 and mfma_count % mps == 0:
                    st_idx = mfma_count // mps
                    if st_idx < n_st:
                        scale_loader.emit_subtile_wait(loader, st_idx)

            if mfma_count % (partition_m * mfmas_per_mi) == 0:
                ctx.comment(
                    f"--- Partition "
                    f"{mfma_count // (partition_m * mfmas_per_mi)} ---")

            for op in schedule.side_ops[i]:
                if op.emit:
                    op.emit()
                if op.kind == OpKind.DS_READ:
                    inflight_lgkm += 1

            if mfma_op.emit:
                mfma_op.emit()
            mfma_count += 1

        for op in schedule.epilogue_ops:
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


# ===================================================================
# Epilogue: output strategy
# ===================================================================

class Epilogue:
    """Base class: stores accumulated results."""

    def emit(self, ctx: AsmContext) -> None:
        raise NotImplementedError


class DirectEpilogue(Epilogue):
    """Write accumulators directly to D via phase_store_d.

    This wraps the existing phase_store_d function. It's invoked
    automatically by the tile tree walker as the workgroup epilogue,
    so this class is a thin compatibility layer.
    """

    def emit(self, ctx: AsmContext) -> None:
        from ..emit.phases import phase_store_d
        phase_store_d(None, ctx)


# ===================================================================
# KernelPipeline: composition
# ===================================================================

class KernelPipeline:
    """Composes TilePartitioner + ComputePipeline + Epilogue.

    This is the top-level kernel structure. The setup phase
    (kernarg loading, thread indexing, LDS addresses) runs before
    this pipeline via the tile tree's prologue phases.
    """

    def __init__(self, partitioner: TilePartitioner,
                 compute: ComputePipeline,
                 epilogue: Optional[Epilogue] = None) -> None:
        self.partitioner = partitioner
        self.compute = compute
        self.epilogue = epilogue

    def emit(self, ctx: AsmContext) -> None:
        ctx.comment("=== KernelPipeline ===")
        self.partitioner.emit(ctx)
        self.compute.emit(ctx)
        # Epilogue is handled by the tile tree's epilogue phases
        # (phase_store_d), not here. This keeps compatibility with
        # the existing tile tree walker.


def pipeline_kloop_phase(level, ctx) -> None:
    """Phase function using KernelPipeline architecture.

    Drop-in replacement for scheduled_kloop_phase.
    """
    tile = ctx._metadata["tile"]
    problem = ctx._metadata["problem"]
    use_dtl = ctx._metadata.get("use_dtl", True)

    loader_cls = ctx._metadata.get("loader_cls",
                                   DTLLoader if use_dtl else BufferLoader)
    loader = loader_cls(ctx, tile, problem)
    swizzle = ctx._metadata.get("swizzle", None)
    reader = LDSReader(ctx, tile, problem, swizzle=swizzle)

    scale_loader = None
    use_real_scales = ctx._metadata.get("use_real_scales", False)
    if use_real_scales and tile.mfma.is_mx:
        from ..memory.scale_loader import VMEMScaleLoader
        swizzled = ctx._metadata.get("swizzled_scales", False)
        scale_loader = VMEMScaleLoader(ctx, tile, swizzled=swizzled)

    pipeline = KernelPipeline(
        partitioner=GridPartitioner(),
        compute=ScheduledCompute(loader, reader, scale_loader),
    )
    pipeline.emit(ctx)
