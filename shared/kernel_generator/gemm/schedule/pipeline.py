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



# ===================================================================
# StreamK partitioner
# ===================================================================

class StreamKPartitioner(TilePartitioner):
    """StreamK: distribute K-tile iterations across all CUs.

    Host precomputes per-WG iteration ranges and passes them via
    kernargs or a side buffer. The GPU does simple lookups.

    For the initial implementation, we use a simpler "data-parallel
    StreamK" approach:
    - Compute how many full waves of tiles fit on the GPU
    - Remaining tiles get their K split across leftover CUs
    - Each WG handles exactly one output tile (possibly partial K)

    This avoids cross-tile iteration and atomic accumulation for the
    common case (large enough grids), while still improving CU
    utilization for the tail wave.

    Args:
        num_cus: Number of compute units on the target GPU.
    """

    def __init__(self, num_cus: int = 304) -> None:
        self.num_cus = num_cus

    def emit(self, ctx: AsmContext) -> None:
        """Emit StreamK work decomposition.

        Reads StreamK params from kernargs (set by host launcher):
        - offset 128: workspace_ptr (u64)
        - offset 136: iter_start (u32) -- this WG's first K-tile iter
        - offset 140: iter_end (u32) -- this WG's last K-tile iter (exclusive)
        - offset 144: k_tiles_per_tile (u32)
        - offset 148: num_m_tiles (u32)
        - offset 152: is_partial (u32) -- 1 if this WG has partial K range

        Sets s_k_tiles = iter_end - iter_start.
        Adjusts SRD bases for K-offset if k_start > 0.
        """
        tile = ctx._metadata["tile"]
        problem = ctx._metadata["problem"]
        elem = problem.element_bytes
        log2_uk = int(math.log2(tile.unroll_k))

        sk_params = ctx._metadata.get("streamk_params")
        if sk_params is None:
            # Fallback: behave like GridPartitioner
            ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
                       comment=f"k_tiles = K / {tile.unroll_k}")
            ctx.raw("")
            return

        ctx.comment("=== StreamK Work Decomposition ===")

        # Load StreamK kernargs
        karg = ctx.sreg("s_kernarg")
        ctx.alloc_sgpr_permanent(2, "s_workspace_ptr")
        ctx.alloc_sgpr_permanent(1, "s_iter_start")
        ctx.alloc_sgpr_permanent(1, "s_iter_end")
        ctx.alloc_sgpr_permanent(1, "s_k_tiles_per_tile")
        ctx.alloc_sgpr_permanent(1, "s_num_m_tiles")
        ctx.alloc_sgpr_permanent(1, "s_is_partial")

        ctx.inst("s_load_dwordx2", ctx.sreg("s_workspace_ptr"), karg, "128",
                 comment="workspace ptr")
        ctx.inst("s_load_dword", ctx.sreg("s_iter_start"), karg, "136",
                 comment="iter_start")
        ctx.inst("s_load_dword", ctx.sreg("s_iter_end"), karg, "140",
                 comment="iter_end")
        ctx.inst("s_load_dword", ctx.sreg("s_k_tiles_per_tile"), karg, "144",
                 comment="k_tiles_per_tile")
        ctx.inst("s_load_dword", ctx.sreg("s_num_m_tiles"), karg, "148",
                 comment="num_m_tiles")
        ctx.inst("s_load_dword", ctx.sreg("s_is_partial"), karg, "152",
                 comment="is_partial (0=full, 1=partial)")
        ctx.s_waitcnt("lgkmcnt(0)", comment="wait SK kernargs")
        ctx.raw("")

        # s_k_tiles = iter_end - iter_start
        ctx.inst("s_sub_u32", ctx.sreg("s_k_tiles"),
                 ctx.sreg("s_iter_end"), ctx.sreg("s_iter_start"),
                 comment="s_k_tiles = iter_end - iter_start")

        # Compute tile index from iter_start
        # tile_idx = iter_start / k_tiles_per_tile (host provides this as int)
        # For simplicity: host packs (wg_id_x, wg_id_y) directly
        # rather than requiring GPU-side division.
        # TODO: support multi-tile StreamK with GPU-side decomposition
        ctx.raw("")

        # Adjust A/B SRD bases for K-offset
        # k_start_within_tile = iter_start % k_tiles_per_tile
        # k_byte_offset = k_start_within_tile * unroll_k * elem
        # For now: host passes k_start directly via iter_start encoding
        ctx.comment("Adjust SRD for K-offset (StreamK partial K)")
        ctx.inst("s_cmp_eq_u32", ctx.sreg("s_iter_start"), "0",
                 comment="skip if k_start == 0")
        ctx.inst("s_cbranch_scc1", "sk_no_k_offset",
                 comment="no K-offset needed")

        # k_byte_offset = iter_start * unroll_k * elem (within-tile K offset)
        # But iter_start is global, need modulo k_tiles_per_tile
        # Simpler: host passes k_start_tiles directly
        # For the single-tile-per-WG case, this is just iter_start
        k_bytes_per_tile_iter = tile.unroll_k * elem
        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_iter_start"),
                  str(k_bytes_per_tile_iter),
                  comment=f"k_offset = iter_start * {k_bytes_per_tile_iter}")
        # Add to SRD A base
        ctx.inst("s_add_u32", ctx.sreg("s_srd_a", 0, 1),
                 ctx.sreg("s_srd_a", 0, 1), ctx.sreg("s_tmp0"),
                 comment="SRD_A += k_offset")
        ctx.inst("s_addc_u32", ctx.sreg("s_srd_a", 1, 1),
                 ctx.sreg("s_srd_a", 1, 1), "0",
                 comment="carry")
        # Add to SRD B base
        ctx.inst("s_add_u32", ctx.sreg("s_srd_b", 0, 1),
                 ctx.sreg("s_srd_b", 0, 1), ctx.sreg("s_tmp0"),
                 comment="SRD_B += k_offset")
        ctx.inst("s_addc_u32", ctx.sreg("s_srd_b", 1, 1),
                 ctx.sreg("s_srd_b", 1, 1), "0",
                 comment="carry")

        ctx.label("sk_no_k_offset")
        ctx.raw("")

    def grid_dims(self, problem, tile):
        """Compute grid dimensions for StreamK launch.

        Uses data-parallel approach: launch enough WGs to fill all CUs
        for complete waves, then handle the remainder.
        """
        tiles_m = problem.m // tile.wg_m
        tiles_n = problem.n // tile.wg_n
        total_tiles = tiles_m * tiles_n

        # For now: just launch all tiles (same as grid partitioner)
        # but with 1D grid for future StreamK extension
        return (total_tiles, 1, 1)

    def compute_sk_params(self, problem, tile):
        """Host-side: compute StreamK parameters.

        Returns a dict with:
        - total_tiles: number of output tiles
        - k_tiles_per_tile: K iterations per output tile
        - total_iters: total K-tile iterations across all tiles
        - dp_tiles: tiles handled by data-parallel (full K)
        - sk_tiles: tiles handled by StreamK (split K)
        - sk_ctas: number of WGs doing StreamK work
        - iters_per_sk_cta: base iterations per StreamK WG
        - extra_iters: number of WGs that get one extra iteration
        """
        tiles_m = problem.m // tile.wg_m
        tiles_n = problem.n // tile.wg_n
        total_tiles = tiles_m * tiles_n
        k_tiles = problem.k // tile.unroll_k
        total_iters = total_tiles * k_tiles

        # Full waves of data-parallel tiles
        full_waves = total_tiles // self.num_cus
        dp_tiles = full_waves * self.num_cus

        # Remaining tiles need StreamK
        sk_tiles = total_tiles - dp_tiles
        if sk_tiles == 0:
            return {
                "total_tiles": total_tiles,
                "k_tiles_per_tile": k_tiles,
                "total_iters": total_iters,
                "dp_tiles": dp_tiles,
                "sk_tiles": 0,
                "sk_ctas": 0,
                "iters_per_sk_cta": 0,
                "extra_iters": 0,
            }

        # StreamK: distribute sk_tiles * k_tiles iterations
        # across num_cus WGs (or fewer if sk_iters < num_cus)
        sk_iters = sk_tiles * k_tiles
        sk_ctas = min(sk_iters, self.num_cus)
        iters_per_cta = sk_iters // sk_ctas
        extra = sk_iters % sk_ctas

        return {
            "total_tiles": total_tiles,
            "k_tiles_per_tile": k_tiles,
            "total_iters": total_iters,
            "dp_tiles": dp_tiles,
            "sk_tiles": sk_tiles,
            "sk_ctas": sk_ctas,
            "iters_per_sk_cta": iters_per_cta,
            "extra_iters": extra,
        }


# ===================================================================
# Atomic epilogue (for StreamK partial tiles)
# ===================================================================

class AtomicEpilogue(Epilogue):
    """Atomic-add accumulators to workspace for StreamK partial tiles.

    Uses global_atomic_add_f32 to accumulate FP32 partial results
    into a workspace buffer. A fixup kernel later converts the
    workspace to the output data type (BF16/FP16).

    For full tiles (is_partial=0), delegates to DirectEpilogue.
    """

    def emit(self, ctx: AsmContext) -> None:
        tile = ctx._metadata["tile"]
        mfma = tile.mfma
        acc_per = mfma.acc_vgprs
        mr = tile.mfma_m_repeat
        nr = tile.mfma_n_repeat

        # Check if this WG has a partial tile
        ctx.inst("s_cmp_eq_u32", ctx.sreg("s_is_partial"), "0",
                 comment="full tile?")
        ctx.inst("s_cbranch_scc1", "sk_direct_store",
                 comment="full tile -> direct store")

        # Partial tile: atomic add to workspace
        ctx.comment("=== StreamK: atomic add to workspace ===")

        # Build workspace SRD
        ctx.alloc_sgpr_permanent(4, "s_srd_ws")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_ws", 0, 1),
                 ctx.sreg("s_workspace_ptr", 0, 1), comment="WS SRD lo")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_ws", 1, 1),
                 ctx.sreg("s_workspace_ptr", 1, 1), comment="WS SRD hi")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_ws", 2, 1),
                 "0xFFFFFFFF", comment="WS limit")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_ws", 3, 1),
                 "0x20000", comment="WS flags")
        ctx.raw("")

        # Compute per-lane workspace offset
        # workspace layout: [num_tiles * wg_m * wg_n] FP32 values
        # Per-WG offset = tile_idx * wg_m * wg_n * 4
        # Per-lane offset within WG = (wave_m * m_per_wave + lane_m) * wg_n
        #                             + (wave_n * n_per_wave + lane_n)
        # times 4 (FP32 bytes)

        # Reuse lane coordinates from store_d setup
        ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.n - 1,
                  comment="lane_n")
        ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
                   int(math.log2(mfma.m)), comment=f"lane_id / {mfma.m}")
        ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_tmp1"), 2,
                   comment="* 4 -> lane_m_base")

        # global_row = wave_m * m_per_wave + lane_m_base
        ctx.v_mul(ctx.vreg("v_tmp2"), str(tile.m_per_wave),
                  ctx.vreg("v_wave_m"), comment=f"wave_m * {tile.m_per_wave}")
        ctx.v_add(ctx.vreg("v_tmp2"), ctx.vreg("v_tmp2"),
                  ctx.vreg("v_tmp1"), comment="+ lane_m_base")

        # global_col = wave_n * n_per_wave + lane_n
        ctx.v_mul(ctx.vreg("v_tmp3"), str(tile.n_per_wave),
                  ctx.vreg("v_wave_n"), comment=f"wave_n * {tile.n_per_wave}")
        ctx.v_add(ctx.vreg("v_tmp3"), ctx.vreg("v_tmp3"),
                  ctx.vreg("v_tmp0"), comment="+ lane_n")

        # ws_base_offset = (global_row * wg_n + global_col) * 4
        ctx.inst("v_mul_lo_u32", ctx.vreg("v_tmp2"),
                 str(tile.wg_n), ctx.vreg("v_tmp2"),
                 comment="row * wg_n")
        ctx.v_add(ctx.vreg("v_tmp2"), ctx.vreg("v_tmp2"),
                  ctx.vreg("v_tmp3"), comment="+ col")
        ctx.v_lshl(ctx.vreg("v_tmp2"), ctx.vreg("v_tmp2"), 2,
                   comment="* 4 -> FP32 byte offset")

        # Add tile offset: wg_id_x * wg_m * wg_n * 4 + wg_id_y * ...
        # For now, use flat tile index from setup
        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"),
                  str(tile.wg_m), comment="wg_base_m")
        ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                 ctx.sreg("s_N"), comment="* N")
        ctx.s_mul(ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_y"),
                  str(tile.wg_n), comment="wg_base_n")
        ctx.inst("s_add_u32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_tmp0"), ctx.sreg("s_tmp1"),
                 comment="tile_offset_elems")
        ctx.s_lshl(ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"), 2,
                    comment="* 4 -> bytes")
        ctx.v_add(ctx.vreg("v_tmp2"), ctx.sreg("s_tmp0"),
                  ctx.vreg("v_tmp2"), comment="+ tile_base -> ws_offset")
        ctx.raw("")

        # Atomic add each accumulator
        ctx.comment(f"Atomic add {mr * nr * acc_per} accumulators")
        for mi in range(mr):
            for ni in range(nr):
                for ai in range(acc_per):
                    acc_idx = (mi * nr + ni) * acc_per + ai
                    row_off = (mi * mfma.m + ai) * tile.wg_n
                    col_off = ni * mfma.n
                    elem_off = (row_off + col_off) * 4  # FP32 bytes

                    ctx.inst("v_accvgpr_read_b32", ctx.vreg("v_tmp0"),
                             ctx.areg("acc_C", acc_idx, 1),
                             comment=f"acc[{acc_idx}]")

                    if elem_off < 4096:
                        ctx.inst("buffer_atomic_add_f32",
                                 ctx.vreg("v_tmp0"),
                                 ctx.vreg("v_tmp2"),
                                 ctx.sreg("s_srd_ws", 0, 4),
                                 "0", f"offen offset:{elem_off}",
                                 comment=f"atomic m{mi}_n{ni}_a{ai}")
                    else:
                        ctx.inst("buffer_atomic_add_f32",
                                 ctx.vreg("v_tmp0"),
                                 ctx.vreg("v_tmp2"),
                                 ctx.sreg("s_srd_ws", 0, 4),
                                 str(elem_off), "offen",
                                 comment=f"atomic m{mi}_n{ni}_a{ai}")

        ctx.s_waitcnt("vmcnt(0)", comment="wait atomics")
        ctx.inst("s_branch", "sk_store_done", comment="skip direct store")
        ctx.raw("")

        # Full tile: direct store
        ctx.label("sk_direct_store")
        DirectEpilogue().emit(ctx)
        ctx.label("sk_store_done")
        ctx.raw("")


# ===================================================================
# Fixup kernel for StreamK workspace -> output conversion
# ===================================================================

def generate_fixup_kernel(m: int, n: int, output_bf16: bool = True) -> str:
    """Generate a simple fixup kernel that converts FP32 workspace to BF16/FP16.

    The fixup kernel:
    1. Reads FP32 values from workspace
    2. Converts to BF16 or FP16
    3. Writes to output D

    Kernargs (32 bytes):
        offset 0: workspace_ptr (u64)
        offset 8: d_ptr (u64)
        offset 16: num_elements (u32)

    Grid: 1D, (num_elements + 255) / 256 workgroups, 256 threads each.
    Each thread converts one element.
    """
    from ..emit.context import AsmContext
    from ..emit.emitter import assemble_kernel

    ctx = AsmContext()
    ctx._metadata = {}

    cvt = "v_cvt_pk_bf16_f32" if output_bf16 else "v_cvt_pk_f16_f32"
    total = m * n

    ctx.comment("=== StreamK Fixup Kernel ===")
    ctx.comment(f"Convert {total} FP32 workspace values to {'BF16' if output_bf16 else 'FP16'}")
    ctx.raw("")

    # Load kernargs
    ctx.inst("s_load_dwordx2", "s[0:1]", "s[4:5]", "0",
             comment="workspace_ptr")
    ctx.inst("s_load_dwordx2", "s[2:3]", "s[4:5]", "8",
             comment="d_ptr")
    ctx.inst("s_load_dword", "s6", "s[4:5]", "16",
             comment="num_elements")
    ctx.inst("s_waitcnt", "lgkmcnt(0)", comment="wait kernargs")
    ctx.raw("")

    # Global ID
    ctx.inst("v_mov_b32", "v0", "s8", comment="block_id")  # s8 from dispatch
    ctx.inst("v_lshlrev_b32", "v0", "8", "v0", comment="* 256")
    ctx.inst("v_add_u32", "v0", "v0", "v1", comment="+ thread_id -> gid")

    # Bounds check
    ctx.inst("v_cmp_lt_u32", "vcc", "v0", "s6", comment="gid < num_elements")
    ctx.inst("s_and_saveexec_b64", "s[10:11]", "vcc",
             comment="mask out-of-bounds lanes")
    ctx.raw("")

    # Read FP32 from workspace
    ctx.inst("v_lshlrev_b32", "v2", "2", "v0", comment="byte offset (FP32)")
    ctx.inst("global_load_dword", "v3", "v2", "s[0:1]", comment="load FP32")
    ctx.inst("s_waitcnt", "vmcnt(0)", comment="wait load")

    # Convert to BF16/FP16 and store
    ctx.inst("v_cvt_f16_f32_e32" if not output_bf16 else "v_cvt_bf16_f32_e32",
             "v3", "v3", comment="FP32 -> output type")
    ctx.inst("v_lshlrev_b32", "v2", "1", "v0", comment="byte offset (output)")
    ctx.inst("global_store_short", "v2", "v3", "s[2:3]",
             comment="store output")

    ctx.inst("s_waitcnt", "vmcnt(0)", comment="wait store")
    ctx.inst("s_endpgm", comment="done")

    return ctx.lines
