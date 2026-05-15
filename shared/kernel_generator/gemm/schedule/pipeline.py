"""Kernel pipeline: TilePartitioner + ComputePipeline.

Separates work distribution, compute, and output into independent
components following the CK/TensileLite/Triton pattern.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from ..emit.context import AsmContext
from ..problem import TileConfig, GemmProblem
from ..memory.global_loader import DTLLoader, BufferLoader
from ..memory.lds_reader import LDSReader
from .tile_ops import (
    emit_decompose_tile_idx, emit_recompute_srds,
    emit_zero_accumulators, emit_reset_kloop_state,
    emit_build_raw_srd,
)

if TYPE_CHECKING:
    from ..memory.global_loader import GlobalLoader
    from ..memory.scale_loader import ScaleLoader
    from ..tile.tree import TileLevel

__all__ = [
    "TilePartitioner", "GridPartitioner",
    "ComputePipeline",
]


# ===================================================================
# TilePartitioner: work distribution
# ===================================================================

class TilePartitioner:
    """Base class: determines what each workgroup computes.

    Must set in ctx.config or ctx._state:
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
        tile = ctx.config.tile
        log2_uk = int(math.log2(tile.unroll_k))
        ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
                   comment=f"k_tiles = K / {tile.unroll_k}")
        ctx.raw("")

    def grid_dims(self, problem: GemmProblem, tile: TileConfig) -> tuple[int, int, int]:
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


# ===================================================================
# KernelPipeline: composition
# ===================================================================


def pipeline_kloop_phase(level: TileLevel, ctx: AsmContext) -> None:
    """Phase function: unified stream architecture K-loop."""
    return pipeline_v2_kloop_phase(level, ctx)


# ── Shared helpers ────────────────────────────────────────────────

def _build_loader_reader_scale(ctx: AsmContext) -> tuple[GlobalLoader, LDSReader, ScaleLoader, int]:
    """Construct the loader, reader, and scale_loader from mainloop.

    Returns (loader, reader, scale_loader, pgr).
    """
    tile = ctx.config.tile
    problem = ctx.config.problem
    mainloop = ctx.config.mainloop

    loader = mainloop.loader_cls(ctx, tile, problem)
    swizzle = mainloop.resolve_swizzle(tile)
    reader = LDSReader(ctx, tile, problem, swizzle=swizzle)
    scale_loader = mainloop.scale_strategy.build_loader(
        ctx, tile, mainloop.lds_data_half(tile))
    return loader, reader, scale_loader, mainloop.pgr




def _emit_sk_iter_range_setup(
    ctx: AsmContext,
    tile: TileConfig,
    pgr: int,
) -> None:
    """One-time setup for pre-computed iteration range StreamK.

    Loads SK kernargs, computes per-WG [iter_start, iter_end),
    and allocates persistent SGPRs for the tile loop.

    Kernarg offsets (TensileLite SK ABI):
        64: workspace_ptr (u64)
        72: flags_ptr (u64)
        140: iters_per_tile (u32) = K / unroll_k
        144: sk_iters_per_wg (u32)
        148: sk_grid (u32) = total WGs launched
        152: sk_tiles (u32) = tiles that need K-splitting
    """
    log2_uk = int(math.log2(tile.unroll_k))
    log2_wgm = int(math.log2(tile.wg_m))
    log2_wgn = int(math.log2(tile.wg_n))
    karg = ctx.sreg("s_kernarg")

    ctx.comment("=== StreamK Pre-Computed Iteration Range Setup ===")

    # Allocate persistent SGPRs
    ctx.alloc_sgpr_permanent(2, "s_workspace_ptr")
    ctx.alloc_sgpr_permanent(2, "s_flags_ptr")
    ctx.alloc_sgpr_permanent(1, "s_iters_per_tile")
    ctx.alloc_sgpr_permanent(1, "s_sk_iters_per_wg")
    ctx.alloc_sgpr_permanent(1, "s_sk_grid")
    ctx.alloc_sgpr_permanent(1, "s_sk_tiles")
    ctx.alloc_sgpr_permanent(1, "s_iter_current")   # current global iter
    ctx.alloc_sgpr_permanent(1, "s_iter_end")        # end of WG's range
    ctx.alloc_sgpr_permanent(1, "s_is_partial")
    ctx.alloc_sgpr_permanent(1, "s_iter_start")      # local K start within tile
    ctx.alloc_sgpr_permanent(1, "s_num_partitions")  # total tiles

    # Load SK kernargs
    ctx.inst("s_load_dwordx2", ctx.sreg("s_workspace_ptr"), karg,
             "64", comment="workspace ptr")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_flags_ptr"), karg,
             "72", comment="flags ptr")
    ctx.inst("s_load_dword", ctx.sreg("s_iters_per_tile"), karg,
             "140", comment="iters_per_tile = K / unroll_k")
    ctx.inst("s_load_dword", ctx.sreg("s_sk_iters_per_wg"), karg,
             "144", comment="sk_iters_per_wg")
    ctx.inst("s_load_dword", ctx.sreg("s_sk_grid"), karg,
             "148", comment="sk_grid (total WGs)")
    ctx.inst("s_load_dword", ctx.sreg("s_sk_tiles"), karg,
             "152", comment="sk_tiles")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait SK kernargs")
    ctx.raw("")

    # Compute total tiles and dp_tiles
    ctx.inst("s_lshr_b32", ctx.sreg("s_tmp0"), ctx.sreg("s_M"),
             str(log2_wgm), comment=f"tiles_m = M / {tile.wg_m}")
    ctx.inst("s_lshr_b32", ctx.sreg("s_tmp1"), ctx.sreg("s_N"),
             str(log2_wgn), comment=f"tiles_n = N / {tile.wg_n}")
    ctx.s_mul(ctx.sreg("s_num_partitions"),
              ctx.sreg("s_tmp0"), ctx.sreg("s_tmp1"),
              comment="total_tiles = tiles_m * tiles_n")
    ctx.inst("s_sub_u32", ctx.sreg("s_tmp0"),
             ctx.sreg("s_num_partitions"), ctx.sreg("s_sk_tiles"),
             comment="dp_tiles = total_tiles - sk_tiles")
    ctx.raw("")

    # ── Determine if this WG is DP or SK ──────────────────────
    # dp_tiles in s_tmp0, wg_id_x is our 1D WG index
    ctx.inst("s_cmp_lt_u32", ctx.sreg("s_wg_id_x"),
             ctx.sreg("s_tmp0"), comment="wg_idx < dp_tiles?")
    ctx.inst("s_cbranch_scc1", "sk_dp_init",
             comment="yes -> DP WG init")
    ctx.raw("")

    # ── SK WG: compute [iter_start, iter_end) ─────────────────
    ctx.comment("SK WG: compute iteration range")
    # sk_idx = wg_idx - dp_tiles
    ctx.inst("s_sub_u32", ctx.sreg("s_tmp1"),
             ctx.sreg("s_wg_id_x"), ctx.sreg("s_tmp0"),
             comment="sk_idx = wg_idx - dp_tiles")

    # extra = sk_tiles * ipt - sk_iters_per_wg * sk_grid
    ctx.s_mul(ctx.sreg("s_iter_current"),
              ctx.sreg("s_sk_tiles"), ctx.sreg("s_iters_per_tile"),
              comment="sk_total = sk_tiles * ipt")
    ctx.s_mul(ctx.sreg("s_iter_end"),
              ctx.sreg("s_sk_iters_per_wg"), ctx.sreg("s_sk_grid"),
              comment="sk_ipw * sk_grid")
    ctx.inst("s_sub_u32", ctx.sreg("s_tmp0"),
             ctx.sreg("s_iter_current"), ctx.sreg("s_iter_end"),
             comment="extra = sk_total - sk_ipw*sk_grid")

    # if sk_idx < extra: iter_start = sk_idx*(ipw+1), num_iters = ipw+1
    # else:              iter_start = sk_idx*ipw + extra, num_iters = ipw
    ctx.inst("s_cmp_lt_u32", ctx.sreg("s_tmp1"),
             ctx.sreg("s_tmp0"), comment="sk_idx < extra?")

    # Branch A: sk_idx < extra
    ctx.inst("s_add_u32", ctx.sreg("s_iter_current"),
             ctx.sreg("s_sk_iters_per_wg"), "1",
             comment="ipw + 1")
    ctx.s_mul(ctx.sreg("s_iter_end"),
              ctx.sreg("s_tmp1"), ctx.sreg("s_iter_current"),
              comment="sk_idx * (ipw+1)")
    # Branch B: sk_idx >= extra
    ctx.s_mul(ctx.sreg("s_iter_current"),
              ctx.sreg("s_tmp1"), ctx.sreg("s_sk_iters_per_wg"),
              comment="sk_idx * ipw")
    ctx.inst("s_add_u32", ctx.sreg("s_iter_current"),
             ctx.sreg("s_iter_current"), ctx.sreg("s_tmp0"),
             comment="+ extra")
    # Select iter_start based on scc (from cmp)
    ctx.inst("s_cselect_b32", ctx.sreg("s_iter_current"),
             ctx.sreg("s_iter_end"), ctx.sreg("s_iter_current"),
             comment="iter_start = select(sk_idx<extra)")
    # num_iters: ipw+1 if sk_idx<extra, else ipw
    ctx.inst("s_add_u32", ctx.sreg("s_tmp1"),
             ctx.sreg("s_sk_iters_per_wg"), "1",
             comment="ipw + 1")
    ctx.inst("s_cselect_b32", ctx.sreg("s_tmp1"),
             ctx.sreg("s_tmp1"), ctx.sreg("s_sk_iters_per_wg"),
             comment="num_iters = select")
    # iter_end = iter_start + num_iters
    ctx.inst("s_add_u32", ctx.sreg("s_iter_end"),
             ctx.sreg("s_iter_current"), ctx.sreg("s_tmp1"),
             comment="iter_end = iter_start + num_iters")

    ctx.inst("s_branch", "sk_init_done", comment="-> tile loop")
    ctx.raw("")

    # ── DP WG: full K, single tile ────────────────────────────
    ctx.label("sk_dp_init")
    ctx.comment("DP WG: iter range covers one full tile")
    # iter_current = wg_idx * iters_per_tile
    ctx.s_mul(ctx.sreg("s_iter_current"),
              ctx.sreg("s_wg_id_x"), ctx.sreg("s_iters_per_tile"),
              comment="iter_start = wg_idx * ipt")
    ctx.inst("s_add_u32", ctx.sreg("s_iter_end"),
             ctx.sreg("s_iter_current"), ctx.sreg("s_iters_per_tile"),
             comment="iter_end = iter_start + ipt")
    ctx.raw("")

    ctx.label("sk_init_done")

    # Save wg_idx for workspace slot addressing in the epilogue
    ctx.alloc_sgpr_permanent(1, "s_wg_idx_save")
    ctx.s_mov(ctx.sreg("s_wg_idx_save"), ctx.sreg("s_wg_id_x"),
              comment="save wg_idx for workspace slot")

    # Save initial K-tiles for drain guard (PGR >= 2)
    if pgr >= 2:
        ctx.alloc_sgpr_permanent(1, "s_k_tiles_init")
    ctx.raw("")


def _emit_sk_tile_from_iter(
    ctx: AsmContext,
    tile: TileConfig,
    mainloop: object,
) -> None:
    """Derive tile coords and K-range from s_iter_current.

    Sets: s_wg_id_x, s_wg_id_y (tile coords)
          s_k_tiles (K iters for this tile)
          s_iter_start (local K offset within tile, for SRD adjustment)
          s_is_partial (1 if partial K, 0 if full tile)
    """
    layout = ctx.config.layout
    problem = ctx.config.problem
    elem = problem.element_bytes

    # tile_idx = dp_tiles + iter_current / iters_per_tile
    # For DP WGs: tile_idx = wg_idx (iter_current = wg_idx * ipt)
    # For SK WGs: tile_idx = dp_tiles + sk_iter / ipt
    ctx.comment("Derive tile_idx and K-range from iter_current")

    # log2(ipt) for shift-based division/modulo
    ctx.inst("s_ff1_i32_b32", ctx.sreg("s_tmp0"),
             ctx.sreg("s_iters_per_tile"), comment="log2(ipt)")

    # tile_idx = iter_current >> log2(ipt)
    ctx.inst("s_lshr_b32", ctx.sreg("s_tmp1"),
             ctx.sreg("s_iter_current"), ctx.sreg("s_tmp0"),
             comment="tile_idx = iter / ipt")

    # local_k_start = iter_current & (ipt - 1)
    ctx.inst("s_sub_u32", ctx.sreg("s_tmp0"),
             ctx.sreg("s_iters_per_tile"), "1",
             comment="ipt - 1 (mask)")
    ctx.inst("s_and_b32", ctx.sreg("s_iter_start"),
             ctx.sreg("s_iter_current"), ctx.sreg("s_tmp0"),
             comment="local_k_start = iter & (ipt-1)")

    # s_k_tiles = min(iter_end - iter_current, ipt - local_k_start)
    ctx.inst("s_sub_u32", ctx.sreg("s_tmp0"),
             ctx.sreg("s_iters_per_tile"), ctx.sreg("s_iter_start"),
             comment="remaining_in_tile = ipt - local_k_start")
    ctx.inst("s_sub_u32", ctx.sreg("s_k_tiles"),
             ctx.sreg("s_iter_end"), ctx.sreg("s_iter_current"),
             comment="remaining_in_range = iter_end - iter_current")
    ctx.inst("s_min_u32", ctx.sreg("s_k_tiles"),
             ctx.sreg("s_k_tiles"), ctx.sreg("s_tmp0"),
             comment="s_k_tiles = min(range, tile)")

    # is_partial = (local_k_start != 0) || (s_k_tiles != ipt)
    # Simplified: is_partial = (s_k_tiles != ipt)
    ctx.inst("s_cmp_eq_u32", ctx.sreg("s_k_tiles"),
             ctx.sreg("s_iters_per_tile"), comment="full tile?")
    ctx.inst("s_cselect_b32", ctx.sreg("s_is_partial"),
             "0", "1", comment="is_partial = (k_tiles != ipt)")
    ctx.raw("")

    # Decompose tile_idx (in s_tmp1) into tile_m, tile_n
    emit_decompose_tile_idx(ctx, tile, tile_idx_reg="s_tmp1")

    # Recompute SRDs from tile coords
    emit_recompute_srds(ctx, tile, mainloop)

    # Apply K-offset for partial tiles (iter_start > 0)
    ctx.inst("s_cmp_eq_u32", ctx.sreg("s_iter_start"), "0",
             comment="skip K-offset if iter_start == 0")
    ctx.inst("s_cbranch_scc1", "sk_no_k_offset",
             comment="no K-offset needed")

    k_bytes = layout.k_offset_bytes(1, tile.unroll_k) if layout else int(tile.unroll_k * elem)
    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_iter_start"),
              str(k_bytes), comment=f"k_off = iter_start * {k_bytes}")
    ctx.inst("s_add_u32", ctx.sreg("s_srd_a", 0, 1),
             ctx.sreg("s_srd_a", 0, 1), ctx.sreg("s_tmp0"),
             comment="SRD_A += k_off")
    ctx.inst("s_addc_u32", ctx.sreg("s_srd_a", 1, 1),
             ctx.sreg("s_srd_a", 1, 1), "0", comment="carry")
    ctx.inst("s_add_u32", ctx.sreg("s_srd_b", 0, 1),
             ctx.sreg("s_srd_b", 0, 1), ctx.sreg("s_tmp0"),
             comment="SRD_B += k_off")
    ctx.inst("s_addc_u32", ctx.sreg("s_srd_b", 1, 1),
             ctx.sreg("s_srd_b", 1, 1), "0", comment="carry")

    if layout and layout.has_scales and ctx.has("s_srd_scale_a"):
        scale_k = tile.unroll_k // layout.scale_block
        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_iter_start"),
                  str(scale_k), comment=f"scale_k_off = iter*{scale_k}")
        ctx.inst("s_add_u32", ctx.sreg("s_srd_scale_a", 0, 1),
                 ctx.sreg("s_srd_scale_a", 0, 1), ctx.sreg("s_tmp0"),
                 comment="scaleA += off")
        ctx.inst("s_addc_u32", ctx.sreg("s_srd_scale_a", 1, 1),
                 ctx.sreg("s_srd_scale_a", 1, 1), "0", comment="carry")
        ctx.inst("s_add_u32", ctx.sreg("s_srd_scale_b", 0, 1),
                 ctx.sreg("s_srd_scale_b", 0, 1), ctx.sreg("s_tmp0"),
                 comment="scaleB += off")
        ctx.inst("s_addc_u32", ctx.sreg("s_srd_scale_b", 1, 1),
                 ctx.sreg("s_srd_scale_b", 1, 1), "0", comment="carry")

    ctx.label("sk_no_k_offset")
    ctx.raw("")


def _emit_persistent_loop(ctx, tile, mainloop, scheduled, buffer_mgr,
                          scale_loader, loader, reader):
    """Emit a persistent tile loop using pre-computed iteration ranges.

    Each WG is assigned [iter_current, iter_end) by the host.
    The loop iterates over tiles within that range:
      1. Derive tile_idx and K-range from iter_current
      2. Decompose tile_idx, recompute SRDs, apply K-offset
      3. Zero accumulators, reset K-loop state
      4. K-loop for s_k_tiles iterations
      5. Store: direct to D (full tile) or workspace (partial)
      6. Advance iter_current past this tile, loop if more
    """
    from .pipeline_emitter import PipelineEmitter
    from ..emit.phases import phase_store_streamk

    _emit_sk_iter_range_setup(ctx, tile, scheduled.pgr)

    # ── Tile loop ─────────────────────────────────────────────
    ctx.label("persistent_loop")
    ctx.comment("=== StreamK Tile Loop ===")

    # Exit if all iterations consumed
    ctx.inst("s_cmp_ge_u32", ctx.sreg("s_iter_current"),
             ctx.sreg("s_iter_end"), comment="iter_current >= iter_end?")
    ctx.inst("s_cbranch_scc1", "persistent_loop_end",
             comment="done -> exit")
    ctx.raw("")

    # Derive tile and K-range from current iteration
    _emit_sk_tile_from_iter(ctx, tile, mainloop)
    # Save k_tiles for PGR drain guard before reset clobbers it
    if scheduled.pgr >= 2 and ctx.has("s_k_tiles_init"):
        ctx.s_mov(ctx.sreg("s_k_tiles_init"), ctx.sreg("s_k_tiles"),
                  comment="save k_tiles for drain guard")
    emit_zero_accumulators(ctx, tile)
    emit_reset_kloop_state(ctx, tile, mainloop, scheduled.pgr)

    # Run K-loop
    PipelineEmitter(scheduled, buffer_mgr, ctx,
                    double_copy=(scheduled.pgr >= 2)).emit()

    # Store: phase_store_streamk handles full/partial dispatch internally
    # (checks s_is_partial, branches to sk_store_direct or workspace path)
    level = ctx._state.get("_tile_level")
    if level is None:
        from ..tile.tree import TileLevel
        level = TileLevel("workgroup", m=tile.wg_m, n=tile.wg_n, k=tile.unroll_k)
    phase_store_streamk(level, ctx)

    # Advance iter_current past this tile (use saved init value since
    # s_k_tiles is decremented to 0 by the K-loop)
    advance_reg = "s_k_tiles_init" if ctx.has("s_k_tiles_init") else "s_k_tiles"
    ctx.inst("s_add_u32", ctx.sreg("s_iter_current"),
             ctx.sreg("s_iter_current"), ctx.sreg(advance_reg),
             comment="iter_current += k_tiles (advance)")

    ctx.inst("s_branch", "persistent_loop", comment="next tile")
    ctx.raw("")

    ctx.label("persistent_loop_end")
    ctx._state["_persistent_store_done"] = True
    ctx.raw("")


def pipeline_v2_kloop_phase(level: TileLevel, ctx: AsmContext) -> None:
    """Phase function using the new unified stream architecture.

    Default pipeline strategy. Uses LDSStream + LDSBufferManager +
    PipelineScheduler + PipelineEmitter.
    """
    ctx._state["_tile_level"] = level
    tile = ctx.config.tile
    mainloop = ctx.config.mainloop

    loader, reader, scale_loader, pgr = \
        _build_loader_reader_scale(ctx)
    ctx._state["_dtl_loader"] = loader

    # --- New architecture ---
    from ..memory.streams import DTLDataStream
    from ..memory.lds_stream import LDSBufferManager
    from .graph_builder import build_kloop_graph
    from .pipeline_scheduler import PipelineScheduler
    from .pipeline_emitter import PipelineEmitter
    from .emit_wiring import wire_emit_callbacks
    from ..memory.mfma_emitter import MFMAEmitter

    problem = ctx.config.problem

    # Scale streams first so scale loads issue before DTL loads.
    # This avoids vmcnt(0) stalls before scale ds_writes:
    # the scale loads complete during the DTL load burst.
    streams = []
    if scale_loader is not None:
        streams.extend(scale_loader.streams(tile))
    streams.append(DTLDataStream("a", tile, problem))
    streams.append(DTLDataStream("b", tile, problem))

    num_buffers = 2 if pgr >= 1 else 1
    buffer_mgr = LDSBufferManager(streams, num_buffers=num_buffers)
    buffer_mgr.compute_layout()

    # Enable ki-phased scheduling for double-copy (PGR>=2, ki>1)
    ki_phased = pgr >= 2 and tile.k_iterations > 1
    mr = tile.mfma_m_repeat

    # For ki-phased: use mr A buffers (no ping-pong WAR deps)
    a_buf_count = mr if ki_phased else num_buffers
    if ki_phased:
        reader.set_num_a_buffers(mr)

    graph = build_kloop_graph(streams, tile, pgr=pgr,
                              num_buffers=a_buf_count, problem=problem)

    # Setup: soffsets, swizzle (must happen before MFMAEmitter
    # creation so scale_names are populated)
    loader.precompute_soffsets()
    if scale_loader is not None and hasattr(scale_loader, 'precompute_soffsets'):
        scale_loader.precompute_soffsets()
    reader.precompute_swizzle_addresses()

    # Create MFMAEmitter -- the scale loader knows which variant to use
    if scale_loader is not None:
        emitter = scale_loader.mfma_emitter(tile.mfma)
    else:
        layout = ctx.config.layout
        if layout and layout.mfma_has_scale_operands:
            emitter = MFMAEmitter.for_mx_constant(tile.mfma)
        else:
            emitter = MFMAEmitter.for_non_mx(tile.mfma)

    # Wire emit callbacks
    wire_emit_callbacks(graph, streams, buffer_mgr, loader, reader,
                        emitter, ctx, scale_loader=scale_loader)

    scheduled = PipelineScheduler(graph, ki_phased=ki_phased).schedule()

    ctx.alloc_sgpr_permanent(1, "s_lds_db_step")
    lds_half_total = mainloop.lds_half_total(tile)
    if not ctx.has("s_rd_db"):
        ctx.alloc_sgpr_permanent(1, "s_rd_db")
    ctx.s_mov(ctx.sreg("s_rd_db"), "0", comment="rd_db = 0")
    ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_half_total),
              comment=f"DB step = {lds_half_total}")
    ctx.raw("")

    # K-tile count: determined by mainloop epilogue type
    from ..mainloop import StreamKStore
    is_streamk = isinstance(mainloop.epilogue, StreamKStore)

    if is_streamk:
        # Persistent loop: grab tile -> recompute SRDs -> K-loop -> store -> repeat
        _emit_persistent_loop(ctx, tile, mainloop, scheduled, buffer_mgr,
                              scale_loader, loader, reader)
    else:
        GridPartitioner().emit(ctx)

        # Save initial k_tiles for drain guard
        if scheduled.pgr >= 2:
            ctx.alloc_sgpr_permanent(1, "s_k_tiles_init")
            ctx.s_mov(ctx.sreg("s_k_tiles_init"), ctx.sreg("s_k_tiles"),
                      comment="save initial k_tiles for drain guard")
            ctx.raw("")

        # Emit K-loop
        PipelineEmitter(scheduled, buffer_mgr, ctx,
                        double_copy=(scheduled.pgr >= 2)).emit()
