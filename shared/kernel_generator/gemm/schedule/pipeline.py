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

if TYPE_CHECKING:
    from ..memory.global_loader import GlobalLoader
    from ..memory.scale_loader import ScaleLoader
    from ..tile.tree import TileLevel

__all__ = [
    "TilePartitioner", "GridPartitioner", "StreamKPartitioner",
    "ComputePipeline",
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
    tile = ctx._metadata["tile"]
    problem = ctx._metadata["problem"]
    mainloop = ctx._metadata["mainloop"]

    loader = mainloop.loader_cls(ctx, tile, problem)
    swizzle = mainloop.resolve_swizzle(tile)
    reader = LDSReader(ctx, tile, problem, swizzle=swizzle)
    scale_loader = mainloop.scale_strategy.build_loader(
        ctx, tile, mainloop.lds_data_half(tile))
    return loader, reader, scale_loader, mainloop.pgr


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
        """Emit Two-Tile StreamK=3 work decomposition.

        Reads pre-computed SK scheduling args from kernargs:
          offset 64:  workspace_ptr (u64)
          offset 72:  flags_ptr (u64)
          offset 140: iters_per_tile (u32) -- K / unroll_k
          offset 144: sk_iters_per_wg (u32) -- K iters per SK WG
          offset 148: sk_grid (u32) -- total WGs launched
          offset 152: sk_tiles (u32) -- tiles that need K-splitting

        Two-Tile SK=3 dispatch (DP-first):
          total_tiles = tiles_m * tiles_n
          dp_tiles = total_tiles - sk_tiles
          if wg_idx < dp_tiles:
            DP mode: tile = wg_idx, full K, direct store
          else:
            SK mode: compute K-slice from global iteration space
            sk_idx = wg_idx - dp_tiles
            global_iter = sk_idx * sk_iters_per_wg (+ extra distribution)
            tile_idx = dp_tiles + (global_iter / iters_per_tile)
            local_k_start = global_iter % iters_per_tile
            s_k_tiles = min(sk_iters_per_wg, iters_per_tile - local_k_start)
            partial store to workspace + set flag
        """
        tile = ctx._metadata["tile"]
        problem = ctx._metadata["problem"]
        layout = ctx._metadata.get("layout")
        elem = problem.element_bytes
        log2_uk = int(math.log2(tile.unroll_k))
        log2_wgm = int(math.log2(tile.wg_m))

        ctx.comment("=== StreamK Two-Tile Dispatch ===")

        # Allocate persistent SGPRs for epilogue
        karg = ctx.sreg("s_kernarg")
        ctx.alloc_sgpr_permanent(2, "s_workspace_ptr")
        ctx.alloc_sgpr_permanent(1, "s_iter_start")
        ctx.alloc_sgpr_permanent(1, "s_is_partial")
        ctx.alloc_sgpr_permanent(1, "s_partition_idx")
        ctx.alloc_sgpr_permanent(2, "s_flags_ptr")
        ctx.alloc_sgpr_permanent(1, "s_num_partitions")

        # Load SK args from kernargs
        ctx.inst("s_load_dwordx2", ctx.sreg("s_workspace_ptr"), karg,
                 "64", comment="workspace ptr")
        ctx.inst("s_load_dwordx2", ctx.sreg("s_flags_ptr"), karg,
                 "72", comment="flags ptr")
        # Load 4 dwords: iters_per_tile, sk_iters_per_wg, sk_grid, sk_tiles
        # from offsets 140-155
        ctx.alloc_sgpr_permanent(1, "s_iters_per_tile")
        ctx.alloc_sgpr_permanent(1, "s_sk_iters_per_wg")
        ctx.alloc_sgpr_permanent(1, "s_sk_grid")
        ctx.alloc_sgpr_permanent(1, "s_sk_tiles")
        ctx.inst("s_load_dword", ctx.sreg("s_iters_per_tile"), karg,
                 "140", comment="iters_per_tile")
        ctx.inst("s_load_dword", ctx.sreg("s_sk_iters_per_wg"), karg,
                 "144", comment="sk_iters_per_wg")
        ctx.inst("s_load_dword", ctx.sreg("s_sk_grid"), karg,
                 "148", comment="sk_grid")
        ctx.inst("s_load_dword", ctx.sreg("s_sk_tiles"), karg,
                 "152", comment="sk_tiles")
        ctx.s_waitcnt("lgkmcnt(0)", comment="wait SK kernargs")
        ctx.raw("")

        # Compute total tiles and dp_tiles
        ctx.inst("s_lshr_b32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_M"), str(log2_wgm),
                 comment=f"tiles_m = M / {tile.wg_m}")
        log2_wgn = int(math.log2(tile.wg_n))
        ctx.inst("s_lshr_b32", ctx.sreg("s_tmp1"),
                 ctx.sreg("s_N"), str(log2_wgn),
                 comment=f"tiles_n = N / {tile.wg_n}")
        ctx.s_mul(ctx.sreg("s_num_partitions"),
                  ctx.sreg("s_tmp0"), ctx.sreg("s_tmp1"),
                  comment="total_tiles = tiles_m * tiles_n")
        ctx.inst("s_sub_u32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_num_partitions"), ctx.sreg("s_sk_tiles"),
                 comment="dp_tiles = total_tiles - sk_tiles")
        ctx.raw("")

        # Check if this WG is DP or SK
        # s_tmp0 = dp_tiles
        ctx.inst("s_cmp_lt_u32", ctx.sreg("s_wg_id_x"),
                 ctx.sreg("s_tmp0"),
                 comment="wg_idx < dp_tiles? (DP WG)")
        ctx.inst("s_cbranch_scc1", "sk_dp_path",
                 comment="yes -> data-parallel")
        ctx.raw("")

        # ── SK path ──────────────────────────────────────────
        ctx.comment("SK WG: compute K-slice from global iteration space")
        # sk_idx = wg_idx - dp_tiles
        ctx.inst("s_sub_u32", ctx.sreg("s_partition_idx"),
                 ctx.sreg("s_wg_id_x"), ctx.sreg("s_tmp0"),
                 comment="sk_idx = wg_idx - dp_tiles")

        # Compute extra iterations for remainder distribution
        # extra = sk_tiles * ipt - sk_iters_per_wg * sk_grid
        ctx.s_mul(ctx.sreg("s_tmp1"),
                  ctx.sreg("s_sk_tiles"), ctx.sreg("s_iters_per_tile"),
                  comment="sk_tiles * iters_per_tile")
        ctx.s_mul(ctx.sreg("s_tmp0"),
                  ctx.sreg("s_sk_iters_per_wg"), ctx.sreg("s_sk_grid"),
                  comment="sk_iters_per_wg * sk_grid")
        ctx.inst("s_sub_u32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_tmp1"), ctx.sreg("s_tmp0"),
                 comment="extra = sk_total - sk_ipw*sk_grid")

        # iter_start computation with extra iter spreading:
        # if sk_idx < extra:
        #   iter_start = sk_idx * (sk_iters_per_wg + 1)
        #   num_iters = sk_iters_per_wg + 1
        # else:
        #   iter_start = sk_idx * sk_iters_per_wg + extra
        #   num_iters = sk_iters_per_wg
        ctx.inst("s_cmp_lt_u32", ctx.sreg("s_partition_idx"),
                 ctx.sreg("s_tmp0"),
                 comment="sk_idx < extra?")
        # Case: sk_idx < extra
        ctx.inst("s_add_u32", ctx.sreg("s_tmp1"),
                 ctx.sreg("s_sk_iters_per_wg"), "1",
                 comment="sk_ipw + 1")
        ctx.s_mul(ctx.sreg("s_iter_start"),
                  ctx.sreg("s_partition_idx"), ctx.sreg("s_tmp1"),
                  comment="iter_start = sk_idx * (sk_ipw+1)")
        # Case: sk_idx >= extra
        ctx.s_mul(ctx.sreg("s_tmp1"),
                  ctx.sreg("s_partition_idx"),
                  ctx.sreg("s_sk_iters_per_wg"),
                  comment="sk_idx * sk_ipw")
        ctx.inst("s_add_u32", ctx.sreg("s_tmp1"),
                 ctx.sreg("s_tmp1"), ctx.sreg("s_tmp0"),
                 comment="+ extra")
        # Select based on scc
        ctx.inst("s_cselect_b32", ctx.sreg("s_iter_start"),
                 ctx.sreg("s_iter_start"), ctx.sreg("s_tmp1"),
                 comment="select iter_start")
        ctx.inst("s_cselect_b32", ctx.sreg("s_tmp1"),
                 ctx.sreg("s_sk_iters_per_wg"), ctx.sreg("s_sk_iters_per_wg"),
                 comment="num_iters = sk_ipw (or +1 handled by iter_start)")

        # Now derive tile_idx and local K range from iter_start
        # tile_idx = dp_tiles + iter_start / iters_per_tile
        ctx.inst("s_lshr_b32", ctx.sreg("s_tmp1"),
                 ctx.sreg("s_iters_per_tile"), "0",
                 comment="iters_per_tile (copy)")
        # Use log2 division (iters_per_tile is power of 2)
        ctx.inst("s_ff1_i32_b32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_iters_per_tile"),
                 comment="log2(iters_per_tile)")
        ctx.inst("s_lshr_b32", ctx.sreg("s_tmp1"),
                 ctx.sreg("s_iter_start"), ctx.sreg("s_tmp0"),
                 comment="iter_start / iters_per_tile")
        # Compute dp_tiles again (was in s_tmp0 but got overwritten)
        ctx.inst("s_sub_u32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_num_partitions"), ctx.sreg("s_sk_tiles"),
                 comment="dp_tiles (recompute)")
        ctx.inst("s_add_u32", ctx.sreg("s_tmp1"),
                 ctx.sreg("s_tmp1"), ctx.sreg("s_tmp0"),
                 comment="tile_idx = dp_tiles + iter/ipt")

        # local_k_start = iter_start % iters_per_tile
        ctx.inst("s_ff1_i32_b32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_iters_per_tile"),
                 comment="log2(iters_per_tile)")
        ctx.inst("s_sub_u32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_iters_per_tile"), "1",
                 comment="ipt - 1 (mask)")
        ctx.inst("s_and_b32", ctx.sreg("s_iter_start"),
                 ctx.sreg("s_iter_start"), ctx.sreg("s_tmp0"),
                 comment="local_k_start = iter_start & (ipt-1)")

        # s_k_tiles = min(sk_iters_per_wg, ipt - local_k_start)
        ctx.inst("s_sub_u32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_iters_per_tile"),
                 ctx.sreg("s_iter_start"),
                 comment="remaining = ipt - local_start")
        ctx.inst("s_min_u32", ctx.sreg("s_k_tiles"),
                 ctx.sreg("s_sk_iters_per_wg"),
                 ctx.sreg("s_tmp0"),
                 comment="s_k_tiles = min(sk_ipw, remaining)")

        ctx.s_mov(ctx.sreg("s_is_partial"), "1",
                  comment="SK WG is partial")

        # Decompose tile_idx -> tile_m, tile_n (s_tmp1 = tile_idx)
        ctx.inst("s_lshr_b32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_M"), str(log2_wgm),
                 comment=f"tiles_m = M / {tile.wg_m}")
        ctx.inst("s_ff1_i32_b32", ctx.sreg("s_is_partial"),
                 ctx.sreg("s_tmp0"), comment="log2(tiles_m)")
        ctx.inst("s_sub_u32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_tmp0"), "1", comment="mask")
        ctx.inst("s_and_b32", ctx.sreg("s_wg_id_x"),
                 ctx.sreg("s_tmp1"), ctx.sreg("s_tmp0"),
                 comment="tile_m = tile_idx & mask")
        ctx.inst("s_lshr_b32", ctx.sreg("s_wg_id_y"),
                 ctx.sreg("s_tmp1"), ctx.sreg("s_is_partial"),
                 comment="tile_n = tile_idx >> log2(tiles_m)")

        ctx.s_mov(ctx.sreg("s_is_partial"), "1",
                  comment="SK WG is partial")

        ctx.inst("s_branch", "sk_recompute_srds",
                 comment="recompute SRDs")
        ctx.raw("")

        # ── DP path ──────────────────────────────────────────
        ctx.label("sk_dp_path")
        ctx.comment("DP WG: full K, one tile")
        ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
                   comment=f"k_tiles = K / {tile.unroll_k}")
        ctx.s_mov(ctx.sreg("s_iter_start"), "0",
                  comment="iter_start = 0")
        ctx.s_mov(ctx.sreg("s_is_partial"), "0",
                  comment="not partial")
        ctx.s_mov(ctx.sreg("s_partition_idx"), "0",
                  comment="partition_idx = 0")
        # tile coords already set by setup phase for DP WGs
        ctx.inst("s_branch", "sk_done", comment="skip SRD recompute")
        ctx.raw("")

        # ── Recompute SRDs for SK tile coords ────────────────
        ctx.label("sk_recompute_srds")
        ctx.comment("Recompute SRD A/B for SK tile coords")
        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"),
                  str(tile.wg_m), comment=f"tile_m * {tile.wg_m}")
        ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                 ctx.sreg("s_k_stride"), comment="* K_stride")
        ctx.inst("s_add_u32", ctx.sreg("s_srd_a", 0, 1),
                 ctx.sreg("s_ptr_A", 0, 1), ctx.sreg("s_tmp0"),
                 comment="SRD_A = ptr_A + row_off")
        ctx.inst("s_addc_u32", ctx.sreg("s_srd_a", 1, 1),
                 ctx.sreg("s_ptr_A", 1, 1), "0", comment="carry")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_a", 2, 1),
                 "0xFFFFFFFF", comment="limit")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_a", 3, 1),
                 "0x20000", comment="flags")

        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_y"),
                  str(tile.wg_n), comment=f"tile_n * {tile.wg_n}")
        ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                 ctx.sreg("s_k_stride"), comment="* K_stride")
        ctx.inst("s_add_u32", ctx.sreg("s_srd_b", 0, 1),
                 ctx.sreg("s_ptr_B", 0, 1), ctx.sreg("s_tmp0"),
                 comment="SRD_B = ptr_B + row_off")
        ctx.inst("s_addc_u32", ctx.sreg("s_srd_b", 1, 1),
                 ctx.sreg("s_ptr_B", 1, 1), "0", comment="carry")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_b", 2, 1),
                 "0xFFFFFFFF", comment="limit")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_b", 3, 1),
                 "0x20000", comment="flags")
        ctx.raw("")

        # Scale SRDs for MX types
        if layout and layout.has_scales and ctx.has("s_srd_scale_a"):
            from ..mainloop import VMEMScaleStrategy
            mainloop = ctx._metadata["mainloop"]
            use_swizzled = (isinstance(mainloop.scale_strategy, VMEMScaleStrategy)
                           and mainloop.scale_strategy.swizzled)
            ctx.comment("Recompute scale SRDs for SK tile")
            mul_m = tile.wg_m // 32 if use_swizzled else tile.wg_m
            ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"),
                      str(mul_m), comment=f"tile_m * {mul_m}")
            ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                     ctx.sreg("s_stride_scale_a"), comment="* stride")
            ctx.inst("s_add_u32", ctx.sreg("s_srd_scale_a", 0, 1),
                     ctx.sreg("s_ptr_scale_a", 0, 1), ctx.sreg("s_tmp0"),
                     comment="SRD_scaleA lo")
            ctx.inst("s_addc_u32", ctx.sreg("s_srd_scale_a", 1, 1),
                     ctx.sreg("s_ptr_scale_a", 1, 1), "0", comment="hi")
            ctx.inst("s_mov_b32", ctx.sreg("s_srd_scale_a", 2, 1),
                     "0xFFFFFFFF", comment="limit")
            ctx.inst("s_mov_b32", ctx.sreg("s_srd_scale_a", 3, 1),
                     "0x20000", comment="flags")

            mul_n = tile.wg_n // 32 if use_swizzled else tile.wg_n
            ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_y"),
                      str(mul_n), comment=f"tile_n * {mul_n}")
            ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                     ctx.sreg("s_stride_scale_b"), comment="* stride")
            ctx.inst("s_add_u32", ctx.sreg("s_srd_scale_b", 0, 1),
                     ctx.sreg("s_ptr_scale_b", 0, 1), ctx.sreg("s_tmp0"),
                     comment="SRD_scaleB lo")
            ctx.inst("s_addc_u32", ctx.sreg("s_srd_scale_b", 1, 1),
                     ctx.sreg("s_ptr_scale_b", 1, 1), "0", comment="hi")
            ctx.inst("s_mov_b32", ctx.sreg("s_srd_scale_b", 2, 1),
                     "0xFFFFFFFF", comment="limit")
            ctx.inst("s_mov_b32", ctx.sreg("s_srd_scale_b", 3, 1),
                     "0x20000", comment="flags")
            ctx.raw("")

        # Apply K-offset for SK WGs
        ctx.inst("s_cmp_eq_u32", ctx.sreg("s_iter_start"), "0",
                 comment="skip K-offset if iter_start == 0")
        ctx.inst("s_cbranch_scc1", "sk_done", comment="no offset")
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

        # Also need s_k_tiles for SK path (compute from iters_per_tile)
        ctx.inst("s_sub_u32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_iters_per_tile"), ctx.sreg("s_iter_start"),
                 comment="remaining = ipt - local_start")
        ctx.inst("s_min_u32", ctx.sreg("s_k_tiles"),
                 ctx.sreg("s_sk_iters_per_wg"), ctx.sreg("s_tmp0"),
                 comment="s_k_tiles = min(ipw, remaining)")

        ctx.label("sk_done")
        ctx.raw("")


    def grid_dims(self, problem: GemmProblem, tile: TileConfig) -> tuple[int, int, int]:
        """Compute grid dimensions for StreamK launch.

        Returns base tile count. The launcher multiplies by
        num_partitions to get the actual 1D grid size.
        """
        total_tiles = (problem.m // tile.wg_m) * (problem.n // tile.wg_n)
        return (total_tiles, 1, 1)


def pipeline_v2_kloop_phase(level: TileLevel, ctx: AsmContext) -> None:
    """Phase function using the new unified stream architecture.

    Default pipeline strategy. Uses LDSStream + LDSBufferManager +
    PipelineScheduler + PipelineEmitter.
    """
    tile = ctx._metadata["tile"]
    mainloop = ctx._metadata["mainloop"]

    loader, reader, scale_loader, pgr = \
        _build_loader_reader_scale(ctx)
    ctx._metadata["_dtl_loader"] = loader

    # --- New architecture ---
    from ..memory.streams import DTLDataStream
    from ..memory.lds_stream import LDSBufferManager
    from .graph_builder import build_kloop_graph
    from .pipeline_scheduler import PipelineScheduler
    from .pipeline_emitter import PipelineEmitter
    from .emit_wiring import wire_emit_callbacks
    from ..memory.mfma_emitter import MFMAEmitter

    problem = ctx._metadata["problem"]

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
        layout = ctx._metadata.get("layout")
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
    if isinstance(mainloop.epilogue, StreamKStore):
        StreamKPartitioner().emit(ctx)
    else:
        GridPartitioner().emit(ctx)

    # Save initial k_tiles for drain guard (PGR>=2 needs to
    # know if enough tiles existed for each ramp-up stage)
    if scheduled.pgr >= 2:
        ctx.alloc_sgpr_permanent(1, "s_k_tiles_init")
        ctx.s_mov(ctx.sreg("s_k_tiles_init"), ctx.sreg("s_k_tiles"),
                  comment="save initial k_tiles for drain guard")
        ctx.raw("")

    # Emit K-loop
    PipelineEmitter(scheduled, buffer_mgr, ctx,
                    double_copy=(scheduled.pgr >= 2)).emit()
