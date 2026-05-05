"""Kernel pipeline: TilePartitioner + ComputePipeline.

Separates work distribution, compute, and output into independent
components following the CK/TensileLite/Triton pattern.
"""
from __future__ import annotations

import math
from typing import Optional

from ..emit.context import AsmContext
from ..problem import TileConfig, GemmProblem
from ..memory.global_loader import GlobalLoader, DTLLoader, BufferLoader
from ..memory.lds_reader import LDSReader

__all__ = [
    "TilePartitioner", "GridPartitioner", "StreamKPartitioner",
    "ComputePipeline",
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
                 epilogue: Optional[object] = None) -> None:
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
    """Phase function: unified stream architecture K-loop."""
    return pipeline_v2_kloop_phase(level, ctx)


# ── Shared helpers ────────────────────────────────────────────────

def _build_loader_reader_scale(ctx):
    """Construct the loader, reader, and scale_loader from ctx metadata.

    Returns (loader, reader, scale_loader, pgr, use_lds_scales).
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
    use_lds_scales = ctx._metadata.get("use_lds_scales", False)
    layout = ctx._metadata.get("layout")
    if use_real_scales and layout.has_scales:
        if use_lds_scales:
            from ..memory.scale_loader import LDSScaleLoader
            lds_data_half = ctx._metadata.get("lds_data_half", 0)
            scale_loader = LDSScaleLoader(ctx, tile,
                                          lds_scale_offset=lds_data_half)
        else:
            from ..memory.scale_loader import VMEMScaleLoader
            swizzled = ctx._metadata.get("swizzled_scales", False)
            scale_loader = VMEMScaleLoader(ctx, tile, swizzled=swizzled)

    pgr_raw = ctx._metadata.get("pgr", None)
    if pgr_raw is None:
        pgr = 2 if ctx._metadata.get("pgr2", False) else 1
    else:
        pgr = int(pgr_raw)

    return loader, reader, scale_loader, pgr, use_lds_scales


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
        """Emit single-launch StreamK work decomposition.

        Each WG computes its own tile and partition from its flat WG ID.
        No per-partition kernarg packing needed on the host.

        Kernarg layout (reuses TensileLite header slots):
          offset 0:  num_partitions (u32)
          offset 4:  unused (u32)
          offset 8:  unused (u32)
          offset 12: total_wgs (u32) -- for WGMXCC
          offset 40: workspace_ptr (u64) -- C slot
          offset 64: flags_ptr lo/hi (u32x2) -- stride D slots

        GPU-side computation from wg_serial (s_wg_id_x):
          tile_idx      = wg_serial >> log2(num_partitions)
          partition_idx = wg_serial & (num_partitions - 1)
          tile_m        = tile_idx % tiles_m
          tile_n        = tile_idx / tiles_m
          k_tiles       = K / unroll_k
          iters_per_part = k_tiles >> log2(num_partitions)
          extra          = k_tiles & (num_partitions - 1)
          iter_start     = partition_idx * iters_per_part
                           + min(partition_idx, extra)
          s_k_tiles      = iters_per_part
                           + (1 if partition_idx < extra else 0)
          is_partial     = (num_partitions > 1) ? 1 : 0

        Requires num_partitions to be a power of 2.
        """
        tile = ctx._metadata["tile"]
        problem = ctx._metadata["problem"]
        elem = problem.element_bytes
        log2_uk = int(math.log2(tile.unroll_k))
        log2_wgm = int(math.log2(tile.wg_m))

        ctx.comment("=== StreamK Single-Launch Work Decomposition ===")

        # Allocate persistent SGPRs used by the epilogue
        karg = ctx.sreg("s_kernarg")
        ctx.alloc_sgpr_permanent(2, "s_workspace_ptr")
        ctx.alloc_sgpr_permanent(1, "s_iter_start")
        ctx.alloc_sgpr_permanent(1, "s_is_partial")
        ctx.alloc_sgpr_permanent(1, "s_partition_idx")
        ctx.alloc_sgpr_permanent(2, "s_flags_ptr")
        ctx.alloc_sgpr_permanent(1, "s_num_partitions")

        # Load num_partitions, workspace_ptr, flags_ptr from kernargs
        ctx.inst("s_load_dword", ctx.sreg("s_num_partitions"), karg, "0",
                 comment="num_partitions (header[0])")
        ctx.inst("s_load_dwordx2", ctx.sreg("s_workspace_ptr"), karg,
                 "40", comment="workspace ptr (C slot)")
        # flags_ptr location depends on kernarg layout:
        # FP16: offset 64 (strideD0/D1), MX: offset 80 (strideD0/D1)
        layout = ctx._metadata.get("layout")
        flags_offset = str(layout.flags_ptr_offset) if layout else "64"
        ctx.inst("s_load_dwordx2", ctx.sreg("s_flags_ptr"), karg,
                 flags_offset, comment="flags ptr (strideD slot)")
        ctx.s_waitcnt("lgkmcnt(0)", comment="wait SK kernargs")
        ctx.raw("")

        # --- Decompose flat wg_serial into tile + partition ---
        # num_partitions is power-of-2; use shift/mask
        ctx.comment("tile_idx = wg_serial / num_partitions")
        ctx.inst("s_ff1_i32_b32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_num_partitions"),
                 comment="log2(num_partitions)")
        ctx.inst("s_lshr_b32", ctx.sreg("s_tmp1"),
                 ctx.sreg("s_wg_id_x"), ctx.sreg("s_tmp0"),
                 comment="tile_idx = wg_serial >> log2(npart)")
        ctx.inst("s_sub_u32", ctx.sreg("s_partition_idx"),
                 ctx.sreg("s_num_partitions"), "1",
                 comment="npart - 1 (mask)")
        ctx.inst("s_and_b32", ctx.sreg("s_partition_idx"),
                 ctx.sreg("s_wg_id_x"), ctx.sreg("s_partition_idx"),
                 comment="partition_idx = wg_serial & (npart-1)")
        # s_tmp1 = tile_idx, s_tmp0 = log2(npart)
        ctx.raw("")

        # Decompose tile_idx -> tile_m, tile_n
        # tiles_m = M >> log2(wg_m) (power-of-2 assumed)
        ctx.comment("Decompose tile_idx into tile_m, tile_n")
        ctx.inst("s_lshr_b32", ctx.sreg("s_iter_start"),
                 ctx.sreg("s_M"), str(log2_wgm),
                 comment=f"tiles_m = M / {tile.wg_m}")
        # tile_m = tile_idx % tiles_m; tile_n = tile_idx / tiles_m
        ctx.inst("s_ff1_i32_b32", ctx.sreg("s_is_partial"),
                 ctx.sreg("s_iter_start"),
                 comment="log2(tiles_m)")
        ctx.inst("s_sub_u32", ctx.sreg("s_iter_start"),
                 ctx.sreg("s_iter_start"), "1",
                 comment="tiles_m - 1 (mask)")
        ctx.inst("s_and_b32", ctx.sreg("s_wg_id_x"),
                 ctx.sreg("s_tmp1"), ctx.sreg("s_iter_start"),
                 comment="tile_m = tile_idx & (tiles_m-1)")
        ctx.inst("s_lshr_b32", ctx.sreg("s_wg_id_y"),
                 ctx.sreg("s_tmp1"), ctx.sreg("s_is_partial"),
                 comment="tile_n = tile_idx >> log2(tiles_m)")
        ctx.raw("")

        # --- Compute K-range for this partition ---
        ctx.comment("Compute iter_start / s_k_tiles from partition_idx")
        ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
                   comment=f"k_tiles = K / {tile.unroll_k}")
        # iters_per_part = k_tiles >> log2(npart)
        # s_tmp0 still holds log2(npart) from above
        ctx.inst("s_lshr_b32", ctx.sreg("s_iter_start"),
                 ctx.sreg("s_k_tiles"), ctx.sreg("s_tmp0"),
                 comment="iters_per_part = k_tiles / npart")
        # extra = k_tiles & (npart - 1)
        ctx.inst("s_sub_u32", ctx.sreg("s_is_partial"),
                 ctx.sreg("s_num_partitions"), "1",
                 comment="npart - 1")
        ctx.inst("s_and_b32", ctx.sreg("s_is_partial"),
                 ctx.sreg("s_k_tiles"), ctx.sreg("s_is_partial"),
                 comment="extra = k_tiles & (npart-1)")
        # iter_start = partition_idx * iters_per_part + min(partition_idx, extra)
        ctx.s_mul(ctx.sreg("s_tmp1"), ctx.sreg("s_partition_idx"),
                  ctx.sreg("s_iter_start"),
                  comment="partition_idx * iters_per_part")
        ctx.inst("s_min_u32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_partition_idx"), ctx.sreg("s_is_partial"),
                 comment="min(partition_idx, extra)")
        ctx.inst("s_add_u32", ctx.sreg("s_iter_start"),
                 ctx.sreg("s_tmp1"), ctx.sreg("s_tmp0"),
                 comment="iter_start = part*ipp + min(part, extra)")
        # s_k_tiles = iters_per_part + (partition_idx < extra ? 1 : 0)
        ctx.inst("s_ff1_i32_b32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_num_partitions"),
                 comment="log2(num_partitions) again")
        ctx.inst("s_lshr_b32", ctx.sreg("s_tmp1"),
                 ctx.sreg("s_k_tiles"), ctx.sreg("s_tmp0"),
                 comment="iters_per_part (recomputed)")
        ctx.inst("s_cmp_lt_u32", ctx.sreg("s_partition_idx"),
                 ctx.sreg("s_is_partial"),
                 comment="partition_idx < extra?")
        ctx.inst("s_cselect_b32", ctx.sreg("s_tmp0"), "1", "0",
                 comment="bonus = scc ? 1 : 0")
        ctx.inst("s_add_u32", ctx.sreg("s_k_tiles"),
                 ctx.sreg("s_tmp1"), ctx.sreg("s_tmp0"),
                 comment="s_k_tiles = iters_per_part + bonus")
        ctx.raw("")

        # is_partial = (num_partitions > 1) ? 1 : 0
        ctx.inst("s_cmp_gt_u32", ctx.sreg("s_num_partitions"), "1",
                 comment="num_partitions > 1?")
        ctx.inst("s_cselect_b32", ctx.sreg("s_is_partial"), "1", "0",
                 comment="is_partial = (npart > 1)")
        ctx.raw("")

        # Recompute SRD A/B bases using the corrected tile coords.
        # The setup phase computed SRDs from the raw s_wg_id_x/y which
        # held the flat serial. Now that we've set them to tile_m/tile_n,
        # recompute: SRD_A = ptr_A + tile_m * wg_m * K * elem
        #            SRD_B = ptr_B + tile_n * wg_n * K * elem
        ctx.comment("Recompute SRD A/B for corrected tile coords")
        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"),
                  str(tile.wg_m), comment=f"tile_m * {tile.wg_m}")
        ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                 ctx.sreg("s_k_stride"),
                 comment="* K_stride (K * elem)")
        ctx.inst("s_add_u32", ctx.sreg("s_srd_a", 0, 1),
                 ctx.sreg("s_ptr_A", 0, 1), ctx.sreg("s_tmp0"),
                 comment="SRD_A = ptr_A + row_offset")
        ctx.inst("s_addc_u32", ctx.sreg("s_srd_a", 1, 1),
                 ctx.sreg("s_ptr_A", 1, 1), "0", comment="carry")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_a", 2, 1),
                 "0xFFFFFFFF", comment="limit")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_a", 3, 1),
                 "0x20000", comment="flags")

        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_y"),
                  str(tile.wg_n), comment=f"tile_n * {tile.wg_n}")
        ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                 ctx.sreg("s_k_stride"),
                 comment="* K_stride (K * elem)")
        ctx.inst("s_add_u32", ctx.sreg("s_srd_b", 0, 1),
                 ctx.sreg("s_ptr_B", 0, 1), ctx.sreg("s_tmp0"),
                 comment="SRD_B = ptr_B + row_offset")
        ctx.inst("s_addc_u32", ctx.sreg("s_srd_b", 1, 1),
                 ctx.sreg("s_ptr_B", 1, 1), "0", comment="carry")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_b", 2, 1),
                 "0xFFFFFFFF", comment="limit")
        ctx.inst("s_mov_b32", ctx.sreg("s_srd_b", 3, 1),
                 "0x20000", comment="flags")
        ctx.raw("")

        # Recompute scale SRDs for MX types
        if layout.has_scales and ctx.has("s_srd_scale_a"):
            use_swizzled = ctx._metadata.get("swizzled_scales", False)
            ctx.comment("Recompute scale SRD A/B for corrected tile coords")
            if use_swizzled:
                ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"),
                          str(tile.wg_m // 32),
                          comment=f"tile_m * {tile.wg_m // 32}")
            else:
                ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"),
                          str(tile.wg_m),
                          comment=f"tile_m * {tile.wg_m}")
            ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                     ctx.sreg("s_stride_scale_a"),
                     comment="* stride_scale_a")
            ctx.inst("s_add_u32", ctx.sreg("s_srd_scale_a", 0, 1),
                     ctx.sreg("s_ptr_scale_a", 0, 1), ctx.sreg("s_tmp0"),
                     comment="SRD_scaleA lo")
            ctx.inst("s_addc_u32", ctx.sreg("s_srd_scale_a", 1, 1),
                     ctx.sreg("s_ptr_scale_a", 1, 1), "0",
                     comment="SRD_scaleA hi")
            ctx.inst("s_mov_b32", ctx.sreg("s_srd_scale_a", 2, 1),
                     "0xFFFFFFFF", comment="limit")
            ctx.inst("s_mov_b32", ctx.sreg("s_srd_scale_a", 3, 1),
                     "0x20000", comment="flags")

            if use_swizzled:
                ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_y"),
                          str(tile.wg_n // 32),
                          comment=f"tile_n * {tile.wg_n // 32}")
            else:
                ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_y"),
                          str(tile.wg_n),
                          comment=f"tile_n * {tile.wg_n}")
            ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                     ctx.sreg("s_stride_scale_b"),
                     comment="* stride_scale_b")
            ctx.inst("s_add_u32", ctx.sreg("s_srd_scale_b", 0, 1),
                     ctx.sreg("s_ptr_scale_b", 0, 1), ctx.sreg("s_tmp0"),
                     comment="SRD_scaleB lo")
            ctx.inst("s_addc_u32", ctx.sreg("s_srd_scale_b", 1, 1),
                     ctx.sreg("s_ptr_scale_b", 1, 1), "0",
                     comment="SRD_scaleB hi")
            ctx.inst("s_mov_b32", ctx.sreg("s_srd_scale_b", 2, 1),
                     "0xFFFFFFFF", comment="limit")
            ctx.inst("s_mov_b32", ctx.sreg("s_srd_scale_b", 3, 1),
                     "0x20000", comment="flags")
            ctx.raw("")

        # Now apply K-offset if iter_start > 0 (separate pass since
        # we just rebuilt the SRD bases)
        ctx.inst("s_cmp_eq_u32", ctx.sreg("s_iter_start"), "0",
                 comment="skip K-offset if iter_start == 0")
        ctx.inst("s_cbranch_scc1", "sk_no_k_offset2",
                 comment="no K-offset needed")
        k_bytes_per_tile_iter = layout.k_offset_bytes(1, tile.unroll_k) if layout else int(tile.unroll_k * elem)
        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_iter_start"),
                  str(k_bytes_per_tile_iter),
                  comment=f"k_offset = iter_start * {k_bytes_per_tile_iter}")
        ctx.inst("s_add_u32", ctx.sreg("s_srd_a", 0, 1),
                 ctx.sreg("s_srd_a", 0, 1), ctx.sreg("s_tmp0"),
                 comment="SRD_A += k_offset")
        ctx.inst("s_addc_u32", ctx.sreg("s_srd_a", 1, 1),
                 ctx.sreg("s_srd_a", 1, 1), "0", comment="carry")
        ctx.inst("s_add_u32", ctx.sreg("s_srd_b", 0, 1),
                 ctx.sreg("s_srd_b", 0, 1), ctx.sreg("s_tmp0"),
                 comment="SRD_B += k_offset")
        ctx.inst("s_addc_u32", ctx.sreg("s_srd_b", 1, 1),
                 ctx.sreg("s_srd_b", 1, 1), "0", comment="carry")
        # Scale SRD K-offset: scale stride is K/mx_block, so
        # k_offset_scale = iter_start * (unroll_k / mx_block) * 1 byte
        if layout.has_scales and ctx.has("s_srd_scale_a"):
            scale_block = layout.scale_block if layout else tile.mfma.mx_block
            scale_k_bytes = tile.unroll_k // scale_block  # 1 byte per scale
            ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_iter_start"),
                      str(scale_k_bytes),
                      comment=f"scale_k_offset = iter_start * {scale_k_bytes}")
            ctx.inst("s_add_u32", ctx.sreg("s_srd_scale_a", 0, 1),
                     ctx.sreg("s_srd_scale_a", 0, 1), ctx.sreg("s_tmp0"),
                     comment="SRD_scaleA += scale_k_offset")
            ctx.inst("s_addc_u32", ctx.sreg("s_srd_scale_a", 1, 1),
                     ctx.sreg("s_srd_scale_a", 1, 1), "0", comment="carry")
            ctx.inst("s_add_u32", ctx.sreg("s_srd_scale_b", 0, 1),
                     ctx.sreg("s_srd_scale_b", 0, 1), ctx.sreg("s_tmp0"),
                     comment="SRD_scaleB += scale_k_offset")
            ctx.inst("s_addc_u32", ctx.sreg("s_srd_scale_b", 1, 1),
                     ctx.sreg("s_srd_scale_b", 1, 1), "0", comment="carry")
        ctx.label("sk_no_k_offset2")
        ctx.raw("")

    def grid_dims(self, problem, tile):
        """Compute grid dimensions for StreamK launch.

        Returns base tile count. The launcher multiplies by
        num_partitions to get the actual 1D grid size.
        """
        total_tiles = (problem.m // tile.wg_m) * (problem.n // tile.wg_n)
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


def pipeline_v2_kloop_phase(level, ctx) -> None:
    """Phase function using the new unified stream architecture.

    Default pipeline strategy. Uses LDSStream + LDSBufferManager +
    PipelineScheduler + PipelineEmitter.
    """
    tile = ctx._metadata["tile"]

    loader, reader, scale_loader, pgr, use_lds_scales = \
        _build_loader_reader_scale(ctx)

    # --- New architecture ---
    from ..memory.streams import DTLDataStream, ScaleStream
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
    if use_lds_scales and scale_loader is not None:
        streams.append(ScaleStream("a", tile))
        streams.append(ScaleStream("b", tile))
    streams.append(DTLDataStream("a", tile, problem))
    streams.append(DTLDataStream("b", tile, problem))

    num_buffers = 2 if pgr >= 1 else 1
    buffer_mgr = LDSBufferManager(streams, num_buffers=num_buffers)
    buffer_mgr.compute_layout()

    graph = build_kloop_graph(streams, tile, pgr=pgr,
                              num_buffers=num_buffers, problem=problem)

    # Setup: soffsets, swizzle (must happen before MFMAEmitter
    # creation so scale_names are populated)
    loader.precompute_soffsets()
    if scale_loader is not None and hasattr(scale_loader, 'precompute_soffsets'):
        scale_loader.precompute_soffsets()
    reader.precompute_swizzle_addresses()

    # Create MFMAEmitter (after scale registers allocated)
    layout = ctx._metadata.get("layout")
    if scale_loader is not None and hasattr(scale_loader, 'scale_names_a'):
        names_a = scale_loader.scale_names_a
        names_b = scale_loader.scale_names_b
        emitter = (MFMAEmitter.for_lds_scales(tile.mfma, names_a, names_b)
                   if use_lds_scales
                   else MFMAEmitter.for_vmem_scales(tile.mfma, names_a, names_b))
    elif layout.mfma_has_scale_operands:
        emitter = MFMAEmitter.for_mx_constant(tile.mfma)
    else:
        emitter = MFMAEmitter.for_non_mx(tile.mfma)

    # Wire emit callbacks
    wire_emit_callbacks(graph, streams, buffer_mgr, loader, reader,
                        emitter, ctx, scale_loader=scale_loader)

    scheduled = PipelineScheduler(graph).schedule()

    ctx.alloc_sgpr_permanent(1, "s_lds_db_step")
    lds_scale_half = ctx._metadata.get("lds_scale_half", 0)
    lds_data_half = ctx._metadata.get("lds_data_half", 0)
    lds_half_total = lds_data_half + lds_scale_half
    if not ctx.has("s_rd_db"):
        ctx.alloc_sgpr_permanent(1, "s_rd_db")
    ctx.s_mov(ctx.sreg("s_rd_db"), "0", comment="rd_db = 0")
    ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_half_total),
              comment=f"DB step = {lds_half_total}")
    ctx.raw("")

    # K-tile count
    if ctx._metadata.get("streamk"):
        StreamKPartitioner().emit(ctx)
    else:
        GridPartitioner().emit(ctx)

    # Emit K-loop
    PipelineEmitter(scheduled, buffer_mgr, ctx).emit()
