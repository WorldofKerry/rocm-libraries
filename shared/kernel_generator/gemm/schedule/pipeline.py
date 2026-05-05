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
    if use_real_scales and tile.mfma.is_mx:
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

        ctx.comment("=== StreamK Work Decomposition ===")

        # Load StreamK kernargs
        karg = ctx.sreg("s_kernarg")
        ctx.alloc_sgpr_permanent(2, "s_workspace_ptr")
        ctx.alloc_sgpr_permanent(1, "s_iter_start")
        ctx.alloc_sgpr_permanent(1, "s_iter_end")
        ctx.alloc_sgpr_permanent(1, "s_k_tiles_per_tile")
        ctx.alloc_sgpr_permanent(1, "s_num_m_tiles")
        ctx.alloc_sgpr_permanent(1, "s_is_partial")
        ctx.alloc_sgpr_permanent(1, "s_partition_idx")

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
        ctx.inst("s_load_dword", ctx.sreg("s_partition_idx"), karg, "156",
                 comment="partition_idx (workspace slot)")
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
    if scale_loader is not None and hasattr(scale_loader, 'scale_names_a'):
        names_a = scale_loader.scale_names_a
        names_b = scale_loader.scale_names_b
        emitter = (MFMAEmitter.for_lds_scales(tile.mfma, names_a, names_b)
                   if use_lds_scales
                   else MFMAEmitter.for_vmem_scales(tile.mfma, names_a, names_b))
    elif tile.mfma.is_mx:
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
