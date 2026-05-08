"""Reusable tile-dependent operations for persistent and non-persistent loops.

Extracts duplicated SRD recompute, tile decomposition, accumulator
zeroing, and K-tile reset into standalone functions.  Used by both
the persistent tile loop and store epilogue.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..emit.context import AsmContext
    from ..mainloop import Mainloop
    from ..problem import TileConfig

__all__ = [
    "emit_decompose_tile_idx",
    "emit_recompute_srds",
    "emit_zero_accumulators",
    "emit_reset_kloop_state",
    "emit_build_raw_srd",
    "emit_compute_tile_serial",
]


def emit_build_raw_srd(
    ctx: 'AsmContext',
    srd_name: str,
    base_lo: str,
    base_hi: str,
) -> None:
    """Build a raw buffer SRD (4 SGPRs) from a 64-bit base pointer."""
    ctx.inst("s_mov_b32", ctx.sreg(srd_name, 0, 1), base_lo,
             comment=f"{srd_name} base lo")
    ctx.inst("s_mov_b32", ctx.sreg(srd_name, 1, 1), base_hi,
             comment=f"{srd_name} base hi")
    ctx.inst("s_mov_b32", ctx.sreg(srd_name, 2, 1), "0xFFFFFFFF",
             comment=f"{srd_name} size")
    ctx.inst("s_mov_b32", ctx.sreg(srd_name, 3, 1), "0x20000",
             comment=f"{srd_name} flags")


def emit_decompose_tile_idx(
    ctx: 'AsmContext',
    tile: 'TileConfig',
    tile_idx_reg: str = "s_tmp0",
) -> None:
    """Decompose a flat tile_idx into (tile_m, tile_n) in s_wg_id_x/y.

    Reads tile_idx from ``tile_idx_reg``.
    Uses s_tmp0, s_tmp1 as scratch.
    """
    ctx.comment("Decompose tile_idx -> tile_m, tile_n")

    # If tile_idx is in s_tmp0, save it before clobbering s_tmp0 with ff1.
    # s_wg_id_x is safe to use as temp since it's overwritten below.
    saved_reg = tile_idx_reg
    if tile_idx_reg == "s_tmp0":
        ctx.s_mov(ctx.sreg("s_wg_id_x"), ctx.sreg("s_tmp0"),
                  comment="save tile_idx")
        saved_reg = "s_wg_id_x"

    log2_wgm = int(math.log2(tile.wg_m))
    ctx.inst("s_lshr_b32", ctx.sreg("s_tmp1"), ctx.sreg("s_M"),
             str(log2_wgm), comment=f"tiles_m = M / {tile.wg_m}")
    ctx.inst("s_ff1_i32_b32", ctx.sreg("s_tmp0"),
             ctx.sreg("s_tmp1"), comment="log2(tiles_m)")
    ctx.inst("s_sub_u32", ctx.sreg("s_tmp1"),
             ctx.sreg("s_tmp1"), "1", comment="tiles_m - 1 (mask)")
    ctx.inst("s_and_b32", ctx.sreg("s_wg_id_x"),
             ctx.sreg(saved_reg), ctx.sreg("s_tmp1"),
             comment="tile_m = tile_idx & mask")
    ctx.inst("s_lshr_b32", ctx.sreg("s_wg_id_y"),
             ctx.sreg(saved_reg), ctx.sreg("s_tmp0"),
             comment="tile_n = tile_idx >> log2(tiles_m)")
    ctx.raw("")


def emit_recompute_data_srd(
    ctx: 'AsmContext',
    tile: 'TileConfig',
    matrix: str,
) -> None:
    """Recompute SRD for matrix A or B from tile_m/tile_n in s_wg_id_x/y."""
    assert matrix in ("a", "b")
    wg_id = "s_wg_id_x" if matrix == "a" else "s_wg_id_y"
    wg_dim = tile.wg_m if matrix == "a" else tile.wg_n
    srd = f"s_srd_{matrix}"
    ptr = f"s_ptr_{matrix.upper()}"

    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg(wg_id),
              str(wg_dim), comment=f"tile * {wg_dim}")
    ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
             ctx.sreg("s_k_stride"), comment="* K_stride")
    ctx.inst("s_add_u32", ctx.sreg(srd, 0, 1),
             ctx.sreg(ptr, 0, 1), ctx.sreg("s_tmp0"),
             comment=f"{srd} lo")
    ctx.inst("s_addc_u32", ctx.sreg(srd, 1, 1),
             ctx.sreg(ptr, 1, 1), "0", comment="hi")
    ctx.inst("s_mov_b32", ctx.sreg(srd, 2, 1), "0xFFFFFFFF",
             comment="limit")
    ctx.inst("s_mov_b32", ctx.sreg(srd, 3, 1), "0x20000",
             comment="flags")


def emit_recompute_scale_srd(
    ctx: 'AsmContext',
    tile: 'TileConfig',
    matrix: str,
    use_swizzled: bool = False,
) -> None:
    """Recompute scale SRD for matrix A or B."""
    assert matrix in ("a", "b")
    srd = f"s_srd_scale_{matrix}"
    ptr = f"s_ptr_scale_{matrix}"
    stride = f"s_stride_scale_{matrix}"
    wg_id = "s_wg_id_x" if matrix == "a" else "s_wg_id_y"
    wg_dim = tile.wg_m if matrix == "a" else tile.wg_n

    if not ctx.has(srd):
        return

    mul_val = wg_dim // 32 if use_swizzled else wg_dim
    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg(wg_id),
              str(mul_val), comment=f"tile * {mul_val}")
    ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
             ctx.sreg(stride), comment=f"* {stride}")
    ctx.inst("s_add_u32", ctx.sreg(srd, 0, 1),
             ctx.sreg(ptr, 0, 1), ctx.sreg("s_tmp0"),
             comment=f"{srd} lo")
    ctx.inst("s_addc_u32", ctx.sreg(srd, 1, 1),
             ctx.sreg(ptr, 1, 1), "0", comment="hi")
    ctx.inst("s_mov_b32", ctx.sreg(srd, 2, 1), "0xFFFFFFFF",
             comment="limit")
    ctx.inst("s_mov_b32", ctx.sreg(srd, 3, 1), "0x20000",
             comment="flags")


def emit_recompute_srds(
    ctx: 'AsmContext',
    tile: 'TileConfig',
    mainloop: 'Mainloop',
) -> None:
    """Recompute all SRDs (A, B, scale_A, scale_B) from s_wg_id_x/y."""
    ctx.comment("Recompute SRDs for tile")
    emit_recompute_data_srd(ctx, tile, "a")
    emit_recompute_data_srd(ctx, tile, "b")

    layout = ctx._metadata.get("layout")
    if layout and layout.has_scales:
        from ..mainloop import VMEMScaleStrategy
        use_swizzled = (isinstance(mainloop.scale_strategy, VMEMScaleStrategy)
                        and mainloop.scale_strategy.swizzled)
        emit_recompute_scale_srd(ctx, tile, "a", use_swizzled)
        emit_recompute_scale_srd(ctx, tile, "b", use_swizzled)
    ctx.raw("")


def emit_zero_accumulators(ctx: 'AsmContext', tile: 'TileConfig') -> None:
    """Zero all accumulator registers."""
    acc_total = tile.mfma_m_repeat * tile.mfma_n_repeat * tile.mfma.acc_vgprs
    ctx.comment(f"Zero {acc_total} accumulators")
    for i in range(acc_total):
        ctx.inst("v_accvgpr_write_b32", ctx.areg("acc_C", i, 1), "0")
    ctx.raw("")


def emit_reset_kloop_state(
    ctx: 'AsmContext',
    tile: 'TileConfig',
    mainloop: 'Mainloop',
    pgr: int,
) -> None:
    """Reset K-tile count and double-buffer state for a new tile.

    For PGR >= 2, restores from s_k_tiles_init.
    For PGR < 2, recomputes from s_K.
    """
    log2_uk = int(math.log2(tile.unroll_k))
    if pgr >= 2:
        ctx.s_mov(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles_init"),
                  comment="reset k_tiles from saved init")
    else:
        ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
                   comment=f"k_tiles = K / {tile.unroll_k}")

    ctx.s_mov(ctx.sreg("s_rd_db"), "0", comment="reset rd_db")
    lds_half_total = mainloop.lds_half_total(tile)
    ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_half_total),
              comment=f"reset DB step = {lds_half_total}")
    ctx.raw("")


def emit_compute_tile_serial(ctx: 'AsmContext', tile: 'TileConfig') -> None:
    """Compute tile_serial = wg_id_y * tiles_m + wg_id_x into s_tmp0.

    Used by StreamK workspace addressing and flag indexing.
    """
    log2_wgm = int(math.log2(tile.wg_m))
    ctx.inst("s_lshr_b32", ctx.sreg("s_tmp0"), ctx.sreg("s_M"),
             str(log2_wgm), comment=f"tiles_m = M / {tile.wg_m}")
    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_y"),
              ctx.sreg("s_tmp0"), comment="wg_id_y * tiles_m")
    ctx.inst("s_add_u32", ctx.sreg("s_tmp0"),
             ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"),
             comment="+ wg_id_x -> tile_serial")
