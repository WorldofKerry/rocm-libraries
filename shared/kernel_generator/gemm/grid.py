"""Grid decomposition strategies.

Each strategy converts hardware WG IDs (s_wg_id_x, s_wg_id_y) into
tile coordinates. The setup phase calls grid.emit(ctx, tile)
unconditionally -- the Grid object decides what to emit.

Usage::

    grid = Grid1DXCC(wg_mapping_xcc=8)
    grid.emit(ctx, tile)
    # s_wg_id_x = tile_m, s_wg_id_y = tile_n
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

from .emit.context import AsmContext
from .problem import TileConfig

__all__ = ["GridDecomposition", "Grid2D", "Grid1DXCC"]


class GridDecomposition(ABC):
    """Base: convert hardware WG IDs to tile coordinates."""

    @abstractmethod
    def emit(self, ctx: AsmContext, tile: TileConfig) -> None:
        """Emit WG ID decomposition. After this call,
        s_wg_id_x = tile_m, s_wg_id_y = tile_n."""

    @abstractmethod
    def grid_dims(self, tiles_m: int, tiles_n: int) -> tuple:
        """Return (grid_x, grid_y) for the host launch."""


class Grid2D(GridDecomposition):
    """2D grid -- hardware WG IDs are already tile coordinates. No-op."""

    def emit(self, ctx: AsmContext, tile: TileConfig) -> None:
        # s_wg_id_x and s_wg_id_y are already tile_m and tile_n
        pass

    def grid_dims(self, tiles_m: int, tiles_n: int) -> tuple:
        return (tiles_m, tiles_n)


class Grid1DXCC(GridDecomposition):
    """1D grid with optional cross-XCC WG remapping for L2 locality.

    Flattens the 2D tile grid into a 1D serial, optionally applies
    WGMXCC interleaving, then decomposes back to (tile_m, tile_n).
    """

    def __init__(self, wg_mapping_xcc: int = 1) -> None:
        self.wg_mapping_xcc = wg_mapping_xcc

    def emit(self, ctx: AsmContext, tile: TileConfig) -> None:
        log2_mt = int(math.log2(tile.wg_m))
        wgmxcc = self.wg_mapping_xcc

        if wgmxcc > 1:
            log2_xcc = int(math.log2(wgmxcc))
            ctx.comment(f"WorkGroupMappingXCC={wgmxcc}: remap for L2 locality")
            ctx.alloc_sgpr_permanent(1, "s_numWG")
            ctx.inst("s_load_dword", ctx.sreg("s_numWG"),
                     ctx.sreg("s_kernarg"), "12",
                     comment="numWG (total workgroups)")
            ctx.s_waitcnt("lgkmcnt(0)", comment="wait numWG")
            ctx.inst("s_lshr_b32", ctx.sreg("s_tmp0"),
                     ctx.sreg("s_wg_id_x"), str(log2_xcc),
                     comment=f"old_wg / {wgmxcc}")
            ctx.inst("s_and_b32", ctx.sreg("s_tmp1"),
                     ctx.sreg("s_wg_id_x"), str(wgmxcc - 1),
                     comment=f"old_wg % {wgmxcc} (XCC lane)")
            ctx.inst("s_lshr_b32", ctx.sreg("s_numWG"),
                     ctx.sreg("s_numWG"), str(log2_xcc),
                     comment=f"numWG / {wgmxcc}")
            ctx.inst("s_mul_i32", ctx.sreg("s_tmp1"),
                     ctx.sreg("s_tmp1"), ctx.sreg("s_numWG"),
                     comment="XCC_lane * (numWG / WGMXCC)")
            ctx.inst("s_add_u32", ctx.sreg("s_wg_id_x"),
                     ctx.sreg("s_tmp0"), ctx.sreg("s_tmp1"),
                     comment="remapped WG serial")
            ctx.raw("")

        # 1D -> 2D: tile_m = serial % tiles_m, tile_n = serial / tiles_m
        ctx.s_mov(ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_x"),
                  comment="save wg_serial")
        ctx.inst("s_add_u32", ctx.sreg("s_tmp0"), ctx.sreg("s_M"),
                 str(tile.wg_m - 1),
                 comment=f"M + {tile.wg_m - 1}")
        ctx.inst("s_lshr_b32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                 str(log2_mt),
                 comment=f"numWG_m = ceil(M/{tile.wg_m})")
        ctx.inst("s_sub_u32", ctx.sreg("s_wg_id_x"),
                 ctx.sreg("s_tmp0"), "1",
                 comment="numWG_m - 1")
        ctx.inst("s_and_b32", ctx.sreg("s_wg_id_y"),
                 ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"),
                 comment="numWG_m & (numWG_m-1) == 0 if power-of-2")
        ctx.inst("s_ff1_i32_b32", ctx.sreg("s_wg_id_y"),
                 ctx.sreg("s_tmp0"),
                 comment="log2(numWG_m)")
        ctx.inst("s_lshr_b32", ctx.sreg("s_wg_id_y"),
                 ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_y"),
                 comment="tile_n = serial >> log2(numWG_m)")
        ctx.inst("s_sub_u32", ctx.sreg("s_wg_id_x"),
                 ctx.sreg("s_tmp0"), "1",
                 comment="numWG_m - 1")
        ctx.inst("s_and_b32", ctx.sreg("s_wg_id_x"),
                 ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_x"),
                 comment="tile_m = serial & (numWG_m - 1)")
        ctx.raw("")

    def grid_dims(self, tiles_m: int, tiles_n: int) -> tuple:
        return (tiles_m * tiles_n, 1)
