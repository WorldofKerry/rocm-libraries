"""Mainloop: composed kernel configuration with no flags.

A Mainloop is a concrete combination of components that fully
determines the generated kernel. Each component owns its own
codegen -- the emit path calls methods unconditionally.

Usage::

    ml = mainloop_mxfp4_pgr1(tile)
    kernel = GemmKernel.build(problem, mainloop=ml)
    # No flags: use_dtl, use_lds_scales, streamk, etc. are all
    # determined by the mainloop's component choices.

Adding a new optimization = adding a new component + factory.
No existing code is edited.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

from .grid import GridDecomposition, Grid2D, Grid1DXCC
from .kernarg_layout import KernargLayout, layout_for, FP16_LAYOUT, BF16_LAYOUT, MXFP4_LAYOUT
from .memory.swizzle import Swizzle, IdentitySwizzle, auto_swizzle, DataLayout, LDS_GFX950
from .problem import DataType, TileConfig, MfmaConfig

__all__ = [
    "Mainloop",
    "mainloop_fp16",
    "mainloop_mxfp4",
]


# ── Store epilogue ────────────────────────────────────────────────

class StoreEpilogue:
    """Base: how accumulators are written to global memory."""

    def phase_func(self):
        """Return the phase function for the tile tree."""
        from .emit.phases import phase_store_d
        return phase_store_d


class DirectStore(StoreEpilogue):
    """Standard store: convert accumulators and write to D."""
    pass  # inherits phase_func -> phase_store_d


class StreamKStore(StoreEpilogue):
    """StreamK 3-way epilogue: sole-owner / partial / owner-reduce."""

    def phase_func(self):
        from .emit.phases import phase_store_streamk
        return phase_store_streamk


# ── Scale strategy (reuses ScaleLoader hierarchy) ─────────────────

class ScaleStrategy:
    """Declares which ScaleLoader to construct at emit time.

    We can't construct the actual ScaleLoader here because it needs
    ctx (AsmContext) which only exists during emission. Instead we
    store the factory parameters and construct during emit.
    """

    def build_loader(self, ctx, tile, lds_data_half: int = 0):
        """Construct the concrete ScaleLoader during emission."""
        from .memory.scale_loader import NullScaleLoader
        return NullScaleLoader()

    @property
    def needs_lds(self) -> bool:
        """Whether this strategy uses LDS for scale data."""
        return False

    def lds_bytes(self, tile: TileConfig, layout: KernargLayout) -> int:
        """LDS bytes per buffer for scale data."""
        return 0


class LDSScaleStrategy(ScaleStrategy):
    """Load scales via DTL into LDS, read via ds_read."""

    def build_loader(self, ctx, tile, lds_data_half: int = 0):
        from .memory.scale_loader import LDSScaleLoader
        return LDSScaleLoader(ctx, tile, lds_scale_offset=lds_data_half)

    @property
    def needs_lds(self) -> bool:
        return True

    def lds_bytes(self, tile: TileConfig, layout: KernargLayout) -> int:
        scale_a = max(tile.wg_m * (tile.unroll_k // layout.scale_block), 4096)
        scale_b = max(tile.wg_n * (tile.unroll_k // layout.scale_block), 4096)
        return scale_a + scale_b


class VMEMScaleStrategy(ScaleStrategy):
    """Load scales directly from VMEM into VGPRs."""

    def __init__(self, swizzled: bool = False):
        self.swizzled = swizzled

    def build_loader(self, ctx, tile, lds_data_half: int = 0):
        from .memory.scale_loader import VMEMScaleLoader
        return VMEMScaleLoader(ctx, tile, swizzled=self.swizzled)


# ── Mainloop ──────────────────────────────────────────────────────

@dataclass
class Mainloop:
    """Fully-determined kernel configuration. No flags.

    Each field is a concrete component that owns its codegen.
    The emit path calls methods on these components unconditionally.
    """
    layout: KernargLayout
    grid: GridDecomposition
    loader_cls: type              # DTLLoader or BufferLoader class
    scale_strategy: ScaleStrategy
    swizzle: Swizzle
    pgr: int
    epilogue: StoreEpilogue
    colmajor_output: bool = False

    @property
    def num_buffers(self) -> int:
        """LDS buffer count (derived from PGR)."""
        return 2 if self.pgr >= 1 else 1

    def lds_scale_half(self, tile: TileConfig) -> int:
        """LDS bytes for scale data per buffer."""
        return self.scale_strategy.lds_bytes(tile, self.layout)

    def resolve_swizzle(self, tile: TileConfig):
        """Return the swizzle, auto-deriving if needed.
        
        Returns None when no swizzle is configured (identity case),
        matching the legacy flag path behavior.
        """
        if not isinstance(self.swizzle, IdentitySwizzle):
            return self.swizzle
        # Check if tile has auto-swizzle configured
        resolved = tile.resolved_swizzle(self.layout.element_bytes())
        return resolved  # None if no swizzle configured


# ── Factory functions ─────────────────────────────────────────────

def mainloop_fp16(
    pgr: int = 1,
    streamk: bool = False,
    wg_mapping_xcc: int = 1,
    colmajor_output: bool = False,
) -> Mainloop:
    """Standard FP16 mainloop. DTL loads, no scales."""
    from .memory.global_loader import DTLLoader

    if wg_mapping_xcc > 1:
        grid = Grid1DXCC(wg_mapping_xcc)
    else:
        grid = Grid2D()

    return Mainloop(
        layout=FP16_LAYOUT,
        grid=grid,
        loader_cls=DTLLoader,
        scale_strategy=ScaleStrategy(),  # null (no scales)
        swizzle=IdentitySwizzle(),       # auto-derived from tile
        pgr=pgr,
        epilogue=StreamKStore() if streamk else DirectStore(),
        colmajor_output=colmajor_output,
    )


def mainloop_bf16(
    pgr: int = 1,
    streamk: bool = False,
    wg_mapping_xcc: int = 1,
    colmajor_output: bool = False,
) -> Mainloop:
    """Standard BF16 mainloop. Same structure as FP16."""
    from .memory.global_loader import DTLLoader

    if wg_mapping_xcc > 1:
        grid = Grid1DXCC(wg_mapping_xcc)
    else:
        grid = Grid2D()

    return Mainloop(
        layout=BF16_LAYOUT,
        grid=grid,
        loader_cls=DTLLoader,
        scale_strategy=ScaleStrategy(),
        swizzle=IdentitySwizzle(),
        pgr=pgr,
        epilogue=StreamKStore() if streamk else DirectStore(),
        colmajor_output=colmajor_output,
    )


def mainloop_mxfp4(
    pgr: int = 1,
    streamk: bool = False,
    wg_mapping_xcc: int = 1,
    colmajor_output: bool = False,
    swizzled_scales: bool = False,
) -> Mainloop:
    """MXFP4 mainloop. DTL loads + LDS scale loading."""
    from .memory.global_loader import DTLLoader

    if wg_mapping_xcc > 1:
        grid = Grid1DXCC(wg_mapping_xcc)
    else:
        grid = Grid2D()

    return Mainloop(
        layout=MXFP4_LAYOUT,
        grid=grid,
        loader_cls=DTLLoader,
        scale_strategy=LDSScaleStrategy(),
        swizzle=IdentitySwizzle(),  # auto-derived from tile
        pgr=pgr,
        epilogue=StreamKStore() if streamk else DirectStore(),
        colmajor_output=colmajor_output,
    )


def mainloop_mxfp4_wave_abi(
    pgr: int = 1,
    swizzled_scales: bool = False,
) -> Mainloop:
    """MXFP4 for rocRoller/hipBLASLt wave ABI path."""
    from .memory.global_loader import DTLLoader

    return Mainloop(
        layout=MXFP4_LAYOUT,
        grid=Grid2D(),
        loader_cls=DTLLoader,
        scale_strategy=VMEMScaleStrategy(swizzled=swizzled_scales),
        swizzle=IdentitySwizzle(),
        pgr=pgr,
        epilogue=DirectStore(),
        colmajor_output=True,
    )
