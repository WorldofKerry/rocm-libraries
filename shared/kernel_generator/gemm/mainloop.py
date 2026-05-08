"""Mainloop: composed kernel configuration with no flags.

A Mainloop is a concrete combination of components that fully
determines the generated kernel. Each component owns its own
codegen -- the emit path calls methods unconditionally.

Usage::

    ml = mainloop_mxfp4(pgr=2, streamk=True)
    kernel = GemmKernel.build(problem, mainloop=ml)

Adding a new optimization = adding a new component + factory.
No existing code is edited.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Type

from .grid import GridDecomposition, Grid2D, Grid1DXCC
from .kernarg_layout import (
    KernargLayout, FP16_LAYOUT, BF16_LAYOUT, MXFP4_LAYOUT, MXFP4_STREAMK_LAYOUT,
)
from .memory.swizzle import Swizzle, IdentitySwizzle
from .problem import TileConfig

__all__ = [
    "Mainloop", "StoreEpilogue", "DirectStore", "StreamKStore",
    "ScaleStrategy", "LDSScaleStrategy", "VMEMScaleStrategy",
    "mainloop_fp16", "mainloop_bf16", "mainloop_mxfp4",
    "mainloop_mxfp4_wave_abi",
]


# ── Store epilogue ────────────────────────────────────────────────

class StoreEpilogue:
    """Base: how accumulators are written to global memory."""

    def phase_func(self) -> Callable:
        """Return the phase function for the tile tree."""
        from .emit.phases import phase_store_d
        return phase_store_d


class DirectStore(StoreEpilogue):
    """Standard store: convert accumulators and write to D."""
    pass


class StreamKStore(StoreEpilogue):
    """StreamK 3-way epilogue: sole-owner / partial / owner-reduce."""

    def phase_func(self) -> Callable:
        from .emit.phases import phase_store_streamk
        return phase_store_streamk


# ── Scale strategy ────────────────────────────────────────────────

class ScaleStrategy:
    """No-op scale strategy for non-MX data types."""

    def build_loader(self, ctx: object, tile: TileConfig,
                     lds_data_half: int = 0) -> object:
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

    def build_loader(self, ctx: object, tile: TileConfig,
                     lds_data_half: int = 0) -> object:
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

    def __init__(self, swizzled: bool = False) -> None:
        self.swizzled: bool = swizzled

    def build_loader(self, ctx: object, tile: TileConfig,
                     lds_data_half: int = 0) -> object:
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
    loader_cls: Type                  # DTLLoader or BufferLoader class
    scale_strategy: ScaleStrategy
    swizzle: Swizzle
    pgr: int
    epilogue: StoreEpilogue
    colmajor_output: bool = False
    wave_abi: bool = False
    tensilelite_abi: bool = False

    @property
    def num_buffers(self) -> int:
        """LDS buffer count (derived from PGR)."""
        return 2 if self.pgr >= 1 else 1

    @property
    def is_streamk(self) -> bool:
        """Whether this mainloop uses StreamK work distribution."""
        return isinstance(self.epilogue, StreamKStore)

    def lds_scale_half(self, tile: TileConfig) -> int:
        """LDS bytes for scale data per buffer."""
        return self.scale_strategy.lds_bytes(tile, self.layout)

    def lds_data_half(self, tile: TileConfig) -> int:
        """LDS bytes for data A + B per buffer.

        Accounts for DTL per-load-line padding when applicable.
        """
        elem = self.layout.element_bytes()
        uk = tile.unroll_k
        pad = tile.lds_pad

        if pad > 0:
            # DTL: per-load-line padding
            threads_per_row = int(uk * elem) // 16
            rows_per_load = tile.block_size // threads_per_row
            num_loads_a = tile.wg_m // rows_per_load
            num_loads_b = tile.wg_n // rows_per_load
            lds_a = int(tile.wg_m * uk * elem) + num_loads_a * pad
            lds_b = int(tile.wg_n * uk * elem) + num_loads_b * pad
            return lds_a + lds_b

        pad_elems = 0
        lds_row_stride = uk + pad_elems
        return int((tile.wg_m + tile.wg_n) * lds_row_stride * elem)

    def lds_half_total(self, tile: TileConfig) -> int:
        """Total LDS bytes per double-buffer slot (data + scale)."""
        return self.lds_data_half(tile) + self.lds_scale_half(tile)

    def lds_total(self, tile: TileConfig) -> int:
        """Total LDS bytes (all buffers)."""
        return self.lds_half_total(tile) * self.num_buffers

    def resolve_swizzle(self, tile: TileConfig) -> Optional[Swizzle]:
        """Return the swizzle, auto-deriving if needed.

        Returns None when no swizzle is configured (identity case).
        """
        if not isinstance(self.swizzle, IdentitySwizzle):
            return self.swizzle
        resolved = tile.resolved_swizzle(self.layout.element_bytes())
        return resolved


# ── Factory functions ─────────────────────────────────────────────

def _make_grid(wg_mapping_xcc: int) -> GridDecomposition:
    """Create grid decomposition from WGMXCC value."""
    if wg_mapping_xcc > 1:
        return Grid1DXCC(wg_mapping_xcc)
    return Grid2D()


def mainloop_fp16(
    *,
    pgr: int = 1,
    streamk: bool = False,
    wg_mapping_xcc: int = 1,
    colmajor_output: bool = False,
) -> Mainloop:
    """Standard FP16 mainloop. DTL loads, no scales."""
    from .memory.global_loader import DTLLoader

    return Mainloop(
        layout=FP16_LAYOUT,
        grid=_make_grid(wg_mapping_xcc),
        loader_cls=DTLLoader,
        scale_strategy=ScaleStrategy(),
        swizzle=IdentitySwizzle(),
        pgr=pgr,
        epilogue=StreamKStore() if streamk else DirectStore(),
        colmajor_output=colmajor_output,
    )


def mainloop_bf16(
    *,
    pgr: int = 1,
    streamk: bool = False,
    wg_mapping_xcc: int = 1,
    colmajor_output: bool = False,
) -> Mainloop:
    """Standard BF16 mainloop. Same structure as FP16."""
    from .memory.global_loader import DTLLoader

    return Mainloop(
        layout=BF16_LAYOUT,
        grid=_make_grid(wg_mapping_xcc),
        loader_cls=DTLLoader,
        scale_strategy=ScaleStrategy(),
        swizzle=IdentitySwizzle(),
        pgr=pgr,
        epilogue=StreamKStore() if streamk else DirectStore(),
        colmajor_output=colmajor_output,
    )


def mainloop_mxfp4(
    *,
    pgr: int = 1,
    streamk: bool = False,
    wg_mapping_xcc: int = 1,
    colmajor_output: bool = False,
) -> Mainloop:
    """MXFP4 mainloop. DTL loads + LDS scale loading."""
    from .memory.global_loader import DTLLoader

    return Mainloop(
        layout=MXFP4_LAYOUT,
        grid=_make_grid(wg_mapping_xcc),
        loader_cls=DTLLoader,
        scale_strategy=LDSScaleStrategy(),
        swizzle=IdentitySwizzle(),
        pgr=pgr,
        epilogue=StreamKStore() if streamk else DirectStore(),
        colmajor_output=colmajor_output,
    )


def mainloop_mxfp4_wave_abi(
    *,
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
        wave_abi=True,
    )


def mainloop_mxfp4_tensilelite(
    *,
    pgr: int = 1,
    wg_mapping_xcc: int = 1,
    colmajor_output: bool = False,
    swizzled_scales: bool = False,
    streamk: bool = False,
) -> Mainloop:
    """MXFP4 mainloop for TensileLite custom kernels with LDS scale loading.

    Uses LDS-based scale loading via true DTL (buffer_load_dwordx4 ... lds)
    for zero VMEM scale loads. Scale data goes directly from global
    memory to LDS, then ds_read_b32 with op_sel byte selection.

    Args:
        swizzled_scales: Accepted for API compatibility. LDS scales
            always use pre-swizzled format (MXScaleFormat=1).
        streamk: Enable StreamK work distribution for better CU utilization.
    """
    from .memory.global_loader import DTLLoader

    layout = MXFP4_STREAMK_LAYOUT if streamk else MXFP4_LAYOUT
    return Mainloop(
        layout=layout,
        grid=_make_grid(wg_mapping_xcc),
        loader_cls=DTLLoader,
        scale_strategy=LDSScaleStrategy(),
        swizzle=IdentitySwizzle(),
        pgr=pgr,
        epilogue=StreamKStore() if streamk else DirectStore(),
        colmajor_output=colmajor_output,
        tensilelite_abi=True,
    )
