# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM kernel as a composable pipeline driven by the tile tree.

This is the middle-ground abstraction between Triton/CK (high-level,
can't touch assembly) and TensileLite (low-level, hard to modify one
piece without understanding everything).

Key ideas:
- The kernel is a **tile tree** with **phases** at each level
- Each phase is a named callable that can be replaced independently
- **MemoryViews** describe how to access tensors at each tile level
  via coordinate transforms
- The tree drives the full codegen: prologue, K-loop, and epilogue
  are all phases on tree nodes
- Named register bindings (never raw register numbers)

The tree structure::

    workgroup(m=wg_m, n=wg_n, parallel=True)
      prologue: [load_kernargs, thread_indexing, ..., k_loop_init, k_loop_label]
      inner: wave(m=m_per_wave, n=n_per_wave, k=unroll_k)
        prologue: [global_load, lds_write]
        inner: mfma(m=16, n=16, k=16)
        epilogue: [k_advance, k_loop_control]
      epilogue: [store_d]

Usage::

    # Default kernel -- just works
    kernel = GemmKernel.build(problem)
    result = kernel.emit()
    result.assemble()

    # Replace a single phase (e.g., custom global load for prefetching)
    kernel = GemmKernel.build(problem)
    kernel.tile_tree = kernel.tile_tree.replace_phase(
        "global_load", my_prefetching_load)
    result = kernel.emit()

    # Replace the MFMA leaf for custom scheduling
    kernel = GemmKernel.build(problem)
    kernel.tile_tree = kernel.tile_tree.replace("mfma", emit=my_mfma)
    result = kernel.emit()

    # Access tensor data at any level via MemoryView
    def my_custom_compute(level, ctx):
        a_view = ctx.get_view("A")  # LDS view at this level
        a_view.emit_read(ctx, dst="v_a", m=mi*16, k=ki*16)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .asm_context import AsmContext
from .asm_transforms import emit_affine, GemmLayouts
from .problem import GemmProblem, TileConfig, MfmaConfig
from .tiling import GemmTiling
from .tile import TileLevel, TilePhase, build_gemm_tile_tree, walk_tile_tree
from .transforms import Embed, Dim

__all__ = ["MemoryView", "GemmKernel", "AsmKernel"]


# ===================================================================
# MemoryView: how to access a tensor at a given tile level
# ===================================================================

@dataclass
class MemoryView:
    """How to read/write a tensor through coordinate transforms.

    At each tile level, tensors have a MemoryView that describes:
    - Where the data lives (global memory, LDS, or accumulators)
    - The Embed transform from tile coordinates to element offset
    - The base register (e.g. LDS base addr, global pointer)

    A researcher at any tile level calls::

        a_view = ctx.get_view("A")
        a_view.emit_read(ctx, ...)

    and gets the right addressing regardless of whether A is in
    global memory, LDS, or registers.
    """
    name: str           # "A", "B", "D"
    source: str         # "global", "lds", "acc"
    layout: Embed       # transform: tile coords -> element offset
    elem_bytes: int     # element size
    base_reg: Optional[str] = None   # base address register
    base_offset: int = 0             # constant offset (e.g. lds_b_offset)

    def emit_offset(self, ctx: AsmContext,
                    bindings: Dict[str, str],
                    result_reg: str) -> None:
        """Emit instructions to compute byte offset from tile coordinates."""
        emit_affine(ctx, self.layout, bindings, result_reg,
                    scale=self.elem_bytes,
                    base=str(self.base_offset) if self.base_offset else None,
                    comment=f"{self.name} offset: {self.layout}")

    def summary(self) -> str:
        return f"{self.name}({self.source}): {self.layout}"


# ===================================================================
# Extend AsmContext with MemoryView registry
# ===================================================================

def _register_view(ctx: AsmContext, view: MemoryView) -> None:
    """Register a MemoryView on the context."""
    if not hasattr(ctx, '_views'):
        ctx._views = {}
    ctx._views[view.name] = view


def _get_view(ctx: AsmContext, name: str) -> MemoryView:
    """Get a registered MemoryView by tensor name."""
    if not hasattr(ctx, '_views') or name not in ctx._views:
        available = list(getattr(ctx, '_views', {}).keys())
        raise KeyError(f"No MemoryView '{name}'. Available: {available}")
    return ctx._views[name]


# Monkey-patch onto AsmContext for convenience
AsmContext.register_view = _register_view
AsmContext.get_view = _get_view


# ===================================================================
# GemmKernel: the full kernel pipeline
# ===================================================================

@dataclass
class AsmKernel:
    """A generated assembly kernel ready to assemble."""
    asm_text: str
    kernel_name: str
    problem: GemmProblem
    tile: TileConfig
    ctx: AsmContext
    lds_bytes: int

    @property
    def vgpr_count(self) -> int:
        return self.ctx._next["v"]

    @property
    def sgpr_count(self) -> int:
        return self.ctx._next["s"]

    @property
    def acc_count(self) -> int:
        return self.ctx._next["acc"]

    def save(self, path: str) -> str:
        with open(path, "w") as f:
            f.write(self.asm_text)
        return path

    def assemble(self, gpu_arch: str = "gfx950",
                 output_path: Optional[str] = None) -> str:
        from .asm_emitter import assemble_kernel
        return assemble_kernel(self.asm_text, gpu_arch, output_path)


@dataclass
class GemmKernel:
    """A GEMM kernel described as a tile tree with phases.

    The entire kernel structure is encoded in the tile tree::

        workgroup
          prologue: [load_kernargs, thread_indexing, ..., k_loop_init, k_loop_label]
          inner: wave
            prologue: [global_load, lds_write]
            inner: mfma (leaf)
            epilogue: [k_advance, k_loop_control]
          epilogue: [store_d]

    Replace any phase to customize one step::

        kernel.tile_tree = kernel.tile_tree.replace_phase(
            "global_load", my_prefetching_load)
    """
    problem: GemmProblem
    tile: TileConfig
    layouts: GemmLayouts
    tile_tree: TileLevel
    kernel_name: str = "gemm_kernel"

    # MFMA visitor for the tile tree walk
    mfma_visitor: Callable = None

    @staticmethod
    def build(problem: GemmProblem,
              tile: Optional[TileConfig] = None,
              kernel_name: str = "gemm_kernel",
              tile_tree: Optional[TileLevel] = None,
              tiling: Optional[GemmTiling] = None) -> GemmKernel:
        """Build a GemmKernel with the full tile tree.

        Args:
            problem: GEMM problem specification.
            tile: Legacy TileConfig (used if tiling is None).
            tiling: GemmTiling with per-dimension TileDim chains.
                    If provided, tile and tile_tree are derived from it.
            kernel_name: Name for the kernel function.
            tile_tree: Custom TileLevel tree (overrides auto-generation).
        """
        if tiling is not None:
            tiling.validate()
            tile = tiling.to_tile_config()
            if tile_tree is None:
                tile_tree = tiling.build_tile_tree()
        if tile is None:
            tile = TileConfig()
        problem.validate(tile)

        layouts = GemmLayouts.build(problem, tile)

        if tile_tree is None:
            from .asm_emitter import build_full_gemm_tree
            tile_tree = build_full_gemm_tree(problem, tile, layouts)

        tile_tree.validate()

        kernel = GemmKernel(
            problem=problem,
            tile=tile,
            layouts=layouts,
            tile_tree=tile_tree,
            kernel_name=kernel_name,
            mfma_visitor=default_mfma_visitor,
        )
        return kernel

    def emit(self) -> AsmKernel:
        """Generate the full kernel assembly.

        Walks the tile tree which handles everything: prologue phases,
        K-loop (via phases), MFMA compute (via visitor), and epilogue.
        """
        from .asm_emitter import (
            _alloc_registers, _emit_header, _emit_descriptor,
        )

        tile = self.tile
        elem = self.problem.element_bytes
        lds_total = (tile.wg_m + tile.wg_n) * tile.unroll_k * elem

        ctx = AsmContext()
        ctx._metadata = {
            "tile": tile,
            "problem": self.problem,
            "layouts": self.layouts,
            "kernel": self,
        }
        _alloc_registers(ctx, self.problem, tile)

        # Register LDS MemoryViews for tensor access at any level
        ctx.register_view(MemoryView(
            name="A", source="lds",
            layout=self.layouts.lds_a,
            elem_bytes=elem,
            base_reg="v_lds_rd_a",
        ))
        ctx.register_view(MemoryView(
            name="B", source="lds",
            layout=self.layouts.lds_b,
            elem_bytes=elem,
            base_reg="v_lds_rd_b",
            base_offset=self.layouts.lds_b_offset,
        ))

        _emit_header(ctx, self.kernel_name)

        # Walk the full tile tree -- phases handle everything
        walk_tile_tree(self.tile_tree, ctx, visitor=self.mfma_visitor)

        ctx.inst("s_endpgm", comment="end of kernel")
        _emit_descriptor(ctx, self.kernel_name, lds_total, tile)

        return AsmKernel(
            asm_text=ctx.asm_text(),
            kernel_name=self.kernel_name,
            problem=self.problem,
            tile=tile,
            ctx=ctx,
            lds_bytes=lds_total,
        )


# ===================================================================
# Default MFMA visitor (backward-compat hook for the tile tree walk)
# ===================================================================

def default_mfma_visitor(level: TileLevel, ctx: AsmContext) -> None:
    """Tile-tree visitor: emit LDS reads + MFMAs at the mfma leaf level."""
    from .asm_emitter import _mfma_visitor
    _mfma_visitor(level, ctx)
