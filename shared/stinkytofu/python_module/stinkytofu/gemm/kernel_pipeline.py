# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM kernel pipeline: build tree, emit assembly, assemble.

Usage::

    kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
    result = kernel.emit()
    co = result.assemble()

    # Replace a phase
    kernel.tile_tree = kernel.tile_tree.replace_phase(
        "global_load", my_prefetching_load)
    result = kernel.emit()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .asm_context import AsmContext
from .asm_emitter import alloc_registers, emit_header, emit_descriptor, assemble_kernel
from .asm_transforms import emit_affine, GemmLayouts
from .phases import (
    WORKGROUP_PROLOGUE_PHASES, WORKGROUP_EPILOGUE_PHASES,
    WAVE_PROLOGUE_PHASES, WAVE_EPILOGUE_PHASES,
    PIPELINED_PROLOGUE_PHASES,
    default_mfma_visitor,
)
from .problem import GemmProblem, TileConfig
from .tiling import GemmTiling
from .tile import TileLevel, TilePhase, walk_tile_tree
from .transforms import Embed, Dim

__all__ = ["MemoryView", "GemmKernel", "AsmKernel", "default_mfma_visitor"]


@dataclass
class MemoryView:
    """How to read/write a tensor through coordinate transforms."""
    name: str
    source: str
    layout: Embed
    elem_bytes: int
    base_reg: Optional[str] = None
    base_offset: int = 0

    def emit_offset(self, ctx, bindings, result_reg):
        emit_affine(ctx, self.layout, bindings, result_reg,
                    scale=self.elem_bytes,
                    base=str(self.base_offset) if self.base_offset else None,
                    comment=f"{self.name} offset: {self.layout}")

    def summary(self) -> str:
        return f"{self.name}({self.source}): {self.layout}"


# Monkey-patch MemoryView registry onto AsmContext
def _register_view(ctx, view):
    if not hasattr(ctx, '_views'):
        ctx._views = {}
    ctx._views[view.name] = view

def _get_view(ctx, name):
    if not hasattr(ctx, '_views') or name not in ctx._views:
        available = list(getattr(ctx, '_views', {}).keys())
        raise KeyError(f"No MemoryView '{name}'. Available: {available}")
    return ctx._views[name]

AsmContext.register_view = _register_view
AsmContext.get_view = _get_view


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
    def vgpr_count(self): return self.ctx._next["v"]
    @property
    def sgpr_count(self): return self.ctx._next["s"]
    @property
    def acc_count(self): return self.ctx._next["acc"]

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.asm_text)
        return path

    def assemble(self, gpu_arch="gfx950", output_path=None):
        return assemble_kernel(self.asm_text, gpu_arch, output_path)


@dataclass
class GemmKernel:
    """A GEMM kernel described as a tile tree with phases.

    The entire kernel structure is encoded in the tile tree.
    Replace any phase to customize one step::

        kernel.tile_tree = kernel.tile_tree.replace_phase(
            "global_load", my_prefetching_load)
    """
    problem: GemmProblem
    tile: TileConfig
    layouts: GemmLayouts
    tile_tree: TileLevel
    kernel_name: str = "gemm_kernel"
    mfma_visitor: Callable = None

    @staticmethod
    def build(problem: GemmProblem,
              tile: Optional[TileConfig] = None,
              kernel_name: str = "gemm_kernel",
              tile_tree: Optional[TileLevel] = None,
              tiling: Optional[GemmTiling] = None,
              pipelined: bool = False) -> GemmKernel:
        """Build a GemmKernel with the full tile tree.

        Args:
            problem: GEMM problem specification.
            tile: TileConfig (derived from tiling if not provided).
            kernel_name: Name for the kernel function.
            tile_tree: Custom TileLevel tree (overrides auto-generation).
            tiling: GemmTiling with per-dimension TileDim chains.
            pipelined: If True, use software-pipelined K-loop.
        """
        if tiling is not None:
            tiling.validate()
            tile = tiling.to_tile_config()
        if tile is None:
            tile = TileConfig()
        problem.validate(tile)

        layouts = GemmLayouts.build(problem, tile)

        if tile_tree is None:
            tile_tree = build_default_tree(tile, pipelined=pipelined)

        tile_tree.validate()

        return GemmKernel(
            problem=problem, tile=tile, layouts=layouts,
            tile_tree=tile_tree, kernel_name=kernel_name,
            mfma_visitor=default_mfma_visitor,
        )

    def emit(self) -> AsmKernel:
        """Generate the full kernel assembly."""
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
        alloc_registers(ctx, self.problem, tile)

        ctx.register_view(MemoryView(
            name="A", source="lds", layout=self.layouts.lds_a,
            elem_bytes=elem, base_reg="v_lds_rd_a"))
        ctx.register_view(MemoryView(
            name="B", source="lds", layout=self.layouts.lds_b,
            elem_bytes=elem, base_reg="v_lds_rd_b",
            base_offset=self.layouts.lds_b_offset))

        emit_header(ctx, self.kernel_name)
        walk_tile_tree(self.tile_tree, ctx, visitor=self.mfma_visitor)
        ctx.inst("s_endpgm", comment="end of kernel")
        emit_descriptor(ctx, self.kernel_name, lds_total, tile)

        return AsmKernel(
            asm_text=ctx.asm_text(), kernel_name=self.kernel_name,
            problem=self.problem, tile=tile, ctx=ctx, lds_bytes=lds_total)


# ===================================================================
# Tree builders
# ===================================================================

def build_default_tree(tile: TileConfig,
                       pipelined: bool = False) -> TileLevel:
    """Build the standard GEMM tile tree with all phases.

    Args:
        tile: TileConfig with tile dimensions.
        pipelined: If True, use software-pipelined K-loop.
    """
    mfma_level = TileLevel(
        "mfma", m=tile.mfma.m, n=tile.mfma.n, k=tile.mfma.k)

    if pipelined:
        # Pipelined: K-loop is a single workgroup prologue phase
        wave_level = TileLevel(
            "wave", m=tile.m_per_wave, n=tile.n_per_wave,
            k=tile.unroll_k, inner=mfma_level)
        workgroup_level = TileLevel(
            "workgroup", m=tile.wg_m, n=tile.wg_n, k=tile.unroll_k,
            inner=wave_level, parallel=True,
            prologue_phases=list(PIPELINED_PROLOGUE_PHASES),
            epilogue_phases=list(WORKGROUP_EPILOGUE_PHASES))
    else:
        # Standard: separate global_load/lds_write/k_advance phases
        wave_level = TileLevel(
            "wave", m=tile.m_per_wave, n=tile.n_per_wave,
            k=tile.unroll_k, inner=mfma_level,
            prologue_phases=list(WAVE_PROLOGUE_PHASES),
            epilogue_phases=list(WAVE_EPILOGUE_PHASES))
        workgroup_level = TileLevel(
            "workgroup", m=tile.wg_m, n=tile.wg_n, k=tile.unroll_k,
            inner=wave_level, parallel=True,
            prologue_phases=list(WORKGROUP_PROLOGUE_PHASES),
            epilogue_phases=list(WORKGROUP_EPILOGUE_PHASES))

    return workgroup_level
