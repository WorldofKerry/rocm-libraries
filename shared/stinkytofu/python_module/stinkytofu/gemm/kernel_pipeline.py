# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM kernel pipeline: build tree, emit assembly, assemble.

GemmTiling is the source of truth.  The TileDim chains determine the
tree structure, and build_tile_tree() attaches the right phases at
each level.

Usage::

    kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
    result = kernel.emit()
    co = result.assemble()

    # Pipelined K-loop (10x faster)
    kernel = GemmKernel.build(problem, pipelined=True)

    # Replace a phase
    kernel.tile_tree = kernel.tile_tree.replace_phase(
        "global_load", my_prefetching_load)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .asm_context import AsmContext
from .asm_emitter import alloc_registers, emit_header, emit_descriptor, assemble_kernel
from .asm_transforms import emit_affine, GemmLayouts
from .phases import default_mfma_visitor
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

    The tile tree is built from GemmTiling (TileDim chains).
    The chains determine the tree structure; phases are assigned
    based on the ScheduleKind at each level.

    Replace any phase to customize one step::

        kernel.tile_tree = kernel.tile_tree.replace_phase(
            "global_load", my_prefetching_load)
    """
    problem: GemmProblem
    tile: TileConfig
    layouts: GemmLayouts
    tile_tree: TileLevel
    tiling: GemmTiling
    kernel_name: str = "gemm_kernel"
    mfma_visitor: Callable = None

    @staticmethod
    def build(problem: GemmProblem,
              tile: Optional[TileConfig] = None,
              kernel_name: str = "gemm_kernel",
              tile_tree: Optional[TileLevel] = None,
              tiling: Optional[GemmTiling] = None,
              pipelined: bool = False,
              optimized: bool = False,
              scheduled: bool = False) -> GemmKernel:
        """Build a GemmKernel.  GemmTiling is the source of truth.

        Args:
            problem: GEMM problem specification.
            tile: TileConfig (if no tiling provided, converted to GemmTiling).
            kernel_name: Name for the kernel function.
            tile_tree: Custom TileLevel tree (overrides auto-generation).
            tiling: GemmTiling with per-dimension TileDim chains.
            pipelined: If True, use software-pipelined K-loop.
            optimized: If True, use all optimizations (DB-LDS +
                       pipelining + interleaved MFMA/LR + fine waitcnt).
            scheduled: If True, use three-layer scheduled codegen with
                       TileOp-based slot placement (DESIGN.md Phase 2).
        """
        # GemmTiling is always the source of truth
        if tiling is None:
            if tile is not None:
                tiling = GemmTiling.from_tile_config(tile)
            else:
                if optimized or scheduled:
                    tiling = GemmTiling.high_perf()
                else:
                    tiling = GemmTiling.standard()

        tiling.validate()
        tile = tiling.to_tile_config()
        problem.validate(tile)

        layouts = GemmLayouts.build(problem, tile)

        # Tree comes from tiling (unless explicitly overridden)
        if tile_tree is None:
            tile_tree = tiling.build_tile_tree(
                pipelined=pipelined, optimized=optimized,
                scheduled=scheduled)

        tile_tree.validate()

        return GemmKernel(
            problem=problem, tile=tile, layouts=layouts,
            tile_tree=tile_tree, tiling=tiling,
            kernel_name=kernel_name,
            mfma_visitor=default_mfma_visitor,
        )

    def emit(self) -> AsmKernel:
        """Generate the full kernel assembly."""
        tile = self.tile
        elem = self.problem.element_bytes
        lds_half = (tile.wg_m + tile.wg_n) * tile.unroll_k * elem
        # Double LDS for double-buffered mode
        is_db = any(p.name in ("optimized_k_loop", "scheduled_k_loop")
                     for p in self.tile_tree.prologue_phases)
        lds_total = lds_half * 2 if is_db else lds_half

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
