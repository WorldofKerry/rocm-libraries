# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM kernel as a composable pipeline of replaceable phases.

This is the middle-ground abstraction between Triton/CK (high-level,
can't touch assembly) and TensileLite (low-level, hard to modify one
piece without understanding everything).

Key ideas:
- The kernel is a pipeline of **Phases** (prologue, k_loop, epilogue)
- Each phase is a callable that can be replaced independently
- **MemoryViews** describe how to access tensors at each tile level
  via coordinate transforms -- the same mechanism works at workgroup,
  wave, or MFMA level
- The tile tree drives the MFMA compute structure
- Named register bindings (never raw register numbers)

Usage::

    # Default kernel -- just works
    kernel = GemmKernel.build(problem, tile)
    result = kernel.emit()
    result.assemble()

    # Replace the K-loop for software pipelining
    kernel = GemmKernel.build(problem, tile)
    kernel.k_loop.global_load = my_prefetching_load
    result = kernel.emit()

    # Replace the MFMA leaf for custom scheduling
    kernel = GemmKernel.build(problem, tile)
    kernel.tile_tree = kernel.tile_tree.replace("mfma", emit=my_mfma)
    result = kernel.emit()

    # Access tensor data at any level via MemoryView
    def my_custom_compute(ctx, kernel):
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
from .tile import TileLevel, build_gemm_tile_tree, walk_tile_tree
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
# KLoop: the K-tile loop with replaceable sub-phases
# ===================================================================

@dataclass
class KLoop:
    """K-tile loop with independently replaceable sub-phases.

    Default structure per iteration:
      1. global_load: fetch A/B from global memory into VGPRs
      2. lds_write:   write VGPRs to LDS + barrier
      3. compute:     walk tile tree (LDS read + MFMA)
      4. k_advance:   advance global pointers + barrier

    Replace any sub-phase for software pipelining, double buffering, etc.
    """
    global_load: Callable = None
    lds_write: Callable = None
    compute: Callable = None
    k_advance: Callable = None
    loop_control: Callable = None

    def emit(self, ctx: AsmContext, kernel: GemmKernel) -> None:
        """Emit the full K-loop."""
        tile = kernel.tile
        elem = kernel.problem.element_bytes

        # Loop setup
        log2_uk = int(math.log2(tile.unroll_k))
        ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
                   comment=f"k_tiles = K / {tile.unroll_k}")
        ctx.raw("")
        ctx.label("k_loop")
        ctx.raw("")

        # Sub-phases
        self.global_load(ctx, kernel)
        self.lds_write(ctx, kernel)
        self.compute(ctx, kernel)
        self.k_advance(ctx, kernel)
        self.loop_control(ctx, kernel)


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
    """A GEMM kernel described as a composable pipeline.

    Each phase is independently replaceable. The kernel structure is::

        prologue  -> k_loop [global_load -> lds_write -> compute -> advance] -> epilogue

    At each level, MemoryViews describe how to access tensor data
    through coordinate transforms.
    """
    problem: GemmProblem
    tile: TileConfig
    layouts: GemmLayouts
    tile_tree: TileLevel
    kernel_name: str = "gemm_kernel"

    # Replaceable phases
    prologue: Callable = None       # (ctx, kernel) -> None
    k_loop: KLoop = field(default_factory=KLoop)
    epilogue: Callable = None       # (ctx, kernel) -> None

    # MFMA visitor for the tile tree walk
    mfma_visitor: Callable = None

    @staticmethod
    def build(problem: GemmProblem,
              tile: Optional[TileConfig] = None,
              kernel_name: str = "gemm_kernel",
              tile_tree: Optional[TileLevel] = None) -> GemmKernel:
        """Build a GemmKernel with all default phases."""
        if tile is None:
            tile = TileConfig()
        problem.validate(tile)

        layouts = GemmLayouts.build(problem, tile)

        if tile_tree is None:
            tile_tree = build_gemm_tile_tree(
                wg_m=tile.wg_m, wg_n=tile.wg_n, unroll_k=tile.unroll_k,
                waves_m=tile.waves_m, waves_n=tile.waves_n,
                mfma_m=tile.mfma.m, mfma_n=tile.mfma.n, mfma_k=tile.mfma.k,
            )
        tile_tree.validate()

        kernel = GemmKernel(
            problem=problem,
            tile=tile,
            layouts=layouts,
            tile_tree=tile_tree,
            kernel_name=kernel_name,
            prologue=default_prologue,
            k_loop=KLoop(
                global_load=default_global_load,
                lds_write=default_lds_write,
                compute=default_compute,
                k_advance=default_k_advance,
                loop_control=default_loop_control,
            ),
            epilogue=default_epilogue,
            mfma_visitor=default_mfma_visitor,
        )
        return kernel

    def emit(self) -> AsmKernel:
        """Generate the full kernel assembly."""
        from .asm_emitter import (
            _alloc_registers, _emit_header, _emit_descriptor,
            assemble_kernel,
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

        # Pipeline: prologue -> k_loop -> epilogue
        self.prologue(ctx, self)
        self.k_loop.emit(ctx, self)
        self.epilogue(ctx, self)

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
# Default phase implementations
# ===================================================================

def default_prologue(ctx: AsmContext, kernel: GemmKernel) -> None:
    """Load kernargs, compute thread/wave indices, set up addresses."""
    from .asm_emitter import (
        _emit_load_kernargs, _emit_thread_indexing,
        _emit_global_load_cluster, _emit_lds_write_addrs,
        _emit_lds_read_addrs, _emit_init_acc,
        _emit_global_addr_a, _emit_global_addr_b,
    )
    _emit_load_kernargs(ctx)
    _emit_thread_indexing(ctx, kernel.tile)
    _emit_global_load_cluster(ctx, kernel.tile, kernel.problem)
    _emit_lds_write_addrs(ctx, kernel.problem, kernel.tile, kernel.layouts)
    _emit_lds_read_addrs(ctx, kernel.problem, kernel.tile, kernel.layouts)
    _emit_init_acc(ctx, kernel.tile)
    _emit_global_addr_a(ctx, kernel.problem, kernel.tile, kernel.layouts)
    _emit_global_addr_b(ctx, kernel.problem, kernel.tile, kernel.layouts)


def default_global_load(ctx: AsmContext, kernel: GemmKernel) -> None:
    """Load A/B tiles from global memory into VGPRs."""
    from .asm_emitter import _emit_global_load
    _emit_global_load(ctx, kernel.problem, kernel.tile)


def default_lds_write(ctx: AsmContext, kernel: GemmKernel) -> None:
    """Write A/B data from VGPRs into LDS + barrier."""
    from .asm_emitter import _emit_lds_write
    _emit_lds_write(ctx, kernel.tile)


def default_compute(ctx: AsmContext, kernel: GemmKernel) -> None:
    """Walk the tile tree for LDS read + MFMA."""
    walk_tile_tree(kernel.tile_tree, ctx, kernel.mfma_visitor)


def default_k_advance(ctx: AsmContext, kernel: GemmKernel) -> None:
    """Advance A/B global pointers by unroll_k."""
    tile = kernel.tile
    k_stride = tile.unroll_k * kernel.problem.element_bytes
    ctx.comment("Advance A, B pointers by unroll_k")
    ctx.inst("v_add_co_u32", ctx.vreg("v_addr_a", 0, 1), "vcc",
             str(k_stride), ctx.vreg("v_addr_a", 0, 1),
             comment=f"A += {k_stride}")
    ctx.inst("v_addc_co_u32", ctx.vreg("v_addr_a", 1, 1), "vcc",
             ctx.vreg("v_addr_a", 1, 1), "0", "vcc",
             comment="carry")
    ctx.inst("v_add_co_u32", ctx.vreg("v_addr_b", 0, 1), "vcc",
             str(k_stride), ctx.vreg("v_addr_b", 0, 1),
             comment=f"B += {k_stride}")
    ctx.inst("v_addc_co_u32", ctx.vreg("v_addr_b", 1, 1), "vcc",
             ctx.vreg("v_addr_b", 1, 1), "0", "vcc",
             comment="carry")
    ctx.raw("")
    ctx.s_barrier(comment="sync before next K-tile LDS write")


def default_loop_control(ctx: AsmContext, kernel: GemmKernel) -> None:
    """Decrement k_tiles counter and branch."""
    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc1", "k_loop",
             comment="branch if k_tiles > 0")
    ctx.raw("")


def default_epilogue(ctx: AsmContext, kernel: GemmKernel) -> None:
    """Store accumulators to D."""
    from .asm_emitter import _emit_store_d
    _emit_store_d(ctx, kernel.problem, kernel.tile)


def default_mfma_visitor(level: TileLevel, ctx: AsmContext) -> None:
    """Tile-tree visitor: emit LDS reads + MFMAs."""
    from .asm_emitter import _mfma_visitor
    _mfma_visitor(level, ctx)
