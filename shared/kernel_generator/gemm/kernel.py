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

    # Scheduled K-loop (default, production path)
    kernel = GemmKernel.build(problem)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .emit.context import AsmContext
from .emit.emitter import alloc_registers, alloc_registers_dtl, emit_header, emit_descriptor, assemble_kernel
from .emit.layouts import GemmLayouts
from .emit.phases import default_mfma_visitor
from .kernarg_layout import layout_for
from .config import KernelConfig
from .mainloop import Mainloop
from .problem import DataType, GemmProblem, MfmaConfig, TileConfig
from .tiling import GemmTiling
from .tile.tree import TileLevel, walk_tile_tree
from .tile.transforms import Embed

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

# Monkey-patch MemoryView registry onto AsmContext
def _register_view(ctx: AsmContext, view: MemoryView) -> None:
    if not hasattr(ctx, '_views'):
        ctx._views = {}
    ctx._views[view.name] = view

def _get_view(ctx: AsmContext, name: str) -> MemoryView:
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
    def vgpr_count(self) -> int: return self.ctx._next["v"]
    @property
    def sgpr_count(self) -> int: return self.ctx._next["s"]
    @property
    def acc_count(self) -> int: return self.ctx._next["acc"]

    def save(self, path: str) -> str:
        with open(path, "w") as f:
            f.write(self.asm_text)
        return path

    def assemble(self, gpu_arch: str = "gfx950", output_path: Optional[str] = None) -> str:
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
              mainloop: Optional[Mainloop] = None,
              # Legacy params (used to auto-construct mainloop if not provided)
              wave_abi: bool = False,
              use_dtl: bool = True,
              pgr2: bool = False,
              pgr: int = 1,
              wg_mapping_xcc: int = 1,
              colmajor_output: bool = False,
              pipeline_strategy: Optional[object] = None,
              streamk: bool = False,
              skip_store: bool = False) -> 'GemmKernel':
        """Build a GemmKernel.  GemmTiling is the source of truth.

        Args:
            problem: GEMM problem specification.
            tile: TileConfig (if no tiling provided, converted to GemmTiling).
            kernel_name: Name for the kernel function.
            tile_tree: Custom TileLevel tree (overrides auto-generation).
            tiling: GemmTiling with per-dimension TileDim chains.
        """
        # GemmTiling is always the source of truth
        if tiling is None:
            if tile is not None:
                tiling = GemmTiling.from_tile_config(tile)
            else:
                # Select MFMA variant matching the problem data type
                if problem.dtype == DataType.BF16:
                    default_mfma = MfmaConfig.bf16_16x16x32()
                else:
                    default_mfma = None  # let tiling factory pick
                if use_dtl:
                    tiling = GemmTiling.high_perf(
                        wg_m=256, wg_n=256, unroll_k=64,
                        mfma=default_mfma)
                else:
                    if default_mfma is not None:
                        tiling = GemmTiling.standard(mfma=default_mfma)
                    else:
                        tiling = GemmTiling.standard()

        tiling.validate()
        tile = tiling.to_tile_config()
        problem.validate(tile)

        layouts = GemmLayouts.build(problem, tile)

        # Tree comes from tiling (unless explicitly overridden)
        if tile_tree is None:
            tile_tree = tiling.build_tile_tree(
                wave_abi=wave_abi)

        tile_tree.validate()

        k = GemmKernel(
            problem=problem, tile=tile, layouts=layouts,
            tile_tree=tile_tree, tiling=tiling,
            kernel_name=kernel_name,
            mfma_visitor=default_mfma_visitor,
        )

        # Construct mainloop if not provided (from legacy flags)
        if mainloop is None:
            from .mainloop import (mainloop_fp16, mainloop_bf16,
                                   mainloop_mxfp4, mainloop_mxfp4_wave_abi)
            effective_pgr = pgr if pgr > 1 else (2 if pgr2 else 1)
            if wave_abi:
                mainloop = mainloop_mxfp4_wave_abi(pgr=effective_pgr)
            elif problem.dtype == DataType.MXFP4:
                mainloop = mainloop_mxfp4(
                    pgr=effective_pgr, streamk=streamk,
                    wg_mapping_xcc=wg_mapping_xcc,
                    colmajor_output=colmajor_output)
            elif problem.dtype == DataType.BF16:
                mainloop = mainloop_bf16(
                    pgr=effective_pgr, streamk=streamk,
                    wg_mapping_xcc=wg_mapping_xcc,
                    colmajor_output=colmajor_output)
            else:
                mainloop = mainloop_fp16(
                    pgr=effective_pgr, streamk=streamk,
                    wg_mapping_xcc=wg_mapping_xcc,
                    colmajor_output=colmajor_output)

        k.mainloop = mainloop
        k._skip_store = skip_store

        # Validate PGR vs buffer count
        if mainloop.pgr > mainloop.num_buffers:
            raise ValueError(
                f"PGR={mainloop.pgr} exceeds num_buffers={mainloop.num_buffers}.")

        # Swap epilogue phase from mainloop
        if skip_store:
            from .mainloop import NoStore
            k.tile_tree = k.tile_tree.replace_phase(
                "store_d", NoStore().phase_func())
        else:
            k.tile_tree = k.tile_tree.replace_phase(
                "store_d", mainloop.epilogue.phase_func())

        # Legacy: store pipeline_strategy for external use
        k.pipeline_strategy = pipeline_strategy

        return k

    def emit(self) -> AsmKernel:
        """Generate the full kernel assembly."""
        tile = self.tile
        layout = self.mainloop.layout if self.mainloop else layout_for(self.problem.dtype)
        mainloop = self.mainloop

        elem = self.problem.element_bytes

        # LDS sizing derived from mainloop
        lds_total = mainloop.lds_total(tile)

        config = KernelConfig(
            tile=tile, problem=self.problem, layout=layout,
            layouts=self.layouts, mainloop=mainloop, kernel=self,
        )
        ctx = AsmContext(config=config)
        # Compat: sync _metadata for code not yet migrated
        ctx._metadata = {
            "tile": tile,
            "problem": self.problem,
            "layout": layout,
            "mainloop": mainloop,
            "layouts": self.layouts,
            "kernel": self,
        }
        # DTL kernels use a different register allocation
        is_dtl = any(p.name in ("dtl_setup", "dtl_interleaved_setup",
                                "wave_abi_setup")
                     for p in self.tile_tree.prologue_phases)
        if is_dtl:
            alloc_registers_dtl(ctx, self.problem, tile, layout)
        else:
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
        # Emit s_endpgm only if the tree has a store phase (complete kernel).
        # When store_d is removed (e.g. for inline ASM), the caller handles
        # the epilogue and program exit.
        has_store = self.tile_tree.get_phase("store_d") is not None
        if has_store and not self._skip_store:
            ctx.inst("s_endpgm", comment="end of kernel")
        emit_descriptor(ctx, self.kernel_name, lds_total, tile, layout)

        return AsmKernel(
            asm_text=ctx.asm_text(), kernel_name=self.kernel_name,
            problem=self.problem, tile=tile, ctx=ctx, lds_bytes=lds_total)
def export_wave_kernel(kernel: "GemmKernel", output_path: str,
                       kernel_name: str = None,
                       gpu_arch: str = "gfx950") -> str:
    """Export kernel as a .co file for the rocRoller WaveGemmKernelArgs path.

    Generates a code object (.co) suitable for hipBLASLt's custom kernel
    registry via hipModuleLoad. Uses the WaveGemmKernelArgs ABI (104 bytes,
    all u64 fields, 2D grid).

    Args:
        kernel: Built GemmKernel (should be built with wave_abi=True).
        output_path: Path for the .co file.
        kernel_name: Override kernel symbol name. Must start with "wave_"
                     for hipBLASLt's ABI dispatch.
        gpu_arch: Target GPU architecture.

    Returns:
        The kernel name used.
    """
    tile = kernel.tile
    mfma = tile.mfma

    if kernel_name is None:
        kl = layout_for(kernel.problem.dtype)
        kernel_name = f"wave_{kl.name}_{tile.wg_m}x{tile.wg_n}x{tile.unroll_k}_kgen"

    kernel.kernel_name = kernel_name
    result = kernel.emit()

    # Assemble .s -> .o -> .co
    co_path = result.assemble(gpu_arch=gpu_arch, output_path=output_path)
    return kernel_name, co_path
