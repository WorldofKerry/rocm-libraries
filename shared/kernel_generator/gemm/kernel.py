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
from .emit.layouts import emit_affine, GemmLayouts
from .emit.phases import default_mfma_visitor
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

    def emit_offset(self, ctx: AsmContext, bindings: dict, result_reg: str) -> None:
        emit_affine(ctx, self.layout, bindings, result_reg,
                    scale=self.elem_bytes,
                    base=str(self.base_offset) if self.base_offset else None,
                    comment=f"{self.name} offset: {self.layout}")

    def summary(self) -> str:
        return f"{self.name}({self.source}): {self.layout}"


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
              wave_abi: bool = False,
              use_dtl: bool = True,
              pgr2: bool = False,
              pgr: int = 1,
              wg_mapping_xcc: int = 1,
              colmajor_output: bool = False,
              pipeline_strategy=None) -> GemmKernel:
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
        # Derive real scales from data type (MX types always need scales)
        k.use_real_scales = problem.dtype.value == 'mxfp4'
        k.use_dtl = use_dtl
        k.pgr2 = pgr2
        k.pgr = pgr if pgr > 1 else (2 if pgr2 else 1)
        k.pipeline_strategy = pipeline_strategy
        # Validate PGR vs buffer count (double-buffered = 2)
        num_lds_buffers = 2
        if k.pgr > num_lds_buffers:
            raise ValueError(
                f"PGR={k.pgr} exceeds num_lds_buffers={num_lds_buffers}. "
                f"Cannot preload more tiles than available buffers.")
        k.wg_mapping_xcc = wg_mapping_xcc
        k.colmajor_output = colmajor_output
        return k

    def emit(self) -> AsmKernel:
        """Generate the full kernel assembly."""
        tile = self.tile
        elem = self.problem.element_bytes
        pad_elems = tile.lds_pad // elem if tile.lds_pad > 0 else 0
        lds_row_stride = tile.unroll_k + pad_elems
        is_dtl = any(p.name in ("dtl_setup", "dtl_interleaved_setup", "wave_abi_setup")
                     for p in self.tile_tree.prologue_phases)
        if is_dtl and tile.lds_pad > 0:
            # DTL uses per-load-line padding (not per-row)
            threads_per_row = int(tile.unroll_k * elem) // 16
            rows_per_load = tile.block_size // threads_per_row
            num_loads_a = tile.wg_m // rows_per_load
            num_loads_b = tile.wg_n // rows_per_load
            lds_a = int(tile.wg_m * tile.unroll_k * elem) + num_loads_a * tile.lds_pad
            lds_b = int(tile.wg_n * tile.unroll_k * elem) + num_loads_b * tile.lds_pad
            lds_half = lds_a + lds_b
        else:
            lds_half = int((tile.wg_m + tile.wg_n) * lds_row_stride * elem)

        # No scale LDS regions needed -- scales loaded directly from global memory
        lds_scale_half = 0

        # Double LDS for double-buffered mode
        is_db = any(p.name in ("scheduled_k_loop",)
                     for p in self.tile_tree.prologue_phases)
        lds_total = lds_half * 2 if is_db else lds_half

        ctx = AsmContext()
        ctx._metadata = {
            "tile": tile,
            "problem": self.problem,
            "layouts": self.layouts,
            "kernel": self,
            "use_dtl": getattr(self, 'use_dtl', True),
            "use_real_scales": getattr(self, 'use_real_scales', False),
            "lds_scale_half": lds_scale_half,
            "lds_data_half": lds_half - lds_scale_half,
            "use_1d_grid": getattr(self, "use_1d_grid", False),
            "swizzled_scales": getattr(self, "swizzled_scales", False),
            "pgr2": getattr(self, "pgr2", False),
            "pgr": getattr(self, "pgr", 1),
            "wg_mapping_xcc": getattr(self, "wg_mapping_xcc", 1),
            "colmajor_output": getattr(self, "colmajor_output", False),
            "pipeline_strategy": getattr(self, "pipeline_strategy", None),
        }
        if is_dtl:
            alloc_registers_dtl(ctx, self.problem, tile)
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
        ctx.inst("s_endpgm", comment="end of kernel")
        emit_descriptor(ctx, self.kernel_name, lds_total, tile)

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
        if mfma.is_mx:
            kernel_name = f"wave_mxfp4_{tile.wg_m}x{tile.wg_n}x{tile.unroll_k}_kgen"
        elif mfma.input_type == "bf16":
            kernel_name = f"wave_bf16_{tile.wg_m}x{tile.wg_n}x{tile.unroll_k}_kgen"
        else:
            kernel_name = f"wave_fp16_{tile.wg_m}x{tile.wg_n}x{tile.unroll_k}_kgen"

    kernel.kernel_name = kernel_name
    result = kernel.emit()

    # Assemble .s -> .o -> .co
    co_path = result.assemble(gpu_arch=gpu_arch, output_path=output_path)
    return kernel_name, co_path
