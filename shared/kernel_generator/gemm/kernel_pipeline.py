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
from .asm_emitter import alloc_registers, alloc_registers_dtl, emit_header, emit_descriptor, assemble_kernel
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
              scheduled: bool = False,
              interleaved: bool = False,
              pgr2: bool = False,
              dtl: bool = False,
              interleaved_large: bool = False,
              auto_scheduled: bool = False,
              pgr2_interleaved: bool = False,
              dtl_interleaved: bool = False,
              dtl_scheduled: bool = False,
              dtl_partitioned: bool = False,
              use_real_scales: bool = False) -> GemmKernel:
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
            interleaved: If True, use fully-interleaved K-loop that
                         distributes ALL overhead between MFMAs.
        """
        # GemmTiling is always the source of truth
        if tiling is None:
            if tile is not None:
                tiling = GemmTiling.from_tile_config(tile)
            else:
                if dtl_interleaved or dtl_scheduled or dtl_partitioned:
                    # DTL variants need 256x256x64 for 128 MFMAs
                    tiling = GemmTiling.high_perf(
                        wg_m=256, wg_n=256, unroll_k=64)
                elif optimized or scheduled or interleaved or pgr2 or dtl or interleaved_large or auto_scheduled or pgr2_interleaved:
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
                scheduled=scheduled, interleaved=interleaved,
                pgr2=pgr2,
                dtl=dtl,
                interleaved_large=interleaved_large,
                auto_scheduled=auto_scheduled,
                pgr2_interleaved=pgr2_interleaved,
                dtl_interleaved=dtl_interleaved,
                dtl_scheduled=dtl_scheduled,
                dtl_partitioned=dtl_partitioned)

        tile_tree.validate()

        k = GemmKernel(
            problem=problem, tile=tile, layouts=layouts,
            tile_tree=tile_tree, tiling=tiling,
            kernel_name=kernel_name,
            mfma_visitor=default_mfma_visitor,
        )
        k.use_real_scales = use_real_scales
        return k

    def emit(self) -> AsmKernel:
        """Generate the full kernel assembly."""
        tile = self.tile
        elem = self.problem.element_bytes
        pad_elems = tile.lds_pad // elem if tile.lds_pad > 0 else 0
        lds_row_stride = tile.unroll_k + pad_elems
        is_dtl = any(p.name in ("dtl_setup", "dtl_interleaved_setup")
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
        is_db = any(p.name in ("optimized_k_loop", "scheduled_k_loop", "fully_interleaved_k_loop", "pgr2_k_loop", "dtl_k_loop", "interleaved_large_k_loop", "auto_scheduled_k_loop", "pgr2_interleaved_k_loop", "dtl_interleaved_k_loop", "dtl_scheduled_k_loop", "dtl_partitioned_k_loop")
                     for p in self.tile_tree.prologue_phases)
        lds_total = lds_half * 2 if is_db else lds_half

        ctx = AsmContext()
        ctx._metadata = {
            "tile": tile,
            "problem": self.problem,
            "layouts": self.layouts,
            "kernel": self,
            "use_real_scales": getattr(self, 'use_real_scales', False),
            "lds_scale_half": lds_scale_half,
            "lds_data_half": lds_half - lds_scale_half,
            "use_1d_grid": getattr(self, "use_1d_grid", False),
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


def export_custom_kernel(kernel: "GemmKernel", output_path: str,
                         kernel_name: str = None) -> str:
    """Export kernel as a TensileLite Custom Kernel .s file.

    Generates a .s file containing:
    1. Assembly code with TensileLite kernarg ABI + 1D WG decomposition
    2. custom.config YAML metadata for TensileLite integration
    3. amdhsa kernel descriptor

    The .s file can be dropped into TensileLite's CustomKernels/ directory
    and referenced from a benchmark YAML.

    Args:
        kernel: Built GemmKernel instance.
        output_path: Path to write the .s file.
        kernel_name: Override kernel symbol name. If None, auto-generated
                     from tile config.

    Returns:
        The kernel name used.
    """
    import os

    tile = kernel.tile
    problem = kernel.problem
    mfma = tile.mfma

    # Generate kernel name matching TensileLite convention
    if kernel_name is None:
        if mfma.is_mx:
            prefix = "Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32"
        else:
            prefix = "Custom_Cijk_Alik_Bljk_HHS"
        kernel_name = (f"{prefix}_MT{tile.wg_m}x{tile.wg_n}x{tile.unroll_k}"
                       f"_MI{mfma.m}x{mfma.n}x1_kgen_gfx950")

    # Enable 1D grid for TensileLite compatibility
    kernel.use_1d_grid = True

    # Emit assembly
    kernel.kernel_name = kernel_name
    result = kernel.emit()
    asm_text = result.asm_text

    # Build custom.config YAML
    if mfma.is_mx:
        data_type = "F4"
        dest_type = "B"  # BF16
        compute_type = "S"  # FP32
        mx_block_a = mfma.mx_block
        mx_block_b = mfma.mx_block
    else:
        data_type = "H"
        dest_type = "H"
        compute_type = "S"
        mx_block_a = 0
        mx_block_b = 0

    mi_input = 32 if mfma.is_mx else mfma.k  # inner K per round
    config = {
        "InternalSupportParams": {"KernArgsVersion": 2},
        "ProblemType": {
            "OperationType": "GEMM",
            "DataType": data_type,
            "DestDataType": dest_type,
            "ComputeDataType": compute_type,
            "HighPrecisionAccumulate": True,
            "TransposeA": 1 if problem.trans_a else 0,
            "TransposeB": 1 if problem.trans_b else 0,
            "UseBeta": True,
            "Batched": True,
        },
        "MatrixInstruction": [mfma.m, mfma.n, mfma.k, 1],
        # MIBlock: [M, N, K_inner, 1, 1, K_rounds]
        # For f8f6f4 16x16x128: K_inner=32, K_rounds=4
        "MIBlock": [mfma.m, mfma.n, 32, 1, 1, mfma.k // 32] if mfma.is_mx else [mfma.m, mfma.n, mfma.k, 1, 1, 1],
        "MIInputPerThread": mi_input,
        "MIInputPerThreadA": mi_input,
        "MIInputPerThreadB": mi_input,
        "WavefrontSize": tile.wave_size,
        "WorkGroupMapping": 16,
        "WorkGroupMappingXCC": 2,
        "WorkGroupMappingXCCGroup": -1,
        "StaggerU": 0,
        "EnableMatrixInstruction": True,
        "MIWaveGroup": [tile.waves_m, tile.waves_n],
        "MIWaveTile": [tile.mfma_m_repeat, tile.mfma_n_repeat],
        "DepthU": tile.unroll_k,
        "DirectToLds": 1,
        "LocalReadVectorWidth": -1,
        "GlobalReadVectorWidthA": 32 if mfma.is_mx else 8,
        "GlobalReadVectorWidthB": 32 if mfma.is_mx else 8,
        "GlobalSplitU": 1,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "GlobalSplitUCoalesced": False,
        "GlobalSplitUWorkGroupMappingRoundRobin": False,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "StreamK": 0,
        "StreamKAtomic": 0,
        "StreamKXCCMapping": 0,
        "TransposeLDS": 0,
        "NoReject": True,  # bypass TL validation (our kernel handles its own LDS)
    }

    if mx_block_a > 0:
        config["ProblemType"]["MXBlockA"] = mx_block_a
        config["ProblemType"]["MXBlockB"] = mx_block_b

    # Format as YAML (inline for simplicity)
    import yaml
    config_yaml = yaml.dump({"custom.config": config},
                            default_flow_style=False, sort_keys=False)

    # Extract assembly lines (between kernel label and .rodata)
    # The full .s includes header, assembly, and metadata
    # We need to restructure: assembly outside ---.../... , metadata inside
    # TensileLite format: YAML between --- and ..., assembly outside

    # The current asm_text has everything in one block.
    # Split into: pre-metadata assembly and metadata
    parts = asm_text.split(".amdgpu_metadata")
    if len(parts) != 2:
        raise RuntimeError("Expected exactly one .amdgpu_metadata section")

    pre_metadata = parts[0]  # assembly + .rodata/.amdhsa_kernel
    metadata_section = ".amdgpu_metadata" + parts[1]

    # Find the --- and ... in the metadata
    meta_lines = metadata_section.split("\n")
    yaml_start = next(i for i, l in enumerate(meta_lines) if l.strip() == "---")
    yaml_end = next(i for i, l in enumerate(meta_lines) if l.strip() == "...")

    # Build the custom kernel .s file
    # Format: assembly code, then YAML block (--- to ...) with both custom.config
    # and amdhsa metadata
    with open(output_path, "w") as f:
        # Write assembly (before .amdgpu_metadata)
        f.write(pre_metadata)

        # Write combined YAML section
        f.write(".amdgpu_metadata\n")
        f.write("---\n")
        f.write(config_yaml)
        # Write amdhsa metadata (skip the --- line, keep everything until ...)
        for line in meta_lines[yaml_start + 1:yaml_end]:
            f.write(line + "\n")
        f.write("...\n")
        f.write(".end_amdgpu_metadata\n")

    return kernel_name
