# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM kernel generator using coordinate transforms and composable phases.

Architecture
------------
1. **Transforms** -- ``Dim``, ``Tile``, ``Flatten``, ``Pad``, ``Embed``,
   ``Xor``, ``TileDescriptor`` -- composable index-space mappings.
2. **Problem** -- ``GemmProblem``, ``TileConfig``, ``MfmaConfig`` --
   mathematical problem + tiling decisions.
3. **Tile Tree** -- ``TileLevel``, ``walk_tile_tree`` -- recursive tile
   hierarchy that drives the MFMA compute structure.
4. **Pipeline** -- ``GemmKernel`` with replaceable phases
   (prologue, k_loop, epilogue). ``MemoryView`` for tensor access
   at any tile level via coordinate transforms.
5. **Assembly** -- ``AsmContext`` for register allocation (named bindings),
   ``emit_affine`` for transform-based address computation,
   ``asm_emitter`` for the full assembly backend.

Quick start::

    from stinkytofu.gemm import GemmKernel, GemmProblem, TileConfig

    # Default kernel
    kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
    result = kernel.emit()
    result.assemble()

    # Replace a phase
    kernel.k_loop.global_load = my_prefetching_load
    result = kernel.emit()
"""
from __future__ import annotations

from .transforms import (
    Dim, Transform, PassThrough, Tile, Flatten, Pad, Embed, Xor,
    TileDescriptor, tile_hierarchy,
)
from .problem import (
    DataType, GemmProblem, TileConfig, MfmaConfig,
    SubTileConfig, PartitionConfig,
)
from .tile import TileLevel, build_gemm_tile_tree, walk_tile_tree
from .context import TileContext, Binding, Lifetime
from .asm_context import AsmContext
from .asm_transforms import emit_affine, GemmLayouts
from .kernel_pipeline import GemmKernel, KLoop, MemoryView
from .tiling import TileDim, GemmTiling, ScheduleKind
from .asm_emitter import emit_gemm_asm, assemble_kernel

__all__ = [
    # transforms
    "Dim", "Transform", "PassThrough", "Tile", "Flatten", "Pad",
    "Embed", "Xor", "TileDescriptor", "tile_hierarchy",
    # problem
    "DataType", "GemmProblem", "TileConfig", "MfmaConfig",
    "SubTileConfig", "PartitionConfig",
    # tile tree
    "TileLevel", "build_gemm_tile_tree", "walk_tile_tree",
    # context
    "TileContext", "Binding", "Lifetime", "AsmContext",
    # transforms -> assembly
    "emit_affine", "GemmLayouts", "MemoryView",
    # pipeline
    "GemmKernel", "KLoop",
    # tiling
    "TileDim", "GemmTiling", "ScheduleKind",
    # assembly backend
    "emit_gemm_asm", "assemble_kernel",
]

__version__ = "0.2.0"
