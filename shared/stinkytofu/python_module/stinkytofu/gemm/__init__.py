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
4. **Pipeline** -- ``GemmKernel`` with tree-driven phases.
   ``MemoryView`` for tensor access at any tile level via
   coordinate transforms. Each phase is independently replaceable.
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
    kernel.tile_tree = kernel.tile_tree.replace_phase(
        "global_load", my_prefetching_load)
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
from .tile import TileLevel, TilePhase, build_gemm_tile_tree, walk_tile_tree
from .context import TileContext, Binding, Lifetime
from .asm_context import AsmContext
from .asm_transforms import emit_affine, GemmLayouts
from .kernel_pipeline import GemmKernel, MemoryView
from .tiling import TileDim, GemmTiling, ScheduleKind
from .asm_emitter import emit_gemm_asm, assemble_kernel, build_full_gemm_tree

__all__ = [
    # transforms
    "Dim", "Transform", "PassThrough", "Tile", "Flatten", "Pad",
    "Embed", "Xor", "TileDescriptor", "tile_hierarchy",
    # problem
    "DataType", "GemmProblem", "TileConfig", "MfmaConfig",
    "SubTileConfig", "PartitionConfig",
    # tile tree
    "TileLevel", "TilePhase", "build_gemm_tile_tree", "walk_tile_tree",
    # context
    "TileContext", "Binding", "Lifetime", "AsmContext",
    # transforms -> assembly
    "emit_affine", "GemmLayouts", "MemoryView",
    # pipeline
    "GemmKernel",
    # tiling
    "TileDim", "GemmTiling", "ScheduleKind",
    # assembly backend
    "emit_gemm_asm", "assemble_kernel", "build_full_gemm_tree",
]

__version__ = "0.2.0"
