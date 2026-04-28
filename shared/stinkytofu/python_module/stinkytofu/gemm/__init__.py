# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM kernel generator using coordinate transforms and composable phases.

Architecture
------------
1. **Transforms** -- ``Dim``, ``Tile``, ``Embed``, ``Xor``, etc.
   Composable index-space mappings that drive ALL address computation.
2. **Problem** -- ``GemmProblem``, ``TileConfig``, ``MfmaConfig``.
3. **Tile Tree** -- ``TileLevel``, ``TilePhase``, ``walk_tile_tree``.
   Recursive hierarchy with prologue/epilogue phases at each level.
4. **Phases** -- Named, replaceable codegen steps.  Each phase uses
   coordinate transforms for address computation via ``emit_affine()``.
5. **Pipeline** -- ``GemmKernel`` orchestrates: build tree, walk it,
   assemble the result.
6. **Assembly** -- ``AsmContext`` for register allocation,
   ``emit_affine`` for transform-based address computation,
   ``assemble_kernel`` for .s -> .co.

Quick start::

    from stinkytofu.gemm import GemmKernel, GemmProblem

    kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
    result = kernel.emit()
    co = result.assemble()

    # Replace a phase
    kernel.tile_tree = kernel.tile_tree.replace_phase(
        "global_load", my_prefetching_load)

    # Pipelined K-loop (10x faster)
    kernel = GemmKernel.build(problem, pipelined=True)
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
from .kernel_pipeline import GemmKernel, MemoryView, default_mfma_visitor
from .tiling import TileDim, GemmTiling, ScheduleKind
from .asm_emitter import assemble_kernel

__all__ = [
    "Dim", "Transform", "PassThrough", "Tile", "Flatten", "Pad",
    "Embed", "Xor", "TileDescriptor", "tile_hierarchy",
    "DataType", "GemmProblem", "TileConfig", "MfmaConfig",
    "SubTileConfig", "PartitionConfig",
    "TileLevel", "TilePhase", "build_gemm_tile_tree", "walk_tile_tree",
    "TileContext", "Binding", "Lifetime", "AsmContext",
    "emit_affine", "GemmLayouts", "MemoryView",
    "GemmKernel", "default_mfma_visitor",
    "TileDim", "GemmTiling", "ScheduleKind",
    "assemble_kernel",
]

__version__ = "0.3.0"
