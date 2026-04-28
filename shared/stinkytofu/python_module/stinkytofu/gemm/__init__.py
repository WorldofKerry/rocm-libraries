# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM kernel generator using coordinate transforms and the StinkyTofu backend.

This package provides a high-level, Python-based GEMM kernel generator that
describes tiling via composable coordinate transforms (inspired by CK and
rocRoller) and emits optimised GPU assembly through StinkyTofu's LogicalModule
IR and pass pipeline.

Architecture
------------
Five layers, each independently overridable:

1. **Transforms** -- ``Dim``, ``Tile``, ``Flatten``, ``Pad``, ``Embed``,
   ``Xor``, ``TileDescriptor`` -- composable index-space mappings.
2. **Problem** -- ``GemmProblem``, ``TileConfig``, ``MfmaConfig`` --
   mathematical problem + tiling decisions.
3. **ThreadMapping** -- transforms -> concrete thread-to-element maps.
4. **Emitter** -- stinkytofu instruction emission per micro-operation.
   Subclass to hand-optimise any section (MFMA block, LDS layout, ...).
5. **Schedule** -- macro-structure (K-loop, prefetch, barriers).
   Subclass for software pipelining, split-K, stream-K, etc.

``generate_gemm_kernel()`` ties everything together.

Quick start::

    from stinkytofu.gemm import generate_gemm_kernel, GemmProblem, TileConfig

    problem = GemmProblem(m=4096, n=4096, k=4096)
    tile    = TileConfig(wg_m=128, wg_n=128, unroll_k=32)

    # Dry run (no stinkytofu binary needed):
    result = generate_gemm_kernel(problem, tile, dry_run=True)
    print(result.summary())

    # Full generation:
    result = generate_gemm_kernel(problem, tile)
    print(result.module.dump())
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
from .codegen import (
    RegisterAllocator, ThreadMapping, Emitter, GemmSchedule, GemmCodegen,
    VGPRTileAllocator, SubtiledSchedule,
)
from .kernel import KernelResult, generate_gemm_kernel

__all__ = [
    # transforms
    "Dim", "Transform", "PassThrough", "Tile", "Flatten", "Pad", "Embed", "Xor",
    "TileDescriptor", "tile_hierarchy",
    # problem
    "DataType", "GemmProblem", "TileConfig", "MfmaConfig",
    "SubTileConfig", "PartitionConfig",
    # codegen layers
    "RegisterAllocator", "ThreadMapping", "Emitter", "GemmSchedule", "GemmCodegen",
    "VGPRTileAllocator", "SubtiledSchedule",
    # top-level
    "KernelResult", "generate_gemm_kernel",
]

__version__ = "0.1.0"
