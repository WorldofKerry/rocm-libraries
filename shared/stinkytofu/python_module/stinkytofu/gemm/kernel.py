# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Top-level kernel generation pipeline.

This module ties together problem description, tiling, coordinate
transforms, and code generation into a single entry point.

Quick start::

    from stinkytofu.gemm import generate_gemm_kernel, GemmProblem, TileConfig

    problem = GemmProblem(m=4096, n=4096, k=4096)
    tile    = TileConfig(wg_m=128, wg_n=128, unroll_k=32)
    result  = generate_gemm_kernel(problem, tile)

    print(result.summary())          # human-readable overview
    print(result.module.dump())      # stinkytofu IR
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

from .codegen import Emitter, GemmCodegen, GemmSchedule, RegisterAllocator, ThreadMapping
from .problem import GemmProblem, MfmaConfig, TileConfig
from .transforms import Dim, TileDescriptor

__all__ = ["KernelResult", "generate_gemm_kernel"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class KernelResult:
    """Everything produced by the kernel generator.

    Attributes:
        module: The stinkytofu ``LogicalModule`` (``None`` in dry-run mode).
        name: Kernel name string.
        problem: The GEMM problem that was compiled.
        tile: The tile configuration used.
        regs: Register allocation summary.
        mapping: Thread-to-element mapping info.
        codegen: The ``GemmCodegen`` instance (for further inspection).
    """
    module: object  # stinkytofu.LogicalModule | None
    name: str
    problem: GemmProblem
    tile: TileConfig
    regs: RegisterAllocator
    mapping: ThreadMapping
    codegen: GemmCodegen

    def summary(self) -> str:
        """Human-readable kernel summary."""
        grid_m, grid_n = self.problem.grid_dims(self.tile)
        lines = [
            f"=== Kernel: {self.name} ===",
            "",
            f"Problem : D[{self.problem.m}, {self.problem.n}] = "
            f"A[{self.problem.m}, {self.problem.k}] @ "
            f"B[{self.problem.k}, {self.problem.n}]",
            f"Dtype   : {self.problem.dtype.value} "
            f"(acc: {self.problem.acc_type.value})",
            f"Grid    : {grid_m} x {grid_n} workgroups "
            f"({grid_m * grid_n} total)",
            "",
            self.tile.summary(),
            "",
            f"LDS     : {self.mapping.lds_size_bytes} bytes "
            f"(A: {self.mapping.lds_offset_b} B: "
            f"{self.mapping.lds_size_bytes - self.mapping.lds_offset_b})",
            "",
            "Registers:",
            self.regs.summary(),
            "",
            "Tile descriptors:",
            f"  M: {self.mapping.m_desc}",
            f"  N: {self.mapping.n_desc}",
            f"  K: {self.mapping.k_desc}",
        ]
        return "\n".join(lines)

    def dry_info(self) -> Dict:
        """Return metadata dict without requiring stinkytofu."""
        return self.codegen.generate_dry()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_gemm_kernel(
    problem: GemmProblem,
    tile: Optional[TileConfig] = None,
    *,
    emitter_cls: Type[Emitter] = Emitter,
    schedule_cls: Type[GemmSchedule] = GemmSchedule,
    dry_run: bool = False,
) -> KernelResult:
    """Generate a GEMM kernel.

    Args:
        problem: The GEMM problem specification (M, N, K, dtype, ...).
        tile: Tile configuration.  If ``None``, a default config for
            ``problem.dtype`` is chosen.
        emitter_cls: Custom ``Emitter`` subclass for instruction-level
            overrides (e.g. hand-tuned MFMA block).
        schedule_cls: Custom ``GemmSchedule`` subclass for structural
            overrides (e.g. software-pipelined K-loop).
        dry_run: If ``True``, skip stinkytofu import and return metadata
            only.  Useful for offline analysis.

    Returns:
        ``KernelResult`` with the generated module and metadata.

    Example -- override the MFMA emitter::

        class MyEmitter(Emitter):
            def emit_mfma_block(self, module, k_iter):
                import stinkytofu as st
                # custom interleaved MFMA + LDS-read schedule
                ...

        result = generate_gemm_kernel(problem, tile, emitter_cls=MyEmitter)

    Example -- override the K-loop schedule::

        class PipelinedSchedule(GemmSchedule):
            def emit_k_loop(self, module):
                import stinkytofu as st
                # double-buffered software pipeline
                ...

        result = generate_gemm_kernel(problem, tile,
                                      schedule_cls=PipelinedSchedule)
    """
    if tile is None:
        tile = _default_tile(problem)

    cg = GemmCodegen(problem, tile,
                     emitter_cls=emitter_cls,
                     schedule_cls=schedule_cls)

    if dry_run:
        module = None
    else:
        module = cg.generate()

    return KernelResult(
        module=module,
        name=cg.kernel_name(),
        problem=problem,
        tile=tile,
        regs=cg.regs,
        mapping=cg.mapping,
        codegen=cg,
    )


def _default_tile(problem: GemmProblem) -> TileConfig:
    """Pick a reasonable default tile config based on the problem."""
    from .problem import DataType

    if problem.dtype == DataType.F16:
        return TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
            vector_width=8,
        )
    # Fallback
    return TileConfig()
