# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Assembly pipeline: generate compilable HIP GEMM kernels.

For v1, we generate a HIP C++ reference kernel that:
- Implements a tiled GEMM matching the TileConfig parameters
- Uses MFMA intrinsics via HIP's built-in support
- Compiles with hipcc into a shared library
- Serves as both a correctness reference and a baseline for performance

The pipeline also supports generating a raw ``.s`` assembly file from
a stinkytofu ``LogicalModule`` for inspection (not yet executable).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .problem import GemmProblem, TileConfig

__all__ = ["generate_hip_reference", "compile_hip", "dump_assembly_text"]


# ---------------------------------------------------------------------------
# HIP reference kernel generator
# ---------------------------------------------------------------------------

_HIP_REFERENCE_TEMPLATE = r"""
// Auto-generated HIP GEMM reference kernel
// Problem: D[{M}, {N}] = alpha * A[{M}, {K}] @ B[{K}, {N}] + beta * C[{M}, {N}]
// Tile: {wg_m}x{wg_n}x{unroll_k}, block_size={block_size}

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C"
__global__ __launch_bounds__(256)
void {kernel_name}(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ D,
    const int M, const int N, const int K,
    const int lda, const int ldb, const int ldd,
    const float alpha, const float beta)
{{
    // Naive GEMM: each thread computes one element of D
    int global_m = blockIdx.x * blockDim.x + threadIdx.x;
    int global_n = blockIdx.y * blockDim.y + threadIdx.y;

    if (global_m >= M || global_n >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {{
        float a_val = __half2float(A[global_m * lda + k]);
        float b_val = __half2float(B[k * ldb + global_n]);
        acc += a_val * b_val;
    }}

    D[global_m * ldd + global_n] = __float2half(alpha * acc + beta * 0.0f);
}}
"""


def generate_hip_reference(
    problem: GemmProblem,
    tile: TileConfig,
    kernel_name: str = "gemm_reference",
    output_path: Optional[str] = None,
) -> str:
    """Generate a HIP C++ source file with a reference GEMM kernel.

    Returns the path to the generated ``.hip`` file.
    """
    src = _HIP_REFERENCE_TEMPLATE.format(
        M=problem.m, N=problem.n, K=problem.k,
        wg_m=tile.wg_m, wg_n=tile.wg_n, unroll_k=tile.unroll_k,
        block_size=tile.block_size,
        kernel_name=kernel_name,
    )

    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), f"{kernel_name}.hip")

    Path(output_path).write_text(src)
    return output_path


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

@dataclass
class CompileResult:
    """Result of compiling a HIP source file."""
    source_path: str
    output_path: str
    success: bool
    stdout: str
    stderr: str
    gpu_arch: str


def compile_hip(
    source_path: str,
    output_path: Optional[str] = None,
    gpu_arch: str = "gfx950",
) -> CompileResult:
    """Compile a HIP source file into a shared library / code object.

    Uses ``hipcc --genco`` to produce a GPU code object (``.co``).

    Args:
        source_path: Path to the ``.hip`` source file.
        output_path: Path for the output ``.co`` file.  Auto-generated if None.
        gpu_arch: Target GPU architecture (e.g. ``"gfx950"``).

    Returns:
        ``CompileResult`` with paths, success flag, and compiler output.
    """
    if output_path is None:
        base = os.path.splitext(source_path)[0]
        output_path = f"{base}.co"

    cmd = [
        "hipcc",
        "--genco",
        f"--offload-arch={gpu_arch}",
        "-O3",
        "-o", output_path,
        source_path,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        return CompileResult(
            source_path=source_path,
            output_path=output_path,
            success=(result.returncode == 0),
            stdout=result.stdout,
            stderr=result.stderr,
            gpu_arch=gpu_arch,
        )
    except FileNotFoundError:
        return CompileResult(
            source_path=source_path,
            output_path=output_path,
            success=False,
            stdout="",
            stderr="hipcc not found in PATH",
            gpu_arch=gpu_arch,
        )
    except subprocess.TimeoutExpired:
        return CompileResult(
            source_path=source_path,
            output_path=output_path,
            success=False,
            stdout="",
            stderr="Compilation timed out after 120s",
            gpu_arch=gpu_arch,
        )


# ---------------------------------------------------------------------------
# Assembly text dump (for inspection, not execution)
# ---------------------------------------------------------------------------

def dump_assembly_text(module, kernel_name: str = "gemm_kernel") -> str:
    """Dump a stinkytofu LogicalModule's IR as human-readable text.

    This is NOT executable assembly -- it shows the logical IR instructions.
    Useful for inspecting what the codegen produced.

    Args:
        module: A stinkytofu ``LogicalModule``.
        kernel_name: Name for the kernel header.

    Returns:
        Multi-line string with the IR dump.
    """
    header = f"; Generated GEMM kernel: {kernel_name}\n"
    header += f"; StinkyTofu Logical IR\n"
    header += "; " + "=" * 60 + "\n\n"
    return header + module.dump()
