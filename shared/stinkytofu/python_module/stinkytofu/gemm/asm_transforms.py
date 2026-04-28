# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Emit assembly from coordinate transforms.

Bridges the declarative transform system (transforms.py) with the
assembly emitter (asm_context.py).  The key function ``emit_affine``
takes an ``Embed`` transform plus register bindings and emits the
multiply-accumulate chain as assembly instructions.

This replaces hardcoded offset formulas in asm_emitter.py with
composable, inspectable transform-based address computation.

Example::

    from .transforms import Dim, Embed
    from .asm_transforms import emit_affine

    # Declare: lds_offset = row * unroll_k + col
    layout = Embed(
        [Dim("row", 128), Dim("col", 32)],
        Dim("lds_offset", 128 * 32),
        [32, 1],
    )

    # Emit assembly
    emit_affine(ctx, layout,
                bindings={"row": ctx.vreg("v_gload_row"),
                           "col": ctx.vreg("v_gload_col")},
                result="v_lds_wr_a",
                scale=2)  # * elem_bytes
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from .asm_context import AsmContext
from .transforms import Embed, Dim
from .problem import GemmProblem, TileConfig

__all__ = [
    "emit_affine", "GemmLayouts",
]


# ===================================================================
# Core: emit assembly for an Embed (affine) transform
# ===================================================================

def emit_affine(
    ctx: AsmContext,
    embed: Embed,
    bindings: Dict[str, str],
    result: str,
    scale: int = 1,
    base: Optional[str] = None,
    comment: str = "",
) -> None:
    """Emit assembly for an affine offset: ``result = sum(dim_i * coeff_i) * scale + base``.

    Args:
        ctx:      AsmContext to emit into.
        embed:    Embed transform describing the affine combination.
        bindings: Maps dimension names to assembly operands (vreg/sreg
                  names like ``"v5"`` or ``"s14"``).  These come from
                  ``ctx.vreg("name")`` / ``ctx.sreg("name")``.
        result:   Destination vreg name (e.g. ``ctx.vreg("v_lds_wr_a")``).
        scale:    Multiply the final sum by this (typically elem_bytes).
                  Must be a power of 2 (emitted as shift).
        base:     Optional base operand to add after scaling.
        comment:  Comment for the first instruction.
    """
    dims = embed.upper_dims
    coeffs = embed._coefficients
    first = True

    for dim, coeff in zip(dims, coeffs):
        if coeff == 0:
            continue
        operand = bindings[dim.name]

        if first:
            # First term: result = operand * coeff
            if coeff == 1:
                ctx.v_mov(result, operand,
                          comment=comment or f"{dim.name}")
            elif _is_pow2(coeff):
                ctx.v_lshl(result, operand, _log2(coeff),
                           comment=comment or f"{dim.name} * {coeff}")
            else:
                ctx.v_mul(result, str(coeff), operand,
                          comment=comment or f"{dim.name} * {coeff}")
            first = False
        else:
            # Subsequent terms: result += operand * coeff
            if coeff == 1:
                ctx.v_add(result, result, operand,
                          comment=f"+ {dim.name}")
            else:
                # Need a temp register for the multiply
                tmp = ctx.vreg("v_tmp0")
                if _is_pow2(coeff):
                    ctx.v_lshl(tmp, operand, _log2(coeff),
                               comment=f"{dim.name} * {coeff}")
                else:
                    ctx.v_mul(tmp, str(coeff), operand,
                              comment=f"{dim.name} * {coeff}")
                ctx.v_add(result, result, tmp,
                          comment=f"+ {dim.name} * {coeff}")

    if first:
        # All coefficients were 0
        ctx.v_mov(result, "0", comment=comment or "zero offset")

    # Scale by element size
    if scale > 1:
        assert _is_pow2(scale), f"scale must be power of 2, got {scale}"
        ctx.v_lshl(result, result, _log2(scale),
                   comment=f"* {scale} (bytes)")

    # Add base
    if base is not None:
        ctx.v_add(result, base, result,
                  comment=f"+ base ({base})")


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _log2(n: int) -> int:
    return int(math.log2(n))


# ===================================================================
# GemmLayouts: all memory layout descriptors for a GEMM kernel
# ===================================================================

@dataclass
class GemmLayouts:
    """Declarative memory layout descriptors for a GEMM kernel.

    Each layout is an ``Embed`` transform that describes the affine
    mapping from tile coordinates to element offsets.  The codegen
    calls ``emit_affine()`` with the appropriate register bindings
    to produce the actual address computation instructions.

    These are the *same* transforms from ``transforms.py``, now
    wired into the assembly pipeline.
    """
    # LDS layouts: (row, col) -> element offset within LDS region
    lds_a: Embed
    lds_b: Embed

    # Global memory: (row, col) -> element offset from base pointer
    # Coefficients for "col" are dynamic (= K from kernarg)
    global_a_row_major: Embed  # A[m, k]: offset = m * K + k
    global_b_row_major: Embed  # B[n, k]: offset = n * K + k

    # Output: (row, col) -> element offset from D base pointer
    # Coefficient for "col" is dynamic (= N from kernarg)
    global_d_row_major: Embed  # D[m, n]: offset = m * N + n

    # LDS byte offset where B region starts
    lds_b_offset: int

    # Element size in bytes
    elem_bytes: int

    @staticmethod
    def build(problem: GemmProblem, tile: TileConfig) -> GemmLayouts:
        """Build all layout descriptors from problem + tile config."""
        elem = problem.element_bytes
        uk = tile.unroll_k

        lds_a_elems = tile.wg_m * uk
        lds_b_offset = lds_a_elems * elem

        return GemmLayouts(
            lds_a=Embed(
                [Dim("row", tile.wg_m), Dim("col", uk)],
                Dim("lds_a_offset", lds_a_elems),
                [uk, 1],
            ),
            lds_b=Embed(
                [Dim("row", tile.wg_n), Dim("col", uk)],
                Dim("lds_b_offset", tile.wg_n * uk),
                [uk, 1],
            ),
            global_a_row_major=Embed(
                [Dim("m", problem.m), Dim("k", problem.k)],
                Dim("a_offset", problem.m * problem.k),
                [problem.k, 1],  # K is dynamic at runtime
            ),
            global_b_row_major=Embed(
                [Dim("n", problem.n), Dim("k", problem.k)],
                Dim("b_offset", problem.n * problem.k),
                [problem.k, 1],
            ),
            global_d_row_major=Embed(
                [Dim("m", problem.m), Dim("n", problem.n)],
                Dim("d_offset", problem.m * problem.n),
                [problem.n, 1],
            ),
            lds_b_offset=lds_b_offset,
            elem_bytes=elem,
        )

    def summary(self) -> str:
        lines = [
            f"LDS A: {self.lds_a}",
            f"LDS B: {self.lds_b} (offset={self.lds_b_offset})",
            f"Global A: {self.global_a_row_major}",
            f"Global B: {self.global_b_row_major}",
            f"Global D: {self.global_d_row_major}",
            f"elem_bytes: {self.elem_bytes}",
        ]
        return "\n".join(lines)
