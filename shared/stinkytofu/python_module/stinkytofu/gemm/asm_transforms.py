# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Emit assembly from coordinate transforms.

Bridges the declarative transform system (transforms.py) with the
assembly emitter (asm_context.py).  The key function ``emit_affine``
takes an ``Embed`` transform plus register bindings and emits the
multiply-accumulate chain as assembly instructions.

Supports both static coefficients (compile-time constants) and
**dynamic coefficients** (runtime values in registers, e.g. K, N
from kernel arguments).  This lets ALL address computation -- LDS,
global load, store -- go through the same transform system.

Example::

    # Static coefficients (LDS layout)
    lds_layout = Embed([Dim("row", 128), Dim("col", 32)],
                       Dim("offset", 4096), [32, 1])
    emit_affine(ctx, lds_layout,
                bindings={"row": "v5", "col": "v6"},
                result="v7", scale=2)

    # Dynamic coefficient (global address: offset = m * K + k)
    global_layout = Embed([Dim("m", M), Dim("k", K)],
                          Dim("offset", M*K), [K, 1])
    emit_affine(ctx, global_layout,
                bindings={"m": "v5", "k": "v6"},
                result="v7", scale=2,
                dynamic_coefficients={"m": "s4"})  # K is in s4
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
    dynamic_coefficients: Optional[Dict[str, str]] = None,
    comment: str = "",
) -> None:
    """Emit assembly for an affine offset.

    Computes: ``result = sum(dim_i * coeff_i) * scale + base``

    Args:
        ctx:      AsmContext to emit into.
        embed:    Embed transform describing the affine combination.
        bindings: Maps dimension names to assembly operands (vreg/sreg
                  names like ``"v5"`` or ``"s14"``).
        result:   Destination vreg name.
        scale:    Multiply the final sum by this (typically elem_bytes).
                  Must be a power of 2 (emitted as shift).
        base:     Optional base operand to add after scaling.
        dynamic_coefficients:
                  Maps dimension names to register operands for runtime
                  coefficients.  When set, the coefficient for that dim
                  comes from the register instead of the static Embed
                  value.  E.g. ``{"m": ctx.sreg("s_K")}`` means the
                  coefficient of "m" is the runtime value of K.
        comment:  Comment for the first instruction.
    """
    dims = embed.upper_dims
    coeffs = embed._coefficients
    dyn = dynamic_coefficients or {}
    first = True

    for dim, coeff in zip(dims, coeffs):
        if coeff == 0 and dim.name not in dyn:
            continue
        operand = bindings[dim.name]

        # Use dynamic coefficient register if provided, else static
        is_dynamic = dim.name in dyn
        coeff_reg = dyn.get(dim.name)

        if first:
            if is_dynamic:
                # result = operand * coeff_reg (runtime multiply)
                ctx.v_mul(result, coeff_reg, operand,
                          comment=comment or f"{dim.name} * {dim.name}_coeff")
            elif coeff == 1:
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
            if is_dynamic:
                tmp = ctx.vreg("v_tmp0")
                ctx.v_mul(tmp, coeff_reg, operand,
                          comment=f"{dim.name} * {dim.name}_coeff")
                ctx.v_add(result, result, tmp,
                          comment=f"+ {dim.name} * dyn")
            elif coeff == 1:
                ctx.v_add(result, result, operand,
                          comment=f"+ {dim.name}")
            else:
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

    For global memory layouts, some coefficients are **dynamic** --
    they depend on runtime values (K, N) loaded from kernel arguments.
    The ``dynamic_dims`` dict maps dimension names to the SGPR that
    holds the runtime coefficient.  Codegen passes this to
    ``emit_affine(dynamic_coefficients=...)``.
    """
    # LDS layouts: (row, col) -> element offset within LDS region
    lds_a: Embed     # A[row, col]: offset = row * unroll_k + col
    lds_b: Embed     # B[row, col]: offset = row * unroll_k + col

    # Global memory layouts (some coefficients are dynamic)
    global_a: Embed  # A[m, k]: offset = m * K + k  (K is dynamic)
    global_b: Embed  # B[n, k]: offset = n * K + k  (K is dynamic)
    global_d: Embed  # D[m, n]: offset = m * N + n  (N is dynamic)

    # Which dimensions have dynamic (runtime) coefficients.
    # Maps: dim_name -> sgpr_name (set after kernarg load).
    dynamic_dims: Dict[str, str]

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
            global_a=Embed(
                [Dim("m", problem.m), Dim("k", problem.k)],
                Dim("a_offset", problem.m * problem.k),
                [problem.k, 1],
            ),
            global_b=Embed(
                [Dim("n", problem.n), Dim("k", problem.k)],
                Dim("b_offset", problem.n * problem.k),
                [problem.k, 1],
            ),
            global_d=Embed(
                [Dim("m", problem.m), Dim("n", problem.n)],
                Dim("d_offset", problem.m * problem.n),
                [problem.n, 1],
            ),
            dynamic_dims={
                "m_for_a": "s_K",  # A's m-coefficient is K (runtime)
                "n_for_b": "s_K",  # B's n-coefficient is K (runtime)
                "m_for_d": "s_N",  # D's m-coefficient is N (runtime)
            },
            lds_b_offset=lds_b_offset,
            elem_bytes=elem,
        )

    # -- Backward compat aliases (used by existing code) --

    @property
    def global_a_row_major(self) -> Embed:
        return self.global_a

    @property
    def global_b_row_major(self) -> Embed:
        return self.global_b

    @property
    def global_d_row_major(self) -> Embed:
        return self.global_d

    def summary(self) -> str:
        lines = [
            f"LDS A: {self.lds_a}",
            f"LDS B: {self.lds_b} (offset={self.lds_b_offset})",
            f"Global A: {self.global_a}  [m coeff is dynamic: K]",
            f"Global B: {self.global_b}  [n coeff is dynamic: K]",
            f"Global D: {self.global_d}  [m coeff is dynamic: N]",
            f"elem_bytes: {self.elem_bytes}",
        ]
        return "\n".join(lines)
