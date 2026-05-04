# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unified MFMA instruction emitter.

``MFMAEmitter`` separates MFMA instruction emission from scale-loading
concerns.  It handles four cases:

1. **Non-MX** (fp16/bf16): plain MFMA without scale operands.
2. **MX with constant scale**: uses ``v_mxscale`` for both A and B scales.
3. **MX with VMEM real scales** (linear layout): per-(mi,ki) and (ni,ki)
   scale VGPRs, no ``op_sel``.
4. **MX with LDS / swizzled real scales**: per-(mi,ki) and (ni,ki) scale
   VGPRs with ``op_sel`` / ``op_sel_hi`` byte selection.

Previously this logic was duplicated across ``VMEMScaleLoader.emit_mfma``,
``LDSScaleLoader.emit_mfma``, and the ``MFMABlock`` closure in
``kloop_graph.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from kernel_generator.gemm.emit.context import AsmContext
from kernel_generator.gemm.problem import MfmaConfig

__all__ = ["MFMAEmitter"]


@dataclass
class MFMAEmitter:
    """Emits a single MFMA instruction with the correct scale operands.

    Args:
        mfma: MFMA instruction configuration (non-MX or MX).
        scale_names_a: Map from ``(mi, ki)`` to a VGPR binding name for
            scale A, or ``None`` to use constant scale.
        scale_names_b: Map from ``(ni, ki)`` to a VGPR binding name for
            scale B, or ``None`` to use constant scale.
        swizzled: If ``True``, emit ``op_sel`` / ``op_sel_hi`` modifiers
            for pre-swizzled (LDS-style) scale byte selection.  Ignored
            when ``scale_names_a`` / ``scale_names_b`` are ``None``.
    """

    mfma: MfmaConfig
    scale_names_a: Optional[Dict[tuple, str]] = None
    scale_names_b: Optional[Dict[tuple, str]] = None
    swizzled: bool = False

    # -- Factory helpers ----------------------------------------------------

    @staticmethod
    def for_non_mx(mfma: MfmaConfig) -> MFMAEmitter:
        """Create an emitter for non-MX MFMA (fp16/bf16)."""
        return MFMAEmitter(mfma=mfma)

    @staticmethod
    def for_mx_constant(mfma: MfmaConfig) -> MFMAEmitter:
        """Create an emitter for MX MFMA with constant scale."""
        return MFMAEmitter(mfma=mfma)

    @staticmethod
    def for_vmem_scales(
        mfma: MfmaConfig,
        scale_names_a: Dict[tuple, str],
        scale_names_b: Dict[tuple, str],
        *,
        swizzled: bool = False,
    ) -> MFMAEmitter:
        """Create an emitter for VMEM-loaded scales.

        When *swizzled* is ``False`` (linear layout), each ``(mi, ki)``
        maps to a distinct scale VGPR and no ``op_sel`` is emitted.

        When *swizzled* is ``True``, the pre-swizzled byte format packs
        multiple mi/ki values into one VGPR, and ``op_sel`` / ``op_sel_hi``
        select the correct byte.
        """
        return MFMAEmitter(
            mfma=mfma,
            scale_names_a=scale_names_a,
            scale_names_b=scale_names_b,
            swizzled=swizzled,
        )

    @staticmethod
    def for_lds_scales(
        mfma: MfmaConfig,
        scale_names_a: Dict[tuple, str],
        scale_names_b: Dict[tuple, str],
    ) -> MFMAEmitter:
        """Create an emitter for LDS-loaded (pre-swizzled) scales.

        LDS scales always use ``op_sel`` byte selection.
        """
        return MFMAEmitter(
            mfma=mfma,
            scale_names_a=scale_names_a,
            scale_names_b=scale_names_b,
            swizzled=True,
        )

    # -- Core emit ----------------------------------------------------------

    def emit(
        self,
        ctx: AsmContext,
        acc: str,
        a_reg: str,
        b_reg: str,
        mi: int,
        ni: int,
        ki: int,
    ) -> None:
        """Emit one MFMA instruction into *ctx*.

        Args:
            ctx: Assembly context to emit into.
            acc: Accumulator register operand string (e.g. ``"acc[0:3]"``).
            a_reg: A-operand VGPR string.
            b_reg: B-operand VGPR string.
            mi: M-repeat index for this MFMA.
            ni: N-repeat index for this MFMA.
            ki: K-iteration index for this MFMA.
        """
        mfma = self.mfma
        comment = f"MFMA m{mi}_n{ni}_k{ki}"

        if not mfma.is_mx:
            # Non-MX: plain MFMA, no scale operands
            ctx.inst(mfma.instruction_name, acc, a_reg, b_reg, acc,
                     comment=comment)
            return

        # Resolve per-index scale VGPR names (None -> constant scale)
        sa_name = (self.scale_names_a or {}).get((mi, ki))
        sb_name = (self.scale_names_b or {}).get((ni, ki))

        if sa_name is None or sb_name is None:
            # MX with constant scale
            ctx.inst(
                mfma.instruction_name, acc, a_reg, b_reg, acc,
                ctx.vreg("v_mxscale"), ctx.vreg("v_mxscale"),
                f"cbsz:{mfma.cbsz} blgp:{mfma.blgp}",
                comment=comment,
            )
            return

        if self.swizzled:
            # Pre-swizzled (LDS or swizzled-VMEM): select byte via op_sel
            a_sel = mi % 2
            b_sel = ni % 2
            hi_a = ki
            hi_b = ki
            ctx.inst(
                mfma.instruction_name, acc, a_reg, b_reg, acc,
                ctx.vreg(sa_name), ctx.vreg(sb_name),
                f"op_sel:[{a_sel},{b_sel}] op_sel_hi:[{hi_a},{hi_b}]"
                f" cbsz:{mfma.cbsz} blgp:{mfma.blgp}",
                comment=comment,
            )
        else:
            # VMEM linear: one VGPR per (mi,ki)/(ni,ki), no byte select
            ctx.inst(
                mfma.instruction_name, acc, a_reg, b_reg, acc,
                ctx.vreg(sa_name), ctx.vreg(sb_name),
                f"cbsz:{mfma.cbsz} blgp:{mfma.blgp}",
                comment=comment,
            )
