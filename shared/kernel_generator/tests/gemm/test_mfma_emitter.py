# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for MFMAEmitter: verify emitted assembly for all four MFMA cases."""
from __future__ import annotations

import pytest

from kernel_generator.gemm.emit.context import AsmContext
from kernel_generator.gemm.memory.mfma_emitter import MFMAEmitter
from kernel_generator.gemm.problem import MfmaConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx_non_mx() -> AsmContext:
    """AsmContext with registers for non-MX MFMA tests."""
    ctx = AsmContext()
    ctx.alloc_vgpr_permanent(2, "v_a")
    ctx.alloc_vgpr_permanent(2, "v_b")
    ctx.alloc_acc_permanent(4, "acc0")
    return ctx


def _make_ctx_mx() -> AsmContext:
    """AsmContext with registers for MX MFMA tests (constant + real scales)."""
    ctx = AsmContext()
    ctx.alloc_vgpr_permanent(4, "v_a")
    ctx.alloc_vgpr_permanent(4, "v_b")
    ctx.alloc_acc_permanent(4, "acc0")
    ctx.alloc_vgpr_permanent(1, "v_mxscale")
    ctx.alloc_vgpr_permanent(1, "v_scale_a_m0_k0")
    ctx.alloc_vgpr_permanent(1, "v_scale_a_m1_k0")
    ctx.alloc_vgpr_permanent(1, "v_scale_a_m0_k1")
    ctx.alloc_vgpr_permanent(1, "v_scale_a_m1_k1")
    ctx.alloc_vgpr_permanent(1, "v_scale_b_n0_k0")
    ctx.alloc_vgpr_permanent(1, "v_scale_b_n1_k0")
    ctx.alloc_vgpr_permanent(1, "v_scale_b_n0_k1")
    ctx.alloc_vgpr_permanent(1, "v_scale_b_n1_k1")
    # LDS-style group registers (one VGPR covers 2 mi/ni x 2 ki)
    ctx.alloc_vgpr_permanent(1, "v_scale_a_g0")
    ctx.alloc_vgpr_permanent(1, "v_scale_b_g0")
    return ctx


def _last_line(ctx: AsmContext) -> str:
    """Return the last emitted line, stripped."""
    return ctx.lines[-1].strip()


# ---------------------------------------------------------------------------
# Non-MX (fp16/bf16)
# ---------------------------------------------------------------------------

class TestNonMX:
    def test_fp16_no_scales(self):
        ctx = _make_ctx_non_mx()
        mfma = MfmaConfig.f16_16x16x16()
        emitter = MFMAEmitter.for_non_mx(mfma)

        acc = ctx.areg("acc0", 0, 4)
        a = ctx.vreg("v_a", 0, 2)
        b = ctx.vreg("v_b", 0, 2)
        emitter.emit(ctx, acc, a, b, mi=0, ni=0, ki=0)

        line = _last_line(ctx)
        assert "v_mfma_f32_16x16x16_f16" in line
        assert "acc[0:3]" in line
        # No scale operands or cbsz/blgp
        assert "cbsz" not in line
        assert "v_mxscale" not in line

    def test_bf16_no_scales(self):
        ctx = _make_ctx_non_mx()
        mfma = MfmaConfig.bf16_16x16x16()
        emitter = MFMAEmitter.for_non_mx(mfma)

        acc = ctx.areg("acc0", 0, 4)
        a = ctx.vreg("v_a", 0, 2)
        b = ctx.vreg("v_b", 0, 2)
        emitter.emit(ctx, acc, a, b, mi=1, ni=2, ki=0)

        line = _last_line(ctx)
        assert "v_mfma_f32_16x16x16_bf16" in line
        assert "MFMA m1_n2_k0" in line


# ---------------------------------------------------------------------------
# MX with constant scale
# ---------------------------------------------------------------------------

class TestMXConstant:
    def test_constant_scale(self):
        ctx = _make_ctx_mx()
        mfma = MfmaConfig.mxfp4_16x16x128()
        emitter = MFMAEmitter.for_mx_constant(mfma)

        acc = ctx.areg("acc0", 0, 4)
        a = ctx.vreg("v_a", 0, 4)
        b = ctx.vreg("v_b", 0, 4)
        emitter.emit(ctx, acc, a, b, mi=0, ni=0, ki=0)

        line = _last_line(ctx)
        assert "v_mfma_scale_f32_16x16x128_f8f6f4" in line
        assert "cbsz:4" in line
        assert "blgp:4" in line
        # Both scale operands are v_mxscale (same register)
        v_mxscale = ctx.vreg("v_mxscale")
        assert line.count(v_mxscale) == 2
        # No op_sel
        assert "op_sel" not in line

    def test_missing_scale_name_falls_back_to_constant(self):
        """If scale_names dicts are provided but miss the (mi,ki) key,
        the emitter falls back to constant scale."""
        ctx = _make_ctx_mx()
        mfma = MfmaConfig.mxfp4_16x16x128()
        # Provide maps that don't cover (mi=3, ki=0)
        emitter = MFMAEmitter.for_vmem_scales(
            mfma,
            scale_names_a={(0, 0): "v_scale_a_m0_k0"},
            scale_names_b={(0, 0): "v_scale_b_n0_k0"},
        )
        acc = ctx.areg("acc0", 0, 4)
        a = ctx.vreg("v_a", 0, 4)
        b = ctx.vreg("v_b", 0, 4)
        emitter.emit(ctx, acc, a, b, mi=3, ni=0, ki=0)

        line = _last_line(ctx)
        v_mxscale = ctx.vreg("v_mxscale")
        assert v_mxscale in line
        assert "op_sel" not in line


# ---------------------------------------------------------------------------
# MX with VMEM real scales (linear, non-swizzled)
# ---------------------------------------------------------------------------

class TestVMEMLinear:
    def _emitter(self, mfma: MfmaConfig) -> MFMAEmitter:
        return MFMAEmitter.for_vmem_scales(
            mfma,
            scale_names_a={
                (0, 0): "v_scale_a_m0_k0",
                (1, 0): "v_scale_a_m1_k0",
                (0, 1): "v_scale_a_m0_k1",
                (1, 1): "v_scale_a_m1_k1",
            },
            scale_names_b={
                (0, 0): "v_scale_b_n0_k0",
                (1, 0): "v_scale_b_n1_k0",
                (0, 1): "v_scale_b_n0_k1",
                (1, 1): "v_scale_b_n1_k1",
            },
            swizzled=False,
        )

    def test_vmem_linear_basic(self):
        ctx = _make_ctx_mx()
        mfma = MfmaConfig.mxfp4_16x16x128()
        emitter = self._emitter(mfma)

        acc = ctx.areg("acc0", 0, 4)
        a = ctx.vreg("v_a", 0, 4)
        b = ctx.vreg("v_b", 0, 4)
        emitter.emit(ctx, acc, a, b, mi=0, ni=0, ki=0)

        line = _last_line(ctx)
        assert "v_mfma_scale_f32_16x16x128_f8f6f4" in line
        assert "cbsz:4 blgp:4" in line
        # Should use the specific scale VGPRs, not v_mxscale
        assert ctx.vreg("v_scale_a_m0_k0") in line
        assert ctx.vreg("v_scale_b_n0_k0") in line
        # No op_sel in linear mode
        assert "op_sel" not in line

    def test_vmem_linear_different_indices(self):
        ctx = _make_ctx_mx()
        mfma = MfmaConfig.mxfp4_16x16x128()
        emitter = self._emitter(mfma)

        acc = ctx.areg("acc0", 0, 4)
        a = ctx.vreg("v_a", 0, 4)
        b = ctx.vreg("v_b", 0, 4)
        emitter.emit(ctx, acc, a, b, mi=1, ni=1, ki=1)

        line = _last_line(ctx)
        assert ctx.vreg("v_scale_a_m1_k1") in line
        assert ctx.vreg("v_scale_b_n1_k1") in line
        assert "MFMA m1_n1_k1" in line

    def test_vmem_linear_shared_scale_a(self):
        """All MFMAs with same (mi, ki) share the same scale A VGPR."""
        ctx = _make_ctx_mx()
        mfma = MfmaConfig.mxfp4_16x16x128()
        emitter = self._emitter(mfma)

        acc = ctx.areg("acc0", 0, 4)
        a = ctx.vreg("v_a", 0, 4)
        b = ctx.vreg("v_b", 0, 4)

        # Two emissions with same (mi=0, ki=0) but different ni
        emitter.emit(ctx, acc, a, b, mi=0, ni=0, ki=0)
        line0 = _last_line(ctx)
        emitter.emit(ctx, acc, a, b, mi=0, ni=1, ki=0)
        line1 = _last_line(ctx)

        # Both should reference the same scale A VGPR
        sa = ctx.vreg("v_scale_a_m0_k0")
        assert sa in line0
        assert sa in line1
        # But different scale B VGPRs
        assert ctx.vreg("v_scale_b_n0_k0") in line0
        assert ctx.vreg("v_scale_b_n1_k0") in line1


# ---------------------------------------------------------------------------
# MX with LDS / swizzled real scales
# ---------------------------------------------------------------------------

class TestLDSSwizzled:
    def _emitter(self, mfma: MfmaConfig) -> MFMAEmitter:
        # LDS groups: one VGPR per pair of mi/ni values
        return MFMAEmitter.for_lds_scales(
            mfma,
            scale_names_a={
                (0, 0): "v_scale_a_g0",
                (1, 0): "v_scale_a_g0",
                (0, 1): "v_scale_a_g0",
                (1, 1): "v_scale_a_g0",
            },
            scale_names_b={
                (0, 0): "v_scale_b_g0",
                (1, 0): "v_scale_b_g0",
                (0, 1): "v_scale_b_g0",
                (1, 1): "v_scale_b_g0",
            },
        )

    def test_lds_op_sel_mi0_ni0_ki0(self):
        ctx = _make_ctx_mx()
        mfma = MfmaConfig.mxfp4_16x16x128()
        emitter = self._emitter(mfma)

        acc = ctx.areg("acc0", 0, 4)
        a = ctx.vreg("v_a", 0, 4)
        b = ctx.vreg("v_b", 0, 4)
        emitter.emit(ctx, acc, a, b, mi=0, ni=0, ki=0)

        line = _last_line(ctx)
        assert "v_mfma_scale_f32_16x16x128_f8f6f4" in line
        # a_sel=0%2=0, b_sel=0%2=0, hi_a=0, hi_b=0
        assert "op_sel:[0,0]" in line
        assert "op_sel_hi:[0,0]" in line
        assert "cbsz:4 blgp:4" in line

    def test_lds_op_sel_mi1_ni0_ki0(self):
        ctx = _make_ctx_mx()
        mfma = MfmaConfig.mxfp4_16x16x128()
        emitter = self._emitter(mfma)

        acc = ctx.areg("acc0", 0, 4)
        a = ctx.vreg("v_a", 0, 4)
        b = ctx.vreg("v_b", 0, 4)
        emitter.emit(ctx, acc, a, b, mi=1, ni=0, ki=0)

        line = _last_line(ctx)
        # a_sel=1%2=1, b_sel=0%2=0
        assert "op_sel:[1,0]" in line
        assert "op_sel_hi:[0,0]" in line

    def test_lds_op_sel_mi0_ni1_ki1(self):
        ctx = _make_ctx_mx()
        mfma = MfmaConfig.mxfp4_16x16x128()
        emitter = self._emitter(mfma)

        acc = ctx.areg("acc0", 0, 4)
        a = ctx.vreg("v_a", 0, 4)
        b = ctx.vreg("v_b", 0, 4)
        emitter.emit(ctx, acc, a, b, mi=0, ni=1, ki=1)

        line = _last_line(ctx)
        # a_sel=0%2=0, b_sel=1%2=1, hi_a=1, hi_b=1
        assert "op_sel:[0,1]" in line
        assert "op_sel_hi:[1,1]" in line

    def test_lds_op_sel_mi1_ni1_ki1(self):
        ctx = _make_ctx_mx()
        mfma = MfmaConfig.mxfp4_16x16x128()
        emitter = self._emitter(mfma)

        acc = ctx.areg("acc0", 0, 4)
        a = ctx.vreg("v_a", 0, 4)
        b = ctx.vreg("v_b", 0, 4)
        emitter.emit(ctx, acc, a, b, mi=1, ni=1, ki=1)

        line = _last_line(ctx)
        # a_sel=1, b_sel=1, hi_a=1, hi_b=1
        assert "op_sel:[1,1]" in line
        assert "op_sel_hi:[1,1]" in line


# ---------------------------------------------------------------------------
# VMEM swizzled (uses op_sel like LDS)
# ---------------------------------------------------------------------------

class TestVMEMSwizzled:
    def test_vmem_swizzled_has_op_sel(self):
        ctx = _make_ctx_mx()
        mfma = MfmaConfig.mxfp4_16x16x128()
        emitter = MFMAEmitter.for_vmem_scales(
            mfma,
            scale_names_a={(0, 0): "v_scale_a_g0"},
            scale_names_b={(0, 0): "v_scale_b_g0"},
            swizzled=True,
        )

        acc = ctx.areg("acc0", 0, 4)
        a = ctx.vreg("v_a", 0, 4)
        b = ctx.vreg("v_b", 0, 4)
        emitter.emit(ctx, acc, a, b, mi=0, ni=0, ki=0)

        line = _last_line(ctx)
        assert "op_sel:[0,0]" in line
        assert "op_sel_hi:[0,0]" in line
        assert "cbsz:4 blgp:4" in line


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------

class TestFactories:
    def test_for_non_mx(self):
        mfma = MfmaConfig.f16_16x16x16()
        e = MFMAEmitter.for_non_mx(mfma)
        assert e.scale_names_a is None
        assert e.scale_names_b is None
        assert not e.swizzled

    def test_for_mx_constant(self):
        mfma = MfmaConfig.mxfp4_16x16x128()
        e = MFMAEmitter.for_mx_constant(mfma)
        assert e.scale_names_a is None
        assert e.mfma.is_mx

    def test_for_lds_always_swizzled(self):
        mfma = MfmaConfig.mxfp4_16x16x128()
        e = MFMAEmitter.for_lds_scales(mfma, {}, {})
        assert e.swizzled is True

    def test_for_vmem_default_not_swizzled(self):
        mfma = MfmaConfig.mxfp4_16x16x128()
        e = MFMAEmitter.for_vmem_scales(mfma, {}, {})
        assert e.swizzled is False
