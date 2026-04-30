# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for MXFP4 (Phase 1 constant-scale) GEMM support."""
import pytest
import numpy as np
from stinkytofu.gemm.problem import DataType, GemmProblem, MfmaConfig, TileConfig
from stinkytofu.gemm.tiling import GemmTiling
from stinkytofu.gemm.kernel_pipeline import GemmKernel


class TestMfmaConfigMXFP4:
    """MfmaConfig.mxfp4_16x16x128 factory and properties."""

    def test_factory_fields(self):
        m = MfmaConfig.mxfp4_16x16x128()
        assert m.m == 16
        assert m.n == 16
        assert m.k == 128
        assert m.input_type == "f8f6f4"
        assert m.acc_type == "f32"
        assert m.a_vgprs == 4
        assert m.b_vgprs == 4
        assert m.acc_vgprs == 4

    def test_mx_flags(self):
        m = MfmaConfig.mxfp4_16x16x128()
        assert m.is_mx is True
        assert m.cbsz == 4
        assert m.blgp == 4
        assert m.element_bits == 4

    def test_element_bytes(self):
        m = MfmaConfig.mxfp4_16x16x128()
        assert m.element_bytes == 0.5

    def test_instruction_name(self):
        m = MfmaConfig.mxfp4_16x16x128()
        assert m.instruction_name == "v_mfma_scale_f32_16x16x128_f8f6f4"

    def test_fp16_unchanged(self):
        """Existing fp16 configs unaffected by new fields."""
        m = MfmaConfig.f16_16x16x16()
        assert m.is_mx is False
        assert m.cbsz == 0
        assert m.blgp == 0
        assert m.element_bits == 16
        assert m.element_bytes == 2.0
        assert m.instruction_name == "v_mfma_f32_16x16x16_f16"

    def test_flops(self):
        m = MfmaConfig.mxfp4_16x16x128()
        assert m.flops_per_instruction == 2 * 16 * 16 * 128


class TestDataTypeMXFP4:
    def test_enum_value(self):
        assert DataType.MXFP4.value == "mxfp4"

    def test_element_bytes(self):
        p = GemmProblem(128, 128, 256, dtype=DataType.MXFP4)
        assert p.element_bytes == 0.5

    def test_validate_ok(self):
        p = GemmProblem(128, 128, 256, dtype=DataType.MXFP4)
        tc = TileConfig(wg_m=128, wg_n=128, unroll_k=256,
                        mfma=MfmaConfig.mxfp4_16x16x128())
        p.validate(tc)  # should not raise

    def test_validate_wrong_mfma(self):
        p = GemmProblem(128, 128, 256, dtype=DataType.MXFP4)
        tc = TileConfig(wg_m=128, wg_n=128, unroll_k=256,
                        mfma=MfmaConfig.f16_16x16x16())
        with pytest.raises(ValueError, match="f8f6f4"):
            p.validate(tc)


class TestGemmTilingMXFP4:
    def test_mxfp4_standard(self):
        t = GemmTiling.mxfp4_standard()
        assert t.wg_m == 128
        assert t.wg_n == 128
        assert t.unroll_k == 256
        assert t.mfma.is_mx is True
        assert t.k_iterations == 2  # 256 / 128

    def test_mxfp4_standard_validates(self):
        t = GemmTiling.mxfp4_standard()
        t.validate()  # should not raise


class TestMXFP4Kernel:
    """Generate an MXFP4 kernel and verify assembly output."""

    def _build_mxfp4_kernel(self):
        tiling = GemmTiling.mxfp4_standard()
        problem = GemmProblem(128, 128, 256, dtype=DataType.MXFP4)
        return GemmKernel.build(
            problem, tiling=tiling, dtl_partitioned=True)

    def test_kernel_builds(self):
        kernel = self._build_mxfp4_kernel()
        assert kernel.tile.mfma.is_mx

    def test_emit_succeeds(self):
        kernel = self._build_mxfp4_kernel()
        result = kernel.emit()
        assert len(result.asm_text) > 0

    def test_has_mfma_scale(self):
        """Assembly must contain v_mfma_scale instruction."""
        kernel = self._build_mxfp4_kernel()
        result = kernel.emit()
        mfma_lines = [l for l in result.ctx.lines
                      if 'v_mfma_scale' in l]
        assert len(mfma_lines) > 0, "No v_mfma_scale instructions found"

    def test_has_cbsz_blgp(self):
        """MFMA instructions must have cbsz and blgp modifiers."""
        kernel = self._build_mxfp4_kernel()
        result = kernel.emit()
        cbsz_lines = [l for l in result.ctx.lines
                      if 'cbsz:4' in l and 'blgp:4' in l]
        assert len(cbsz_lines) > 0, "No cbsz:4 blgp:4 modifiers found"

    def test_has_scale_init(self):
        """Assembly must initialize v_mxscale with 0x7F7F7F7F."""
        kernel = self._build_mxfp4_kernel()
        result = kernel.emit()
        scale_lines = [l for l in result.ctx.lines
                       if '0x7F7F7F7F' in l]
        assert len(scale_lines) > 0, "No scale init (0x7F7F7F7F) found"

    def test_no_v_mfma_plain(self):
        """Should NOT have plain v_mfma_ (without _scale_) instructions."""
        kernel = self._build_mxfp4_kernel()
        result = kernel.emit()
        plain_mfma = [l for l in result.ctx.lines
                      if 'v_mfma_f32' in l and 'v_mfma_scale' not in l]
        assert len(plain_mfma) == 0, \
            f"Found plain MFMA instructions: {plain_mfma[:3]}"

    def test_mfma_count(self):
        """Correct number of MFMA instructions emitted."""
        kernel = self._build_mxfp4_kernel()
        result = kernel.emit()
        mfma_lines = [l for l in result.ctx.lines
                      if 'v_mfma_scale' in l]
        tile = kernel.tile
        expected = (tile.mfma_m_repeat * tile.mfma_n_repeat *
                    tile.k_iterations)
        assert len(mfma_lines) == expected, \
            f"Expected {expected} MFMAs, got {len(mfma_lines)}"

    def test_lds_size(self):
        """LDS allocation is correct for MXFP4."""
        kernel = self._build_mxfp4_kernel()
        result = kernel.emit()
        # 128x128x256 tile, 0.5 bytes/elem, double-buffered
        # lds_half = (128+128) * 256 * 0.5 = 32768
        # lds_total = 32768 * 2 = 65536
        assert result.lds_bytes == 65536

    def test_accumulators(self):
        """Accumulator count matches tile config."""
        kernel = self._build_mxfp4_kernel()
        result = kernel.emit()
        # mr=4, nr=4, acc_vgprs=4 -> 64 acc VGPRs
        assert result.acc_count == 64

    def test_fp16_still_works(self):
        """Existing fp16 dtl_partitioned path still works."""
        tiling = GemmTiling.high_perf(wg_m=128, wg_n=128, unroll_k=64)
        problem = GemmProblem(128, 128, 64)
        kernel = GemmKernel.build(problem, tiling=tiling,
                                  dtl_partitioned=True)
        result = kernel.emit()
        mfma_lines = [l for l in result.ctx.lines
                      if 'v_mfma_f32_16x16x32_f16' in l]
        assert len(mfma_lines) > 0
        # Should NOT have v_mfma_scale
        scale_lines = [l for l in result.ctx.lines
                       if 'v_mfma_scale' in l]
        assert len(scale_lines) == 0


class TestMXFP4Launcher:
    """Test launcher data generation for MXFP4."""

    def test_generate_inputs(self):
        from stinkytofu.gemm.launcher import GemmLauncher
        p = GemmProblem(128, 128, 256, dtype=DataType.MXFP4)
        tc = TileConfig(wg_m=128, wg_n=128, unroll_k=256,
                        mfma=MfmaConfig.mxfp4_16x16x128())
        launcher = GemmLauncher(p, tc)
        A, B, C = launcher.generate_inputs()
        # A is packed: [M, K//2] = [128, 128]
        assert A.shape == (128, 128)
        assert A.dtype == np.uint8
        assert B.shape == (128, 128)  # [N, K//2] for trans_b
        assert B.dtype == np.uint8
        # C is fp16 output
        assert C.shape == (128, 128)
        assert C.dtype == np.float16
