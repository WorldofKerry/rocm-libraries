# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for problem description, tile config, and performance modelling."""
import pytest
from kernel_generator.gemm.problem import (
    DataType, GemmProblem, MfmaConfig, TileConfig,
)
from kernel_generator.gemm.tile.transforms import Dim


# ===========================================================================
# MfmaConfig
# ===========================================================================

class TestMfmaConfig:
    def test_f16_16x16x16(self):
        m = MfmaConfig.f16_16x16x16()
        assert (m.m, m.n, m.k) == (16, 16, 16)
        assert m.input_type == "f16"
        assert m.acc_type == "f32"
        assert m.acc_vgprs == 4

    def test_f16_32x32x8(self):
        m = MfmaConfig.f16_32x32x8()
        assert (m.m, m.n, m.k) == (32, 32, 8)
        assert m.acc_vgprs == 16

    def test_flops_per_mfma(self):
        m = MfmaConfig.f16_16x16x16()
        # Each MFMA does m * n * k * 2 FLOPs (multiply + add)
        assert m.flops_per_instruction == 16 * 16 * 16 * 2

    def test_flops_per_mfma_32x32(self):
        m = MfmaConfig.f16_32x32x8()
        assert m.flops_per_instruction == 32 * 32 * 8 * 2


# ===========================================================================
# TileConfig
# ===========================================================================

class TestTileConfig:
    def test_defaults(self):
        t = TileConfig()
        assert t.block_size == 256  # 2 * 2 * 64

    def test_derived_quantities(self):
        t = TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
        )
        assert t.m_per_wave == 64
        assert t.n_per_wave == 64
        assert t.mfma_m_repeat == 4   # 64 / 16
        assert t.mfma_n_repeat == 4
        assert t.k_iterations == 2    # 32 / 16
        assert t.total_mfma_per_wave == 4 * 4 * 2

    def test_validate_ok(self):
        t = TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
        )
        t.validate()  # should not raise

    def test_validate_bad_waves(self):
        t = TileConfig(wg_m=100, waves_m=3)
        with pytest.raises(ValueError, match="divisible"):
            t.validate()

    def test_validate_bad_mfma(self):
        t = TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig(m=32, n=32, k=8, blocks=1,
                            input_type="f16", acc_type="f32",
                            a_vgprs=2, b_vgprs=2, acc_vgprs=16),
        )
        # m_per_wave = 64, mfma.m = 32 -> 64/32 = 2, ok
        t.validate()

    def test_build_m_descriptor(self):
        t = TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
        )
        desc = t.build_m_descriptor()
        names = [d.name for d in desc.visible_dims]
        assert "M_wave_id" in names
        assert "M_mfma_id" in names
        assert "M_mfma" in names

    def test_summary(self):
        t = TileConfig()
        s = t.summary()
        assert "Workgroup tile" in s
        assert "MFMA" in s

    def test_flops_per_wave_per_unroll(self):
        t = TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
        )
        expected = t.total_mfma_per_wave * t.mfma.flops_per_instruction
        assert t.flops_per_wave_per_unroll == expected


# ===========================================================================
# GemmProblem
# ===========================================================================

class TestGemmProblem:
    def test_basic(self):
        p = GemmProblem(m=1024, n=2048, k=512)
        assert p.m == 1024
        assert p.dtype == DataType.F16
        assert p.element_bytes == 2

    def test_strides_nn(self):
        """A=row-major (not transposed), B=col-major (transposed)."""
        p = GemmProblem(m=64, n=128, k=256, trans_a=False, trans_b=True)
        assert p.a_stride_row == 256   # lda = K
        assert p.a_stride_col == 1
        assert p.b_stride_row == 1     # B^T: row stride = 1
        assert p.b_stride_col == 256   # B^T: col stride = K

    def test_grid_dims(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        t = TileConfig(wg_m=128, wg_n=128)
        assert p.grid_dims(t) == (32, 32)

    def test_grid_dims_non_multiple(self):
        p = GemmProblem(m=1000, n=500, k=256)
        t = TileConfig(wg_m=128, wg_n=128)
        gm, gn = p.grid_dims(t)
        assert gm == 8   # ceil(1000 / 128)
        assert gn == 4   # ceil(500 / 128)

    def test_total_flops(self):
        p = GemmProblem(m=1024, n=1024, k=1024)
        # GEMM FLOPs = 2 * M * N * K
        assert p.total_flops == 2 * 1024 * 1024 * 1024

    def test_validate_ok(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        t = TileConfig(mfma=MfmaConfig.f16_16x16x16())
        p.validate(t)  # should not raise

    def test_validate_type_mismatch(self):
        p = GemmProblem(m=64, n=64, k=64, dtype=DataType.F16)
        t = TileConfig(mfma=MfmaConfig(
            m=16, n=16, k=16, blocks=1,
            input_type="bf16", acc_type="f32",
            a_vgprs=2, b_vgprs=2, acc_vgprs=4,
        ))
        with pytest.raises(ValueError, match="f16 MFMA"):
            p.validate(t)

    def test_arithmetic_intensity(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        # AI = 2*M*N*K / (bytes_read + bytes_written)
        # bytes_read = (M*K + K*N) * elem_size
        # bytes_written = M*N * elem_size
        ai = p.arithmetic_intensity
        assert ai > 0
        # For large square GEMM, AI should be high
        assert ai > 100


# ===========================================================================
# Performance model
# ===========================================================================

class TestPerformanceModel:
    def test_theoretical_peak_tflops(self):
        """Sanity check: peak throughput estimation."""
        p = GemmProblem(m=4096, n=4096, k=4096)
        t = TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
        )
        grid_m, grid_n = p.grid_dims(t)
        total_wgs = grid_m * grid_n
        k_tiles = p.k // t.unroll_k

        # Total MFMAs across entire problem
        waves_per_wg = t.waves_m * t.waves_n
        mfmas_total = (total_wgs * waves_per_wg
                       * t.total_mfma_per_wave * k_tiles)
        flops_total = mfmas_total * t.mfma.flops_per_instruction

        # Should equal 2*M*N*K
        assert flops_total == p.total_flops
