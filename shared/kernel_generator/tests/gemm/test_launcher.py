# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the launcher and assembly pipeline."""
from __future__ import annotations


import numpy as np
import pytest

from kernel_generator.gemm.launcher import GemmLauncher
from kernel_generator.gemm.problem import GemmProblem, TileConfig

# Check for stinkytofu C extension
try:
    import stinkytofu as _st
    HAS_ST = hasattr(_st, "LogicalModule")
except ImportError:
    HAS_ST = False

# Check for hipcc
import shutil
HAS_HIPCC = shutil.which("hipcc") is not None

# Check for GPU
HAS_GPU = False
try:
    import subprocess
    out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
    HAS_GPU = "gfx" in out.stdout
except Exception:
    pass

requires_hipcc = pytest.mark.skipif(not HAS_HIPCC, reason="hipcc not found")
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="No GPU detected")
requires_st = pytest.mark.skipif(not HAS_ST, reason="stinkytofu C extension not available")


# ===========================================================================
# CPU reference tests (always run)
# ===========================================================================

class TestReferenceNumpy:
    def test_basic_shapes(self):
        p = GemmProblem(m=64, n=32, k=16, trans_b=False)
        launcher = GemmLauncher(p, TileConfig())
        A, B, C, D_ref = launcher.reference_numpy()

        assert A.shape == (64, 16)
        assert B.shape == (16, 32)
        assert C.shape == (64, 32)
        assert D_ref.shape == (64, 32)
        assert A.dtype == np.float16
        assert D_ref.dtype == np.float16

    def test_identity_like(self):
        """Small enough to verify manually."""
        p = GemmProblem(m=4, n=4, k=4, alpha=1.0, beta=0.0, trans_b=False)
        launcher = GemmLauncher(p, TileConfig())
        A, B, C, D_ref = launcher.reference_numpy()

        # Verify against direct numpy matmul
        D_check = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
        np.testing.assert_allclose(D_ref, D_check, atol=1e-2, rtol=1e-2)

    def test_reproducible(self):
        """Same seed produces same inputs."""
        p = GemmProblem(m=128, n=128, k=64)
        l1 = GemmLauncher(p, TileConfig(), seed=42)
        l2 = GemmLauncher(p, TileConfig(), seed=42)

        A1, B1, _, D1 = l1.reference_numpy()
        A2, B2, _, D2 = l2.reference_numpy()

        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_array_equal(D1, D2)

    def test_different_seeds(self):
        p = GemmProblem(m=64, n=64, k=32)
        l1 = GemmLauncher(p, TileConfig(), seed=1)
        l2 = GemmLauncher(p, TileConfig(), seed=2)

        A1, _, _, _ = l1.reference_numpy()
        A2, _, _, _ = l2.reference_numpy()

        assert not np.array_equal(A1, A2)

    def test_with_beta(self):
        p = GemmProblem(m=32, n=32, k=16, alpha=2.0, beta=0.5, trans_b=False)
        launcher = GemmLauncher(p, TileConfig())
        A, B, C, D_ref = launcher.reference_numpy()

        # C is zeros, so beta contribution is zero
        D_check = (2.0 * A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
        np.testing.assert_allclose(D_ref, D_check, atol=1e-2, rtol=1e-2)

    def test_transposed_b(self):
        """Default is trans_b=True, meaning B is stored column-major."""
        p = GemmProblem(m=32, n=32, k=16, trans_b=True)
        launcher = GemmLauncher(p, TileConfig())
        A, B, C, D_ref = launcher.reference_numpy()

        # With trans_b=True, op(B) = B^T
        D_check = (A.astype(np.float32) @ B.T.astype(np.float32)).astype(np.float16)
        np.testing.assert_allclose(D_ref, D_check, atol=1e-2, rtol=1e-2)


# ===========================================================================
# Verification tests
# ===========================================================================

class TestVerification:
    def test_correct_result(self):
        p = GemmProblem(m=32, n=32, k=16)
        launcher = GemmLauncher(p, TileConfig())
        _, _, _, D_ref = launcher.reference_numpy()

        result = launcher.verify(D_ref, D_ref)
        assert result.correct is True
        assert result.max_abs_error == 0.0

    def test_incorrect_result(self):
        p = GemmProblem(m=32, n=32, k=16)
        launcher = GemmLauncher(p, TileConfig())
        _, _, _, D_ref = launcher.reference_numpy()

        D_wrong = D_ref + 100.0
        result = launcher.verify(D_wrong, D_ref)
        assert result.correct is False
        assert result.max_abs_error > 50.0


# ===========================================================================
# Performance estimation tests
# ===========================================================================

class TestPerformance:
    def test_tflops_estimate(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        launcher = GemmLauncher(p, TileConfig())

        # 1ms execution -> should be a large number of TFLOPS
        tflops = launcher.estimate_tflops(0.001)
        assert tflops > 100  # 137 TFLOPS for 4096^3 in 1ms

    def test_tflops_zero_time(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        launcher = GemmLauncher(p, TileConfig())
        assert launcher.estimate_tflops(0.0) == 0.0

    def test_print_performance(self, capsys):
        p = GemmProblem(m=1024, n=1024, k=1024)
        launcher = GemmLauncher(p, TileConfig())
        launcher.print_performance(0.005)

        captured = capsys.readouterr()
        assert "TFLOPS" in captured.out
        assert "1024" in captured.out


# ===========================================================================
# Assembly pipeline tests
# ===========================================================================


# ===========================================================================
# WorkGroupMappingXCC / 1D grid tests
# ===========================================================================

class TestUse1dGrid:
    """Tests for the use_1d_grid (WorkGroupMappingXCC) launcher support."""

    def test_run_asm_kernel_accepts_use_1d_grid(self):
        """run_asm_kernel signature includes use_1d_grid parameter."""
        import inspect
        sig = inspect.signature(GemmLauncher.run_asm_kernel)
        assert "use_1d_grid" in sig.parameters
        assert sig.parameters["use_1d_grid"].default is False

    def test_1d_grid_dimensions(self):
        """1D grid flattens grid_m * grid_n into a single dimension."""
        p = GemmProblem(m=256, n=256, k=64)
        tile = TileConfig(wg_m=128, wg_n=128)
        grid_m, grid_n = p.grid_dims(tile)
        total_wgs = grid_m * grid_n

        assert grid_m == 2
        assert grid_n == 2
        assert total_wgs == 4

    def test_1d_grid_non_square(self):
        """Non-square problem produces correct 1D flattening."""
        p = GemmProblem(m=512, n=128, k=64)
        tile = TileConfig(wg_m=128, wg_n=128)
        grid_m, grid_n = p.grid_dims(tile)
        total_wgs = grid_m * grid_n

        assert grid_m == 4
        assert grid_n == 1
        assert total_wgs == 4

    def test_1d_grid_ceil_division(self):
        """Non-tile-aligned problem sizes round up correctly."""
        p = GemmProblem(m=300, n=200, k=64)
        tile = TileConfig(wg_m=128, wg_n=128)
        grid_m, grid_n = p.grid_dims(tile)
        total_wgs = grid_m * grid_n

        # ceil(300/128)=3, ceil(200/128)=2
        assert grid_m == 3
        assert grid_n == 2
        assert total_wgs == 6

    def test_1d_grid_large_problem(self):
        """Larger problem matching MI355X multi-XCC scenario."""
        p = GemmProblem(m=4096, n=4096, k=4096)
        tile = TileConfig(wg_m=128, wg_n=128)
        grid_m, grid_n = p.grid_dims(tile)
        total_wgs = grid_m * grid_n

        assert grid_m == 32
        assert grid_n == 32
        assert total_wgs == 1024
