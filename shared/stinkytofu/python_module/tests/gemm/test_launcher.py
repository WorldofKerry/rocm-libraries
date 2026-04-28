# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the launcher and assembly pipeline."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from stinkytofu.gemm.launcher import GemmLauncher, GemmResult
from stinkytofu.gemm.assemble import (
    generate_hip_reference, compile_hip, dump_assembly_text,
)
from stinkytofu.gemm.problem import DataType, GemmProblem, TileConfig, MfmaConfig

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

class TestAssemblyPipeline:
    def test_generate_hip_reference(self):
        p = GemmProblem(m=256, n=256, k=128)
        t = TileConfig(wg_m=128, wg_n=128, unroll_k=32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_hip_reference(
                p, t, output_path=os.path.join(tmpdir, "test.hip"),
            )
            assert os.path.exists(path)

            src = open(path).read()
            assert "gemm_reference" in src
            assert "__global__" in src
            assert "__launch_bounds__" in src
            assert "256" in src  # block_size

    def test_generate_different_configs(self):
        configs = [
            (GemmProblem(m=128, n=128, k=64), TileConfig(wg_m=64, wg_n=64)),
            (GemmProblem(m=4096, n=4096, k=4096), TileConfig(wg_m=256, wg_n=128)),
        ]
        for p, t in configs:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = generate_hip_reference(
                    p, t, output_path=os.path.join(tmpdir, "k.hip"),
                )
                src = open(path).read()
                assert str(t.wg_m) in src

    @requires_hipcc
    def test_compile_hip(self):
        p = GemmProblem(m=256, n=256, k=128)
        t = TileConfig(wg_m=128, wg_n=128, unroll_k=32)

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = generate_hip_reference(
                p, t, output_path=os.path.join(tmpdir, "test.hip"),
            )
            result = compile_hip(src_path, gpu_arch="gfx950")

            assert result.success, f"Compilation failed:\n{result.stderr}"
            assert os.path.exists(result.output_path)

    @requires_st
    def test_dump_assembly_text(self):
        import stinkytofu as st
        from stinkytofu.gemm.codegen_v2 import generate_from_tree

        p = GemmProblem(m=128, n=128, k=32)
        result = generate_from_tree(p)
        text = dump_assembly_text(result.module, "test_kernel")

        assert "test_kernel" in text
        assert "MFMA" in text


# ===========================================================================
# End-to-end GPU test (requires hipcc + GPU)
# ===========================================================================

@requires_hipcc
@requires_gpu
class TestGPUExecution:
    @pytest.mark.skip(reason="ctypes HIP launcher needs arg packing fix; segfaults")
    def test_reference_kernel_correctness(self):
        """Compile and run the HIP reference kernel, verify against numpy.

        Note: The ctypes-based HIP launcher is fragile. This test is
        marked xfail until we have a more robust launch mechanism
        (e.g., via hiprtc or a compiled C helper).
        """
        p = GemmProblem(m=256, n=256, k=128, trans_b=False)
        t = TileConfig(wg_m=16, wg_n=16, unroll_k=32,
                       waves_m=1, waves_n=1, wave_size=64,
                       mfma=MfmaConfig.f16_16x16x16())

        launcher = GemmLauncher(p, t)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate and compile
            src_path = generate_hip_reference(
                p, t,
                kernel_name="gemm_reference",
                output_path=os.path.join(tmpdir, "ref.hip"),
            )
            comp = compile_hip(src_path, gpu_arch="gfx950")
            assert comp.success, f"Compile failed:\n{comp.stderr}"

            # Run on GPU
            result = launcher.run_hip_reference(comp.output_path, "gemm_reference")

            # Verify
            _, _, _, D_ref = launcher.reference_numpy()
            verified = launcher.verify(result.D, D_ref, atol=0.5, rtol=0.1)

            launcher.print_performance(result.time_seconds)
            print(f"  Correct  : {verified.correct}")
            print(f"  Max |err|: {verified.max_abs_error:.4f}")
            print(f"  Max %err : {verified.max_rel_error:.4f}")

            assert verified.correct, (
                f"GPU result incorrect: max_abs={verified.max_abs_error:.4f}, "
                f"max_rel={verified.max_rel_error:.4f}"
            )
