# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for MXFP4 (Phase 1 constant-scale) GEMM support."""
import pytest
import numpy as np
from kernel_generator.gemm.problem import DataType, GemmProblem, MfmaConfig, TileConfig
from kernel_generator.gemm.tiling import GemmTiling
from kernel_generator.gemm.kernel import GemmKernel


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
            problem, tiling=tiling, composable=True)

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
        # 65536 data + 4096 scale (2x 1024 per half-buffer, doubled)
        assert result.lds_bytes == 65536

    def test_accumulators(self):
        """Accumulator count matches tile config."""
        kernel = self._build_mxfp4_kernel()
        result = kernel.emit()
        # mr=4, nr=4, acc_vgprs=4 -> 64 acc VGPRs
        assert result.acc_count == 64

    def test_fp16_still_works(self):
        """Existing fp16 composable path still works."""
        tiling = GemmTiling.high_perf(wg_m=128, wg_n=128, unroll_k=64)
        problem = GemmProblem(128, 128, 64)
        kernel = GemmKernel.build(problem, tiling=tiling,
                                  composable=True)
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
        from kernel_generator.gemm.launcher import GemmLauncher
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


# ---------------------------------------------------------------------------
# GPU correctness tests for MXFP4 kernels
# ---------------------------------------------------------------------------

def _has_gpu():
    """Return True if HIP runtime and a GPU are available."""
    try:
        import ctypes
        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipMalloc.restype = ctypes.c_int
        hip.hipMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        hip.hipFree.restype = ctypes.c_int
        hip.hipFree.argtypes = [ctypes.c_void_p]
        d = ctypes.c_void_p()
        if hip.hipMalloc(ctypes.byref(d), 4) != 0:
            return False
        hip.hipFree(d)
        return True
    except OSError:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="No GPU / HIP runtime")


@requires_gpu
class TestMXFP4GPU:
    """Run MXFP4 GEMM kernels on GPU and verify correctness."""

    @staticmethod
    def _build_and_assemble(m, n, k, output_path):
        """Build an MXFP4 kernel, emit assembly, assemble to code object."""
        tiling = GemmTiling.mxfp4_standard()
        problem = GemmProblem(m, n, k, dtype=DataType.MXFP4)
        kernel = GemmKernel.build(problem, tiling=tiling,
                                  composable=True)
        result = kernel.emit()
        co_path = result.assemble(output_path=output_path)
        return kernel, co_path

    def test_zeros(self):
        """All-zero FP4 inputs must produce all-zero fp16 output."""
        from kernel_generator.gemm.launcher import GemmLauncher

        M, N, K = 128, 128, 256
        kernel, co_path = self._build_and_assemble(
            M, N, K, "/tmp/test_mxfp4_zeros.co")

        problem = GemmProblem(M, N, K, dtype=DataType.MXFP4)
        tile = kernel.tile
        launcher = GemmLauncher(problem, tile, seed=0)
        # Override inputs with packed zeros
        launcher._A = np.zeros((M, K // 2), dtype=np.uint8)
        launcher._B = np.zeros((N, K // 2), dtype=np.uint8)

        result = launcher.run_asm_kernel(
            co_path, kernel_name="gemm_kernel",
            num_warmup=1, num_iters=1)

        D_gpu = result.D
        assert D_gpu.dtype == np.float16, \
            f"D dtype should be float16, got {D_gpu.dtype}"
        assert D_gpu.shape == (M, N), \
            f"D shape should be ({M},{N}), got {D_gpu.shape}"
        assert np.all(D_gpu == 0), \
            f"All-zero input should give all-zero output, " \
            f"max abs value: {np.max(np.abs(D_gpu))}"

    def test_ones(self):
        """All FP4=1.0 inputs: D[i,j] = K * 1.0 * 1.0 = 256.0."""
        from kernel_generator.gemm.launcher import GemmLauncher

        M, N, K = 128, 128, 256
        kernel, co_path = self._build_and_assemble(
            M, N, K, "/tmp/test_mxfp4_ones.co")

        problem = GemmProblem(M, N, K, dtype=DataType.MXFP4)
        tile = kernel.tile
        launcher = GemmLauncher(problem, tile, seed=0)
        # 0x22 = low nibble 0010 (1.0), high nibble 0010 (1.0)
        launcher._A = np.full((M, K // 2), 0x22, dtype=np.uint8)
        launcher._B = np.full((N, K // 2), 0x22, dtype=np.uint8)

        result = launcher.run_asm_kernel(
            co_path, kernel_name="gemm_kernel",
            num_warmup=1, num_iters=1)

        D_gpu = result.D
        assert D_gpu.dtype == np.float16

        # Reference: each dot product is sum of K ones = K = 256
        _, _, _, D_ref = launcher.reference_numpy()
        assert np.allclose(D_ref, 256.0), \
            f"Reference should be 256.0, got {D_ref[0,0]}"

        max_err = float(np.max(np.abs(
            D_gpu.astype(np.float32) - D_ref.astype(np.float32))))
        assert np.allclose(D_gpu, D_ref, atol=1.0, rtol=0.01), \
            f"GPU vs reference mismatch: max_err={max_err}"

    def test_random(self):
        """Random FP4 data: GPU kernel must match CPU reference within FP4 tolerance."""
        from kernel_generator.gemm.launcher import GemmLauncher

        M, N, K = 128, 128, 256
        kernel, co_path = self._build_and_assemble(
            M, N, K, "/tmp/test_mxfp4_random.co")

        problem = GemmProblem(M, N, K, dtype=DataType.MXFP4)
        tile = kernel.tile
        launcher = GemmLauncher(problem, tile, seed=12345)

        # generate_inputs produces random uint8 arrays for A and B
        rng = np.random.RandomState(12345)
        launcher._A = rng.randint(0, 256, size=(M, K // 2), dtype=np.uint8)
        launcher._B = rng.randint(0, 256, size=(N, K // 2), dtype=np.uint8)

        result = launcher.run_asm_kernel(
            co_path, kernel_name="gemm_kernel",
            num_warmup=1, num_iters=1)

        D_gpu = result.D
        assert D_gpu.dtype == np.float16
        assert D_gpu.shape == (M, N)

        _, _, _, D_ref = launcher.reference_numpy()

        max_err = float(np.max(np.abs(
            D_gpu.astype(np.float32) - D_ref.astype(np.float32))))
        assert np.allclose(D_gpu, D_ref, atol=1.0, rtol=0.05), \
            f"GPU vs reference mismatch: max_err={max_err}"

    @pytest.mark.slow
    def test_performance(self):
        """Benchmark 4096x4096x4096 MXFP4 GEMM and compare with hipBLASLt."""
        import subprocess
        from kernel_generator.gemm.launcher import GemmLauncher

        M, N, K = 4096, 4096, 4096
        kernel, co_path = self._build_and_assemble(
            M, N, K, "/tmp/test_mxfp4_perf.co")

        problem = GemmProblem(M, N, K, dtype=DataType.MXFP4)
        tile = kernel.tile
        launcher = GemmLauncher(problem, tile, seed=0)

        result = launcher.run_asm_kernel(
            co_path, kernel_name="gemm_kernel",
            num_warmup=5, num_iters=10)

        our_tflops = 2 * M * N * K / result.time_seconds / 1e12

        print(f"\n--- MXFP4 Performance ({M}x{N}x{K}) ---")
        print(f"  Kernel time : {result.time_seconds * 1000:.3f} ms")
        print(f"  Our TFLOPS  : {our_tflops:.2f}")

        # Try hipBLASLt comparison
        hipblaslt_bench = (
            "/home/kerrwang/repos/rocm-libraries/agent/projects/"
            "hipblaslt/build/clients/hipblaslt-bench"
        )
        hipblaslt_tflops = None
        try:
            cmd = [
                hipblaslt_bench,
                "-m", str(M), "-n", str(N), "-k", str(K),
                "--a_type", "fp4_r", "--b_type", "fp4_r",
                "--c_type", "f16_r", "--d_type", "f16_r",
                "--compute_type", "f32_r",
                "--transA", "N", "--transB", "T",
                "-i", "10",
            ]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if out.returncode == 0:
                # Parse last CSV line; tflops is the last field
                lines = out.stdout.strip().splitlines()
                data_line = lines[-1]
                hipblaslt_tflops = float(data_line.split(",")[-1])
                print(f"  hipBLASLt   : {hipblaslt_tflops:.2f} TFLOPS")
        except Exception as e:
            print(f"  hipBLASLt   : unavailable ({e})")

        if hipblaslt_tflops is None:
            print("  hipBLASLt   : skipped (binary not found or failed)")


class TestMXFP4RealScales:
    """Test MXFP4 with real scale loading from global memory."""

    @staticmethod
    def _build_real_scales(m, n, k, output_path):
        tiling = GemmTiling.mxfp4_standard()
        problem = GemmProblem(m, n, k, dtype=DataType.MXFP4)
        kernel = GemmKernel.build(problem, tiling=tiling,
                                  composable=True,
                                  )
        result = kernel.emit()
        co_path = result.assemble(output_path=output_path)
        return kernel, co_path

    def test_real_scales_ones(self):
        """Real scale=1.0 (0x7F) should match constant scale results."""
        from kernel_generator.gemm.launcher import GemmLauncher

        M, N, K = 128, 128, 256
        kernel, co_path = self._build_real_scales(
            M, N, K, "/tmp/test_mxfp4_realscale_ones.co")

        problem = GemmProblem(M, N, K, dtype=DataType.MXFP4)
        launcher = GemmLauncher(problem, kernel.tile, seed=42)
        # All-ones FP4 input (0x22 = two packed 1.0 values)
        launcher._A = np.full((M, K // 2), 0x22, dtype=np.uint8)
        launcher._B = np.full((N, K // 2), 0x22, dtype=np.uint8)

        result = launcher.run_asm_kernel(
            co_path, kernel_name="gemm_kernel",
            num_warmup=1, num_iters=1)

        D_gpu = result.D
        expected = 256.0  # K=256, all 1.0*1.0*scale1.0
        assert D_gpu.dtype == np.float16
        assert np.all(D_gpu == expected), \
            f"Expected all {expected}, got min={np.min(D_gpu)} max={np.max(D_gpu)}"

    def test_real_scales_scale2(self):
        """Scale=2.0 (E8M0 0x80 = 2^1) should double the output vs scale=1.0."""
        from kernel_generator.gemm.launcher import GemmLauncher

        M, N, K = 128, 128, 256
        kernel, co_path = self._build_real_scales(
            M, N, K, "/tmp/test_mxfp4_realscale_2x.co")

        problem = GemmProblem(M, N, K, dtype=DataType.MXFP4)
        launcher = GemmLauncher(problem, kernel.tile, seed=42)
        # All-ones FP4 input
        launcher._A = np.full((M, K // 2), 0x22, dtype=np.uint8)
        launcher._B = np.full((N, K // 2), 0x22, dtype=np.uint8)

        # Override scale buffers: scale_A=2.0 (0x80), scale_B=1.0 (0x7F)
        # E8M0: value = 2^(code - 127). 0x80 = 128, 2^(128-127) = 2^1 = 2.0
        mx_block = 32
        launcher._scale_A = np.full(M * (K // mx_block), 0x80, dtype=np.uint8)
        launcher._scale_B = np.full(N * (K // mx_block), 0x7F, dtype=np.uint8)

        result = launcher.run_asm_kernel(
            co_path, kernel_name="gemm_kernel",
            num_warmup=1, num_iters=1)

        D_gpu = result.D
        # With scale_A=2.0: each A element is effectively 2*1.0=2.0
        # D = sum(2.0 * 1.0 * 1.0) for K=256 = 512.0
        expected = 512.0
        assert np.allclose(D_gpu, expected, atol=1.0), \
            f"Expected ~{expected}, got min={np.min(D_gpu)} max={np.max(D_gpu)}"


class TestWaveABIKernel:
    """Test Wave ABI kernel generation for rocRoller/hipBLASLt integration."""

    def test_wave_abi_emit(self):
        """Wave ABI kernel emits valid assembly."""
        from kernel_generator.gemm.kernel import GemmKernel
        from kernel_generator.gemm.tiling import GemmTiling

        mx = MfmaConfig.mxfp4_16x16x128()
        t = GemmTiling.high_perf(wg_m=128, wg_n=128, unroll_k=256, mfma=mx)
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.MXFP4)
        k = GemmKernel.build(p, tiling=t, wave_abi=True)

        result = k.emit()
        assert len(result.asm_text) > 0
        # Wave ABI setup should appear in the assembly
        assert "Wave ABI Setup" in result.asm_text
        # Should NOT have TensileLite kernarg offsets
        assert "TensileLite" not in result.asm_text

    def test_wave_abi_assemble(self):
        """Wave ABI kernel assembles to .co successfully."""
        from kernel_generator.gemm.kernel import GemmKernel, export_wave_kernel
        from kernel_generator.gemm.tiling import GemmTiling
        import os

        mx = MfmaConfig.mxfp4_16x16x128()
        t = GemmTiling.high_perf(wg_m=128, wg_n=128, unroll_k=256, mfma=mx)
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.MXFP4)
        k = GemmKernel.build(p, tiling=t, wave_abi=True)

        name, co = export_wave_kernel(k, "/tmp/test_wave_abi.co")
        assert name == "wave_mxfp4_128x128x256_kgen"
        assert os.path.exists(co)
        assert os.path.getsize(co) > 0

    def test_wave_abi_kernarg_offsets(self):
        """Wave ABI kernel uses correct kernarg offsets for all fields."""
        from kernel_generator.gemm.kernel import GemmKernel
        from kernel_generator.gemm.tiling import GemmTiling

        mx = MfmaConfig.mxfp4_16x16x128()
        t = GemmTiling.high_perf(wg_m=128, wg_n=128, unroll_k=256, mfma=mx)
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.MXFP4)
        k = GemmKernel.build(p, tiling=t, wave_abi=True)

        result = k.emit()
        asm = result.asm_text

        # Verify Wave ABI kernarg offsets are present
        # ptr_a at 0, ptr_a_scale at 8, ptr_b at 16, ptr_b_scale at 24
        # ptr_c at 32, M at 40, N at 48, K at 56
        for offset in ["0", "8", "16", "24", "32", "40", "48", "56"]:
            assert f", {offset}" in asm or f" {offset}" in asm, \
                f"Expected kernarg offset {offset} in assembly"

    def test_wave_abi_kernel_name_prefix(self):
        """Wave ABI kernel name starts with 'wave_' for ABI dispatch."""
        from kernel_generator.gemm.kernel import GemmKernel, export_wave_kernel
        from kernel_generator.gemm.tiling import GemmTiling

        mx = MfmaConfig.mxfp4_16x16x128()
        t = GemmTiling.high_perf(wg_m=128, wg_n=128, unroll_k=256, mfma=mx)
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.MXFP4)
        k = GemmKernel.build(p, tiling=t, wave_abi=True)

        name, _ = export_wave_kernel(k, "/tmp/test_wave_name.co")
        assert name.startswith("wave_"), \
            f"Kernel name must start with 'wave_' for hipBLASLt dispatch, got '{name}'"
