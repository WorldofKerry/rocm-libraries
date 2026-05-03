# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for BF16 GEMM support.

BF16 uses the same MFMA tile shapes and LDS layout as FP16, with
different MFMA instruction suffix (_bf16 instead of _f16) and
v_cvt_pk_bf16_f32 for the store epilogue.
"""
import pytest
import numpy as np
from kernel_generator.gemm.problem import DataType, GemmProblem, MfmaConfig, TileConfig
from kernel_generator.gemm.tiling import GemmTiling
from kernel_generator.gemm.kernel import GemmKernel


# ===========================================================================
# MfmaConfig BF16 factories
# ===========================================================================

class TestMfmaConfigBF16:
    """BF16 MfmaConfig factory methods and properties."""

    def test_bf16_16x16x16_fields(self):
        m = MfmaConfig.bf16_16x16x16()
        assert (m.m, m.n, m.k) == (16, 16, 16)
        assert m.input_type == "bf16"
        assert m.acc_type == "f32"
        assert m.a_vgprs == 2
        assert m.b_vgprs == 2
        assert m.acc_vgprs == 4

    def test_bf16_16x16x32_fields(self):
        m = MfmaConfig.bf16_16x16x32()
        assert (m.m, m.n, m.k) == (16, 16, 32)
        assert m.input_type == "bf16"
        assert m.acc_type == "f32"
        assert m.a_vgprs == 4
        assert m.b_vgprs == 4
        assert m.acc_vgprs == 4

    def test_instruction_name_16x16x16(self):
        m = MfmaConfig.bf16_16x16x16()
        assert m.instruction_name == "v_mfma_f32_16x16x16_bf16"

    def test_instruction_name_16x16x32(self):
        m = MfmaConfig.bf16_16x16x32()
        assert m.instruction_name == "v_mfma_f32_16x16x32_bf16"

    def test_not_mx(self):
        m = MfmaConfig.bf16_16x16x16()
        assert m.is_mx is False
        assert m.cbsz == 0
        assert m.blgp == 0

    def test_element_bytes(self):
        m = MfmaConfig.bf16_16x16x16()
        assert m.element_bytes == 2.0

    def test_flops_16x16x16(self):
        m = MfmaConfig.bf16_16x16x16()
        assert m.flops_per_instruction == 2 * 16 * 16 * 16

    def test_flops_16x16x32(self):
        m = MfmaConfig.bf16_16x16x32()
        assert m.flops_per_instruction == 2 * 16 * 16 * 32

    def test_matches_f16_shape(self):
        """BF16 and F16 variants must have identical tile shapes."""
        for bf, fp in [
            (MfmaConfig.bf16_16x16x16(), MfmaConfig.f16_16x16x16()),
            (MfmaConfig.bf16_16x16x32(), MfmaConfig.f16_16x16x32()),
        ]:
            assert (bf.m, bf.n, bf.k) == (fp.m, fp.n, fp.k)
            assert bf.a_vgprs == fp.a_vgprs
            assert bf.b_vgprs == fp.b_vgprs
            assert bf.acc_vgprs == fp.acc_vgprs


# ===========================================================================
# DataType.BF16 with GemmProblem
# ===========================================================================

class TestDataTypeBF16:
    def test_enum_value(self):
        assert DataType.BF16.value == "bf16"

    def test_element_bytes(self):
        p = GemmProblem(128, 128, 128, dtype=DataType.BF16)
        assert p.element_bytes == 2

    def test_validate_ok(self):
        p = GemmProblem(128, 128, 128, dtype=DataType.BF16)
        tc = TileConfig(wg_m=128, wg_n=128, unroll_k=32,
                        mfma=MfmaConfig.bf16_16x16x16())
        p.validate(tc)  # should not raise

    def test_validate_wrong_mfma(self):
        """BF16 problem must reject F16 MFMA."""
        p = GemmProblem(128, 128, 128, dtype=DataType.BF16)
        tc = TileConfig(wg_m=128, wg_n=128, unroll_k=32,
                        mfma=MfmaConfig.f16_16x16x16())
        with pytest.raises(ValueError, match="bf16 MFMA"):
            p.validate(tc)

    def test_f16_rejects_bf16_mfma(self):
        """F16 problem must reject BF16 MFMA."""
        p = GemmProblem(128, 128, 128, dtype=DataType.F16)
        tc = TileConfig(wg_m=128, wg_n=128, unroll_k=32,
                        mfma=MfmaConfig.bf16_16x16x16())
        with pytest.raises(ValueError, match="f16 MFMA"):
            p.validate(tc)


# ===========================================================================
# GemmTiling with BF16
# ===========================================================================

class TestGemmTilingBF16:
    def test_standard_bf16(self):
        t = GemmTiling.standard(mfma=MfmaConfig.bf16_16x16x16())
        assert t.mfma.input_type == "bf16"
        t.validate()

    def test_high_perf_bf16(self):
        t = GemmTiling.high_perf(
            wg_m=256, wg_n=256, unroll_k=64,
            mfma=MfmaConfig.bf16_16x16x32())
        assert t.mfma.input_type == "bf16"
        assert t.k_iterations == 2  # 64 / 32
        t.validate()


# ===========================================================================
# Kernel build + emit (CPU-only assembly verification)
# ===========================================================================

class TestBF16Kernel:
    """Generate a BF16 kernel and verify assembly output."""

    def _build_bf16_kernel(self, wg_m=256, wg_n=256, unroll_k=64):
        problem = GemmProblem(wg_m, wg_n, unroll_k, dtype=DataType.BF16)
        return GemmKernel.build(problem)

    def test_auto_select_mfma(self):
        """GemmKernel.build() auto-selects bf16 MFMA for BF16 problems."""
        kernel = self._build_bf16_kernel()
        assert kernel.tile.mfma.input_type == "bf16"

    def test_kernel_builds(self):
        kernel = self._build_bf16_kernel()
        assert kernel.tile.mfma.input_type == "bf16"
        assert not kernel.tile.mfma.is_mx

    def test_emit_succeeds(self):
        kernel = self._build_bf16_kernel()
        result = kernel.emit()
        assert len(result.asm_text) > 0

    def test_has_bf16_mfma(self):
        """Assembly must contain v_mfma_f32_16x16x32_bf16."""
        kernel = self._build_bf16_kernel()
        result = kernel.emit()
        mfma_lines = [l for l in result.ctx.lines
                      if 'v_mfma_f32_16x16x32_bf16' in l]
        assert len(mfma_lines) > 0, "No v_mfma_f32_16x16x32_bf16 found"

    def test_no_f16_mfma(self):
        """BF16 kernel should NOT have f16 MFMA instructions."""
        kernel = self._build_bf16_kernel()
        result = kernel.emit()
        f16_mfma = [l for l in result.ctx.lines
                    if 'v_mfma_f32_16x16x32_f16' in l
                    or 'v_mfma_f32_16x16x16_f16' in l]
        assert len(f16_mfma) == 0, \
            f"Found f16 MFMA instructions in bf16 kernel: {f16_mfma[:3]}"

    def test_has_bf16_store(self):
        """Assembly must use v_cvt_pk_bf16_f32 for output conversion."""
        kernel = self._build_bf16_kernel()
        result = kernel.emit()
        cvt_lines = [l for l in result.ctx.lines
                     if 'v_cvt_pk_bf16_f32' in l]
        assert len(cvt_lines) > 0, "No v_cvt_pk_bf16_f32 found in store"

    def test_no_f16_store(self):
        """BF16 kernel should NOT have f16 output conversion."""
        kernel = self._build_bf16_kernel()
        result = kernel.emit()
        f16_cvt = [l for l in result.ctx.lines
                   if 'v_cvt_pk_f16_f32' in l
                   or 'v_cvt_f16_f32' in l]
        assert len(f16_cvt) == 0, \
            f"Found f16 convert in bf16 kernel: {f16_cvt[:3]}"

    def test_mfma_count(self):
        """Correct number of MFMA instructions emitted."""
        kernel = self._build_bf16_kernel()
        result = kernel.emit()
        mfma_lines = [l for l in result.ctx.lines
                      if 'v_mfma_f32_16x16x32_bf16' in l]
        tile = kernel.tile
        expected = (tile.mfma_m_repeat * tile.mfma_n_repeat
                    * tile.k_iterations)
        assert len(mfma_lines) == expected, \
            f"Expected {expected} MFMAs, got {len(mfma_lines)}"

    def test_accumulators(self):
        """Accumulator count matches tile config."""
        kernel = self._build_bf16_kernel()
        result = kernel.emit()
        # 256x256 tile, 2x2 waves, 16x16 mfma -> mr=8, nr=8, acc=4
        assert result.acc_count == 8 * 8 * 4  # 256

    def test_fp16_still_works(self):
        """Existing FP16 path unaffected by BF16 additions."""
        problem = GemmProblem(256, 256, 64, dtype=DataType.F16)
        tiling = GemmTiling.high_perf(wg_m=256, wg_n=256, unroll_k=64)
        kernel = GemmKernel.build(problem, tiling=tiling)
        result = kernel.emit()
        mfma_lines = [l for l in result.ctx.lines
                      if 'v_mfma_f32_16x16x32_f16' in l]
        assert len(mfma_lines) > 0
        # Should NOT have bf16 MFMA
        bf16_lines = [l for l in result.ctx.lines
                      if '_bf16' in l and 'v_mfma' in l]
        assert len(bf16_lines) == 0

    def test_16x16x16_variant(self):
        """BF16 with the smaller 16x16x16 MFMA also works."""
        tiling = GemmTiling.standard(mfma=MfmaConfig.bf16_16x16x16())
        problem = GemmProblem(128, 128, 32, dtype=DataType.BF16)
        kernel = GemmKernel.build(problem, tiling=tiling)
        result = kernel.emit()
        mfma_lines = [l for l in result.ctx.lines
                      if 'v_mfma_f32_16x16x16_bf16' in l]
        assert len(mfma_lines) > 0


class TestBF16ExportWaveKernel:
    """Verify wave ABI kernel naming for BF16."""

    def test_wave_kernel_name(self):
        from kernel_generator.gemm.kernel import export_wave_kernel
        tiling = GemmTiling.high_perf(
            wg_m=128, wg_n=128, unroll_k=64,
            mfma=MfmaConfig.bf16_16x16x32())
        problem = GemmProblem(4096, 4096, 4096, dtype=DataType.BF16)
        kernel = GemmKernel.build(problem, tiling=tiling)
        name, co = export_wave_kernel(kernel, "/tmp/test_wave_bf16.co")
        assert name == "wave_bf16_128x128x64_kgen"
        assert name.startswith("wave_")


# ===========================================================================
# GPU tests (skipped if no HIP runtime)
# ===========================================================================

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


def _float32_to_bf16_bytes(arr_f32):
    """Convert float32 array to bfloat16 as a uint16 array.

    BF16 is the upper 16 bits of IEEE 754 float32.
    """
    return (arr_f32.view(np.uint32) >> 16).astype(np.uint16)


def _bf16_bytes_to_float32(arr_u16):
    """Convert bfloat16 (uint16) back to float32."""
    return (arr_u16.astype(np.uint32) << 16).view(np.float32)


def _build_tl_args(d_A, d_B, d_D, M, N, K, grid_m, grid_n):
    """Build TensileLite-compatible kernarg for BF16 (same layout as FP16)."""
    import ctypes
    total_wgs = grid_m * grid_n
    vals = [
        ctypes.c_uint32(0), ctypes.c_uint32(0),
        ctypes.c_uint32(0), ctypes.c_uint32(total_wgs),
        ctypes.c_uint32(M), ctypes.c_uint32(N),
        ctypes.c_uint32(1), ctypes.c_uint32(K),
        ctypes.c_void_p(d_D.value), ctypes.c_void_p(d_D.value),
        ctypes.c_void_p(d_A.value), ctypes.c_void_p(d_B.value),
        ctypes.c_uint32(N), ctypes.c_uint32(M * N),
        ctypes.c_uint32(N), ctypes.c_uint32(M * N),
        ctypes.c_uint32(K), ctypes.c_uint32(M * K),
        ctypes.c_uint32(K), ctypes.c_uint32(N * K),
        ctypes.c_float(1.0), ctypes.c_float(0.0),
    ]
    args = (ctypes.c_void_p * len(vals))()
    for i, v in enumerate(vals):
        args[i] = ctypes.cast(ctypes.pointer(v), ctypes.c_void_p)
    return vals, args


@requires_gpu
class TestBF16GPU:
    """Run BF16 GEMM kernels on GPU and verify correctness."""

    @staticmethod
    def _run_bf16_gemm(M, N, K):
        """Generate BF16 kernel, launch on GPU, return (D_gpu_f32, D_ref_f32)."""
        import ctypes
        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipMalloc.restype = ctypes.c_int
        hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        hip.hipFree.restype = ctypes.c_int
        hip.hipFree.argtypes = [ctypes.c_void_p]
        hip.hipMemcpy.restype = ctypes.c_int
        hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_size_t, ctypes.c_int]
        hip.hipMemset.restype = ctypes.c_int
        hip.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        hip.hipDeviceSynchronize.restype = ctypes.c_int
        hip.hipDeviceSynchronize.argtypes = []
        hip.hipModuleLoad.restype = ctypes.c_int
        hip.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.c_char_p]
        hip.hipModuleGetFunction.restype = ctypes.c_int
        hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                              ctypes.c_void_p, ctypes.c_char_p]
        hip.hipModuleLaunchKernel.restype = ctypes.c_int
        hip.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
        ]
        hip.hipModuleUnload.restype = ctypes.c_int
        hip.hipModuleUnload.argtypes = [ctypes.c_void_p]

        problem = GemmProblem(m=M, n=N, k=K, dtype=DataType.BF16)
        kernel = GemmKernel.build(problem)
        result = kernel.emit()
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            co_path = os.path.join(tmpdir, f"test_bf16_{M}_{N}_{K}.co")
            co = result.assemble(output_path=co_path)

            tile = kernel.tile
            elem = 2  # bf16 = 2 bytes

            d_A = ctypes.c_void_p()
            d_B = ctypes.c_void_p()
            d_D = ctypes.c_void_p()
            hip.hipMalloc(ctypes.byref(d_A), M * K * elem)
            hip.hipMalloc(ctypes.byref(d_B), N * K * elem)
            hip.hipMalloc(ctypes.byref(d_D), M * N * elem)

            # Prepare BF16 input data from float32
            rng = np.random.RandomState(42)
            scale = 1.0 / np.sqrt(K)
            A_f32 = (rng.randn(M, K) * scale).astype(np.float32)
            B_f32 = (rng.randn(N, K) * scale).astype(np.float32)

            # Convert to bf16 (truncate to upper 16 bits of f32)
            A_bf16 = _float32_to_bf16_bytes(A_f32)
            B_bf16 = _float32_to_bf16_bytes(B_f32)

            hip.hipMemcpy(d_A, A_bf16.ctypes.data_as(ctypes.c_void_p),
                          M * K * elem, 1)
            hip.hipMemcpy(d_B, B_bf16.ctypes.data_as(ctypes.c_void_p),
                          N * K * elem, 1)
            hip.hipMemset(d_D, 0, M * N * elem)

            module = ctypes.c_void_p()
            hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
            func = ctypes.c_void_p()
            hip.hipModuleGetFunction(ctypes.byref(func), module,
                                      result.kernel_name.encode())

            gm, gn = problem.grid_dims(tile)
            _vals, args = _build_tl_args(d_A, d_B, d_D, M, N, K, gm, gn)
            ret = hip.hipModuleLaunchKernel(
                func, gm, gn, 1, tile.block_size, 1, 1, 0, None, args, None)
            assert ret == 0, f"Launch failed: {ret}"
            ret = hip.hipDeviceSynchronize()
            assert ret == 0, f"Sync failed: {ret}"

            D_bf16 = np.zeros((M, N), dtype=np.uint16)
            hip.hipMemcpy(D_bf16.ctypes.data_as(ctypes.c_void_p), d_D,
                           M * N * elem, 2)

            hip.hipFree(d_A)
            hip.hipFree(d_B)
            hip.hipFree(d_D)
            hip.hipModuleUnload(module)

        D_gpu_f32 = _bf16_bytes_to_float32(D_bf16)

        # Reference: convert bf16 inputs back to f32 for matmul
        A_ref = _bf16_bytes_to_float32(A_bf16)
        B_ref = _bf16_bytes_to_float32(B_bf16)
        # B is transposed (trans_b=True): D = A @ B^T
        D_ref_f32 = A_ref @ B_ref.T

        return D_gpu_f32, D_ref_f32

    @pytest.mark.parametrize("M,N,K", [
        (256, 256, 256),
        (512, 512, 512),
    ])
    def test_correctness(self, M, N, K):
        D_gpu, D_ref = self._run_bf16_gemm(M, N, K)
        max_err = np.max(np.abs(D_gpu - D_ref))
        assert max_err < 1.0, \
            f"BF16 {M}x{N}x{K}: max_err={max_err}"
