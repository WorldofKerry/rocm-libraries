"""Tests for BF16 GEMM kernel generation."""
import os
import tempfile

import numpy as np
import pytest

from kernel_generator.gemm.problem import GemmProblem, DataType, MfmaConfig
from kernel_generator.gemm.tiling import GemmTiling
from kernel_generator.gemm.kernel import GemmKernel
from kernel_generator.gemm.mainloop import mainloop_bf16


class TestBF16Emit:
    """BF16 kernel emission and assembly tests."""

    def test_bf16_emits(self) -> None:
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.BF16)
        k = GemmKernel.build(p)
        result = k.emit()
        assert result.vgpr_count > 0
        assert result.sgpr_count > 0
        assert "v_mfma_f32_16x16x32_bf16" in result.asm_text

    def test_bf16_assembles(self) -> None:
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.BF16)
        k = GemmKernel.build(p)
        co = k.emit().assemble()
        assert co.endswith(".co")

    def test_bf16_mfma_config(self) -> None:
        mfma = MfmaConfig.bf16_16x16x32()
        assert mfma.m == 16
        assert mfma.n == 16
        assert mfma.k == 32
        assert mfma.input_type == "bf16"
        assert mfma.acc_type == "f32"
        assert not mfma.is_mx

    def test_bf16_mainloop(self) -> None:
        ml = mainloop_bf16()
        assert ml.layout.name == "bf16"
        assert ml.pgr == 1
        assert not ml.is_streamk

    def test_bf16_pgr2(self) -> None:
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.BF16)
        k = GemmKernel.build(p, pgr=2)
        result = k.emit()
        co = result.assemble()
        assert co.endswith(".co")

    def test_bf16_streamk(self) -> None:
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.BF16)
        k = GemmKernel.build(p, streamk=True)
        result = k.emit()
        assert "StreamK" in result.asm_text or "sk_" in result.asm_text
        co = result.assemble()
        assert co.endswith(".co")

    def test_bf16_mainloop_api(self) -> None:
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.BF16)
        ml = mainloop_bf16(pgr=2, streamk=True)
        k = GemmKernel.build(p, mainloop=ml)
        result = k.emit()
        co = result.assemble()
        assert co.endswith(".co")

    def test_bf16_register_counts(self) -> None:
        """BF16 should use same register counts as FP16 (same tile shape)."""
        p_bf16 = GemmProblem(4096, 4096, 4096, dtype=DataType.BF16)
        p_fp16 = GemmProblem(4096, 4096, 4096, dtype=DataType.F16)
        r_bf16 = GemmKernel.build(p_bf16).emit()
        r_fp16 = GemmKernel.build(p_fp16).emit()
        assert r_bf16.vgpr_count == r_fp16.vgpr_count
        assert r_bf16.sgpr_count == r_fp16.sgpr_count

    def test_bf16_tile_sizes(self) -> None:
        """BF16 works with various problem sizes."""
        for m, n, k in [(512, 512, 512), (1024, 2048, 4096), (4096, 4096, 4096)]:
            p = GemmProblem(m, n, k, dtype=DataType.BF16)
            result = GemmKernel.build(p).emit()
            co = result.assemble()
            assert co.endswith(".co"), f"Failed for {m}x{n}x{k}"


# ---------------------------------------------------------------------------
# GPU correctness tests for BF16 kernels
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


def _run_bf16(M, N, K, *, fill=None, seed=42):
    """Build, emit, assemble, and launch a BF16 GEMM kernel.

    Args:
        fill: If ``"zeros"`` or ``"ones"``, override inputs instead of random.
        seed: RNG seed for random inputs.

    Returns:
        (D_gpu, D_ref) as np.float16 arrays.
    """
    from kernel_generator.gemm.launcher import GemmLauncher

    problem = GemmProblem(M, N, K, dtype=DataType.BF16)
    kernel = GemmKernel.build(problem)
    result = kernel.emit()

    with tempfile.TemporaryDirectory() as tmpdir:
        co_path = os.path.join(tmpdir, f"test_bf16_{M}_{N}_{K}.co")
        co_path = result.assemble(output_path=co_path)

        launcher = GemmLauncher(problem, kernel.tile, seed=seed)

        if fill == "zeros":
            launcher._A = np.zeros((M, K), dtype=np.uint16)
            launcher._B = np.zeros((N, K), dtype=np.uint16)
        elif fill == "ones":
            # BF16 1.0 = 0x3F80
            launcher._A = np.full((M, K), 0x3F80, dtype=np.uint16)
            launcher._B = np.full((N, K), 0x3F80, dtype=np.uint16)

        gpu_result = launcher.run_asm_kernel(
            co_path, kernel_name="gemm_kernel",
            num_warmup=1, num_iters=1)

        _, _, _, D_ref = launcher.reference_numpy()

    return gpu_result.D, D_ref


@requires_gpu
class TestBF16GPU:
    """GPU correctness tests for BF16 GEMM kernels."""

    def test_zeros(self):
        """All-zero inputs must produce all-zero output."""
        D_gpu, D_ref = _run_bf16(256, 256, 256, fill="zeros")
        assert D_gpu.shape == (256, 256)
        assert np.all(D_gpu == 0), \
            f"Expected all zeros, max abs: {np.max(np.abs(D_gpu))}"

    def test_ones(self):
        """All-ones inputs: D[i,j] = K."""
        M, N, K = 256, 256, 256
        D_gpu, D_ref = _run_bf16(M, N, K, fill="ones")
        assert D_gpu.shape == (M, N)
        assert np.allclose(D_gpu, float(K), atol=1.0), \
            f"Expected {K}, got min={np.min(D_gpu)} max={np.max(D_gpu)}"

    @pytest.mark.parametrize("M,N,K", [
        (256, 256, 256),
        (512, 512, 512),
    ])
    def test_random(self, M, N, K):
        """Random data: GPU must match numpy reference."""
        D_gpu, D_ref = _run_bf16(M, N, K, seed=42)
        max_err = float(np.max(np.abs(
            D_gpu.astype(np.float32) - D_ref.astype(np.float32))))
        assert np.allclose(D_gpu, D_ref, atol=1.0, rtol=0.05), \
            f"BF16 {M}x{N}x{K}: max_err={max_err}"

    def test_large(self):
        """Large problem: 4096x4096x4096."""
        M, N, K = 4096, 4096, 4096
        D_gpu, D_ref = _run_bf16(M, N, K, seed=42)
        max_err = float(np.max(np.abs(
            D_gpu.astype(np.float32) - D_ref.astype(np.float32))))
        assert np.allclose(D_gpu, D_ref, atol=1.0, rtol=0.05), \
            f"BF16 {M}x{N}x{K}: max_err={max_err}"
