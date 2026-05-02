# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU correctness and performance tests.

These tests require a GPU (MI355X / gfx950) and ROCm.
Skipped automatically if hipMalloc fails.
"""
import ctypes
import pytest
import numpy as np

from kernel_generator.gemm.kernel import GemmKernel
from kernel_generator.gemm.problem import GemmProblem



def _build_tl_args(d_A, d_B, d_D, M, N, K, grid_m, grid_n):
    """Build TensileLite-compatible kernarg as void** array for non-MX FP16.

    Layout: header(16B) + sizes(16B) + ptrs(D,C,A,B) + strides + alpha/beta.
    Uses 2D grid (no WG decomposition needed).
    """
    total_wgs = grid_m * grid_n
    vals = [
        # Header (16 bytes, ignored by kernel)
        ctypes.c_uint32(0), ctypes.c_uint32(0),
        ctypes.c_uint32(0), ctypes.c_uint32(total_wgs),
        # Sizes: M, N, batch=1, K
        ctypes.c_uint32(M), ctypes.c_uint32(N),
        ctypes.c_uint32(1), ctypes.c_uint32(K),
        # Pointers: D, C(=D), A, B
        ctypes.c_void_p(d_D.value), ctypes.c_void_p(d_D.value),
        ctypes.c_void_p(d_A.value), ctypes.c_void_p(d_B.value),
        # Strides: D0,D1, C0,C1, A0,A1, B0,B1
        ctypes.c_uint32(N), ctypes.c_uint32(M * N),
        ctypes.c_uint32(N), ctypes.c_uint32(M * N),
        ctypes.c_uint32(K), ctypes.c_uint32(M * K),
        ctypes.c_uint32(K), ctypes.c_uint32(N * K),
        # alpha=1.0, beta=0.0
        ctypes.c_float(1.0), ctypes.c_float(0.0),
    ]
    args = (ctypes.c_void_p * len(vals))()
    for i, v in enumerate(vals):
        args[i] = ctypes.cast(ctypes.pointer(v), ctypes.c_void_p)
    return vals, args  # return vals to keep references alive

# --- HIP runtime helpers ---

def _load_hip():
    """Load HIP runtime, return None if unavailable."""
    try:
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
        # Quick test
        d = ctypes.c_void_p()
        ret = hip.hipMalloc(ctypes.byref(d), 4)
        if ret != 0:
            return None
        hip.hipFree(d)
        return hip
    except OSError:
        return None


HIP = _load_hip()
requires_gpu = pytest.mark.skipif(HIP is None, reason="No GPU / HIP runtime")


def _run_gemm(M, N, K, tile=None):
    """Generate, assemble, launch GEMM kernel, return (D_gpu, D_ref)."""
    hip = HIP
    problem = GemmProblem(m=M, n=N, k=K)
    kernel = GemmKernel.build(problem, tile)
    result = kernel.emit()
    co = result.assemble(output_path=f"/tmp/test_gpu_{M}_{N}_{K}.co")

    tile_cfg = kernel.tile
    elem = 2
    d_A, d_B, d_D = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    hip.hipMalloc(ctypes.byref(d_A), M * K * elem)
    hip.hipMalloc(ctypes.byref(d_B), N * K * elem)
    hip.hipMalloc(ctypes.byref(d_D), M * N * elem)

    rng = np.random.RandomState(42)
    scale = 1.0 / np.sqrt(K)
    A = (rng.randn(M, K) * scale).astype(np.float16)
    B = (rng.randn(N, K) * scale).astype(np.float16)

    hip.hipMemcpy(d_A, A.ctypes.data_as(ctypes.c_void_p), M * K * elem, 1)
    hip.hipMemcpy(d_B, B.ctypes.data_as(ctypes.c_void_p), N * K * elem, 1)
    hip.hipMemset(d_D, 0, M * N * elem)

    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"gemm_kernel")

    lds = (tile_cfg.wg_m + tile_cfg.wg_n) * tile_cfg.unroll_k * elem
    gm, gn = M // tile_cfg.wg_m, N // tile_cfg.wg_n
    _vals, args = _build_tl_args(d_A, d_B, d_D, M, N, K, gm, gn)

    ret = hip.hipModuleLaunchKernel(
        func, gm, gn, 1, tile_cfg.block_size, 1, 1, lds, None, args, None)
    assert ret == 0, f"Launch failed: {ret}"
    ret = hip.hipDeviceSynchronize()
    assert ret == 0, f"Sync failed: {ret}"

    D_gpu = np.zeros((M, N), dtype=np.float16)
    hip.hipMemcpy(D_gpu.ctypes.data_as(ctypes.c_void_p), d_D, M * N * elem, 2)

    hip.hipFree(d_A)
    hip.hipFree(d_B)
    hip.hipFree(d_D)
    hip.hipModuleUnload(module)

    D_ref = (A.astype(np.float32) @ B.astype(np.float32).T).astype(np.float16)
    return D_gpu, D_ref


# --- Tests ---

@requires_gpu
class TestGPUCorrectness:
    """Verify GEMM produces correct results on GPU."""

    @pytest.mark.parametrize("M,N,K", [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ])
    def test_square(self, M, N, K):
        D_gpu, D_ref = _run_variant(M, N, K, scheduled=True)
        assert np.allclose(D_gpu, D_ref, atol=1.0, rtol=0.05), \
            f"Max error: {np.max(np.abs(D_gpu.astype(np.float32) - D_ref.astype(np.float32)))}"

    def test_large(self):
        D_gpu, D_ref = _run_variant(4096, 4096, 4096, scheduled=True)
        assert np.allclose(D_gpu, D_ref, atol=1.0, rtol=0.05)

    def test_identity(self):
        """A = I, B = I => D = I (for the first K rows/cols)."""
        M, N, K = 256, 256, 256
        problem = GemmProblem(m=M, n=N, k=K)
        kernel = GemmKernel.build(problem)
        result = kernel.emit()
        co = result.assemble(output_path="/tmp/test_gpu_identity.co")
        tile = kernel.tile
        elem = 2
        hip = HIP

        d_A, d_B, d_D = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        hip.hipMalloc(ctypes.byref(d_A), M * K * elem)
        hip.hipMalloc(ctypes.byref(d_B), N * K * elem)
        hip.hipMalloc(ctypes.byref(d_D), M * N * elem)

        A = np.eye(M, K, dtype=np.float16)
        B = np.eye(N, K, dtype=np.float16)
        hip.hipMemcpy(d_A, A.ctypes.data_as(ctypes.c_void_p), M * K * elem, 1)
        hip.hipMemcpy(d_B, B.ctypes.data_as(ctypes.c_void_p), N * K * elem, 1)
        hip.hipMemset(d_D, 0, M * N * elem)

        module = ctypes.c_void_p()
        hip.hipModuleLoad(ctypes.byref(module), co.encode())
        func = ctypes.c_void_p()
        hip.hipModuleGetFunction(ctypes.byref(func), module, b"gemm_kernel")
        gm, gn = M // tile.wg_m, N // tile.wg_n
        _vals, args = _build_tl_args(d_A, d_B, d_D, M, N, K, gm, gn)
        hip.hipModuleLaunchKernel(
            func, gm, gn, 1, tile.block_size, 1, 1, 0, None, args, None)
        hip.hipDeviceSynchronize()

        D = np.zeros((M, N), dtype=np.float16)
        hip.hipMemcpy(D.ctypes.data_as(ctypes.c_void_p), d_D, M * N * elem, 2)
        hip.hipFree(d_A); hip.hipFree(d_B); hip.hipFree(d_D)
        hip.hipModuleUnload(module)

        D_ref = np.eye(M, N, dtype=np.float16)
        assert np.allclose(D, D_ref, atol=0.01)

    def test_all_ones(self):
        """A = 1, B = 1 => D[i,j] = K for all i,j."""
        M, N, K = 256, 256, 256
        problem = GemmProblem(m=M, n=N, k=K)
        kernel = GemmKernel.build(problem)
        result = kernel.emit()
        co = result.assemble(output_path="/tmp/test_gpu_ones.co")
        tile = kernel.tile
        elem = 2
        hip = HIP

        d_A, d_B, d_D = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        hip.hipMalloc(ctypes.byref(d_A), M * K * elem)
        hip.hipMalloc(ctypes.byref(d_B), N * K * elem)
        hip.hipMalloc(ctypes.byref(d_D), M * N * elem)

        A = np.ones((M, K), dtype=np.float16)
        B = np.ones((N, K), dtype=np.float16)
        hip.hipMemcpy(d_A, A.ctypes.data_as(ctypes.c_void_p), M * K * elem, 1)
        hip.hipMemcpy(d_B, B.ctypes.data_as(ctypes.c_void_p), N * K * elem, 1)
        hip.hipMemset(d_D, 0, M * N * elem)

        module = ctypes.c_void_p()
        hip.hipModuleLoad(ctypes.byref(module), co.encode())
        func = ctypes.c_void_p()
        hip.hipModuleGetFunction(ctypes.byref(func), module, b"gemm_kernel")
        gm, gn = M // tile.wg_m, N // tile.wg_n
        _vals, args = _build_tl_args(d_A, d_B, d_D, M, N, K, gm, gn)
        hip.hipModuleLaunchKernel(
            func, gm, gn, 1, tile.block_size, 1, 1, 0, None, args, None)
        hip.hipDeviceSynchronize()

        D = np.zeros((M, N), dtype=np.float16)
        hip.hipMemcpy(D.ctypes.data_as(ctypes.c_void_p), d_D, M * N * elem, 2)
        hip.hipFree(d_A); hip.hipFree(d_B); hip.hipFree(d_D)
        hip.hipModuleUnload(module)

        assert np.allclose(D, float(K), atol=1.0)


@requires_gpu
class TestGPUPerformance:
    """TFLOPS sanity checks."""

    def test_tflops_4096(self):
        """4096^3 should achieve > 50 TFLOPS."""
        import time
        M = N = K = 4096
        problem = GemmProblem(m=M, n=N, k=K)
        kernel = GemmKernel.build(problem)
        result = kernel.emit()
        co = result.assemble(output_path="/tmp/test_gpu_perf.co")
        tile = kernel.tile
        elem = 2
        hip = HIP

        d_A, d_B, d_D = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        hip.hipMalloc(ctypes.byref(d_A), M * K * elem)
        hip.hipMalloc(ctypes.byref(d_B), N * K * elem)
        hip.hipMalloc(ctypes.byref(d_D), M * N * elem)

        module = ctypes.c_void_p()
        hip.hipModuleLoad(ctypes.byref(module), co.encode())
        func = ctypes.c_void_p()
        hip.hipModuleGetFunction(ctypes.byref(func), module, b"gemm_kernel")
        gm, gn = M // tile.wg_m, N // tile.wg_n
        _vals, args = _build_tl_args(d_A, d_B, d_D, M, N, K, gm, gn)

        # Warmup
        for _ in range(3):
            hip.hipModuleLaunchKernel(
                func, gm, gn, 1, tile.block_size, 1, 1,
                0, None, args, None)
        hip.hipDeviceSynchronize()

        # Timed
        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            hip.hipModuleLaunchKernel(
                func, gm, gn, 1, tile.block_size, 1, 1,
                0, None, args, None)
        hip.hipDeviceSynchronize()
        avg = (time.perf_counter() - t0) / iters
        tflops = 2 * M * N * K / avg / 1e12

        hip.hipFree(d_A); hip.hipFree(d_B); hip.hipFree(d_D)
        hip.hipModuleUnload(module)

        assert tflops > 50, f"Only {tflops:.1f} TFLOPS, expected > 50"


@requires_gpu
class TestScheduledKernelGPU:
    """Test the scheduled K-loop variant on GPU."""

    @pytest.mark.parametrize("M,N,K", [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ])
    def test_scheduled_correct(self, M, N, K):
        D_gpu, D_ref = _run_variant(M, N, K, scheduled=True)
        max_err = np.max(np.abs(D_gpu.astype(np.float32) - D_ref.astype(np.float32)))
        assert max_err < 1.0, \
            f"scheduled {M}x{N}x{K}: max_err={max_err}"


# --- Variant kernel tests ---

def _run_variant(M, N, K, **build_kwargs):
    """Generate, assemble, launch a kernel variant, return (D_gpu, D_ref)."""
    hip = HIP
    problem = GemmProblem(m=M, n=N, k=K)
    kernel = GemmKernel.build(problem, **build_kwargs)
    result = kernel.emit()
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        co_path = os.path.join(tmpdir, f"test_{M}_{N}_{K}.co")
        co = result.assemble(output_path=co_path)

        tile = kernel.tile
        elem = 2
        d_A, d_B, d_D = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        hip.hipMalloc(ctypes.byref(d_A), M * K * elem)
        hip.hipMalloc(ctypes.byref(d_B), N * K * elem)
        hip.hipMalloc(ctypes.byref(d_D), M * N * elem)

        rng = np.random.RandomState(42)
        scale = 1.0 / np.sqrt(K)
        A = (rng.randn(M, K) * scale).astype(np.float16)
        B = (rng.randn(N, K) * scale).astype(np.float16)

        hip.hipMemcpy(d_A, A.ctypes.data_as(ctypes.c_void_p), M * K * elem, 1)
        hip.hipMemcpy(d_B, B.ctypes.data_as(ctypes.c_void_p), N * K * elem, 1)
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

        D_gpu = np.zeros((M, N), dtype=np.float16)
        hip.hipMemcpy(D_gpu.ctypes.data_as(ctypes.c_void_p), d_D,
                       M * N * elem, 2)

        hip.hipFree(d_A)
        hip.hipFree(d_B)
        hip.hipFree(d_D)
        hip.hipModuleUnload(module)

    D_ref = (A.astype(np.float32) @ B.astype(np.float32).T).astype(np.float16)
    return D_gpu, D_ref


@requires_gpu
class TestBaselineKernel:
    """Scheduled kernel at various sizes."""

    @pytest.mark.parametrize("M,N,K", [
        (256, 256, 64),
        (256, 256, 256),
        (512, 512, 512),
    ])
    def test_correct(self, M, N, K):
        D_gpu, D_ref = _run_variant(M, N, K, scheduled=True)
        max_err = np.max(np.abs(D_gpu.astype(np.float32)
                                - D_ref.astype(np.float32)))
        assert max_err < 1.0, f"Scheduled {M}x{N}x{K}: max_err={max_err}"


@requires_gpu
@requires_gpu
class TestComposablePartitionedKernel:
    """Composable K-loop with partition-based scheduling."""

    @pytest.mark.parametrize("M,N,K", [
        (256, 256, 64),
        (256, 256, 128),
        (4096, 4096, 4096),
    ])
    def test_correct(self, M, N, K):
        D_gpu, D_ref = _run_variant(M, N, K, composable=True)
        max_err = np.max(np.abs(D_gpu.astype(np.float32)
                                - D_ref.astype(np.float32)))
        assert max_err < 0.001, \
            f"composable {M}x{N}x{K}: max_err={max_err}"

    def test_emit_structure(self):
        """Verify the kernel uses the expected tile config."""
        problem = GemmProblem(m=4096, n=4096, k=4096)
        kernel = GemmKernel.build(problem, composable=True)
        assert kernel.tile.wg_m == 256
        assert kernel.tile.wg_n == 256
        assert kernel.tile.unroll_k == 64
        result = kernel.emit()
        assert result.vgpr_count <= 128, \
            f"VGPR count {result.vgpr_count} too high"
        assert result.acc_count == 256


@requires_gpu
class TestSmallBaseline:
    """Scheduled kernel at the smallest valid tile size (256x256x64)."""

    def test_correct(self):
        D_gpu, D_ref = _run_variant(256, 256, 64, scheduled=True)
        max_err = np.max(np.abs(D_gpu.astype(np.float32)
                                - D_ref.astype(np.float32)))
        assert max_err < 1.0, f"Small scheduled: max_err={max_err}"


@requires_gpu
class TestComposableKernel:
    """Composable K-loop with modular building blocks.."""

    @pytest.mark.parametrize("M,N,K", [
        (256, 256, 64),
        (256, 256, 128),
    ])
    def test_correct(self, M, N, K):
        D_gpu, D_ref = _run_variant(M, N, K, composable=True)
        max_err = np.max(np.abs(D_gpu.astype(np.float32)
                                - D_ref.astype(np.float32)))
        assert max_err < 0.001, \
            f"composable {M}x{N}x{K}: max_err={max_err}"



@requires_gpu
class TestScheduledKernel:
    """Dependency-driven scheduled K-loop on GPU."""

    @pytest.mark.parametrize("M,N,K", [
        (256, 256, 64),
        (256, 256, 128),
        (4096, 4096, 4096),
    ])
    def test_correct(self, M, N, K):
        D_gpu, D_ref = _run_variant(M, N, K, scheduled=True)
        max_err = np.max(np.abs(D_gpu.astype(np.float32)
                                - D_ref.astype(np.float32)))
        assert max_err < 0.001, \
            f"scheduled {M}x{N}x{K}: max_err={max_err}"

    def test_emit_structure(self):
        """Verify the kernel uses the expected tile config."""
        problem = GemmProblem(m=4096, n=4096, k=4096)
        kernel = GemmKernel.build(problem, scheduled=True)
        assert kernel.tile.wg_m == 256
        assert kernel.tile.wg_n == 256
        assert kernel.tile.unroll_k == 64
        result = kernel.emit()
        assert result.vgpr_count <= 128, \
            f"VGPR count {result.vgpr_count} too high"
        assert result.acc_count == 256
