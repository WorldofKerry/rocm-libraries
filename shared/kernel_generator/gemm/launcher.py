# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""HIP host launcher: run GEMM kernels on GPU and verify correctness.

Provides:
- CPU reference via numpy
- GPU execution via HIP runtime (hipModuleLoad + hipModuleLaunchKernel)
- Correctness verification against numpy reference
- Performance measurement and TFLOPS estimation
"""
from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .problem import DataType, GemmProblem, TileConfig

__all__ = ["GemmLauncher", "GemmResult"]


# FP4 E2M1 lookup: 4-bit index -> float32 value
_FP4_E2M1_TABLE = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,   # 0000..0111 (positive)
   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # 1000..1111 (negative)
], dtype=np.float32)


def _unpack_fp4_to_float(packed: np.ndarray) -> np.ndarray:
    """Unpack uint8 array of packed FP4 E2M1 values to float32.

    Each byte holds 2 FP4 values: low nibble = first element,
    high nibble = second element.

    Input shape:  ``[rows, cols_packed]``
    Output shape: ``[rows, cols_packed * 2]``
    """
    lo = (packed & 0x0F).astype(np.intp)
    hi = ((packed >> 4) & 0x0F).astype(np.intp)
    rows, cols = packed.shape
    result = np.empty((rows, cols * 2), dtype=np.float32)
    result[:, 0::2] = _FP4_E2M1_TABLE[lo]
    result[:, 1::2] = _FP4_E2M1_TABLE[hi]
    return result


def _load_hip() -> ctypes.CDLL:
    """Load libamdhip64.so with proper argtypes to avoid 64-bit pointer truncation."""
    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError:
        raise RuntimeError("libamdhip64.so not found; is ROCm installed?")

    VP = ctypes.c_void_p
    PVP = ctypes.POINTER(ctypes.c_void_p)
    SZ = ctypes.c_size_t
    INT = ctypes.c_int
    CHAR_P = ctypes.c_char_p

    hip.hipSetDevice.argtypes = [INT]
    hip.hipSetDevice.restype = INT
    hip.hipMalloc.argtypes = [PVP, SZ]
    hip.hipMalloc.restype = INT
    hip.hipFree.argtypes = [VP]
    hip.hipFree.restype = INT
    hip.hipMemcpy.argtypes = [VP, VP, SZ, INT]
    hip.hipMemcpy.restype = INT
    hip.hipMemset.argtypes = [VP, INT, SZ]
    hip.hipMemset.restype = INT
    hip.hipModuleLoad.argtypes = [PVP, CHAR_P]
    hip.hipModuleLoad.restype = INT
    hip.hipModuleGetFunction.argtypes = [PVP, VP, CHAR_P]
    hip.hipModuleGetFunction.restype = INT
    # hipModuleLaunchKernel(func, gridX, gridY, gridZ, blockX, blockY, blockZ,
    #                       sharedMem, stream, kernelParams, extra)
    hip.hipModuleLaunchKernel.argtypes = [
        VP,   # function
        INT, INT, INT,  # grid dims
        INT, INT, INT,  # block dims
        INT,  # shared mem bytes
        VP,   # stream
        VP,   # kernelParams (void**)
        VP,   # extra (void**)
    ]
    hip.hipModuleLaunchKernel.restype = INT
    hip.hipDeviceSynchronize.restype = INT
    hip.hipModuleUnload.argtypes = [VP]
    hip.hipModuleUnload.restype = INT
    hip.hipEventCreate.argtypes = [PVP]
    hip.hipEventCreate.restype = INT
    hip.hipEventRecord.argtypes = [VP, VP]
    hip.hipEventRecord.restype = INT
    hip.hipEventSynchronize.argtypes = [VP]
    hip.hipEventSynchronize.restype = INT
    hip.hipEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), VP, VP]
    hip.hipEventElapsedTime.restype = INT
    hip.hipEventDestroy.argtypes = [VP]
    hip.hipEventDestroy.restype = INT

    return hip



@dataclass
class GemmResult:
    """Result of a GEMM execution."""
    D: np.ndarray                   # output matrix
    time_seconds: float = 0.0       # kernel execution time
    correct: Optional[bool] = None  # None if not verified
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0


class GemmLauncher:
    """Launch GEMM kernels on GPU and verify against CPU reference.

    Usage::

        from kernel_generator.gemm.launcher import GemmLauncher
        from kernel_generator.gemm.problem import GemmProblem, TileConfig

        problem = GemmProblem(m=1024, n=1024, k=1024)
        tile = TileConfig()
        launcher = GemmLauncher(problem, tile)

        # CPU reference
        A, B, C, D_ref = launcher.reference_numpy()

        # Verify (placeholder until GPU launch works)
        launcher.print_performance(0.001)  # hypothetical 1ms
    """

    def __init__(self, problem: GemmProblem, tile: TileConfig,
                 seed: int = 42) -> None:
        self.problem = problem
        self.tile = tile
        self.seed = seed
        self._A: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None
        self._scale_A: Optional[np.ndarray] = None
        self._scale_B: Optional[np.ndarray] = None

    # -- numpy dtype mapping ------------------------------------------------

    @staticmethod
    def _np_dtype(dt: DataType) -> np.dtype:
        if dt == DataType.MXFP4:
            return np.uint8  # packed: 2 FP4 elements per byte
        return {
            DataType.F16: np.float16,
            DataType.BF16: np.float16,  # numpy has no bfloat16; use f16 approx
            DataType.F32: np.float32,
        }[dt]

    # -- Input generation ---------------------------------------------------

    def generate_inputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random input matrices A, B, C.

        Uses a fixed seed for reproducibility.  A and B are cached so
        repeated calls return the same data.

        For MXFP4: generates random uint8 data (2 FP4 elements per byte).
        Buffer shapes use K//2 for the packed K dimension.
        """
        p = self.problem
        dtype = self._np_dtype(p.dtype)
        rng = np.random.RandomState(self.seed)

        if self._A is None:
            if p.dtype == DataType.MXFP4:
                # Packed FP4: 2 elements per byte, shape [M, K//2]
                self._A = rng.randint(0, 256, size=(p.m, p.k // 2),
                                      dtype=np.uint8)
                if p.trans_b:
                    self._B = rng.randint(0, 256, size=(p.n, p.k // 2),
                                          dtype=np.uint8)
                else:
                    self._B = rng.randint(0, 256, size=(p.k // 2, p.n),
                                          dtype=np.uint8)
            else:
                # Scale inputs to avoid overflow in f16
                scale = 1.0 / np.sqrt(p.k)
                self._A = (rng.randn(p.m, p.k) * scale).astype(dtype)
                if p.trans_b:
                    self._B = (rng.randn(p.n, p.k) * scale).astype(dtype)
                else:
                    self._B = (rng.randn(p.k, p.n) * scale).astype(dtype)

        if p.dtype == DataType.MXFP4:
            # Output is fp16 (MXFP4 GEMM outputs fp16/fp32, not fp4)
            C = np.zeros((p.m, p.n), dtype=np.float16)
        else:
            C = np.zeros((p.m, p.n), dtype=dtype)
        return self._A, self._B, C

    # -- CPU reference ------------------------------------------------------

    def reference_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute reference GEMM on CPU with numpy.

        ``D = alpha * op(A) @ op(B) + beta * C``

        Accumulation is done in float32 for accuracy.

        Returns:
            ``(A, B, C, D_ref)`` where D_ref is the reference output.
        """
        p = self.problem
        A, B, C = self.generate_inputs()

        if p.dtype == DataType.MXFP4:
            # Unpack FP4 -> float32 before matmul
            # A is [M, K//2] packed -> [M, K]
            A_op = _unpack_fp4_to_float(A)
            if p.trans_a:
                A_op = A_op.T
            # B: unpack along the K (packed) dimension, then arrange as [K, N]
            if p.trans_b:
                # B stored as [N, K//2] -> unpack -> [N, K] -> transpose -> [K, N]
                B_op = _unpack_fp4_to_float(B).T
            else:
                # B stored as [K//2, N] -> .T -> [N, K//2] -> unpack -> [N, K] -> .T -> [K, N]
                B_op = _unpack_fp4_to_float(B.T).T
        else:
            # A stored as [M, K]; if trans_a, op(A) = A^T -> [K, M]
            A_op = A.astype(np.float32)
            if p.trans_a:
                A_op = A_op.T

            # B stored as [N, K] when trans_b, [K, N] otherwise
            # op(B): if trans_b, B^T -> [K, N]; if not, B -> [K, N]
            B_op = B.astype(np.float32)
            if p.trans_b:
                B_op = B_op.T  # [N, K] -> [K, N]

        # Accumulate in f32
        D_f32 = p.alpha * (A_op @ B_op) + p.beta * C.astype(np.float32)

        # Output is always fp16 for MXFP4, otherwise match input dtype
        out_dtype = np.float16 if p.dtype == DataType.MXFP4 else self._np_dtype(p.dtype)
        D_ref = D_f32.astype(out_dtype)
        return A, B, C, D_ref

    # -- Verification -------------------------------------------------------

    def verify(self, D_actual: np.ndarray, D_ref: np.ndarray,
               atol: float = 1e-2, rtol: float = 1e-2) -> GemmResult:
        """Verify GPU result against CPU reference.

        Args:
            D_actual: GPU output matrix.
            D_ref: CPU reference matrix.
            atol: Absolute tolerance.
            rtol: Relative tolerance.

        Returns:
            ``GemmResult`` with correctness flag and error metrics.
        """
        D_a = D_actual.astype(np.float32)
        D_r = D_ref.astype(np.float32)

        abs_err = np.abs(D_a - D_r)
        max_abs = float(np.max(abs_err))

        # Relative error (avoid division by zero)
        denom = np.maximum(np.abs(D_r), 1e-7)
        rel_err = abs_err / denom
        max_rel = float(np.max(rel_err))

        correct = bool(np.allclose(D_a, D_r, atol=atol, rtol=rtol))

        return GemmResult(
            D=D_actual,
            correct=correct,
            max_abs_error=max_abs,
            max_rel_error=max_rel,
        )

    # -- Performance --------------------------------------------------------

    def estimate_tflops(self, time_seconds: float) -> float:
        """Estimate TFLOPS from kernel execution time."""
        if time_seconds <= 0:
            return 0.0
        return self.problem.total_flops / time_seconds / 1e12

    def print_performance(self, time_seconds: float) -> None:
        """Print performance summary."""
        p = self.problem
        tflops = self.estimate_tflops(time_seconds)
        grid_m, grid_n = p.grid_dims(self.tile)
        print(f"  Problem    : {p.m}x{p.n}x{p.k} {p.dtype.value}")
        print(f"  Tile       : {self.tile.wg_m}x{self.tile.wg_n}x{self.tile.unroll_k}")
        print(f"  Grid       : {grid_m}x{grid_n} ({grid_m * grid_n} WGs)")
        print(f"  Time       : {time_seconds * 1000:.3f} ms")
        print(f"  TFLOPS     : {tflops:.2f}")
        print(f"  Total FLOPs: {p.total_flops:,}")
        print(f"  Arith. Int.: {p.arithmetic_intensity:.1f} FLOPs/byte")

    # -- GPU launch (placeholder for future) --------------------------------

    def run_hip_reference(
        self,
        code_object_path: str,
        kernel_name: str = "gemm_reference",
    ) -> GemmResult:
        """Launch a compiled HIP kernel on the GPU.

        Uses ``hipModuleLoad`` / ``hipModuleLaunchKernel`` via ctypes
        to load and run the kernel.

        Args:
            code_object_path: Path to the ``.co`` code object.
            kernel_name: Name of the kernel function in the code object.

        Returns:
            ``GemmResult`` with the output matrix and timing.

        Raises:
            RuntimeError: If HIP runtime calls fail.
        """
        p = self.problem
        A, B, C = self.generate_inputs()

        hip = _load_hip()

        def _check(ret: int, msg: str = "HIP error") -> None:
            if ret != 0:
                raise RuntimeError(f"{msg}: error code {ret}")

        # hipMalloc / hipMemcpy
        d_A = ctypes.c_void_p()
        d_B = ctypes.c_void_p()
        d_D = ctypes.c_void_p()
        elem = p.element_bytes
        # D output is always fp16 (2 bytes) for MXFP4
        d_elem = 2 if p.dtype == DataType.MXFP4 else elem

        _check(hip.hipMalloc(ctypes.byref(d_A), int(p.m * p.k * elem)), "hipMalloc A")
        _check(hip.hipMalloc(ctypes.byref(d_B), int(p.k * p.n * elem)), "hipMalloc B")
        _check(hip.hipMalloc(ctypes.byref(d_D), int(p.m * p.n * d_elem)), "hipMalloc D")

        _check(hip.hipMemcpy(d_A, A.ctypes.data, int(p.m * p.k * elem), 1), "H2D A")
        _check(hip.hipMemcpy(d_B, B.ctypes.data, int(p.k * p.n * elem), 1), "H2D B")

        # hipModuleLoad + hipModuleGetFunction
        module = ctypes.c_void_p()
        _check(hip.hipModuleLoad(ctypes.byref(module),
                                 code_object_path.encode()), "hipModuleLoad")

        func = ctypes.c_void_p()
        _check(hip.hipModuleGetFunction(ctypes.byref(func), module,
                                        kernel_name.encode()), "hipModuleGetFunction")

        # Kernel arguments
        M = ctypes.c_int(p.m)
        N = ctypes.c_int(p.n)
        K = ctypes.c_int(p.k)
        lda = ctypes.c_int(p.k)  # row-major A[M,K]
        ldb = ctypes.c_int(p.n)  # row-major B[K,N]
        ldd = ctypes.c_int(p.n)  # row-major D[M,N]
        alpha = ctypes.c_float(p.alpha)
        beta = ctypes.c_float(p.beta)

        # Pack args for hipModuleLaunchKernel.
        # Each element must be a pointer to the argument value.
        # We keep references alive by storing them in a list first.
        _arg_vals = [d_A, d_B, d_D, M, N, K, lda, ldb, ldd, alpha, beta]
        args = (ctypes.c_void_p * len(_arg_vals))()
        for i, v in enumerate(_arg_vals):
            args[i] = ctypes.cast(ctypes.pointer(v), ctypes.c_void_p)

        # Grid / block dims
        grid_m, grid_n = p.grid_dims(self.tile)
        # Reference kernel: one thread per element, block = (wg_m, wg_n)
        # Clamp block dims to hardware limits
        block_x = min(self.tile.wg_m, 16)
        block_y = min(self.tile.wg_n, 16)
        import math
        grid_x = math.ceil(p.m / block_x)
        grid_y = math.ceil(p.n / block_y)

        # Warmup
        _check(hip.hipModuleLaunchKernel(
            func,
            grid_x, grid_y, 1,           # grid
            block_x, block_y, 1,         # block
            0, None,                     # shared mem, stream
            args, None,                  # args
        ), "hipModuleLaunchKernel warmup")
        _check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize warmup")

        # Timed run
        t0 = time.perf_counter()
        _check(hip.hipModuleLaunchKernel(
            func,
            grid_x, grid_y, 1,
            block_x, block_y, 1,
            0, None,
            args, None,
        ), "hipModuleLaunchKernel")
        _check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
        t1 = time.perf_counter()

        # Copy result back
        out_dtype = np.float16 if p.dtype == DataType.MXFP4 else self._np_dtype(p.dtype)
        D_out = np.zeros((p.m, p.n), dtype=out_dtype)
        _check(hip.hipMemcpy(D_out.ctypes.data, d_D, int(p.m * p.n * d_elem), 2), "D2H D")

        # Cleanup
        hip.hipFree(d_A)
        hip.hipFree(d_B)
        hip.hipFree(d_D)
        hip.hipModuleUnload(module)

        return GemmResult(D=D_out, time_seconds=t1 - t0)

    def run_asm_kernel(
        self,
        code_object_path: str,
        kernel_name: str = "gemm_kernel",
        num_warmup: int = 3,
        num_iters: int = 10,
        lds_bytes: int = 0,
        use_1d_grid: bool = False,
    ) -> GemmResult:
        """Launch the generated GEMM assembly kernel on the GPU.

        TensileLite kernarg layout (KernArgsVersion >= 1):
          offset  0-15: header (Gemm info, kernel info0/1, numWG)
          offset 16: M, 20: N, 24: batch, 28: K
          offset 32: D, 40: C, 48: A, [56: MXSA], 64: B, [72: MXSB]
          offset 80+: strides, alpha, beta
        Grid: 1D (total_wgs, 1, 1) when use_1d_grid=True,
              2D (grid_m, grid_n, 1) otherwise.
        Block: block_size threads.

        Args:
            use_1d_grid: When True, launch with a 1D grid
                ``(grid_m * grid_n, 1, 1)`` for WorkGroupMappingXCC
                L2 locality across XCCs.
        """
        import struct

        p = self.problem
        A, B, C = self.generate_inputs()
        elem = p.element_bytes

        hip = _load_hip()

        def _check(ret: int, msg: str = "HIP error") -> None:
            if ret != 0:
                raise RuntimeError(f"{msg}: error code {ret}")

        # Allocate device memory
        d_A = ctypes.c_void_p()
        d_B = ctypes.c_void_p()
        d_D = ctypes.c_void_p()
        a_bytes = int(p.m * p.k * elem)
        b_bytes = int(p.n * p.k * elem)  # B is [N, K] for trans_b
        # D output is always fp16 (2 bytes/element) for MXFP4,
        # regardless of input element size.
        d_bytes = int(p.m * p.n * (2 if p.dtype == DataType.MXFP4 else elem))

        _check(hip.hipMalloc(ctypes.byref(d_A), a_bytes), "hipMalloc A")
        _check(hip.hipMalloc(ctypes.byref(d_B), b_bytes), "hipMalloc B")
        _check(hip.hipMalloc(ctypes.byref(d_D), d_bytes), "hipMalloc D")

        # Copy inputs to device
        # A is [M, K] row-major
        _check(hip.hipMemcpy(d_A, A.ctypes.data, a_bytes, 1), "H2D A")
        # B is [N, K] row-major (trans_b=True means stored as N x K)
        _check(hip.hipMemcpy(d_B, B.ctypes.data, b_bytes, 1), "H2D B")
        # Zero D
        _check(hip.hipMemset(d_D, 0, d_bytes), "memset D")

        # Load module
        module = ctypes.c_void_p()
        _check(hip.hipModuleLoad(ctypes.byref(module),
                                 code_object_path.encode()),
               "hipModuleLoad")

        func = ctypes.c_void_p()
        _check(hip.hipModuleGetFunction(ctypes.byref(func), module,
                                        kernel_name.encode()),
               "hipModuleGetFunction")

        # Pack TensileLite kernarg layout (KernArgsVersion >= 1)
        is_mx = hasattr(self.tile, 'mfma') and getattr(self.tile.mfma, 'is_mx', False)
        d_scale_A = ctypes.c_void_p()
        d_scale_B = ctypes.c_void_p()
        grid_m, grid_n = p.grid_dims(self.tile)
        total_wgs = grid_m * grid_n
        block_size = self.tile.block_size

        if is_mx:
            mx_block = self.tile.mfma.mx_block
            scale_a_cols = p.k // mx_block
            scale_b_cols = p.k // mx_block
            scale_a_bytes = p.m * scale_a_cols
            scale_b_bytes = p.n * scale_b_cols
            _check(hip.hipMalloc(ctypes.byref(d_scale_A), scale_a_bytes),
                   "hipMalloc scale_A")
            _check(hip.hipMalloc(ctypes.byref(d_scale_B), scale_b_bytes),
                   "hipMalloc scale_B")
            if self._scale_A is not None:
                _check(hip.hipMemcpy(d_scale_A, self._scale_A.ctypes.data,
                                     scale_a_bytes, 1), "H2D scale_A")
            else:
                _check(hip.hipMemset(d_scale_A, 0x7F, scale_a_bytes),
                       "memset scale_A=1.0")
            if self._scale_B is not None:
                _check(hip.hipMemcpy(d_scale_B, self._scale_B.ctypes.data,
                                     scale_b_bytes, 1), "H2D scale_B")
            else:
                _check(hip.hipMemset(d_scale_B, 0x7F, scale_b_bytes),
                       "memset scale_B=1.0")
            # TensileLite MXFP4 kernarg (136 bytes):
            # header(16) + sizes(16) + ptrs(48) + strides(48) + alpha/beta(8)
            kernarg = struct.pack(
                "<IIII"       # header: gemm_info, info0, info1, numWG
                "IIII"        # sizes: M, N, batch, K
                "QQ"          # D, C ptrs
                "QQ"          # A, MXSA ptrs
                "QQ"          # B, MXSB ptrs
                "IIIIIIII"    # strides: D0,D1,C0,C1,A0,A1,MXSA0,MXSA1
                "IIII"        # strides: B0,B1,MXSB0,MXSB1
                "ff"          # alpha, beta
                ,
                0, 0, 0, total_wgs,                           # header
                p.m, p.n, 1, p.k,                             # sizes (batch=1)
                d_D.value, d_D.value,                         # D, C (C=D for beta=0)
                d_A.value, d_scale_A.value,                   # A, MXSA
                d_B.value, d_scale_B.value,                   # B, MXSB
                p.m, p.m * p.n, p.m, p.m * p.n,              # strideD0/1, strideC0/1
                p.k, p.m * p.k, scale_a_cols, p.m * scale_a_cols,  # strideA0/1, MXSA0/1
                p.k, p.n * p.k, scale_b_cols, p.n * scale_b_cols,  # strideB0/1, MXSB0/1
                1.0, 0.0,                                     # alpha, beta
            )
            kernarg_size = 136
        else:
            # TensileLite FP16 kernarg (104 bytes)
            kernarg = struct.pack(
                "<IIII"       # header
                "IIII"        # sizes
                "QQ"          # D, C ptrs
                "QQ"          # A, B ptrs
                "IIIIIIII"    # strides: D0,D1,C0,C1,A0,A1,B0,B1
                "ff"          # alpha, beta
                ,
                0, 0, 0, total_wgs,
                p.m, p.n, 1, p.k,
                d_D.value, d_D.value,
                d_A.value, d_B.value,
                p.n, p.m * p.n, p.n, p.m * p.n,
                p.k, p.m * p.k, p.k, p.n * p.k,
                1.0, 0.0,
            )
            kernarg_size = 104
        kernarg_buf = (ctypes.c_char * kernarg_size)(*kernarg)
        kernarg_ptr = ctypes.cast(kernarg_buf, ctypes.c_void_p)

        # Use hipExtModuleLaunchKernel with flat kernarg buffer
        # Build void** args array matching TensileLite kernarg order
        # hipModuleLaunchKernel with args=void** packs each arg sequentially
        # We use the flat buffer approach via HIP_LAUNCH_PARAM_BUFFER_POINTER
        d_kernarg = ctypes.c_void_p()
        _check(hip.hipMalloc(ctypes.byref(d_kernarg), kernarg_size),
               "hipMalloc kernarg")
        _check(hip.hipMemcpy(d_kernarg, kernarg_ptr, kernarg_size, 1),
               "H2D kernarg")

        # Flatten grid to 1D (TensileLite KernArgsVersion >= 1)
        args = None  # will use flat kernarg via hipExtModuleLaunchKernel

        # Assembly kernels declare LDS in the kernel descriptor
        # (.amdhsa_group_segment_fixed_size). Passing non-zero sharedMemBytes
        # to hipModuleLaunchKernel ADDS to the descriptor value, potentially
        # exceeding hardware limits. Always pass 0 for asm kernels.
        lds_size = 0

        # Build args array for hipModuleLaunchKernel (void** of pointers to each arg)
        _header = [ctypes.c_uint32(0), ctypes.c_uint32(0),
                   ctypes.c_uint32(0), ctypes.c_uint32(total_wgs)]
        _sizes = [ctypes.c_uint32(p.m), ctypes.c_uint32(p.n),
                  ctypes.c_uint32(1), ctypes.c_uint32(p.k)]
        _ptrs = [ctypes.c_void_p(d_D.value), ctypes.c_void_p(d_D.value)]  # D, C
        if is_mx:
            _ptrs += [ctypes.c_void_p(d_A.value), ctypes.c_void_p(d_scale_A.value),
                      ctypes.c_void_p(d_B.value), ctypes.c_void_p(d_scale_B.value)]
            _strides = [
                ctypes.c_uint32(p.m), ctypes.c_uint32(p.m * p.n),       # D
                ctypes.c_uint32(p.m), ctypes.c_uint32(p.m * p.n),       # C
                ctypes.c_uint32(p.k), ctypes.c_uint32(p.m * p.k),       # A
                ctypes.c_uint32(scale_a_cols), ctypes.c_uint32(p.m * scale_a_cols),  # MXSA
                ctypes.c_uint32(p.k), ctypes.c_uint32(p.n * p.k),       # B
                ctypes.c_uint32(scale_b_cols), ctypes.c_uint32(p.n * scale_b_cols),  # MXSB
            ]
        else:
            _ptrs += [ctypes.c_void_p(d_A.value), ctypes.c_void_p(d_B.value)]
            _strides = [
                ctypes.c_uint32(p.n), ctypes.c_uint32(p.m * p.n),
                ctypes.c_uint32(p.n), ctypes.c_uint32(p.m * p.n),
                ctypes.c_uint32(p.k), ctypes.c_uint32(p.m * p.k),
                ctypes.c_uint32(p.k), ctypes.c_uint32(p.n * p.k),
            ]
        _alpha_beta = [ctypes.c_float(1.0), ctypes.c_float(0.0)]
        _all_args = _header + _sizes + _ptrs + _strides + _alpha_beta
        args = (ctypes.c_void_p * len(_all_args))()
        for i, v in enumerate(_all_args):
            args[i] = ctypes.cast(ctypes.pointer(v), ctypes.c_void_p)

        # Dispatch grid dimensions
        if use_1d_grid:
            # 1D grid for WorkGroupMappingXCC: kernel decomposes the
            # flat WG index back into (tile_m, tile_n) in its setup phase.
            launch_grid_x = total_wgs
            launch_grid_y = 1
        else:
            launch_grid_x = grid_m
            launch_grid_y = grid_n

        for _ in range(num_warmup):
            _check(hip.hipModuleLaunchKernel(
                func,
                launch_grid_x, launch_grid_y, 1,
                block_size, 1, 1,
                lds_size, None,
                args, None,
            ), "hipModuleLaunchKernel warmup")
        _check(hip.hipDeviceSynchronize(), "sync warmup")

        # Timed runs using HIP events for accurate GPU timing
        ev_start = ctypes.c_void_p()
        ev_stop = ctypes.c_void_p()
        _check(hip.hipEventCreate(ctypes.byref(ev_start)), "hipEventCreate")
        _check(hip.hipEventCreate(ctypes.byref(ev_stop)), "hipEventCreate")

        _check(hip.hipEventRecord(ev_start, None), "hipEventRecord start")
        for _ in range(num_iters):
            _check(hip.hipModuleLaunchKernel(
                func,
                launch_grid_x, launch_grid_y, 1,
                block_size, 1, 1,
                lds_size, None,
                args, None,
            ), "hipModuleLaunchKernel")
        _check(hip.hipEventRecord(ev_stop, None), "hipEventRecord stop")
        _check(hip.hipEventSynchronize(ev_stop), "hipEventSynchronize")

        elapsed_ms = ctypes.c_float()
        _check(hip.hipEventElapsedTime(ctypes.byref(elapsed_ms),
                                        ev_start, ev_stop),
               "hipEventElapsedTime")
        avg_time = elapsed_ms.value / 1000.0 / num_iters

        hip.hipEventDestroy(ev_start)
        hip.hipEventDestroy(ev_stop)

        # Copy result back
        out_dtype = np.float16 if p.dtype == DataType.MXFP4 else self._np_dtype(p.dtype)
        D_out = np.zeros((p.m, p.n), dtype=out_dtype)
        _check(hip.hipMemcpy(D_out.ctypes.data, d_D, d_bytes, 2), "D2H D")

        # Cleanup
        hip.hipFree(d_A)
        hip.hipFree(d_B)
        hip.hipFree(d_D)
        if is_mx:
            hip.hipFree(d_scale_A)
            hip.hipFree(d_scale_B)
        hip.hipFree(d_kernarg)
        hip.hipModuleUnload(module)

        return GemmResult(D=D_out, time_seconds=avg_time)

    def run_streamk(self, co_path: str, kernel_name: str = "gemm_kernel",
                    num_warmup: int = 10, num_iters: int = 100,
                    num_cus: int = 304):
        """Launch kernel with StreamK work distribution.

        Allocates workspace, launches the GEMM kernel with StreamK
        parameters, then (for partial tiles) would launch a fixup kernel.

        For the initial implementation, this uses a simplified approach:
        each WG handles one complete output tile (no partial K), but
        the grid is sized to fill all CUs even when tile count < CU count.

        Args:
            co_path: Path to compiled code object.
            kernel_name: Name of the kernel function.
            num_warmup: Number of warmup iterations.
            num_iters: Number of timed iterations.
            num_cus: Number of compute units on target GPU.

        Returns:
            GemmResult with timing and correctness info.
        """
        from .schedule.pipeline import StreamKPartitioner

        p = self.problem
        tile = self.tile
        sk = StreamKPartitioner(num_cus=num_cus)
        params = sk.compute_sk_params(p, tile)

        # For now: just use data-parallel (full tiles) since
        # the AtomicEpilogue path needs more validation.
        # Launch with grid_dims from StreamKPartitioner.
        grid = sk.grid_dims(p, tile)

        # Delegate to run_asm_kernel with 1D grid if beneficial
        if params["sk_tiles"] == 0:
            # All tiles fit in full waves, no StreamK benefit
            return self.run_asm_kernel(
                co_path, kernel_name=kernel_name,
                num_warmup=num_warmup, num_iters=num_iters)

        # StreamK would launch here with workspace allocation
        # For now, fall back to regular launch
        return self.run_asm_kernel(
            co_path, kernel_name=kernel_name,
            num_warmup=num_warmup, num_iters=num_iters)
