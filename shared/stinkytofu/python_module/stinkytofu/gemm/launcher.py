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
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .problem import DataType, GemmProblem, TileConfig

__all__ = ["GemmLauncher", "GemmResult"]


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

        from stinkytofu.gemm.launcher import GemmLauncher
        from stinkytofu.gemm.problem import GemmProblem, TileConfig

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

        # Apply transposes
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

        D_ref = D_f32.astype(self._np_dtype(p.dtype))
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

        def _check(ret, msg="HIP error"):
            if ret != 0:
                raise RuntimeError(f"{msg}: error code {ret}")

        # hipMalloc / hipMemcpy
        d_A = ctypes.c_void_p()
        d_B = ctypes.c_void_p()
        d_D = ctypes.c_void_p()
        elem = p.element_bytes

        _check(hip.hipMalloc(ctypes.byref(d_A), int(p.m * p.k * elem)), "hipMalloc A")
        _check(hip.hipMalloc(ctypes.byref(d_B), int(p.k * p.n * elem)), "hipMalloc B")
        _check(hip.hipMalloc(ctypes.byref(d_D), int(p.m * p.n * elem)), "hipMalloc D")

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
        D_out = np.zeros((p.m, p.n), dtype=self._np_dtype(p.dtype))
        _check(hip.hipMemcpy(D_out.ctypes.data, d_D, int(p.m * p.n * elem), 2), "D2H D")

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
    ) -> GemmResult:
        """Launch the generated GEMM assembly kernel on the GPU.

        The kernel expects these arguments packed in kernarg segment:
          offset  0: A ptr (8 bytes)
          offset  8: B ptr (8 bytes)
          offset 16: D ptr (8 bytes)
          offset 24: M     (4 bytes)
          offset 28: N     (4 bytes)
          offset 32: K     (4 bytes)
        Total kernarg size: 64 bytes (padded).

        Grid: (M / wg_m, N / wg_n) workgroups
        Block: block_size threads (e.g., 256)
        """
        import struct

        p = self.problem
        A, B, C = self.generate_inputs()
        elem = p.element_bytes

        hip = _load_hip()

        def _check(ret, msg="HIP error"):
            if ret != 0:
                raise RuntimeError(f"{msg}: error code {ret}")

        # Allocate device memory
        d_A = ctypes.c_void_p()
        d_B = ctypes.c_void_p()
        d_D = ctypes.c_void_p()
        a_bytes = int(p.m * p.k * elem)
        b_bytes = int(p.n * p.k * elem)  # B is [N, K] for trans_b
        d_bytes = int(p.m * p.n * elem)

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

        # Pack kernel arguments into a flat buffer matching the kernarg layout
        # struct { void* A, void* B, void* D, int M, int N, int K }
        kernarg = struct.pack("QQQiii",
                              d_A.value, d_B.value, d_D.value,
                              p.m, p.n, p.k)
        # Pad to 64 bytes
        kernarg += b'\x00' * (64 - len(kernarg))
        kernarg_buf = (ctypes.c_char * 64)(*kernarg)
        kernarg_ptr = ctypes.cast(kernarg_buf, ctypes.c_void_p)

        # Allocate kernarg on device
        d_kernarg = ctypes.c_void_p()
        _check(hip.hipMalloc(ctypes.byref(d_kernarg), 64), "hipMalloc kernarg")
        _check(hip.hipMemcpy(d_kernarg, kernarg_ptr, 64, 1), "H2D kernarg")

        # Grid/block dims
        grid_m, grid_n = p.grid_dims(self.tile)
        block_size = self.tile.block_size

        # For hipModuleLaunchKernel with kernarg:
        # We pass NULL for args and use the kernarg segment
        # Actually, hipModuleLaunchKernel needs args as void** array
        # But for assembly kernels using kernarg segment, we use
        # hipExtModuleLaunchKernel or pass args as individual pointers.
        # Simpler: use the args array approach with each element pointer.
        _arg_A = d_A
        _arg_B = d_B
        _arg_D = d_D
        _arg_M = ctypes.c_int(p.m)
        _arg_N = ctypes.c_int(p.n)
        _arg_K = ctypes.c_int(p.k)

        _arg_vals = [_arg_A, _arg_B, _arg_D, _arg_M, _arg_N, _arg_K]
        args = (ctypes.c_void_p * len(_arg_vals))()
        for i, v in enumerate(_arg_vals):
            args[i] = ctypes.cast(ctypes.pointer(v), ctypes.c_void_p)

        # Assembly kernels declare LDS in the kernel descriptor
        # (.amdhsa_group_segment_fixed_size). Passing non-zero sharedMemBytes
        # to hipModuleLaunchKernel ADDS to the descriptor value, potentially
        # exceeding hardware limits. Always pass 0 for asm kernels.
        lds_size = 0

        # Warmup
        for _ in range(num_warmup):
            _check(hip.hipModuleLaunchKernel(
                func,
                grid_m, grid_n, 1,
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
                grid_m, grid_n, 1,
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
        D_out = np.zeros((p.m, p.n), dtype=self._np_dtype(p.dtype))
        _check(hip.hipMemcpy(D_out.ctypes.data, d_D, d_bytes, 2), "D2H D")

        # Cleanup
        hip.hipFree(d_A)
        hip.hipFree(d_B)
        hip.hipFree(d_D)
        hip.hipFree(d_kernarg)
        hip.hipModuleUnload(module)

        return GemmResult(D=D_out, time_seconds=avg_time)
