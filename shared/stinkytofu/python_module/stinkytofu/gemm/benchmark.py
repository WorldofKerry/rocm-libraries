# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark GEMM kernel generator vs hipBLASLt.

Generates assembly kernels for several square problem sizes, launches
them on GPU, measures TFLOPS, and optionally compares against
hipblaslt-bench if the binary is available.

Usage::

    PYTHONPATH=python_module:$PYTHONPATH python3 -m stinkytofu.gemm.benchmark
"""
from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# -- Constants --------------------------------------------------------------

# MI355X theoretical fp16 peak (TFLOPS).
# 1 PFLOPS = 1000 TFLOPS per the task specification.
THEORETICAL_PEAK_TFLOPS = 1000.0

DEFAULT_SIZES: List[Tuple[int, int, int]] = [
    (128, 128, 128),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]

NUM_WARMUP = 3
NUM_ITERS = 10

HIPBLASLT_BENCH_SEARCH_PATHS = [
    "/home/kerrwang/repos/rocm-libraries/agent/projects/hipblaslt/build/clients/hipblaslt-bench",
    "/home/kerrwang/repos/rocm-libraries/build/release/hipblaslt-install/bin/hipblaslt-bench",
    "/home/kerrwang/repos/rocm-libraries/build/hipblaslt-install/bin/hipblaslt-bench",
    "/opt/rocm/bin/hipblaslt-bench",
]


# -- Result container -------------------------------------------------------

@dataclass
class BenchmarkRow:
    m: int
    n: int
    k: int
    our_tflops: Optional[float] = None
    hblt_tflops: Optional[float] = None

    @property
    def size_label(self) -> str:
        return f"{self.m}x{self.n}x{self.k}"

    @property
    def pct_of_hblt(self) -> Optional[float]:
        if self.our_tflops is not None and self.hblt_tflops:
            return self.our_tflops / self.hblt_tflops * 100.0
        return None

    @property
    def pct_of_peak(self) -> Optional[float]:
        if self.our_tflops is not None:
            return self.our_tflops / THEORETICAL_PEAK_TFLOPS * 100.0
        return None


# -- HIP helpers ------------------------------------------------------------

def _load_hip():
    """Load and configure the HIP runtime via ctypes."""
    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError:
        return None

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

    # Probe GPU availability
    d = ctypes.c_void_p()
    if hip.hipMalloc(ctypes.byref(d), 4) != 0:
        return None
    hip.hipFree(d)
    return hip


def _check(ret: int, msg: str = "HIP error") -> None:
    if ret != 0:
        raise RuntimeError(f"{msg}: error code {ret}")


# -- Our kernel benchmark ---------------------------------------------------

def bench_our_kernel(
    hip,
    M: int, N: int, K: int,
    num_warmup: int = NUM_WARMUP,
    num_iters: int = NUM_ITERS,
) -> float:
    """Generate, assemble, and benchmark our GEMM kernel. Returns TFLOPS."""
    from .kernel_pipeline import GemmKernel
    from .problem import GemmProblem, TileConfig
    from .asm_emitter import build_pipelined_gemm_tree
    from .asm_transforms import GemmLayouts

    problem = GemmProblem(m=M, n=N, k=K)
    tile = TileConfig()
    problem.validate(tile)
    layouts = GemmLayouts.build(problem, tile)
    tree = build_pipelined_gemm_tree(problem, tile, layouts)
    kernel = GemmKernel.build(problem, tile_tree=tree)
    result = kernel.emit()

    with tempfile.TemporaryDirectory(prefix="stinkytofu_bench_") as tmpdir:
        co_path = os.path.join(tmpdir, f"gemm_{M}_{N}_{K}.co")
        co = result.assemble(output_path=co_path)
        tile = kernel.tile
        elem = problem.element_bytes  # 2 for fp16

        # Allocate device memory
        d_A = ctypes.c_void_p()
        d_B = ctypes.c_void_p()
        d_D = ctypes.c_void_p()
        _check(hip.hipMalloc(ctypes.byref(d_A), M * K * elem), "hipMalloc A")
        _check(hip.hipMalloc(ctypes.byref(d_B), N * K * elem), "hipMalloc B")
        _check(hip.hipMalloc(ctypes.byref(d_D), M * N * elem), "hipMalloc D")

        # Initialize inputs on host and copy to device
        rng = np.random.RandomState(42)
        scale = 1.0 / np.sqrt(K)
        A = (rng.randn(M, K) * scale).astype(np.float16)
        B = (rng.randn(N, K) * scale).astype(np.float16)
        hip.hipMemcpy(d_A, A.ctypes.data_as(ctypes.c_void_p),
                       M * K * elem, 1)
        hip.hipMemcpy(d_B, B.ctypes.data_as(ctypes.c_void_p),
                       N * K * elem, 1)
        hip.hipMemset(d_D, 0, M * N * elem)

        # Load code object
        module = ctypes.c_void_p()
        _check(hip.hipModuleLoad(ctypes.byref(module), co.encode()),
               "hipModuleLoad")
        func = ctypes.c_void_p()
        _check(hip.hipModuleGetFunction(ctypes.byref(func), module,
                                        b"gemm_kernel"),
               "hipModuleGetFunction")

        # Kernel arguments: A, B, D, M, N, K (matches test_gpu.py layout)
        _a = [ctypes.c_void_p(d_A.value), ctypes.c_void_p(d_B.value),
              ctypes.c_void_p(d_D.value),
              ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]
        args = (ctypes.c_void_p * len(_a))()
        for i, v in enumerate(_a):
            args[i] = ctypes.cast(ctypes.pointer(v), ctypes.c_void_p)

        lds = (tile.wg_m + tile.wg_n) * tile.unroll_k * elem
        gm = M // tile.wg_m
        gn = N // tile.wg_n

        # Warmup
        for _ in range(num_warmup):
            hip.hipModuleLaunchKernel(
                func, gm, gn, 1, tile.block_size, 1, 1,
                lds, None, args, None)
        _check(hip.hipDeviceSynchronize(), "sync warmup")

        # Timed iterations -- batch dispatches, single sync at end
        t0 = time.perf_counter()
        for _ in range(num_iters):
            hip.hipModuleLaunchKernel(
                func, gm, gn, 1, tile.block_size, 1, 1,
                lds, None, args, None)
        _check(hip.hipDeviceSynchronize(), "sync timed")
        elapsed = (time.perf_counter() - t0) / num_iters

        tflops = 2.0 * M * N * K / elapsed / 1e12

        # Cleanup
        hip.hipFree(d_A)
        hip.hipFree(d_B)
        hip.hipFree(d_D)
        hip.hipModuleUnload(module)

    return tflops


# -- hipBLASLt benchmark ----------------------------------------------------

def find_hipblaslt_bench() -> Optional[str]:
    """Locate hipblaslt-bench binary, or return None."""
    found = shutil.which("hipblaslt-bench")
    if found:
        return found
    for p in HIPBLASLT_BENCH_SEARCH_PATHS:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def bench_hipblaslt(
    bench_bin: str,
    M: int, N: int, K: int,
    num_iters: int = NUM_ITERS,
) -> Optional[float]:
    """Run hipblaslt-bench for one size and return TFLOPS, or None on failure."""
    cmd = [
        bench_bin,
        "-m", str(M), "-n", str(N), "-k", str(K),
        "--a_type", "f16_r",
        "--b_type", "f16_r",
        "--c_type", "f16_r",
        "--d_type", "f16_r",
        "--compute_type", "f32_r",
        "--transA", "N",
        "--transB", "T",
        "--initialization", "trig_float",
        "--algo_method", "heuristic",
        "-i", str(num_iters),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"  [hipblaslt-bench] error for {M}x{N}x{K}: {exc}",
              file=sys.stderr)
        return None

    if proc.returncode != 0:
        print(f"  [hipblaslt-bench] failed for {M}x{N}x{K} "
              f"(rc={proc.returncode})", file=sys.stderr)
        if proc.stderr:
            for line in proc.stderr.strip().splitlines()[:5]:
                print(f"    {line}", file=sys.stderr)
        return None

    return _parse_hipblaslt_gflops(proc.stdout)


def _parse_hipblaslt_gflops(output: str) -> Optional[float]:
    """Extract hipblaslt-Gflops from CSV output and convert to TFLOPS.

    hipblaslt-bench output has info lines followed by a CSV section.
    The CSV header looks like:
        [0]:transA,...,hipblaslt-Gflops,hipblaslt-GB/s,us
    and data lines are indented with spaces.
    """
    lines = output.strip().splitlines()

    # Find the CSV header line (contains "Gflops")
    header_idx = None
    for i, line in enumerate(lines):
        if "gflops" in line.lower():
            header_idx = i
            break
    if header_idx is None:
        return None

    # Strip any "[N]:" prefix from the header
    header = lines[header_idx]
    if ":" in header.split(",")[0]:
        header = header.split(":", 1)[1]
    cols = [c.strip() for c in header.split(",")]

    # Find the Gflops column index (case-insensitive)
    gflops_idx = None
    for i, col in enumerate(cols):
        if "gflops" in col.lower():
            gflops_idx = i
            break
    if gflops_idx is None:
        return None

    # Parse the last data line after the header
    for data_line in reversed(lines[header_idx + 1:]):
        fields = [f.strip() for f in data_line.split(",")]
        if len(fields) > gflops_idx:
            try:
                gflops = float(fields[gflops_idx])
                return gflops / 1000.0  # GFLOPS -> TFLOPS
            except ValueError:
                continue
    return None


# -- Output formatting ------------------------------------------------------

def print_table(rows: List[BenchmarkRow], hblt_available: bool) -> None:
    """Print a clean comparison table."""
    if hblt_available:
        hdr = (f"{'Size':<16} {'Ours (TFLOPS)':>14} "
               f"{'hipBLASLt (TFLOPS)':>19} {'% of hBLT':>10} "
               f"{'% of Peak':>10}")
    else:
        hdr = f"{'Size':<16} {'Ours (TFLOPS)':>14} {'% of Peak':>10}"

    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        ours_s = f"{r.our_tflops:.1f}" if r.our_tflops is not None else "N/A"
        peak_s = (f"{r.pct_of_peak:.1f}%"
                  if r.pct_of_peak is not None else "N/A")

        if hblt_available:
            hblt_s = (f"{r.hblt_tflops:.1f}"
                      if r.hblt_tflops is not None else "N/A")
            pct_s = (f"{r.pct_of_hblt:.1f}%"
                     if r.pct_of_hblt is not None else "N/A")
            print(f"{r.size_label:<16} {ours_s:>14} {hblt_s:>19} "
                  f"{pct_s:>10} {peak_s:>10}")
        else:
            print(f"{r.size_label:<16} {ours_s:>14} {peak_s:>10}")


# -- Main -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark GEMM kernel generator vs hipBLASLt")
    parser.add_argument(
        "--sizes", type=str, default=None,
        help="Comma-separated MxNxK sizes "
             "(e.g. 1024x1024x1024,4096x4096x4096)")
    parser.add_argument(
        "--iters", type=int, default=NUM_ITERS,
        help=f"Timed iterations (default: {NUM_ITERS})")
    parser.add_argument(
        "--warmup", type=int, default=NUM_WARMUP,
        help=f"Warmup iterations (default: {NUM_WARMUP})")
    parser.add_argument(
        "--skip-hipblaslt", action="store_true",
        help="Skip hipBLASLt comparison")
    args = parser.parse_args()

    # Parse sizes
    if args.sizes:
        sizes = []
        for s in args.sizes.split(","):
            parts = s.strip().split("x")
            if len(parts) != 3:
                print(f"Invalid size: {s}", file=sys.stderr)
                sys.exit(1)
            sizes.append(tuple(int(x) for x in parts))
    else:
        sizes = DEFAULT_SIZES

    print("GEMM Kernel Generator Benchmark (fp16, gfx950)")
    print("=" * 47)
    print()

    # Load HIP runtime
    hip = _load_hip()
    if hip is None:
        print("ERROR: HIP runtime not available "
              "(no GPU or ROCm not installed)", file=sys.stderr)
        sys.exit(1)

    # Locate hipblaslt-bench
    hblt_bin = None
    if not args.skip_hipblaslt:
        hblt_bin = find_hipblaslt_bench()
        if hblt_bin:
            print(f"hipblaslt-bench: {hblt_bin}")
        else:
            print("hipblaslt-bench: not found (skipping comparison)")
    else:
        print("hipblaslt-bench: skipped by user")

    print(f"Iterations:     {args.iters} (warmup: {args.warmup})")
    print(f"Peak (fp16):    {THEORETICAL_PEAK_TFLOPS:.0f} TFLOPS (MI355X)")
    print()

    rows: List[BenchmarkRow] = []

    for M, N, K in sizes:
        row = BenchmarkRow(m=M, n=N, k=K)
        label = f"{M}x{N}x{K}"

        # Our kernel
        print(f"[{label}] generating + benchmarking our kernel ... ",
              end="", flush=True)
        try:
            row.our_tflops = bench_our_kernel(
                hip, M, N, K,
                num_warmup=args.warmup, num_iters=args.iters)
            print(f"{row.our_tflops:.1f} TFLOPS")
        except Exception as exc:
            print(f"FAILED ({exc})")

        # hipBLASLt
        if hblt_bin:
            print(f"[{label}] running hipblaslt-bench ... ",
                  end="", flush=True)
            row.hblt_tflops = bench_hipblaslt(
                hblt_bin, M, N, K, num_iters=args.iters)
            if row.hblt_tflops is not None:
                print(f"{row.hblt_tflops:.1f} TFLOPS")
            else:
                print("FAILED")

        rows.append(row)

    # Summary table
    print()
    print_table(rows, hblt_available=hblt_bin is not None)
    print()


if __name__ == "__main__":
    main()
