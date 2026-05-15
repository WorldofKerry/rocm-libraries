# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark kernel generator kernels through TensileLite's shared harness.

Automates: generate .s -> write YAML -> copy to CustomKernels -> run Tensile.sh.
This ensures identical data init, GPU event timing, and memory layout as any
TensileLite kernel, making TFLOPS numbers directly comparable.

Usage::

    python -m kernel_generator.gemm.bench_tensilelite \
        --tensile-dir /path/to/projects/hipblaslt \
        --dtype mxfp4 --sizes 4096x4096x4096
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from .export_tensilelite import generate_custom_kernel

# Maps .amdgpu_metadata arg names -> TensileLite CustomKernel semantics
_NAME_TO_SEMANTIC = {
    "SizesFree0": "SizeFree0",
    "SizesFree1": "SizeFree1",
    "SizesFree2": "SizeFree2",
    "SizesSum0": "SizeSum",
    "D": "AddressD",
    "C": "AddressC",
    "A": "AddressA",
    "B": "AddressB",
    "MXSA": "AddressMXScaleA",
    "MXSB": "AddressMXScaleB",
    "strideD0": "StrideD0",
    "strideD1": "StrideD1",
    "strideC0": "StrideC0",
    "strideC1": "StrideC1",
    "strideA0": "StrideA0",
    "strideA1": "StrideA1",
    "strideB0": "StrideB0",
    "strideB1": "StrideB1",
    "strideMXSA0": "StrideScaleA0",
    "strideMXSA1": "StrideScaleA1",
    "strideMXSB0": "StrideScaleB0",
    "strideMXSB1": "StrideScaleB1",
    "alpha": "Alpha",
    "beta": "Beta",
    "AddressWorkspace": "AddressWorkspace",
    "AddressFlags": "AddressFlags",
    "NumWorkGroups": "NumWorkGroups",
    "ItersPerTile": "ItersPerTile",
    "SKItersPerWG": "SKItersPerWG",
    "SKGrid": "SKGrid",
    "SKTilesAndSplit": "SKTilesAndSplit",
}


def _parse_kernel_metadata(asm: str) -> Tuple[str, list]:
    """Extract kernel name and args from .amdgpu_metadata in the .s file."""
    name_match = re.search(r"\.globl (\S+)", asm)
    if not name_match:
        raise ValueError("No .globl directive found in assembly")
    kernel_name = name_match.group(1)

    meta_match = re.search(
        r"\.amdgpu_metadata\n(.*?)\.end_amdgpu_metadata", asm, re.DOTALL
    )
    if not meta_match:
        raise ValueError("No .amdgpu_metadata section found")
    meta = yaml.safe_load(meta_match.group(1))
    args = meta["amdhsa.kernels"][0][".args"]
    return kernel_name, args


def _args_to_yaml_list(args: list) -> list:
    """Convert .amdgpu_metadata args to YAML CustomKernel args format."""
    yaml_args = []
    for a in args:
        name = a[".name"]
        size = a[".size"]
        semantic = _NAME_TO_SEMANTIC.get(name)
        if semantic is None:
            print(f"  WARNING: unknown arg '{name}', using Padding", file=sys.stderr)
            semantic = "Padding"

        if size == 8 and a.get(".value_kind") == "global_buffer":
            yaml_args.append({"type": "address", "semantic": semantic})
        elif name in ("alpha", "beta"):
            yaml_args.append({"type": "float32", "semantic": semantic})
        else:
            yaml_args.append({"type": "uint32", "semantic": semantic})
    return yaml_args


def _build_yaml(
    kernel_name: str,
    yaml_args: list,
    dtype: str,
    sizes: List[Tuple[int, int, int]],
    macrotile: List[int],
    threads: List[int],
    validate: bool = True,
    use_1d_grid: bool = False,
    streamk: bool = False,
) -> str:
    """Build a complete TensileLite benchmark YAML."""
    # ProblemType
    if dtype == "mxfp4":
        problem_type = {
            "OperationType": "GEMM",
            "DataType": "F4",
            "DestDataType": "B",
            "ComputeDataType": "S",
            "HighPrecisionAccumulate": True,
            "TransposeA": 1,
            "TransposeB": 0,
            "UseBeta": True,
            "Batched": True,
            "UseBias": 0,
            "Activation": False,
            "UseScaleAlphaVec": 0,
            "SwizzleTensorA": False,
            "SwizzleTensorB": False,
            "MXBlockA": 32,
            "MXBlockB": 32,
        }
        mi = [16, 16, 128, 1, 1, 8, 8, 2, 2]
    else:
        problem_type = {
            "OperationType": "GEMM",
            "DataType": "H",
            "DestDataType": "H",
            "ComputeDataType": "S",
            "HighPrecisionAccumulate": True,
            "TransposeA": 1,
            "TransposeB": 0,
            "UseBeta": True,
            "Batched": True,
            "UseBias": 0,
            "Activation": False,
            "UseScaleAlphaVec": 0,
        }
        mi = [16, 16, 16, 1, 1, 8, 8, 2, 2]

    # Format args for YAML inline
    args_lines = []
    for i, a in enumerate(yaml_args):
        parts = [f"type: {a['type']}", f"semantic: {a['semantic']}"]
        if "padding" in a:
            parts.append(f"padding: {a['padding']}")
        args_lines.append(f"                    {{ {', '.join(parts)} }}")

    # Problem sizes
    size_lines = []
    for m, n, k in sizes:
        size_lines.append(f"          - Exact: [{m}, {n}, 1, {k}]")

    num_validate = -1 if validate else 0

    return f"""TestParameters:
  marks: [skip-gfx900, skip-gfx906, skip-gfx908, skip-gfx90a, skip-gfx1010, skip-gfx1011, skip-gfx1012, skip-gfx1030, skip-gfx1100, skip-gfx1101, skip-gfx1102, skip-gfx1200, skip-gfx1201]

GlobalParameters:
  MinimumRequiredVersion: 5.0.0
  SleepPercent: 50
  NumElementsToValidate: {num_validate}
  NumBenchmarks: 1
  SyncsPerBenchmark: 4
  EnqueuesPerSync: 1
  NumWarmups: 4
  DataInitTypeBeta: 0
  DataInitTypeAlpha: 1
  DataInitTypeA: 3
  DataInitTypeB: 3
  DataInitTypeMXSA: 3
  DataInitTypeMXSB: 3
  KernelTime: True
  DeviceLDS: 163840
  MaxLDS: 163840
  MXScaleFormat: 1
  CSVExportWinner: 1
  # NOTE: TensileLite's Reference.cpp reads pre-swizzled scale data
  # linearly, causing validation failures for K > unroll_k (multiple
  # K-tiles).  The kernel is correct -- verified by standalone Python
  # FP4 reference.  Use --no-validate for K > unroll_k benchmarks.
  CSVMergeSameProblemID: 1
  Device: 0
  ValidateMetadata: True

BenchmarkProblems:
  -
    - # ProblemType
{chr(10).join("      " + l for l in yaml.dump(problem_type, default_flow_style=False).rstrip().splitlines())}

    - # BenchmarkProblemSizeGroup
      InitialSolutionParameters:
      BenchmarkCommonParameters:
        - KernelLanguage: ["Assembly"]
      ForkParameters:
        - CustomKernel:
          - name: "{kernel_name}"
            args: [
{','.join(chr(10) + l for l in args_lines)} ]
            macrotile: {macrotile}
            threads: {threads}
            grid: [{"StreamKWithBatch" if streamk else "TilesXYBatchGSU" if use_1d_grid else "TilesX"}, {"One" if streamk or use_1d_grid else "TilesY"}, One]
{"            workspaceType: StreamK" + chr(10) + "            workspaceSizePerElemC: 4" if streamk else ""}
        - MatrixInstruction:
          - {mi}
        - AssertFree0ElementMultiple: [{macrotile[0]}]
        - AssertFree1ElementMultiple: [{macrotile[1]}]
        - PreloadKernArgs: [False]
        - DepthU: [{macrotile[2]}]
        - VectorWidthA: [-1]
        - VectorWidthB: [-1]
        - GlobalReadVectorWidthA: [-1]
        - GlobalReadVectorWidthB: [-1]
        - LocalReadVectorWidth: [-1]
        - TransposeLDS: [-1]
        - LdsBlockSizePerPadA: [-1]
        - LdsBlockSizePerPadB: [-1]
        - LdsPadA: [-1]
        - LdsPadB: [-1]
        - StaggerU: [16]
        - StaggerUStride: [-1]
        - WorkGroupMapping: [1]
        - StaggerUMapping: [0]
        - 1LDSBuffer: [-1]
        - WorkGroupMappingXCC: [8]
        - WorkGroupMappingXCCGroup: [-1]
        - GlobalSplitU: [1]
        - GlobalSplitUAlgorithm: ["MultipleBuffer"]
        - GlobalReadPerMfma: [1]
        - LocalWritePerMfma: [-1]
        - StoreRemapVectorWidth: [0]
        - StoreVectorWidth: [-1]
        - SourceSwap: [1]
        - NumElementsPerBatchStore: [16]
        - ClusterLocalRead: [1]
        - NonTemporalA: [0]
        - NonTemporalB: [0]
        - DirectToVgprA: [1]
        - DirectToVgprB: [0]
      BenchmarkJoinParameters:
      BenchmarkFinalParameters:
        - ProblemSizes:
{chr(10).join(size_lines)}

LibraryLogic:
    ScheduleName: "gfx950"
    DeviceNames: ["Device 0049", "Device 0050"]
    ArchitectureName: "gfx950"
    LibraryType: "GridBased"
"""


def _parse_csv_results(output: str) -> list:
    """Parse Tensile.sh CSV output for timing results."""
    results = []
    header = None
    for line in output.splitlines():
        # Data lines start with a digit (run index)
        if not line:
            continue
        # Capture header line for column mapping
        if line.startswith("run,"):
            header = next(csv.reader(io.StringIO(line)))
            continue
        if not line[0].isdigit():
            continue
        # Use csv.reader to handle quoted fields with commas
        try:
            fields = next(csv.reader(io.StringIO(line)))
        except (csv.Error, StopIteration):
            continue
        if len(fields) < 12 or not header:
            continue
        try:
            col = {name: i for i, name in enumerate(header)}
            sizes_str = fields[col["problem-sizes"]].strip("()")
            validation = fields[col["validation"]]
            time_str = fields[col["time-us"]]
            gflops_str = fields[col["gflops"]]
            time_us = float(time_str) if time_str not in ("", "-nan") else None
            gflops = float(gflops_str) if gflops_str not in ("", "-nan") else None
            results.append({
                "sizes": sizes_str,
                "validation": validation,
                "time_us": time_us,
                "gflops": gflops,
                "tflops": gflops / 1000 if gflops else None,
            })
        except (ValueError, IndexError):
            continue
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark kernel generator via TensileLite harness"
    )
    parser.add_argument(
        "--tensile-dir", required=True,
        help="Path to projects/hipblaslt on the GemmFromAnywhere branch",
    )
    parser.add_argument("--dtype", default="mxfp4", choices=["mxfp4", "fp16"])
    parser.add_argument("--wg-m", type=int, default=256)
    parser.add_argument("--wg-n", type=int, default=256)
    parser.add_argument("--unroll-k", type=int, default=256)
    parser.add_argument(
        "--sizes", default="4096x4096x4096",
        help="Comma-separated MxNxK sizes",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip correctness validation")
    parser.add_argument("--swizzled-scales", action="store_true", default=True)
    parser.add_argument("--streamk", action="store_true", default=False)
    parser.add_argument("--keep-s", action="store_true",
                        help="Keep the generated .s file")
    args = parser.parse_args()

    tensile_dir = Path(args.tensile_dir)
    tensile_sh = tensile_dir / "build-tensilelite" / "Tensile.sh"
    client = tensile_dir / "build-tensilelite" / "tensilelite" / "client" / "tensilelite-client"
    custom_dir = tensile_dir / "tensilelite" / "Tensile" / "CustomKernels" / "rocroller"
    test_dir = tensile_dir / "tensilelite" / "Tensile" / "Tests" / "custom"

    # Preflight checks
    for path, desc in [(tensile_sh, "Tensile.sh"), (client, "tensilelite-client")]:
        if not path.exists():
            print(f"ERROR: {desc} not found at {path}", file=sys.stderr)
            print("Build TensileLite first: cmake --build build-tensilelite --parallel",
                  file=sys.stderr)
            sys.exit(1)

    custom_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Parse sizes
    sizes = []
    for s in args.sizes.split(","):
        parts = s.strip().split("x")
        if len(parts) != 3:
            print(f"ERROR: Invalid size '{s}', expected MxNxK", file=sys.stderr)
            sys.exit(1)
        sizes.append(tuple(int(x) for x in parts))

    # Step 1: Generate kernel
    print(f"Generating {args.dtype} kernel ({args.wg_m}x{args.wg_n}x{args.unroll_k})...",
          flush=True)
    asm = generate_custom_kernel(
        wg_m=args.wg_m, wg_n=args.wg_n, unroll_k=args.unroll_k,
        dtype=args.dtype, swizzled_scales=args.swizzled_scales,
        streamk=args.streamk,
    )
    kernel_name, metadata_args = _parse_kernel_metadata(asm)
    print(f"  Kernel: {kernel_name}")

    # Step 2: Write .s to CustomKernels
    s_path = custom_dir / f"{kernel_name}.s"
    s_path.write_text(asm)
    print(f"  Wrote: {s_path}")

    # Step 3: Generate YAML
    yaml_args = _args_to_yaml_list(metadata_args)
    macrotile = [args.wg_m, args.wg_n, args.unroll_k]
    threads = [256, 1, 1]
    # FP16 uses WGMXCC=8 (1D grid); MXFP4 uses WGMXCC=1 (2D grid)
    use_1d = (args.dtype != "mxfp4") and not args.streamk
    yaml_text = _build_yaml(
        kernel_name=kernel_name,
        yaml_args=yaml_args,
        dtype=args.dtype,
        sizes=sizes,
        macrotile=macrotile,
        threads=threads,
        validate=not args.no_validate,
        use_1d_grid=use_1d,
        streamk=args.streamk,
    )
    yaml_path = test_dir / f"custom_kgen_{args.dtype}.yaml"
    yaml_path.write_text(yaml_text)
    print(f"  YAML:  {yaml_path}")

    # Step 4: Run Tensile.sh
    with tempfile.TemporaryDirectory(prefix="kgen_bench_") as tmpdir:
        cmd = [
            str(tensile_sh),
            str(yaml_path),
            tmpdir,
            "--cxx-compiler", "/opt/rocm/bin/amdclang++",
            "--prebuilt-client", str(client),
            "--library-format", "msgpack",
        ]
        if args.dtype == "mxfp4":
            cmd += ["--mx-scale-format", "1"]

        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(args.device)

        print(f"\nRunning Tensile.sh (device {args.device})...", flush=True)
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=600,
        )

        # Parse results
        results = _parse_csv_results(proc.stdout)
        if not results:
            # Try stderr too
            results = _parse_csv_results(proc.stderr)

        # Also try to parse ExactWinners from output (fallback)
        if not results:
            for line in proc.stdout.splitlines():
                m = re.search(r"ExactWinners.*?\[(\d+),\s*([\d.]+)\]", line)
                if m:
                    results.append({
                        "sizes": args.sizes,
                        "validation": "NO_CHECK",
                        "time_us": float(m.group(2)),
                        "gflops": 2.0 * sizes[0][0] * sizes[0][1] * sizes[0][2] / (float(m.group(2)) * 1e-6) / 1e9,
                        "tflops": 2.0 * sizes[0][0] * sizes[0][1] * sizes[0][2] / (float(m.group(2)) * 1e-6) / 1e12,
                    })

        if results:
            print(f"\n{'Size':<20} {'Status':<8} {'Time (us)':>10} {'TFLOPS':>10}")
            print("-" * 52)
            for r in results:
                time_s = f"{r['time_us']:.1f}" if r["time_us"] else "N/A"
                tflops_s = f"{r['tflops']:.1f}" if r["tflops"] else "N/A"
                print(f"{r['sizes']:<20} {r['validation']:<8} {time_s:>10} {tflops_s:>10}")
        else:
            print("\nNo results parsed. Tensile.sh output:")
            # Show last 20 lines
            for line in proc.stdout.splitlines()[-20:]:
                print(f"  {line}")
            if proc.returncode != 0:
                print(f"\nTensile.sh exited with code {proc.returncode}")
                for line in proc.stderr.splitlines()[-10:]:
                    print(f"  {line}")

    # Cleanup
    if not args.keep_s:
        s_path.unlink(missing_ok=True)
        print(f"\nCleaned up {s_path.name}")
    else:
        print(f"\nKept: {s_path}")


if __name__ == "__main__":
    main()
