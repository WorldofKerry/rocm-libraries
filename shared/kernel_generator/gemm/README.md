# GEMM Kernel Generator

Python-based GEMM kernel generator for AMD MI355X (gfx950). Emits
optimized GCN assembly with dependency-driven instruction scheduling.

## Features

- FP16, BF16, MXFP4 data types
- Direct-To-LDS (DTL) global loads
- Double-buffered LDS with swizzle-aware reads
- Ping-pong A buffers with automatic partition scheduling
- MX scale loading (linear and pre-swizzled)
- TensileLite custom kernel export

## Quick Start

```bash
cd kernel-generator
PYTHONPATH=shared python3 -c "
from kernel_generator.gemm.problem import GemmProblem
from kernel_generator.gemm.kernel import GemmKernel

k = GemmKernel.build(GemmProblem(4096, 4096, 4096))
result = k.emit()
co = result.assemble()
print(f'Code object: {co}')
"
```

## GPU Launch

```python
from kernel_generator.gemm.problem import GemmProblem
from kernel_generator.gemm.kernel import GemmKernel
from kernel_generator.gemm.launcher import GemmLauncher

p = GemmProblem(4096, 4096, 4096)
k = GemmKernel.build(p)
co = k.emit().assemble()

launcher = GemmLauncher(p, k.tile, seed=42)
r = launcher.run_asm_kernel(co, kernel_name='gemm_kernel',
                            num_warmup=100, num_iters=500)
tflops = 2 * 4096**3 / (r.time_seconds * 1e12)
print(f'{tflops:.0f} TFLOPS, {r.time_seconds*1e6:.1f} us')
```

## MXFP4

```python
from kernel_generator.gemm.problem import GemmProblem, DataType, MfmaConfig
from kernel_generator.gemm.tiling import GemmTiling
from kernel_generator.gemm.kernel import GemmKernel

mx = MfmaConfig.mxfp4_16x16x128()
t = GemmTiling.high_perf(wg_m=256, wg_n=256, unroll_k=256, mfma=mx,
                          lds_swizzle=True)
p = GemmProblem(4096, 4096, 4096, dtype=DataType.MXFP4)
k = GemmKernel.build(p, tiling=t)
co = k.emit().assemble()
```

## Tests

```bash
PYTHONPATH=shared pytest shared/kernel_generator/tests/gemm/ -q
```

## Architecture

See [DESIGN.md](DESIGN.md) for the full architecture documentation.

Key concept: building blocks (`MFMABlock`, `DSReadBlock`,
`GlobalLoadBlock`, `SuffixBlock`) declare operations and dependencies
into a `KLoopGraph`. The `KLoopScheduler` derives instruction ordering
from the dependency DAG -- no magic constants or manual slot placement.
