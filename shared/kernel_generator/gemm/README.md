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
- StreamK work decomposition (K-splitting with GPU-side reduction)

## Setup

Create a virtual environment and install dependencies (requires [uv](https://docs.astral.sh/uv/)):

```bash
cd kernel-generator
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e shared/kernel_generator[dev]
```

This installs `numpy` and `pytest` into an isolated venv. After
activation, `PYTHONPATH` is no longer needed -- the editable install
puts `kernel_generator` on the path automatically.

## Quick Start

```bash
cd kernel-generator
python3 -c "
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

## StreamK

Toggle StreamK with `streamk=True` on `GemmKernel.build`. The kernel
swaps its epilogue to a 3-way conditional store that supports both
data-parallel (DP) and K-split modes from the same code object.

```python
from kernel_generator.gemm.problem import GemmProblem, DataType
from kernel_generator.gemm.kernel import GemmKernel
from kernel_generator.gemm.launcher import GemmLauncher

p = GemmProblem(4096, 4096, 4096, dtype=DataType.F16)
k = GemmKernel.build(p, streamk=True)
co = k.emit().assemble()

launcher = GemmLauncher(p, k.tile, seed=42)
# DP mode (num_partitions=1): each WG computes full K
r = launcher.run_streamk(co, num_partitions=1)
# K-split mode: each tile's K is split across partitions
r = launcher.run_streamk(co, num_partitions=2)
```

StreamK epilogue paths:
1. **Sole owner** (is_partial=0): direct store to D
2. **Non-owner partial** (is_partial=1, iter_start>0): store f32 to workspace, set flag
3. **Owner partial** (is_partial=1, iter_start=0): poll flags, load partials, accumulate, store to D

Multi-partition launches use separate HIP streams for concurrent execution.

## Tests

```bash
pytest shared/kernel_generator/tests/gemm/ -q
```

## Benchmarking via TensileLite (Recommended)

For fair performance comparison against TensileLite/hipBLASLt kernels,
use the TensileLite client as a shared harness. This ensures identical
data initialization, GPU event timing, and memory layout -- our own
`GemmLauncher` uses different timing (host-side `perf_counter`) and
data init, so numbers are not directly comparable.

### 1. Export a custom kernel

```python
from kernel_generator.gemm.export_tensilelite import generate_custom_kernel

# MXFP4
s = generate_custom_kernel(256, 256, 256, dtype='mxfp4')
open('my_kernel.s', 'w').write(s)

# FP16
s = generate_custom_kernel(256, 256, 64, dtype='fp16')
open('my_kernel_fp16.s', 'w').write(s)
```

The `.s` file includes embedded `custom.config` metadata (ProblemType,
MatrixInstruction, kernarg layout) so TensileLite can load it directly.

### 2. Place the kernel

Copy the `.s` file into the GemmFromAnywhere branch's CustomKernels
directory:

```bash
cp my_kernel.s <rocm-libraries>/projects/hipblaslt/tensilelite/Tensile/CustomKernels/rocroller/
```

### 3. Build TensileLite client (one-time)

```bash
cd <rocm-libraries>/projects/hipblaslt
cmake -B build-tensilelite -S . \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS=gfx950 \
  -DHIPBLASLT_ENABLE_FETCH=ON \
  -DHIPBLASLT_BUILD_TESTING=OFF \
  -DHIPBLASLT_ENABLE_CLIENT=OFF \
  -DHIPBLASLT_ENABLE_DEVICE=OFF \
  -DHIPBLASLT_ENABLE_HOST=OFF \
  -DHIPBLASLT_ENABLE_ROCROLLER=OFF \
  -DTENSILELITE_ENABLE_CLIENT=ON \
  -DTENSILELITE_BUILD_TESTING=OFF \
  -DTENSILELITE_ENABLE_AUTOBUILD=ON
cmake --build build-tensilelite --parallel
```

### 4. Write a benchmark YAML

Create a test YAML under `Tensile/Tests/custom/` that references the
kernel by name and declares its kernarg layout. The `args` list must
match the `.args` section in the kernel's `.amdgpu_metadata`. See
existing examples in `Tensile/Tests/custom/custom_aiter_f4.yaml`.

### 5. Run

```bash
HIP_VISIBLE_DEVICES=0 ./build-tensilelite/Tensile.sh \
  tensilelite/Tensile/Tests/custom/my_test.yaml \
  /tmp/bench_output \
  --cxx-compiler /opt/rocm/bin/amdclang++ \
  --prebuilt-client ./build-tensilelite/tensilelite/client/tensilelite-client \
  --library-format msgpack \
  --mx-scale-format 1
```

Output is a CSV line with time (us) and GFLOPS -- directly comparable
to any other TensileLite kernel run through the same harness.

## Architecture

See [DESIGN.md](DESIGN.md) for the full architecture documentation.

Key concept: building blocks (`MFMABlock`, `DSReadBlock`,
`GlobalLoadBlock`, `SuffixBlock`) declare operations and dependencies
into a `KLoopGraph`. The `KLoopScheduler` derives instruction ordering
from the dependency DAG -- no magic constants or manual slot placement.
