# stinkytofu.gemm -- GEMM Kernel Generator

A Python GEMM kernel generator that describes tiling via composable
coordinate transforms and emits optimised GPU assembly through
StinkyTofu's LogicalModule IR.

## Quick Start

No build required for analysis and dry-run mode:

```bash
cd shared/stinkytofu
PYTHONPATH=python_module:$PYTHONPATH python3 -W ignore::ImportWarning -c "
from stinkytofu.gemm import generate_gemm_kernel, GemmProblem, TileConfig, MfmaConfig

problem = GemmProblem(m=4096, n=4096, k=4096)
tile    = TileConfig(wg_m=128, wg_n=128, unroll_k=32, mfma=MfmaConfig.f16_16x16x16())
result  = generate_gemm_kernel(problem, tile, dry_run=True)
print(result.summary())
"
```

## Running Tests

```bash
cd shared/stinkytofu
PYTHONPATH=python_module:$PYTHONPATH python3 -W ignore::ImportWarning \
  -m pytest python_module/tests/gemm/ -v
```

86 tests run in pure Python.  6 additional tests require the compiled
stinkytofu C extension and are skipped automatically when it is absent.

## Running the Example

```bash
cd shared/stinkytofu
PYTHONPATH=python_module:$PYTHONPATH python3 -W ignore::ImportWarning \
  python_module/stinkytofu/gemm/examples/basic_gemm.py
```

This prints:
- Dry-run kernel summary with register allocation and tile descriptors
- FLOP count and arithmetic intensity analysis
- Tile configuration comparison across MFMA variants
- Coordinate transform walkthrough

## Full Generation (requires stinkytofu build)

Build stinkytofu with Python bindings, then generate assembly:

```bash
cd shared/stinkytofu
cmake -S . -B build -GNinja \
  -DCMAKE_CXX_COMPILER=amdclang++ \
  -DCMAKE_C_COMPILER=amdclang \
  -DSTINKYTOFU_BUILD_PYTHON=ON
cmake --build build --target stinkytofu_python -j12

PYTHONPATH=python_module:build/lib:$PYTHONPATH python3 -c "
from stinkytofu.gemm import generate_gemm_kernel, GemmProblem

result = generate_gemm_kernel(GemmProblem(m=4096, n=4096, k=4096))
print(result.module.dump())
"
```

To run all tests including the C-extension-dependent ones:

```bash
PYTHONPATH=python_module:build/lib:$PYTHONPATH python3 \
  -m pytest python_module/tests/gemm/ -v
```

## Architecture

Five layers, each independently overridable via subclassing:

```
transforms.py   Dim, Tile, Flatten, Pad, Embed, Xor, TileDescriptor
     |
problem.py      GemmProblem, TileConfig, MfmaConfig
     |
codegen.py      RegisterAllocator -> ThreadMapping -> Emitter -> GemmSchedule -> GemmCodegen
     |
kernel.py       generate_gemm_kernel() -> KernelResult
```

### Overriding a Layer

**Custom MFMA emission** (hand-tuned instruction sequence):

```python
from stinkytofu.gemm import Emitter, generate_gemm_kernel, GemmProblem

class MyEmitter(Emitter):
    def emit_mfma_block(self, module, k_iter):
        # your hand-optimised MFMA + LDS-read interleaving
        ...

result = generate_gemm_kernel(GemmProblem(4096, 4096, 4096), emitter_cls=MyEmitter)
```

**Custom K-loop** (software pipelining, split-K):

```python
from stinkytofu.gemm import GemmSchedule, generate_gemm_kernel, GemmProblem

class PipelinedSchedule(GemmSchedule):
    def emit_k_loop(self, module):
        # double-buffered software pipeline
        ...

result = generate_gemm_kernel(GemmProblem(4096, 4096, 4096), schedule_cls=PipelinedSchedule)
```

**Inject raw assembly** at a labelled point:

```python
from stinkytofu.gemm import GemmCodegen, GemmProblem, TileConfig

cg = GemmCodegen(GemmProblem(4096, 4096, 4096), TileConfig())
cg.inject_before("k_loop", my_prefetch_instructions)
module = cg.generate()
```

## Tile Hierarchy

```
Problem (M x N x K)
  Workgroup tile (wg_m x wg_n x unroll_k)      -- one per workgroup
    Wave tile (m_per_wave x n_per_wave)          -- one per wave
      MFMA repeat (mfma_m_repeat x mfma_n_repeat) -- multiple MFMAs per wave
        MFMA tile (mfma.m x mfma.n x mfma.k)    -- single hardware instruction
```

All levels are expressed as coordinate transforms and can be inspected:

```python
from stinkytofu.gemm import TileConfig, MfmaConfig

tile = TileConfig(wg_m=128, wg_n=128, unroll_k=32, mfma=MfmaConfig.f16_16x16x16())
print(tile.build_m_descriptor())
# TileDescriptor(M_tile: [M_wave_id:2, M_mfma_id:4, M_mfma:16])
```
