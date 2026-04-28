# stinkytofu.gemm -- GEMM Kernel Generator

Python-based GEMM kernel generator that emits GPU assembly (`.s` files) for
AMD gfx950 (MI355X).  Each kernel is a composable pipeline of replaceable
phases -- swap one function to test a new optimization without touching the
rest.

## Quick Start

No build step required.  Pure Python, no C extension dependency.

```bash
cd shared/stinkytofu

# Generate assembly and print the first 500 chars
PYTHONPATH=python_module:$PYTHONPATH python3 -c "
from stinkytofu.gemm import GemmKernel, GemmProblem

kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
result = kernel.emit()
print(result.asm_text[:500])
"

# Assemble into a code object (requires amdclang in PATH)
PYTHONPATH=python_module:$PYTHONPATH python3 -c "
from stinkytofu.gemm import GemmKernel, GemmProblem

kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
result = kernel.emit()
co = result.assemble()           # produces .o code object
print(f'Code object: {co}')
"

# Launch on GPU (requires HIP runtime + MI355X)
PYTHONPATH=python_module:$PYTHONPATH python3 -c "
from stinkytofu.gemm import GemmKernel, GemmProblem
from stinkytofu.gemm.launcher import GemmLauncher

problem = GemmProblem(4096, 4096, 4096)
kernel  = GemmKernel.build(problem)
result  = kernel.emit()
co      = result.assemble()

launcher = GemmLauncher(problem, kernel.tile)
gpu_result = launcher.run(co)
print(f'Correct: {gpu_result.correct}  Time: {gpu_result.time_seconds:.4f}s')
"
```

## Architecture

Five layers, bottom-up.  Each layer depends only on the ones below it.

```
Layer 1: transforms.py    Composable coordinate transforms
Layer 2: problem.py       GEMM problem + tile config
Layer 3: tile.py           Recursive tile tree
Layer 4: kernel_pipeline.py   GemmKernel pipeline + MemoryView
Layer 5: asm_emitter.py   Assembly backend (emits .s text)
         asm_context.py   Register allocation + instruction emit
         asm_transforms.py   emit_affine + GemmLayouts
```

### Layer 1: Coordinate Transforms

Seven composable transforms describe how indices map across tile levels:
`Dim`, `Tile`, `Flatten`, `Pad`, `Embed`, `Xor`, `TileDescriptor`.
All tiling decisions (workgroup mapping, LDS layout, register layout)
are compositions of these transforms.

### Layer 2: Problem + Tile Config

`GemmProblem` captures sizes, data types, and transposes.
`TileConfig` captures tiling and mapping decisions.
`MfmaConfig` describes the hardware MFMA instruction.

### Layer 3: Tile Tree

Recursive `TileLevel` hierarchy that mirrors the hardware mapping:

```
grid
  workgroup  (m=128, n=128, k=32)
    wave       (m=64,  n=64)
      subtile    (m=16,  n=16)
        mfma       (m=16,  n=16,  k=16)    -- leaf
```

### Layer 4: Kernel Pipeline

`GemmKernel` is a pipeline of callable phases:

```
prologue -> k_loop [ global_load -> lds_write -> compute -> k_advance ] -> epilogue
```

`MemoryView` provides tensor access at any tile level via coordinate
transforms -- the same interface works whether data is in global memory,
LDS, or accumulators.

### Layer 5: Assembly Emission

`AsmContext` manages named register bindings (never raw register numbers),
emits instructions, and tracks resource usage.  `emit_affine` turns
coordinate transforms into scalar/vector address arithmetic.
`asm_emitter` orchestrates the full `.s` file generation.

## Customizing Kernels

The design principle: a middle ground between Triton/CK (can't touch
assembly) and TensileLite (hard to modify one piece without understanding
everything).  Each pipeline phase is independently replaceable.

### Replace a K-loop Sub-phase

```python
from stinkytofu.gemm import GemmKernel, GemmProblem

kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))

# Replace just the global load phase
def my_prefetching_load(ctx, kernel):
    ctx.comment("Custom prefetching global load")
    # ... emit your own global_load_dwordx4 sequence ...

kernel.k_loop.global_load = my_prefetching_load
result = kernel.emit()
```

### Replace the Prologue or Epilogue

```python
def my_epilogue(ctx, kernel):
    ctx.comment("Custom epilogue with activation fusion")
    # ... custom store + activation logic ...

kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
kernel.epilogue = my_epilogue
result = kernel.emit()
```

### Replace the MFMA Leaf

```python
kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
kernel.tile_tree = kernel.tile_tree.replace("mfma", emit=my_mfma)
result = kernel.emit()
```

### Access Tensor Data at Any Level

Inside any phase, use `MemoryView` to read/write tensors through
coordinate transforms:

```python
def my_custom_compute(ctx, kernel):
    a_view = ctx.get_view("A")     # LDS view at this level
    b_view = ctx.get_view("B")
    # a_view.emit_offset(ctx, bindings, result_reg)
```

## File Inventory

| File | Lines | Purpose |
|------|------:|---------|
| `transforms.py` | 484 | Composable coordinate transforms (Dim, Tile, Flatten, Pad, Embed, Xor) |
| `problem.py` | 472 | GemmProblem, TileConfig, MfmaConfig, SubTileConfig |
| `tile.py` | 587 | Recursive TileLevel tree, walk_tile_tree |
| `context.py` | 347 | TileContext with scoped register allocation |
| `kernel_pipeline.py` | 399 | GemmKernel pipeline, KLoop, MemoryView |
| `asm_emitter.py` | 860 | Full assembly backend (.s generation) |
| `asm_context.py` | 193 | AsmContext: instruction emit + register tracking |
| `asm_transforms.py` | 219 | emit_affine, GemmLayouts |
| `addressing.py` | 534 | Pure-Python offset calculators |
| `launcher.py` | 457 | HIP GPU launcher + correctness verification |

## Running Tests

151 tests, pure Python.  No GPU or C extension required.

```bash
cd shared/stinkytofu
PYTHONPATH=python_module:$PYTHONPATH python3 -m pytest python_module/tests/gemm/ -v
```

Test files cover each layer independently:

| Test file | Count | Coverage |
|-----------|------:|----------|
| `test_transforms.py` | 35 | Coordinate transforms |
| `test_addressing.py` | 26 | Offset calculators |
| `test_context.py` | 22 | Scoped register allocation |
| `test_problem.py` | 21 | Problem + tile config validation |
| `test_subtile.py` | 20 | Sub-tile partitioning |
| `test_pipeline.py` | 16 | Kernel pipeline + phase replacement |
| `test_launcher.py` | 11 | Launcher (CPU reference, mock GPU) |

## Performance

On MI355X (gfx950), 8192x8192x8192 fp16:

- **75 TFLOPS** (~7.5% of theoretical peak)
- This is a **naive kernel**: no software pipelining, no instruction scheduling
- Baseline for measuring the impact of individual optimizations

## Design Constraints

- **No stinkytofu/rocisa dependency** -- the assembly layer is self-contained.
  Assembly is emitted as plain text, assembled by `amdclang`.
- **Pure Python** -- no build step, no C extension for generation.
  The C extension is only needed if using the stinkytofu LogicalModule IR path.
- **gfx950 target** -- MFMA instructions and addressing assume MI355X.
  Supporting other targets requires updating `MfmaConfig` and the assembly
  templates in `asm_emitter.py`.
