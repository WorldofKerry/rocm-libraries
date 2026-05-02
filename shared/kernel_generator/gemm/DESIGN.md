# Kernel Generator -- Architecture

## Overview

Python-based GEMM kernel generator targeting MI355X (gfx950). Produces
GCN assembly (`.s`) from a dependency graph of K-loop operations. No IR --
assembly is emitted as text by callback functions, then assembled by
`amdclang`.

Supported data types: FP16, BF16, MXFP4 (with real scale loading).

## Pipeline

```
GemmKernel.build(problem, tiling)
    |
GemmTiling.build_tile_tree()     -- tile hierarchy with phase callbacks
    |
KLoopGraph                       -- ops + dependencies (DAG)
    |
KLoopScheduler.schedule()        -- MFMA backbone + side-op placement
    |
scheduled_kloop_phase()          -- emit assembly from schedule
    |
AsmContext -> .s -> amdclang -> .co
```

## Dependency-Driven Scheduling

Building blocks declare operations and their dependencies into a
`KLoopGraph`. The `KLoopScheduler` produces an instruction sequence.

**Building blocks:**
- `MFMABlock`: mr * nr * ki MFMAs with RAW deps on A/B operands
- `DSReadBlock`: ds_reads with SYNC on barrier + WAR for A ping-pong
- `GlobalLoadBlock`: next-iter DTL loads (iteration=1, conditional skip)
- `SuffixBlock`: vmcnt wait, toggle, negate (placed backward)

**Key insight:** WAR deps on A ping-pong buffers automatically create
the partition structure. `read_a(mi=2, buf=0)` can't issue until
`mfma(mi=0, ..., last)` finishes, so the scheduler groups MFMAs by mi
with A-prefetch interleaving -- no magic constants.

**Scheduling phases:**
1. MFMA backbone ordered by (mi, ki, ni)
2. Side ops classified (pre-body B reads, preamble, prefetch, body reads)
3. Forward placement of A-prefetch reads respecting WAR deps
4. Backward placement of suffix ops (maximize overlap)
5. Auto-wait insertion from dependency distances

See `schedule/DESIGN_SCHEDULER.md` for the full design.

## K-loop Structure

```
Prologue:
  DTL load tile 0 -> vmcnt(0) -> barrier

k_loop:
  B[ki=0] reads (overlap with arriving loads)
  conditional: advance + toggle + DTL load next tile
  barrier
  preamble: A[m0,k0] + B[ki=1] + A[m0,k1]
  lgkmcnt(N)

  [scheduled MFMA body with interleaved ds_reads + suffix]
    for each MFMA:
      side ops (A-prefetch reads, suffix vmcnt/toggle/negate)
      MFMA instruction
      auto-wait at mi boundaries

  barrier
  branch k_loop
```

## Memory Architecture

**Global -> LDS:** Direct-To-LDS (`buffer_load ... ,lds`) bypasses
VGPRs. Double-buffered LDS with XOR toggle.

**LDS -> VGPRs:** Swizzle-aware `ds_read_b128` with per-ki XOR offsets
to avoid bank conflicts. A operands use ping-pong buffers (2 slots).
B operands are single-buffered (loaded in preamble).

**Scales (MXFP4):** Loaded directly from global memory into VGPRs via
`buffer_load_dword`. Subtile-level prefetch hides latency. Supports
both linear and pre-swizzled (AITER) addressing.

## File Map

```
gemm/
  kernel.py              GemmKernel.build() + emit() entry point
  problem.py             GemmProblem, TileConfig, MfmaConfig, DataType
  tiling.py              GemmTiling, TileDim chains, build_tile_tree()
  launcher.py            HIP ctypes launcher + correctness + timing
  benchmark.py           Benchmark harness (ours vs hipblaslt-bench)
  export_tensilelite.py  Export as TensileLite custom kernel

  emit/
    context.py           AsmContext -- register naming + instruction emit
    emitter.py           assemble_kernel() (amdclang), emit_header()
    layouts.py           emit_affine(), GemmLayouts (coordinate transforms)
    phases.py            Prologue phases + store epilogue

  tile/
    tree.py              TileLevel, TilePhase, walk_tile_tree()
    transforms.py        Dim, Tile, Flatten, Pad, Embed, Xor
    context.py           TileContext -- scoped register allocator

  kloop/
    setup.py             DTL/wave-ABI setup phases + phase_mx_scale_setup

  schedule/
    kloop_graph.py       KLoopGraph, KLoopOp, Dep, building blocks
    kloop_scheduler.py   KLoopScheduler + scheduled_kloop_phase (emitter)
    slot_placer.py       SlotPlacer -- instruction interleaving engine

  memory/
    global_loader.py     DTLLoader, BufferLoader (global -> LDS)
    lds_reader.py        LDSReader (LDS -> VGPRs, swizzle-aware)
    scale_loader.py      VMEMScaleLoader, NullScaleLoader (MX scales)
    swizzle.py           RotationSwizzle, XorSwizzle, IdentitySwizzle
```

## Usage

```python
from kernel_generator.gemm.problem import GemmProblem, DataType, MfmaConfig
from kernel_generator.gemm.tiling import GemmTiling
from kernel_generator.gemm.kernel import GemmKernel

# FP16
p = GemmProblem(4096, 4096, 4096)
k = GemmKernel.build(p, scheduled=True)
asm = k.emit()
co = asm.assemble()

# MXFP4
mx = MfmaConfig.mxfp4_16x16x128()
t = GemmTiling.high_perf(wg_m=256, wg_n=256, unroll_k=256, mfma=mx)
p = GemmProblem(4096, 4096, 4096, dtype=DataType.MXFP4)
k = GemmKernel.build(p, tiling=t, scheduled=True)
asm = k.emit()
co = asm.assemble()
```

## Tests

```bash
cd kernel-generator
PYTHONPATH=shared pytest shared/kernel_generator/tests/gemm/ -q
```
