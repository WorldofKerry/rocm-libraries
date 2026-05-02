# Dependency-Driven K-Loop Scheduler

## Problem

The current `ComposableKLoop` manually specifies:
- Loop structure (prologue, early B reads, conditional load-skip, preamble, body, postamble)
- Slot placement with magic constants (`min(4, ...)`, `min(20, ...)`, `2 + i*4`)
- Where to insert `s_waitcnt` (hardcoded MFMA-count boundaries)
- A-buffer ping-pong timing

This makes the code fragile and hard to extend (new data types, tile sizes,
swizzle patterns). A change to the MFMA count or partition structure requires
manual recalibration of all the magic numbers.

## Goal

Replace the manual loop construction with a **dependency-driven scheduler**:
1. Building blocks **declare** operations and their data dependencies
2. The scheduler **derives** the instruction ordering, loop structure, and wait placement
3. Validate by comparing output against the current manual code

## Architecture

```
BuildingBlocks          KLoopGraph           Scheduler          Emitter
┌──────────┐       ┌──────────────┐     ┌─────────────┐    ┌──────────┐
│DTLLoader │──────>│              │     │             │    │          │
│LDSReader │──────>│  Ops + Deps  │────>│ List-sched  │───>│ emit asm │
│ScaleLoader│─────>│  (DAG)       │     │ + auto-wait │    │          │
│MFMABlock │──────>│              │     │             │    │          │
└──────────┘       └──────────────┘     └─────────────┘    └──────────┘
```

## Core Types

```python
class OpKind(Enum):
    MFMA = "mfma"              # matrix multiply-accumulate
    DS_READ = "ds_read"        # LDS -> VGPR (lgkmcnt)
    GLOBAL_LOAD = "global_load"  # VMEM -> LDS/VGPR (vmcnt)
    BARRIER = "barrier"        # s_barrier
    SCALAR = "scalar"          # pointer advance, toggle, negate
    SCALE_LOAD = "scale_load"  # buffer_load for MX scales (vmcnt)

@dataclass
class KLoopOp:
    name: str
    kind: OpKind
    emit: Callable[[AsmContext], None]
    iteration: int = 0         # 0 = current iter, 1 = next iter (prefetch)

@dataclass
class Dep:
    producer: str              # op name
    consumer: str              # op name
    kind: DepKind              # RAW, WAR, SYNC
    min_cycles: int = 0        # latency constraint (e.g. ds_read -> MFMA: ~20 cycles)
```

### Dependency kinds

- **RAW** (Read-After-Write): consumer needs data produced by producer.
  Example: `ds_read_a(mi=0,ki=0)` → `mfma(mi=0,ni=0,ki=0)`.
- **WAR** (Write-After-Read): consumer overwrites register that producer reads.
  Example: `mfma(mi=0,ni=last,ki=last)` → `ds_read_a(mi=2,ki=0)` (A ping-pong reuse).
- **SYNC**: ordering enforced by barrier/waitcnt.
  Example: `global_load` → `barrier` → `ds_read`.

## Building Block Interface

Each building block registers its ops and deps into a shared `KLoopGraph`:

```python
class BuildingBlock(Protocol):
    def register(self, graph: KLoopGraph) -> None:
        """Add ops and deps to the graph."""
        ...
```

### MFMABlock

Declares all `mr * nr * ki_count` MFMAs with RAW deps on their A/B operands:

```python
class MFMABlock(BuildingBlock):
    def register(self, g):
        for mi in range(mr):
            buf = mi % 2
            for ki in range(ki_count):
                for ni in range(nr):
                    name = f"mfma_m{mi}_n{ni}_k{ki}"
                    g.add_op(KLoopOp(name, OpKind.MFMA, emit=...))
                    g.add_dep(f"read_a_m{mi}_k{ki}_buf{buf}", name, DepKind.RAW)
                    g.add_dep(f"read_b_n{ni}_k{ki}", name, DepKind.RAW)
```

### DSReadBlock (wraps LDSReader)

Declares ds_reads with:
- RAW dep on barrier (data must be in LDS)
- WAR dep from last-consumer MFMA of the same A buffer (ping-pong)

```python
class DSReadBlock(BuildingBlock):
    def register(self, g):
        for mi in range(mr):
            buf = mi % 2
            for ki in range(ki_count):
                name = f"read_a_m{mi}_k{ki}_buf{buf}"
                g.add_op(KLoopOp(name, OpKind.DS_READ, emit=...))
                g.add_dep("barrier", name, DepKind.SYNC)

                # A ping-pong WAR: can't overwrite buf until prior mi using
                # same buf is done. mi=2 reuses buf0 from mi=0.
                if mi >= 2:
                    prior_mi = mi - 2
                    last_consumer = f"mfma_m{prior_mi}_n{nr-1}_k{ki_count-1}"
                    g.add_dep(last_consumer, name, DepKind.WAR)

        for ni in range(nr):
            for ki in range(ki_count):
                name = f"read_b_n{ni}_k{ki}"
                g.add_op(KLoopOp(name, OpKind.DS_READ, emit=...))
                g.add_dep("barrier", name, DepKind.SYNC)
```

The WAR deps on A buffers automatically create the partition structure --
the scheduler will naturally group MFMAs by mi and interleave A-prefetch
reads for mi+2 within mi's compute window.

### GlobalLoadBlock (wraps GlobalLoader)

Declares next-iteration loads with cross-iteration deps:

```python
class GlobalLoadBlock(BuildingBlock):
    def register(self, g):
        # Next-iteration ops (iteration=1)
        g.add_op(KLoopOp("advance", OpKind.SCALAR, ..., iteration=1))
        g.add_op(KLoopOp("toggle", OpKind.SCALAR, ..., iteration=1))
        g.add_op(KLoopOp("global_load_next", OpKind.GLOBAL_LOAD, ..., iteration=1))
        g.add_op(KLoopOp("barrier", OpKind.BARRIER, ...))

        g.add_dep("advance", "global_load_next", DepKind.RAW)
        g.add_dep("toggle", "global_load_next", DepKind.RAW)
        g.add_dep("global_load_next", "barrier", DepKind.SYNC)

        # Toggle must happen after last ds_read from current buffer
        # (scheduler resolves this from WAR deps on reader toggle regs)
```

### ScaleBlock (wraps ScaleLoader)

Declares scale loads with RAW deps to their consuming MFMAs:

```python
class ScaleBlock(BuildingBlock):
    def register(self, g):
        for mi in range(mr):
            for ki in range(ki_count):
                name = f"scale_a_m{mi}_k{ki}"
                g.add_op(KLoopOp(name, OpKind.SCALE_LOAD, emit=...))
                # Scale must be ready before MFMA that uses it
                for ni in range(nr):
                    g.add_dep(name, f"mfma_m{mi}_n{ni}_k{ki}", DepKind.RAW)
```

## Scheduler Algorithm

### Phase 1: Pipeline extraction

Partition ops by `iteration` field:
- `iteration=0`: current-iteration ops (ds_reads, MFMAs, scales)
- `iteration=1`: next-iteration ops (advance, toggle, global_load)

Next-iteration ops can execute **in parallel** with current-iteration compute
(they use VMEM/SALU, while compute uses LDS/MFMA).

### Phase 2: MFMA backbone ordering

MFMAs are the fixed backbone. Their order is determined by walking the
dependency graph: `mi` order is forced by A ping-pong WAR deps, `ki` and
`ni` order within each mi follows the natural iteration.

Result: a fixed MFMA sequence with "slots" between adjacent MFMAs.

### Phase 3: Side-op placement (list scheduling)

For each non-MFMA op, find the **earliest legal slot** that satisfies:
1. All producer deps are already scheduled (topological order)
2. Enough cycles between ds_read and consuming MFMA (latency hiding)
3. Resource constraints (≤1 ds_read per MFMA interval, HW hazards)

Special handling:
- **Suffix ops** (vmcnt wait, toggle, negate) are placed **backward** from the
  end -- delay as long as possible to maximize overlap.
- **Next-iteration loads** are placed in the middle of the compute body
  to overlap with MFMA execution.
- **Last-iteration skip**: next-iter ops are wrapped in a conditional branch
  that the emitter generates automatically.

### Phase 4: Auto-wait insertion

Instead of hardcoded `s_waitcnt` at MFMA-count boundaries, the scheduler:
1. Tracks in-flight `lgkmcnt` and `vmcnt` counters as it walks the schedule
2. Before each MFMA, checks if its operand ds_reads have completed
   (counter distance ≥ latency requirement)
3. Inserts `s_waitcnt lgkmcnt(N)` only when actually needed
4. Inserts `s_waitcnt vmcnt(N)` before barriers based on in-flight load count

This eliminates all hardcoded wait positions.

## User API

```python
def my_kloop_phase(level, ctx):
    tile = ctx._metadata["tile"]
    problem = ctx._metadata["problem"]

    # 1. Choose building blocks
    loader = DTLLoader(ctx, tile, problem)
    reader = LDSReader(ctx, tile, problem, swizzle=RotationSwizzle())
    scales = VMEMScaleLoader(ctx, tile)

    # 2. Build dependency graph
    graph = KLoopGraph(tile, problem)
    GlobalLoadBlock(loader).register(graph)
    DSReadBlock(reader).register(graph)
    MFMABlock(ctx, tile, scales).register(graph)
    ScaleBlock(scales).register(graph)

    # 3. Schedule and emit
    scheduler = KLoopScheduler(graph)
    scheduled = scheduler.schedule()
    scheduled.emit(ctx)
```

No manual slot placement, no magic constants, no hardcoded waits.

## Validation Strategy

Before switching over, validate that the new scheduler produces equivalent
output to the current manual `ComposableKLoop`:

1. **Structural comparison**: For a reference tile config (256x256x64 fp16,
   256x256x256 mxfp4), emit both the manual and scheduled versions, then
   compare the instruction sequences:
   - Same MFMA order
   - Same ds_read operands (same register names, offsets)
   - waitcnt positions within ±2 MFMAs of manual version
   - Same total instruction count (±5%)

2. **Assembly comparison**: Assemble both versions and diff the `.s` files.
   Structural ops (MFMAs, ds_reads, buffer_loads) must match exactly.
   Wait/barrier placement may differ slightly.

3. **GPU correctness**: Run the existing 288 test suite with the scheduled
   version -- all must pass.

4. **Performance parity**: Benchmark 4096^3 through Tensile.sh -- scheduled
   version must be within 2% of manual version.

## Implementation Plan

1. `KLoopGraph` + `KLoopOp` + `Dep` data structures
2. `MFMABlock`, `DSReadBlock`, `GlobalLoadBlock` building blocks
3. `KLoopScheduler` with phases 1-4
4. Comparison test: manual vs scheduled output for reference configs
5. `ScaleBlock` for MXFP4
6. Switch `composable_kloop_phase` to use scheduler, gate behind flag
7. Validate, benchmark, remove manual path

## What This Enables

- **New data types**: just implement a new `BuildingBlock` that declares
  its ops/deps -- scheduler handles the rest
- **New tile sizes**: no magic constants to recalibrate
- **New swizzle patterns**: swizzle affects ds_read offsets (in LDSReader),
  not the schedule structure
- **Future GPU archs**: change latency constants and resource constraints,
  scheduler adapts automatically
- **Experimentation**: users can add/remove blocks and the scheduler
  produces a valid loop -- no manual rewiring

## Implementation Status

### Completed
- [x] `KLoopGraph` + `KLoopOp` + `Dep` data structures (`kloop_graph.py`)
- [x] `MFMABlock`: declares MFMAs with RAW deps on A/B operands
- [x] `DSReadBlock`: declares ds_reads with SYNC on barrier + WAR for A ping-pong
- [x] `GlobalLoadBlock`: declares next-iter loads with cross-iter overlap
- [x] `SuffixBlock`: declares vmcnt wait, toggle, negate (placed backward)
- [x] `KLoopScheduler` with phases 1-5 (backbone, classify, forward, backward, auto-wait)
- [x] `scheduled_kloop_phase`: emitter that builds graph, schedules, emits assembly
- [x] `scheduled=True` flag on `GemmKernel.build()`
- [x] Comparison tests: manual vs scheduled output matches exactly
  (128 MFMAs, 32 ds_reads, 2 barriers, 32 buffer_loads -- all identical)
- [x] MXFP4 support: emits, assembles, MFMA count matches composable
- [x] 27 tests covering graph construction, scheduling, and comparison

### Remaining
- [ ] GPU correctness test (`scheduled=True` on actual hardware)
- [ ] Performance benchmark (4096^3 through Tensile.sh)
- [ ] ScaleBlock: register scale loads as graph ops with RAW deps to MFMAs
- [ ] Remove manual `ComposableKLoop` once scheduled path is validated
