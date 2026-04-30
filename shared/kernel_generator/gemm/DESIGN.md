# Kernel Generator Architecture

## Current Performance

Target: MI355X (gfx950), fp16 GEMM, NT layout.

| Size | Ours (TFLOPS) | hipBLASLt (TFLOPS) | Ratio |
|------|---:|---:|---:|
| 4096x4096x4096 | 1104 | 1195 | 92.4% |
| 8192x8192x8192 | 1195 | 1192 | 100.3% |

Best kernel variant: `dtl_partitioned` (256x256x64 tile, 128 MFMAs,
partition-based scheduling with ping-pong A buffers, SlotPlacer
interleaving).

## Architecture Overview

The generator produces GCN assembly from a Python description of the
GEMM tile structure. The pipeline is:

```
GemmTiling -> TileTree -> walk + phase callbacks -> AsmContext -> .s -> .co
```

Phase callbacks emit assembly directly into an `AsmContext` text buffer.
There is no IR. The K-loop body uses a partition-based scheduler where
the partition is the scheduling unit and operations from adjacent
partitions are interleaved.

## Partition-Based Scheduling

The core scheduling architecture (inspired by TensileLite's
`SubtileBasedScheduler`) divides the macrotile into partitions and
schedules data movement around partition boundaries.

### Concepts

**Partition:** A rectangle of subtiles processed together. With
`partition_m=2` and an 8x8 subtile grid (mr=8, nr=8), we get 4
partitions, each computing 2 mi * 8 ni * 2 ki = 32 MFMAs.

**Ping-pong A buffers:** Two VGPR buffer slots for A operands. Within
each partition, mi values are processed sequentially. The current mi
reads from one buffer while the next mi's data is prefetched into the
other. At partition boundaries, the roles swap.

**Cross-partition prefetch:** Each partition's MFMAs are interleaved
with ds_reads that load the *next* partition's A data. By the time
partition N finishes, partition N+1's data is already in registers.

**B is read-only:** B operands are shared across all partitions and
loaded once per K-tile iteration in the preamble. No partition-scoped
B management is needed.

### K-loop Structure

```
Prologue:
  DTL load tile 0 -> vmcnt(0) -> barrier

k_loop:
  // --- DTL prefix (conditional) ---
  advance SRD pointers
  toggle LDS write addresses
  issue 16 DTL loads (8 A + 8 B)

  // --- Preamble ---
  ds_read B[ki=0] (8 reads)
  ds_read A[mi=0, ki=0] into buf0
  ds_read B[ki=1] (8 reads)
  ds_read A[mi=0, ki=1] into buf0
  lgkmcnt(9)  // wait for B[ki=0] + A[m0,k0] only

  // --- Partition-structured body ---
  for mi in 0..7:
    if mi == 0, ki == 1: lgkmcnt(0)  // wait for B[ki=1]
    for ki in 0..1:
      for ni in 0..7:
        // Prefetch A[mi+1] at slots 2 and 10
        if slot == 2:  ds_read A[mi+1, ki=0] into next_buf
        if slot == 10: ds_read A[mi+1, ki=1] into next_buf
        // Suffix ops in last mi group (vmcnt, toggle, negate)
        MFMA(buf[mi%2], B[ni][ki])
    lgkmcnt(0)  // wait for A[mi+1] prefetch
    swap buf

  // --- Postamble ---
  barrier
  branch k_loop
```

### Module Breakdown

The scheduling infrastructure is split into three layers:

**PartitionPlan** (`partition_plan.py`) -- Derives from tile config what
each partition computes and loads. Contains `VGPRTileAllocator` with
separate A/B ID spaces and free-list reuse across partitions. Purely
declarative -- no assembly emission.

```python
plan = PartitionPlan.from_tiling(tile, partition_m=2)
# plan.partitions[0].tile_a_indices = [0, 1]
# plan.partitions[0].lr_a_targets = [2, 3]  (prefetch for P1)
# plan.partitions[3].lr_a_targets = []       (P0's data loaded in preamble)
```

**MainloopScheduler** (`mainloop_scheduler.py`) -- Builds `ScheduleModule`
objects (MFMA, LR, GR) per partition per subIterK, wires dependency edges
across partition boundaries, and derives NGLL/NLL by filtering modules.
Each module carries `emit_fn` closures -- the scheduler decides *order*,
closures decide *content*.

```python
sched = MainloopScheduler(plan)
sched.build_modules(make_mfma_fn, make_lr_fn, make_gr_fn)
sched.wire_dependencies()
modules = sched.mainloop_modules()  # flat list for SlotPlacer
```

**SlotPlacer** (`slot_placer.py`) -- Interleaves non-MFMA instructions
between MFMAs using 2 slots per interval. Supports forward placement
(GR paths -- issue early) and backward placement (wait paths -- delay
late). Rules are injected as callbacks:
- `one_ds_read_per_interval` -- LDS bank conflict avoidance
- `spread_buffer_loads` -- even distribution across slots
- `no_m0_with_buffer_load` -- hardware hazard
- `min_gap_ds_read_to_wait` -- latency hiding

### How It Fits Together

`dtl_partitioned.py` is the integration layer. The K-loop body flows
through the full scheduling stack:

```
PartitionPlan.from_tiling()       # derive partition structure
        |
  emit closures created           # MFMA, LR (A-prefetch), suffix ops
  per mi/ni/ki with ping-pong     # each closure captures its registers
        |
  SlotPlacer.place_path()         # LR paths placed within constrained
                                  # mi ranges (anti-dep safe)
                                  # suffix path placed backward
        |
  schedule.build()                # flat list of (side_ops, mfma) intervals
        |
  emission loop                   # walks intervals, emits ops,
                                  # inserts lgkmcnt at mi/partition boundaries
```

**Preamble** (manual): B[ki=0] + A[m0,k0] + B[ki=1] + A[m0,k1] with
`lgkmcnt(9)` -- this split-ki structure is a proven pattern that doesn't
benefit from automation.

**MFMA body** (scheduled): The `SlotPlacer` places each mi's 2 A-prefetch
ds_reads within that mi's 16-MFMA interval range. Placement is constrained
so writes to a ping-pong buffer only occur after the previous user of
that buffer has completed (anti-dependency safety). The suffix ops (vmcnt,
toggle, negate) are placed backward from the end of the last mi group.

**Waitcnt** (auto): The emission loop tracks in-flight lgkm ops and
inserts `lgkmcnt(0)` at mi boundaries (before the next mi's MFMAs consume
freshly-loaded A data).

## File Map

| File | Purpose |
|------|---------|
| `kernel_pipeline.py` | `GemmKernel.build()` + `emit()` entry point |
| `tiling.py` | `GemmTiling`, `TileDim` chains, `build_tile_tree()` |
| `tile.py` | `TileLevel`, `TilePhase`, `walk_tile_tree()` |
| `problem.py` | `GemmProblem`, `TileConfig`, `MfmaConfig` |
| `asm_context.py` | `AsmContext` -- assembly emission + register naming |
| `context.py` | `TileContext` -- scoped register allocator |
| `asm_transforms.py` | `GemmLayouts`, coordinate transforms |
| `asm_emitter.py` | `assemble_kernel()` (clang), `emit_header()`, `emit_descriptor()` |
| `phases.py` | Prologue phases + older K-loop variants + store epilogue |
| **`partition_plan.py`** | **`PartitionPlan`, `VGPRTileAllocator` -- partition structure** |
| **`mainloop_scheduler.py`** | **`MainloopScheduler`, `ScheduleModule` -- module + deps** |
| **`slot_placer.py`** | **`SlotPlacer`, `SchedulingRules` -- instruction interleaving** |
| **`dtl_partitioned.py`** | **Partition-based DTL K-loop (current best)** |
| `dtl_scheduled.py` | Flat auto-scheduled DTL K-loop (legacy, same perf) |
| `dtl_interleaved.py` | Hand-tuned DTL K-loop (legacy) |
| `auto_scheduler.py` | `ScheduleGraph` + SPREAD (used by `dtl_scheduled`) |
| `launcher.py` | HIP ctypes launch + verification + HIP-event timing |
| `benchmark.py` | Benchmark harness (our kernel vs hipblaslt-bench) |

## K-loop Variant Hierarchy

```
pipelined            basic software-pipelined
pgr2                 PGR=2 double-buffered
dtl                  Direct-To-LDS (no ds_write)
dtl_interleaved      DTL + hand-tuned interleaving (legacy)
dtl_scheduled        DTL + flat ScheduleGraph (legacy)
dtl_partitioned      DTL + partition-based scheduling (current best)
```

## What's Next

### 3-Barrier DTL Overlap (~3-5%)

Move DTL loads from the manual prefix into the partition body so they
can overlap with ds_reads from the opposite matrix region:

```
Partition 0-1:  compute + ds_read A
BARRIER 1:      A reads done -> safe to DTL-write A
Partition 1-2:  compute + DTL A + ds_read B
BARRIER 2:      B reads done -> safe to DTL-write B
Partition 2-3:  compute + DTL B
BARRIER 3:      DTL done -> safe to ds_read new buffer
```

This is a localized change to the partition schedule -- add GR modules
to specific partitions and wire SYNC dep edges at barrier points.

### Vectorized Store (~1%)

Replace 256 `global_store_short` with ~28 `buffer_store_dwordx4` using
SRD-based column-major stride and `v_pack_b32_f16`.

### Even/Odd Wave Scheduling (~1-2%)

Read `HW_REG_HW_ID` SIMD bit to stagger DTL/ds_read order across waves.
