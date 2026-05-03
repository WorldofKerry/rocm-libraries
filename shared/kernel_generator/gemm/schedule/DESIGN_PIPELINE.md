# Generic Software-Pipelined K-Loop

## Relationship to KLoopGraph

The existing `KLoopGraph` handles **intra-iteration** scheduling:
how MFMAs, ds_reads, and suffix ops are ordered within one loop
body iteration. It has `KLoopOp`, `Dep`, `BuildingBlock`.

This design adds **inter-iteration** pipeline scheduling: which
stages are active in each iteration, how they overlap across
iterations, and how many ramp-up/drain iterations are needed.

The two are orthogonal and composable:
- `SoftwarePipeline` decides the loop structure (which stages per iter)
- `KLoopScheduler` decides instruction order within the R+M stage

## Core Abstraction

The K-loop is a pipeline of **stages**. Each stage operates on a
tile at some offset from the "current" tile. Stages have data
dependencies that determine the minimum pipeline depth.

```python
@dataclass
class PipelineStage:
    name: str                       # "global_load", "lds_read", "mfma"
    emit: Callable                  # emits instructions for this stage
    resource: Optional[str] = None  # shared resource, e.g. "lds_buf"
    resource_mode: str = "none"     # "read", "write", or "none"

@dataclass
class StageDep:
    producer: str       # stage name
    consumer: str       # stage name
    distance: int = 1   # tile distance: consumer uses producer's output
                        # from `distance` iterations ago
```

### GEMM Example

```python
stages = [
    PipelineStage("G", emit_global_load, resource="lds_buf", resource_mode="write"),
    PipelineStage("R", emit_lds_read,    resource="lds_buf", resource_mode="read"),
    PipelineStage("M", emit_mfma),
]

deps = [
    StageDep("G", "R", distance=1),  # R reads what G wrote 1 iter ago
    StageDep("R", "M", distance=0),  # M uses R's output (same iter)
]
```

## What the Framework Derives

### 1. Pipeline Depth

The minimum PGR is the longest path through the dependency graph
measured in tile distances:

```
min_pgr = max over all paths (sum of distances along path)
```

For G(d=1)→R(d=0)→M: depth = 1. Min PGR = 1.

### 2. Buffer Lifecycle (loads_before_reads)

For each resource with both read and write stages:
- Count `num_buffers` for that resource (default 2, configurable)
- If `pgr < num_buffers`: write can go before read (free buffer)
- If `pgr == num_buffers`: write must go after read (reuse buffer)
- If `pgr > num_buffers`: **error** (impossible)

This is derived, not configured. The user sets PGR and buffer count;
the framework computes whether loads go before or after reads.

### 3. Loop Phases

The K-loop has three phases. "Prologue" and "epilogue" are avoided
as names since they're overloaded with kernel setup/store phases.

```
Phase         Active Stages   Iterations   Description
----------    -------------   ----------   -----------
Ramp-up       G only          PGR          Fill the pipeline
Steady-state  G + R + M       T - 2*PGR+1  All stages overlapped
Drain         R + M only      PGR - 1      Empty the pipeline
```

Ramp-up: PGR iterations of G-only, each loading one tile.
Steady-state: all stages active, the inductive step.
Drain: last PGR-1 iterations where G is skipped (tiles already
loaded). The drain is implicit -- the load skip condition
`k_tiles > PGR-1` naturally produces it.

### 4. Stage Ordering Within Body

Within the inductive step:

```
if loads_before_reads:
    G(i+pgr)  →  sync  →  R(i) + M(i)  →  suffix
else:
    sync  →  R(i) + M(i)  →  lgkmcnt(0)  →  G(i+pgr)  →  suffix
```

The ONLY difference is G's position. R+M scheduling (from
`KLoopScheduler`) is identical in both cases.

## Extension: More Stages

### Scale Prefetch

```python
stages = [
    PipelineStage("G", emit_global_load, resource="lds_buf", resource_mode="write"),
    PipelineStage("S", emit_scale_load),
    PipelineStage("R", emit_lds_read,    resource="lds_buf", resource_mode="read"),
    PipelineStage("M", emit_mfma),
]

deps = [
    StageDep("G", "R", distance=1),
    StageDep("S", "M", distance=1),  # scales needed 1 iter ahead
    StageDep("R", "M", distance=0),
]
```

Both G and S have distance=1, so they're both issued 1 tile ahead.
min_pgr = 1. Ramp-up: [G(0)+S(0)].

If scales have higher latency:

```python
    StageDep("S", "M", distance=2),  # scales need 2 iters
```

Now min_pgr = 2. Ramp-up: [G(0)+S(0)], [G(1)+S(1)].

### Format Conversion

```python
    PipelineStage("C", emit_convert, resource="vgpr_buf", resource_mode="write"),
```

Adds a conversion stage with its own resource. Dependencies and
buffer counts are independent of the LDS pipeline.

## Integration with KLoopGraph

The `KLoopGraph` already has `iteration=0` and `iteration=1` on ops.
This maps to the pipeline:

- `iteration=0` ops: R+M stage (current tile)
- `iteration=1` ops: G stage (future tile, conditionally skipped)

The `SoftwarePipeline` would set `iteration` tags based on the
stage's distance from the current tile. An op in stage G with
distance=1 gets `iteration=1`. An op in stage S with distance=2
would get `iteration=2`.

The `KLoopScheduler` already classifies ops by iteration tag and
places iteration=1 ops in the prefetch section. Extending to
iteration=N is straightforward.

## Implementation Path

### Phase 1 (current)
- PGR as integer, `loads_before_reads` derived from PGR vs buffers
- Generic load condition `k_tiles > PGR-1`
- Single `_emit_global_load_ops` shared by both load paths
- Ramp-up loop in prologue emits PGR stages

### Phase 2 (formalize)
- `PipelineStage` and `StageDep` dataclasses
- `SoftwarePipeline.from_stages(stages, deps, pgr, buffers)`
- Validates min_pgr, buffer constraints
- Generates ramp-up, body template, drain condition
- Replaces `ScheduledCompute._emit_prologue` and `_emit_loop`

### Phase 3 (user-extensible)
- Users declare custom stages with deps
- Framework auto-derives pipeline structure
- `KLoopGraph.iteration` extended to support distance > 1
- Per-resource buffer count configuration

### Phase 4 (triple buffering)
- `num_buffers=3` for LDS
- Rotation-based toggle (mod 3) instead of XOR (mod 2)
- PGR=2 with 3 buffers: load-before-read (optimal)
