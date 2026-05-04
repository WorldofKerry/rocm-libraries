# Unified LDS Stream Architecture

## Problem

The current data/scale loading subsystem has two parallel class hierarchies
(`GlobalLoader` and `ScaleLoader`) with ad-hoc bridging via
`DTLLoader.attach_scale_loader()`. This creates:

- 10+ `isinstance` checks across 4 files
- Dual composition mechanisms (attach-path vs graph-path)
- Duplicated SRD advance, toggle, sync logic
- MFMA emission split between `ScaleLoader` and `MFMABlock`
- No clear ownership of scale lifecycle

## Core Insight

Data (A/B matrices) and scales (MX E8M0 factors) share the same lifecycle
through the K-loop: global load → LDS write → LDS read → SRD advance →
DB toggle → barrier. The differences (DTL vs 2-step, swizzled vs linear
reads, ds_read_b128 vs ds_read_b32) are implementation details within
each stream, not fundamental structural differences.

## Architecture: Three Layers

```
┌──────────────────────────────────────────────────┐
│  PipelineScheduler                               │
│  Derives ramp-up / body / drain from distances.  │
│  Computes all vmcnt/lgkmcnt from RAW deps.       │
├──────────────────────────────────────────────────┤
│  KLoopGraph                                      │
│  Ops with distance annotations + RAW/WAR/SYNC    │
│  edges. Built uniformly from streams.            │
├──────────────────────────────────────────────────┤
│  LDSStream + LDSBufferManager                    │
│  Uniform interface for any data channel in LDS.  │
│  Streams don't know about each other.            │
└──────────────────────────────────────────────────┘
```

### Layer 1: LDS Streams

```python
class LDSStream(ABC):
    """One data channel occupying a region in double-buffered LDS."""
    name: str               # "data_a", "scale_b", etc.
    region_size: int         # bytes per buffer

    # Global → LDS (two-phase for load/write overlap)
    def emit_global_loads(ctx)   # issue async loads (vmcnt-tracked)
    def emit_lds_writes(ctx)     # write VGPRs to LDS after vmcnt drain
                                 # (no-op for true DTL streams)

    # LDS → VGPR reads
    def emit_read(ctx, idx, ki)  # one ds_read from this stream

    # K-loop lifecycle
    def advance(ctx)              # SRD += k_stride
    def toggle_write(ctx, step)   # write base += step
    def toggle_read(ctx, step)    # read base += step

    # Metadata for graph construction
    num_global_loads: int         # vmcnt contribution per emit_global_loads
    needs_lds_write: bool         # False for DTL, True for buffer+ds_write
```

Concrete implementations:
- `DTLDataStream` - A or B matrix data via buffer_load_dwordx4 ... lds
- `BufferDataStream` - A or B via buffer_load_dwordx4 to VGPRs + ds_write
- `ScaleStream` - MX scale via buffer_load_dword + ds_write_b32
- `NullStream` - no-op (non-MX kernels, zero cost)

No stream knows about any other stream. No `attach_scale_loader`.

### LDSBufferManager

Owns all streams, computes the LDS layout, and provides bulk operations.

```python
class LDSBufferManager:
    streams: List[LDSStream]
    num_buffers: int              # 1 (PGR=0), 2 (PGR=1-2), 3 (PGR=3+)
    buffer_size: int              # sum of all stream region_sizes
    db_step: int                  # = buffer_size (for N=2)

    def compute_layout()          # assign region offsets within buffer
    def emit_all_loads()          # issue all streams' global loads
    def emit_all_writes()         # after vmcnt drain, write all to LDS
    def toggle_all_writes()       # toggle all write bases
    def toggle_all_reads()        # toggle all read bases
    def advance_all()             # advance all SRDs
    def emit_barrier()            # shared barrier
```

Toggle mechanism by buffer count:
- N=2: ADD + negate (current approach, 2 instructions)
- N=3: increment + conditional wrap (3 instructions)
- N=1: no toggle

### Layer 2: Dependency Graph

One unified graph for ALL ops in a K-loop iteration. Each op carries a
`distance` annotation for cross-iteration relationships.

```python
@dataclass
class Op:
    name: str
    kind: OpKind       # VMEM, DS_READ, DS_WRITE, MFMA, SCALAR, SYNC
    distance: int      # 0 = this iteration, pgr = prefetched ahead
    emit: Callable

@dataclass
class Dep:
    producer: str
    consumer: str
    kind: DepKind      # RAW, WAR, SYNC, ORDER
    distance: int = 0  # cross-iteration distance
```

#### Distance semantics

- `Op.distance = pgr` on a global_load means "this load fetches data
  for pgr iterations in the future."
- `Dep.distance = 2` on a WAR edge means "the consumer in iteration i
  conflicts with the producer in iteration i+2" (ping-pong buffer reuse).

#### Key dependency patterns

```
Producer ops (distance = pgr):
  advance_srd → toggle_write → global_load → [lds_write] → barrier

Consumer ops (distance = 0):
  barrier → ds_read_* → mfma_* → toggle_read

Cross-iteration (WAR):
  mfma_m{mi} --(WAR, dist=num_buffers)-→ ds_read_a_m{mi}
  (ping-pong: A read into buf0 can't happen until MFMA consuming buf0
   from num_buffers iterations ago has completed)

B matrix: no WAR dep (read-only, no ping-pong)
```

#### Unified graph construction

```python
def build_kloop_graph(streams, tile, pgr, num_buffers):
    g = KLoopGraph()

    # Producer ops for each stream (distance = pgr)
    for s in streams:
        g.add_op(Op(f"load_{s.name}", VMEM, distance=pgr, ...))
        if s.needs_lds_write:
            g.add_op(Op(f"write_{s.name}", DS_WRITE, distance=pgr, ...))
            g.add_dep(f"load_{s.name}", f"write_{s.name}", RAW)
        g.add_op(Op(f"advance_{s.name}", SCALAR, distance=pgr, ...))
        g.add_op(Op(f"toggle_wr_{s.name}", SCALAR, distance=pgr, ...))

    # Barrier (all writes must land before any reads)
    g.add_op(Op("barrier", SYNC, distance=0))
    for s in streams:
        write_op = f"write_{s.name}" if s.needs_lds_write else f"load_{s.name}"
        g.add_dep(write_op, "barrier", SYNC)

    # Consumer ops: ds_reads + MFMAs (distance = 0)
    for mi, ni, ki in tile.mfma_indices():
        # ... register read ops, mfma ops, RAW/WAR deps ...

    # Toggle read ops (after all MFMAs)
    for s in streams:
        g.add_op(Op(f"toggle_rd_{s.name}", SCALAR, distance=0, ...))

    return g
```

Scale ops and data ops are registered identically. The scheduler doesn't
need to know the difference.

### Layer 3: Pipeline Scheduler

Derives the complete K-loop structure from the graph.

```python
class PipelineScheduler:
    def schedule(self, graph) -> ScheduledLoop:
        # 1. max_distance = max producer distance (= pgr)
        # 2. ramp_up: emit producers for iters 0..pgr-1
        # 3. body: topological sort of all ops
        #    - produce-first vs consume-first falls out automatically
        #      from distance-aware topological ordering
        # 4. drain: body without producers (final iterations)
        # 5. waitcnts: derived from issue position + RAW deps
```

#### Auto-derived loop order

The scheduler topologically sorts ops. The `distance` annotation
determines whether producers go before or after consumers:

- PGR=1: producers have distance=1. They must complete before the
  barrier of the NEXT iteration. Earliest valid placement: before the
  barrier in THIS iteration. Result: **produce-first**.

- PGR=2: producers have distance=2. They don't need to complete until
  2 iterations later. They can go after consumers. Result:
  **consume-first** (barrier at top of loop).

No `loads_before_reads` flag needed.

#### Auto-derived waitcnts

For each consumer op that depends on a vmcnt/lgkmcnt-tracked producer:

```
wait_value = total_issued_before_consumer - issue_position_of_dep - 1
```

This works identically for:
- ds_read depending on global_load (vmcnt)
- MFMA depending on ds_read (lgkmcnt)
- MFMA depending on scale ds_read (lgkmcnt)
- Cross-iteration deps (the scheduler unrolls the pipeline to compute
  issue positions across iteration boundaries)

The vmcnt(0) before barrier in PGR=2, the vmcnt(extra) in PGR=1, and
all lgkmcnt values for MFMA operands are all computed by this one formula.
No special-casing per PGR level.

### MFMA Emission (Separate Concern)

Currently `ScaleLoader.emit_mfma()` mixes data loading with compute.
Separate into:

```python
class MFMAEmitter:
    """Formats MFMA instructions from operand + scale VGPRs."""

    def emit(ctx, mfma, acc, a_reg, b_reg, scale_a, scale_b, mi, ni, ki):
        # Non-MX: v_mfma_f32_16x16x16_f16 acc, a, b, acc
        # MX without real scales: ... v_mxscale, v_mxscale cbsz:4 blgp:4
        # MX with real scales: ... scale_a, scale_b op_sel:[a_sel,b_sel] ...
```

Streams provide the operand VGPR names. MFMAEmitter formats the
instruction. Clean separation of "how to load" from "how to compute."

## What This Eliminates

| Current code | After |
|---|---|
| `DTLLoader.attach_scale_loader()` | Gone. Streams are independent. |
| 10+ `isinstance` checks | Gone. Polymorphism via LDSStream. |
| `SoftwarePipeline.loads_before_reads` | Gone. Derived from distances. |
| Per-PGR branching in `emit_ramp_up` | Gone. Ramp-up from max distance. |
| Manual vmcnt in `emit_consume` | Gone. Computed from graph. |
| Separate `ScaleBlock`/`DSReadBlock` | Gone. Unified stream-based builder. |
| `ScaleLoader.emit_mfma()` | Moved to `MFMAEmitter`. |
| Dual composition (attach + graph) | Single path: graph only. |

## Extension Scenarios

**New stream type** (bias, output scales):
1. Implement `BiasStream(LDSStream)`
2. Add to streams list
3. Graph builder adds ops + deps automatically
4. Scheduler places them

**New PGR level** (PGR=3):
1. Set `distance=3` on producer ops
2. Set `num_buffers=3`
3. Scheduler derives triple-buffer ramp-up, body, drain, waitcnts

**New data type** (fp8, bf16):
1. Implement appropriate `LDSStream` subclass
2. Implement `MFMAEmitter` variant
3. No changes to graph or scheduler

## Migration Path

Each step keeps all tests passing:

1. **Add LDSStream ABC + concrete wrappers** around existing loaders.
   Old code still runs. New code is unused.

2. **Add LDSBufferManager** that wraps existing loader + scale_loader.
   Delegates to old code internally.

3. **Add unified graph builder** alongside old ScaleBlock/DSReadBlock.
   Verify equivalent graph output.

4. **Extend scheduler** with distance annotations and auto-waitcnt.
   Verify same assembly output.

5. **Replace SoftwarePipeline** with PipelineScheduler.
   Remove old code, isinstance checks, attach pattern.

6. **Extract MFMAEmitter** from ScaleLoader.
