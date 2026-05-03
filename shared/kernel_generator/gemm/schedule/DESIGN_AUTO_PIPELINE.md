# Auto-Pipelined K-Loop Design

Generic software-pipelined K-loop where users declare stages and
dependencies, and the framework auto-derives the loop structure.
Switchable against the manual `ScheduledCompute` for A/B comparison.

## Two-Level Scheduling

```
Level 1: Inter-iteration pipeline        Level 2: Intra-iteration scheduling
(SoftwarePipeline)                        (KLoopScheduler)

  Stages: G, R+M                           Ops: 128 MFMAs, 32 ds_reads,
  Deps: G->R+M (dist=1)                         suffix, scale loads
  Derives: loads_before_reads,              Deps: RAW, WAR, SYNC edges
           ramp-up count,                   Derives: instruction ordering,
           drain condition,                          lgkmcnt placement,
           barrier placement                         read/MFMA interleaving
```

Level 1 is structurally deterministic: given PGR and buffer count,
there is exactly one correct loop structure. No heuristics needed.

Level 2 is the performance-critical part: ds_read placement relative
to consuming MFMAs determines latency hiding. The KLoopScheduler
uses greedy forward placement with WAR/RAW constraints, matching
what a human would do.

## Why This Should Match Hand-Tuned Performance

1. **G stage placement**: derived from `loads_before_reads`. There's
   only one correct answer for a given (PGR, num_buffers) pair.
   Identical to hand-tuned.

2. **Barrier/waitcnt placement**: derived from pipeline structure.
   `s_barrier` goes between G and R+M (or vice versa). `s_waitcnt`
   values are computed from dependency distances. Identical to
   hand-tuned.

3. **MFMA ordering** (mi/ki/ni): determined by WAR dependencies
   (A ping-pong). The scheduler produces the same order as hand-tuned
   kernels because the constraints force it.

4. **ds_read placement**: the scheduler issues reads ~5 MFMAs ahead
   of their consumer (DS_READ_LATENCY / MFMA_CYCLES). This is what
   a human would do.

5. **Global loads are fire-and-forget**: DTL loads go to LDS, tracked
   by vmcnt. Their exact position within the body doesn't affect
   performance as long as they're issued before the barrier.

Risk: the greedy scheduler could make suboptimal choices at edge
cases (unusual tile sizes, high register pressure). Mitigation:
A/B comparison against the manual version catches these.

## Stage Declaration API

```python
@dataclass(frozen=True)
class PipelineStage:
    """A coarse pipeline stage."""
    name: str                          # "G", "RM", "S"
    distance: int = 0                  # tile offset from current
    resource: Optional[str] = None     # shared resource name
    mode: str = "none"                 # "read", "write", "none"
    wait_counter: str = "none"         # "vmcnt", "lgkmcnt", "none"

@dataclass(frozen=True)
class StageDep:
    """Dependency between pipeline stages."""
    producer: str
    consumer: str
    distance: int = 1  # minimum iterations of separation

@dataclass
class ResourceConfig:
    """Shared resource with buffer count."""
    name: str
    num_buffers: int = 2
    buf_size: int = 0  # bytes per buffer slice
```

### GEMM Stage Declarations

```python
# FP16 GEMM
stages = [
    PipelineStage("G", distance=1, resource="lds", mode="write",
                  wait_counter="vmcnt"),
    PipelineStage("RM", distance=0, resource="lds", mode="read",
                  wait_counter="lgkmcnt"),
]
deps = [StageDep("G", "RM", distance=1)]
resources = {"lds": ResourceConfig("lds", num_buffers=2)}

# MXFP4 with scale prefetch
stages = [
    PipelineStage("G",  distance=1, resource="lds", mode="write",
                  wait_counter="vmcnt"),
    PipelineStage("S",  distance=1, resource=None,
                  wait_counter="vmcnt"),
    PipelineStage("RM", distance=0, resource="lds", mode="read",
                  wait_counter="lgkmcnt"),
]
deps = [
    StageDep("G", "RM", distance=1),
    StageDep("S", "RM", distance=1),
]
```

## SoftwarePipeline Class

```python
class SoftwarePipeline:
    """Derives loop structure from declared stages and dependencies."""

    def __init__(self, stages, deps, resources, pgr=None):
        self.stages = {s.name: s for s in stages}
        self.deps = deps
        self.resources = {r.name: r for r in resources}

        # Auto-derive minimum PGR from dependency graph
        self.min_pgr = self._compute_min_pgr()
        self.pgr = pgr if pgr is not None else self.min_pgr

        # Auto-derive buffer lifecycle
        self.loads_before_reads = self._compute_loads_before_reads()

    def _compute_min_pgr(self) -> int:
        """Longest path through dependency graph (sum of distances)."""
        # ASAP scheduling: stage_num[s] = max over predecessors
        stage_num = {s: 0 for s in self.stages}
        changed = True
        while changed:
            changed = False
            for dep in self.deps:
                new_val = stage_num[dep.producer] + dep.distance
                if new_val > stage_num[dep.consumer]:
                    stage_num[dep.consumer] = new_val
                    changed = True
        return max(stage_num.values()) if stage_num else 0

    def _compute_loads_before_reads(self) -> bool:
        """True when PGR < num_buffers for all shared resources."""
        for dep in self.deps:
            prod = self.stages[dep.producer]
            cons = self.stages[dep.consumer]
            if (prod.resource and prod.resource == cons.resource
                    and prod.mode == "write" and cons.mode == "read"):
                bufs = self.resources[prod.resource].num_buffers
                if self.pgr >= bufs:
                    return False
        return True

    # -- Emission interface --

    def emit_ramp_up(self, ctx, stage_emitters):
        """Emit PGR ramp-up stages."""
        ...

    def emit_body(self, ctx, stage_emitters):
        """Emit one loop body iteration (inductive step)."""
        if self.loads_before_reads:
            # Producer stages first, then barrier, then consumer stages
            self._emit_skip_check(ctx)
            for s in self._producer_stages():
                stage_emitters[s.name].emit_prefetch(ctx)
            self._emit_barrier(ctx)
            for s in self._consumer_stages():
                stage_emitters[s.name].emit_compute(ctx)
        else:
            # Consumer first, then producer
            self._emit_barrier(ctx)
            for s in self._consumer_stages():
                stage_emitters[s.name].emit_compute(ctx)
            self._emit_drain_wait(ctx)
            self._emit_skip_check(ctx)
            for s in self._producer_stages():
                stage_emitters[s.name].emit_prefetch(ctx)

    def emit_kloop(self, ctx, stage_emitters):
        """Emit the complete K-loop."""
        self.emit_ramp_up(ctx, stage_emitters)
        ctx.label("k_loop")
        self.emit_body(ctx, stage_emitters)
        self._emit_suffix(ctx)
        self._emit_branch(ctx)
```

## StageEmitter Interface

Each stage provides an emitter that knows how to emit its
instructions. The pipeline framework calls these at the right
points.

```python
class StageEmitter(Protocol):
    def emit_ramp_up(self, ctx, stage_index: int) -> None:
        """Emit ramp-up for this stage (called once per PGR stage)."""

    def emit_prefetch(self, ctx) -> None:
        """Emit prefetch ops (G: advance + toggle + load)."""

    def emit_compute(self, ctx) -> None:
        """Emit compute ops (RM: ds_reads + MFMAs from scheduler)."""
```

### Concrete Emitters

```python
class GlobalLoadEmitter(StageEmitter):
    def __init__(self, loader, schedule):
        self.loader = loader
        self.schedule = schedule

    def emit_ramp_up(self, ctx, stage_index):
        self.loader.emit_loads()
        if stage_index == 0:
            ctx.s_waitcnt("vmcnt(0)")
            ctx.s_barrier()
        else:
            self.loader.advance()
            self.loader.toggle_write()
            self.loader.emit_loads()

    def emit_prefetch(self, ctx):
        for op in self.schedule.prefetch_ops:
            if op.emit: op.emit()


class ReadComputeEmitter(StageEmitter):
    def __init__(self, schedule, reader, loader, scale_loader):
        self.schedule = schedule
        self.reader = reader
        self.loader = loader
        self.scale_loader = scale_loader

    def emit_compute(self, ctx):
        # Delegates to existing _emit_read_compute logic
        # (preamble, MFMA body with interleaved ds_reads, suffix)
        ...
```

## Switchable Design

```python
def pipeline_kloop_phase(level, ctx) -> None:
    """Phase function: supports both manual and auto-pipelined modes."""
    auto_pipeline = ctx._metadata.get("auto_pipeline", False)

    if auto_pipeline:
        # Generic: derive structure from stage declarations
        pipeline_stages = _build_stages(ctx)
        sw_pipeline = SoftwarePipeline(
            stages=pipeline_stages,
            deps=_build_deps(ctx),
            resources=_build_resources(ctx),
            pgr=ctx._metadata.get("pgr", None),
        )
        emitters = _build_emitters(ctx, loader, reader, schedule)
        compute = AutoPipelinedCompute(sw_pipeline, emitters)
    else:
        # Manual: existing hardcoded ScheduledCompute
        compute = ScheduledCompute(loader, reader, scale_loader, pgr=pgr)

    pipeline = KernelPipeline(
        partitioner=GridPartitioner(),
        compute=compute,
    )
    pipeline.emit(ctx)
```

Usage:
```python
# Manual (current, reference)
k = GemmKernel.build(p, pgr=1)

# Auto-pipelined (new, for comparison)
k = GemmKernel.build(p, pgr=1, auto_pipeline=True)
```

## Verification Strategy

1. **Assembly diff**: emit both versions for the same config,
   diff the assembly. They should be instruction-identical for
   simple cases (PGR=1, FP16).

2. **Correctness**: both versions must produce correct results
   on GPU (max_abs_error within tolerance).

3. **Performance**: benchmark both via hipblaslt-bench or our
   launcher. Any regression > 1% triggers investigation.

4. **Progressive migration**: once auto-pipelined matches manual
   for all configs, deprecate the manual path. Keep it behind a
   flag for regression testing.

## Extension Points

Adding a new pipeline stage:
1. Declare a `PipelineStage` with distance and resource
2. Implement a `StageEmitter` with ramp_up/prefetch/compute
3. Register ops in KLoopGraph (for intra-iteration scheduling)
4. The pipeline framework handles everything else

Example: adding format conversion between R and M:
```python
stages.append(PipelineStage("C", distance=0, resource="vgpr_buf",
                            mode="write"))
deps.append(StageDep("R", "C", distance=0))
deps.append(StageDep("C", "M", distance=0))
```

The framework sees C has distance=0 from both R and M, so it's
co-scheduled with them in the same iteration. The KLoopGraph
handles the fine-grained interleaving of conversion ops with
MFMAs.
