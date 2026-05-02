# K-Loop Architecture V2: Separation of Concerns

## Lessons from existing frameworks

### Composable Kernel (CK)
CK's `UniversalGemmKernel` is parameterized by 3 template types:
```cpp
template <TilePartitioner, GemmPipeline, EpiloguePipeline>
struct UniversalGemmKernel { ... };
```
- **TilePartitioner**: maps WG ID -> (m_tile, n_tile). StreamK uses a
  different partitioner that also computes K range.
- **GemmPipeline**: the K-loop body. Takes block windows for A/B and
  returns accumulated C tile. Same pipeline works for regular and StreamK.
- **EpiloguePipeline**: stores results. StreamK uses atomic accumulation.

StreamK reuses the exact same `GemmPipeline` -- it just wraps it:
```cpp
while (iter_start < iter_end) {
    tile_idx = partitioner.get_tile_index(iter_start);
    k_range = partitioner.get_k_range(iter_start, iter_end);
    BaseGemm(tile_idx, k_range);  // same pipeline
    iter_start += k_range;
}
```

### TensileLite
StreamK is handled at the host level (hipBLASLt library), not in the
kernel codegen. The kernel receives partial K ranges via kernargs.
The assembly K-loop body is identical regardless of StreamK -- only
the prologue (kernarg decode) and epilogue (store vs atomic) differ.

### Triton
Triton's `tl.program_id()` + manual tile indexing puts work decomposition
in user code. The autotuner varies tile sizes. StreamK is a user-level
pattern, not a compiler feature.

### MLIR (linalg → gpu)
MLIR separates tiling, fusion, and distribution as distinct passes.
The tiling pass creates loops. The distribution pass maps loops to
GPU threads. StreamK would be a different distribution strategy applied
to the same tiled computation.

## Common pattern

All frameworks separate the same 3 concerns:
1. **Work distribution** (which WG computes what)
2. **Compute pipeline** (the K-loop body: load + MFMA)
3. **Output strategy** (direct store vs atomic accumulate)

The compute pipeline (#2) is always reusable across StreamK/non-StreamK.

## Current architecture problem

Our `scheduled_kloop_phase()` is a 230-line function that mixes all 3:

```python
def scheduled_kloop_phase(level, ctx):
    # === Work distribution (hardcoded: full K range) ===
    ctx.s_lshr(s_k_tiles, s_K, log2_uk)           # k_tiles = K / unroll_k
    
    # === Compute pipeline (reusable) ===
    loader.emit_loads()                             # prologue
    ctx.label("k_loop")                             # loop
    ...128 MFMAs with interleaved ds_reads...       # body
    ctx.s_cbranch_scc1("k_loop")                   # branch
    
    # === Output strategy (implicit: caller does phase_store_d) ===
```

Adding StreamK would require duplicating the entire compute pipeline
section just to change the 2 lines of work distribution and the store.

## Proposed architecture

```python
class KernelPipeline:
    """Top-level kernel structure. Composes 3 independent concerns."""
    
    def __init__(self,
                 partitioner: TilePartitioner,
                 compute: ComputePipeline,
                 epilogue: Epilogue):
        self.partitioner = partitioner
        self.compute = compute
        self.epilogue = epilogue
    
    def emit(self, ctx: AsmContext) -> None:
        self.partitioner.emit(ctx)   # sets s_tile_m, s_tile_n, s_k_start, s_k_end
        self.compute.emit(ctx)       # K-loop from s_k_start to s_k_end
        self.epilogue.emit(ctx)      # store accumulators

class TilePartitioner(Protocol):
    """Determines what each workgroup computes."""
    def emit(self, ctx: AsmContext) -> None: ...
    def grid_dims(self, problem: GemmProblem, tile: TileConfig) -> tuple[int, int, int]: ...

class ComputePipeline(Protocol):
    """The K-loop: loads data and computes MFMAs."""
    def emit(self, ctx: AsmContext) -> None: ...

class Epilogue(Protocol):
    """Stores accumulated results."""
    def emit(self, ctx: AsmContext) -> None: ...
```

### TilePartitioner variants

```python
class GridPartitioner(TilePartitioner):
    """Simple 2D grid. Each WG gets one (m_tile, n_tile), full K."""
    def emit(self, ctx):
        # s_tile_m = wg_id_x * wg_m
        # s_tile_n = wg_id_y * wg_n
        # s_k_start = 0, s_k_end = K
    
    def grid_dims(self, problem, tile):
        return (problem.m // tile.wg_m, problem.n // tile.wg_n, 1)

class StreamKPartitioner(TilePartitioner):
    """StreamK. Distributes K-tile iterations across all CUs."""
    def __init__(self, num_cus: int = 304): ...
    
    def emit(self, ctx):
        # flat_id = blockIdx.x
        # Compute (tile_m, tile_n, k_start, k_end) from flat_id
        # May process multiple output tiles in a loop (like CK)
    
    def grid_dims(self, problem, tile):
        total_iters = tiles_m * tiles_n * k_tiles
        return (min(total_iters, self.num_cus), 1, 1)
```

### ComputePipeline

This is the reusable K-loop. It reads `s_k_start`, `s_k_end` from
registers set by the partitioner. Everything inside is identical
regardless of StreamK vs Grid:

```python
class ScheduledComputePipeline(ComputePipeline):
    """K-loop using KLoopGraph + KLoopScheduler."""
    def __init__(self, loader, reader, scale_loader):
        self.loader = loader
        self.reader = reader
        self.scale_loader = scale_loader
    
    def emit(self, ctx):
        # Build graph + schedule (existing code)
        graph = KLoopGraph(...)
        GlobalLoadBlock(self.loader).register(graph)
        DSReadBlock(self.reader).register(graph)
        MFMABlock(ctx, tile, self.scale_loader).register(graph)
        SuffixBlock(self.reader, ...).register(graph)
        schedule = KLoopScheduler(graph).schedule()
        
        # Emit K-loop using s_k_tiles from partitioner
        self._emit_prologue(ctx)
        self._emit_loop(ctx, schedule)
        # Does NOT emit store -- that's the epilogue's job
```

### Epilogue variants

```python
class DirectEpilogue(Epilogue):
    """Write accumulators to D. For full tiles."""
    def emit(self, ctx):
        # Existing phase_store_d logic
        ...

class AtomicEpilogue(Epilogue):
    """Atomic-add accumulators to workspace buffer."""
    def emit(self, ctx):
        # Convert acc to output type
        # global_atomic_add to workspace[tile_offset]
        ...

class ConditionalEpilogue(Epilogue):
    """For StreamK: direct store for full tiles, atomic for partial."""
    def __init__(self, direct: Epilogue, atomic: Epilogue): ...
    def emit(self, ctx):
        # if is_full_tile: direct.emit(ctx)
        # else: atomic.emit(ctx)
```

## What changes vs current code

The `scheduled_kloop_phase` function gets split into:
1. Move work decomposition into `GridPartitioner` (~10 lines)
2. Move K-loop body into `ScheduledComputePipeline` (~150 lines)
3. Move store into `DirectEpilogue` (already separate as `phase_store_d`)

The existing `phase_dtl_interleaved_setup` does kernarg loading and
thread indexing -- this stays as a setup phase that runs before the
KernelPipeline.

## StreamK implementation with this architecture

```python
# Non-StreamK (current behavior, no code change needed)
pipeline = KernelPipeline(
    partitioner=GridPartitioner(),
    compute=ScheduledComputePipeline(loader, reader, scale_loader),
    epilogue=DirectEpilogue(),
)

# StreamK
pipeline = KernelPipeline(
    partitioner=StreamKPartitioner(num_cus=304),
    compute=ScheduledComputePipeline(loader, reader, scale_loader),  # SAME
    epilogue=ConditionalEpilogue(
        direct=DirectEpilogue(),
        atomic=AtomicEpilogue(workspace_ptr=...),
    ),
)
```

The `ScheduledComputePipeline` is **identical** in both cases. Only
the partitioner and epilogue change.

## Migration path

1. Extract `ScheduledComputePipeline` from `scheduled_kloop_phase`
2. Create `GridPartitioner` (trivial, wraps current setup)
3. Wrap `phase_store_d` as `DirectEpilogue`
4. Create `KernelPipeline` that composes all three
5. Validate: identical output to current code
6. Implement `StreamKPartitioner` + `AtomicEpilogue`
7. Benchmark on 2048^2 and other small sizes
