# Stream-K Design

## Problem

Standard data-parallel GEMM launches one workgroup per output tile.
When total_tiles is not a multiple of num_CUs, the last "wave" of
tiles under-utilizes the GPU. Stream-K distributes K-iterations
across CUs to eliminate this tail effect.

## Architecture Integration

Stream-K is a **partitioning strategy**, not a compute strategy.
It changes HOW work is distributed, not WHAT computation happens.

```
KernelPipeline = TilePartitioner + ComputePipeline + Epilogue
                  ^^^^^^^^^^^^^                       ^^^^^^^^
                  StreamK changes these two.
                  ComputePipeline (PipelineScheduler + PipelineEmitter) is unchanged.
```

### Components

1. **StreamKPartitioner** (GPU-side, exists):
   - Loads per-WG iteration range from kernargs (iter_start, iter_end)
   - Sets `s_k_tiles = iter_end - iter_start`
   - Adjusts A/B SRD bases for K-offset when iter_start > 0
   - The K-loop body runs exactly `s_k_tiles` iterations (unchanged)

2. **StreamKEpilogue** (GPU-side, needed):
   - Full-K WGs: store D directly (same as GridPartitioner)
   - Partial-K WGs: atomic-add partial accumulators to workspace
   - Last partial WG for each tile: also writes final result to D

3. **StreamKHost** (host-side, exists as `compute_sk_params`):
   - Computes per-WG (iter_start, iter_end) ranges
   - Allocates workspace buffer for partial accumulation
   - Packs StreamK kernargs
   - Launches fixup kernel if needed

## Execution Flow

```
Host:
  1. Compute SK params: dp_tiles, sk_tiles, sk_ctas, iter ranges
  2. Allocate workspace: sk_tiles * tile_m * tile_n * sizeof(f32)
  3. Launch main kernel with 1D grid: dp_tiles + sk_ctas WGs
  4. Launch fixup kernel (if atomic path used)

GPU main kernel:
  WG serial_id < dp_tiles:
    → GridPartitioner path: tile_m/n from 2D decomp, full K
  WG serial_id >= dp_tiles:
    → StreamK path:
      a. Load iter_start/iter_end from SK params
      b. Derive tile_m/n from global iteration index
      c. Set s_k_tiles = iter_end - iter_start
      d. Adjust SRDs for K-offset
      e. Run K-loop (identical to data-parallel)
      f. Epilogue: atomic-add partial result to workspace

GPU fixup kernel:
  For each SK tile: load workspace partial, add to D
```

## Implementation Plan

### Phase 1: Single-tile-per-WG StreamK (simplest)
Each SK WG handles a contiguous K-range of ONE output tile.
No cross-tile iteration. Requires atomic accumulation.

Changes needed:
- `StreamKEpilogue`: check `s_is_partial`, branch to atomic-add path
- `GemmLauncher.run_streamk()`: host-side orchestration
- Test with small grids where tail wave matters

### Phase 2: Multi-tile StreamK
SK WGs may span multiple output tiles. Requires GPU-side
tile index computation from global iteration index.

### Phase 3: Non-atomic StreamK (deterministic)
Use workspace partitioning instead of atomics.
Each partial result written to a unique workspace slot.
Fixup kernel reduces all partials per tile.

## Key Constraints

- ComputePipeline (K-loop) is UNCHANGED -- it just runs fewer iterations
- The auto-pipeline scheduler doesn't need to know about StreamK
- StreamK only affects: partitioner setup, epilogue store, host launch
- Workspace size: sk_tiles * MT_M * MT_N * 4 bytes (f32 accumulators)
  For 256x256 tile: 256KB per SK tile. Typically < 100 SK tiles.
