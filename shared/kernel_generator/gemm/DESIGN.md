# GEMM Kernel Generator -- Architecture

Python-based assembly generator for GEMM kernels targeting AMD gfx950
(MI355X). Produces `.s` assembly that assembles to `.co` code objects
loadable via hipModule.

## Pipeline Overview

```
GemmProblem + GemmTiling
        |
   GemmKernel.build()       # kernel.py, tiling.py
        |
   TileTree walker           # tile/tree.py -- emits prologue phases
        |
   pipeline_kloop_phase()    # schedule/pipeline.py -- K-loop entry point
        |
   +-- _build_loader_reader_scale()   # shared setup
   |
   +-- build_kloop_graph()            # schedule/graph_builder.py
   |       builds KLoopGraph from LDSStream declarations
   |
   +-- PipelineScheduler.schedule()   # schedule/pipeline_scheduler.py
   |       derives body order, ramp-up, drain, waitcnts
   |
   +-- wire_emit_callbacks()          # schedule/emit_wiring.py
   |       connects graph ops to codegen primitives
   |
   +-- PipelineEmitter.emit()        # schedule/pipeline_emitter.py
   |       emits ramp-up + body loop + drain as assembly
   |       uses interleave helpers for MFMA/DTL spreading
   |
   +-- tile_ops (optional)           # schedule/tile_ops.py
           reusable SRD recompute, tile decompose, acc zeroing
           used by persistent loop + StreamKPartitioner
```

## Directory Layout

```
gemm/
  kernel.py          Top-level build + emit entry point
  problem.py         GemmProblem, TileConfig, MfmaConfig, DataType
  tiling.py          GemmTiling: composable tile hierarchy
  launcher.py        GPU launch + correctness checking
  benchmark.py       Performance measurement
  export_tensilelite.py  TensileLite ABI-compatible export

  emit/              Assembly emission primitives
    context.py       AsmContext: register allocation, instruction emit
    emitter.py       Kernel metadata (.amdhsa_kernel, .amdgpu_metadata)
    layouts.py       LDS layout computation
    phases.py        Tile tree phase functions (setup, store)

  tile/              Tile tree (hierarchical code generation)
    tree.py          TileLevel, TilePhase -- walk and emit
    transforms.py    Dim, Tile, TileDescriptor, Embed
    context.py       Thread/wave index computation

  kloop/             K-loop setup helpers
    setup.py         DTL interleaved setup, wave ABI, scale setup

  memory/            Data movement codegen
    global_loader.py DTLLoader (direct-to-LDS), BufferLoader (VGPR path)
    lds_reader.py    Swizzle-aware ds_read for A/B operands
    lds_stream.py    LDSStream ABC + LDSBufferManager
    streams.py       DTLDataStream, ScaleStream, NullStream
    mfma_emitter.py  Unified MFMA emission (all scale variants)
    scale_loader.py  LDSScaleLoader, VMEMScaleLoader (MX scale handling)
    swizzle.py       Bank-conflict-free LDS swizzle framework

  schedule/          K-loop scheduling
    kloop_graph.py   OpKind, DepKind, KLoopOp, Dep, KLoopGraph
    graph_builder.py build_kloop_graph() from LDSStream list
    pipeline_scheduler.py  PipelineScheduler -> ScheduledPipeline
    pipeline_emitter.py    PipelineEmitter: assembly from schedule
    emit_wiring.py   Connects graph ops to codegen callbacks
    pipeline.py      Entry point + TilePartitioner + StreamKPartitioner
    tile_ops.py      Reusable tile operations (decompose, SRD recompute, zero acc)
    interleave.py    MFMA/DTL interleaving helpers (shared by all copy strategies)
```

## K-Loop Scheduling

The scheduler works in three layers:

1. **Graph construction** (`graph_builder.py`): Declares ops (global_load,
   ds_write, barrier, ds_read, mfma, toggle) and dependencies (RAW, WAR,
   SYNC) from LDSStream metadata. No codegen at this stage.

2. **Scheduling** (`pipeline_scheduler.py`): Derives loop structure from
   the graph:
   - Pre-body reads (B ki=0) placed before skip-check for latency hiding
   - Produce-first vs consume-first order from PGR depth
   - Ramp-up / drain stages from max iteration distance
   - Waitcnts computed from counter tracking + RAW deps

3. **Emission** (`pipeline_emitter.py`): Walks the `ScheduledPipeline`
   and emits assembly. The emitter is a thin layer -- all scheduling
   decisions come from the scheduler.

Op-to-instruction mapping is handled by `emit_wiring.py`, which connects
each graph op's `emit` callback to the actual codegen method on
`GlobalLoader`, `LDSReader`, `MFMAEmitter`, or `ScaleStream`.

## Double Buffering

LDS is split into two equal buffers. The `LDSBufferManager` owns layout
and toggle. ADD-based toggle (negate step each iteration) supports
non-power-of-2 buffer sizes needed for scale data alongside matrix data.

A-matrix uses ping-pong within each buffer (2 VGPR sets for even/odd mi).
B-matrix reads are not ping-ponged.

## Supported Configurations

- **Data types**: FP16, BF16, MXFP4
- **PGR**: 0 (no prefetch), 1 (produce-first), 2 (consume-first)
- **Tile sizes**: 256x256 default; configurable via GemmTiling
- **LDS swizzle**: RowRotation, PairedRowRotation, XOR, Identity
- **Scale loading**: LDS-based (DTL + ds_write), VMEM-based (buffer_load)

## Shared Tile Operations (tile_ops.py)

Reusable assembly-level primitives for tile-dependent state management:

- `emit_decompose_tile_idx` -- flat tile_idx -> (tile_m, tile_n) in s_wg_id_x/y
- `emit_recompute_srds` -- recompute A/B/scale SRDs from tile coords
- `emit_zero_accumulators` -- zero all acc VGPRs
- `emit_reset_kloop_state` -- reset k_tiles, rd_db, lds_db_step
- `emit_build_raw_srd` -- build a 4-SGPR raw buffer SRD from a pointer
- `emit_compute_tile_serial` -- tile_serial = wg_id_y * tiles_m + wg_id_x

Used by both `_emit_persistent_loop` (StreamK) and `StreamKPartitioner`
to eliminate code duplication. The persistent loop body reads as:

```python
_emit_atomic_tile_grab(ctx, tile)
emit_decompose_tile_idx(ctx, tile, tile_idx_reg="s_tmp0")
emit_recompute_srds(ctx, tile, mainloop)
emit_zero_accumulators(ctx, tile)
emit_reset_kloop_state(ctx, tile, mainloop, scheduled.pgr)
PipelineEmitter(scheduled, ...).emit()
phase_store_d(level, ctx)
```

## Interleaving Helpers (interleave.py)

Extracted from `PipelineEmitter` to eliminate duplicated MFMA/load
interleaving code between `_emit_copy_interleaved` and `_emit_copy_ki_phased`:

- `classify_body_ops` -- partition ops into ki0/ki1 reads, MFMAs, scalars, loads
- `build_dtl_sequence` -- build ordered DTL load sequence (scale first, then A/B alternating)
- `emit_mfmas_with_dtl_interleaved` -- spread DTL loads evenly among MFMA ops
- `emit_mfmas_with_reads_interleaved` -- spread ds_reads evenly among MFMA ops

These helpers make scheduling policies pluggable: changing how loads are
spread among MFMAs only requires editing the helpers, not the emitter.
