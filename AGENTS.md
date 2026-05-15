# AGENTS.md -- Kernel Generator

Python-based GPU assembly GEMM kernel generator targeting AMD MI355X (gfx950).
Produces `.s` assembly files benchmarkable through TensileLite's shared harness
via `generate_custom_kernel()` export.

## Architecture Summary

The kernel generator uses a graph-driven, subtile-based pipeline to emit
software-pipelined K-loop assembly. The pipeline has three layers:

1. **Graph construction** (`graph_builder.py`): Declares ops (global_load,
   ds_write, barrier, ds_read, mfma, toggle) and dependencies (RAW, WAR,
   SYNC) from `LDSStream` declarations. No codegen at this stage.

2. **Scheduling** (`pipeline_scheduler.py`): `PipelineScheduler` derives the
   loop structure from the graph -- body op order, ramp-up stages, drain
   iterations, and waitcnt placement. Produces a `ScheduledPipeline`.

3. **Emission** (`pipeline_emitter.py`): `PipelineEmitter` walks the
   `ScheduledPipeline` and emits assembly. All scheduling decisions come
   from the scheduler; the emitter is a thin translation layer.

State is split across three layers on `AsmContext`:

- `ctx.config` (`KernelConfig`) -- immutable config: tile, problem, layout, mainloop
- `ctx._state` -- runtime state for pipeline phases
- `ctx._bindings` -- register allocation

Op-to-instruction mapping is handled by `emit_wiring.py`, which connects
each graph op's `emit` callback to codegen methods on `GlobalLoader`,
`LDSReader`, `MFMAEmitter`, or `ScaleStream`.

```
GemmProblem + GemmTiling
        |
   GemmKernel.build()             # kernel.py
        |
   TileTree walker                # tile/tree.py
        |
   pipeline_kloop_phase()         # schedule/pipeline.py
        |
   +-- build_kloop_graph()        # schedule/graph_builder.py
   |       KLoopGraph from LDSStream declarations
   |
   +-- PipelineScheduler          # schedule/pipeline_scheduler.py
   |       -> ScheduledPipeline (body, ramp-up, drain, waitcnts)
   |
   +-- wire_emit_callbacks()      # schedule/emit_wiring.py
   |       graph ops -> codegen primitives
   |
   +-- PipelineEmitter.emit()     # schedule/pipeline_emitter.py
           assembly output
```

## Directory Layout

```
kernel-generator/
  AGENTS.md              This file
  shared/
    kernel_generator/
      gemm/
        DESIGN.md          Main architecture doc
        kernel.py          Top-level build + emit entry point
        config.py          KernelConfig, DTypeConfig, DTYPE_REGISTRY, KLoopContract
        problem.py         GemmProblem, TileConfig, MfmaConfig, DataType
        tiling.py          GemmTiling: composable tile hierarchy
        launcher.py        GPU launch + correctness checking
        benchmark.py       Performance measurement
        export_tensilelite.py  TensileLite ABI-compatible .s export
        bench_tensilelite.py   Benchmark via Tensile.sh harness

        emit/              Assembly emission primitives
          context.py       AsmContext: register allocation, instruction emit
          emitter.py       Kernel metadata (.amdhsa_kernel, .amdgpu_metadata)
          layouts.py       LDS layout computation
          phases.py        Tile tree phase functions (setup, store)
          mainloop.py      Mainloop variants (FP16, BF16, MXFP4, NoStore)

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
          DESIGN_PGR.md       PGR pipeline design doc
          DESIGN_STREAMK.md   StreamK design doc
          kloop_graph.py      OpKind, DepKind, KLoopOp, Dep, KLoopGraph
          graph_builder.py    build_kloop_graph() from LDSStream list
          pipeline_scheduler.py  PipelineScheduler -> ScheduledPipeline
          pipeline_emitter.py    PipelineEmitter: assembly from schedule
          emit_wiring.py     Connects graph ops to codegen callbacks
          pipeline.py        Entry point + TilePartitioner + StreamKPartitioner
          tile_ops.py        Reusable tile ops (decompose, SRD recompute, zero acc)
          interleave.py      MFMA/DTL interleaving helpers

      tests/
        gemm/              pytest tests for the GEMM generator
```

## Key Abstractions

Defined in `config.py`:

- **KernelConfig**: Typed, immutable replacement for `ctx._metadata`. User-constructable
  via `KernelConfig.from_problem(problem, tiling)`.
- **DTypeConfig** + **DTYPE_REGISTRY**: Per-data-type constants registry (MFMA config,
  TensileLite metadata, element sizing). Adding a new data type = one registry entry.
- **KLoopContract**: Declares the register interface between setup and K-loop phases.
- **setup_kloop()**: Standalone K-loop setup without `GemmKernel.build()`.

Defined in `problem.py`:

- **DataType**: Enum -- `FP16`, `BF16`, `MXFP4`
- **MfmaConfig**: MFMA instruction geometry (m, n, k, blocks, input/output types)
- **SubTileConfig**: Per-wave sub-tile dimensions and MFMA counts
- **PartitionConfig**: Wave partitioning within a workgroup
- **TileConfig**: Full tile description (macro tile M/N, unroll K, sub-tiles, partitions)
- **GemmProblem**: Top-level problem spec (M, N, K, data types, tile config, PGR depth)

Defined in `memory/`:

- **LDSStream** (`lds_stream.py`): Abstract base for anything that flows through LDS
  (DTLDataStream for matrix data, ScaleStream for MX scales, NullStream placeholder)
- **LDSBufferManager** (`lds_stream.py`): Owns LDS layout + double-buffer toggle.
  ADD-based toggle supports non-power-of-2 buffer sizes.

Defined in `memory/swizzle.py` -- swizzle hierarchy:

- **IdentitySwizzle**: No remapping
- **XorSwizzle**: XOR-based bank conflict avoidance
- **RowRotationSwizzle**: Row-based rotation
- **PairedRowRotationSwizzle**: Paired rotation for wider tiles
- **DTLRotationSwizzle**: Rotation applied to global read voffset (not LDS write),
  because DTL writes have hardware-fixed sequential LDS addresses
- **auto_swizzle()** / **auto_swizzle_dtl()**: Select optimal swizzle for a layout

Defined in `schedule/`:

- **KLoopGraph** (`kloop_graph.py`): Dependency graph of K-loop operations
- **PipelineScheduler** (`pipeline_scheduler.py`): Derives ramp-up/body/drain from graph
- **PipelineEmitter** (`pipeline_emitter.py`): Emits assembly from scheduled pipeline

## Library Mode

Components can be used standalone without `GemmKernel.build()`:

```python
cfg = KernelConfig.from_problem(problem, tiling)
ctx = AsmContext(config=cfg)
contract = setup_kloop(ctx, cfg)
# compose graph, schedule, wire, emit
```

`skip_store=True` on `GemmKernel.build()` (or `tile_tree.remove_phase("store_d")`)
emits the mainloop only -- no store/endpgm. The `NoStore` epilogue in
`emit/mainloop.py` supports this path.

Tile tree methods: `replace_phase()` swaps a phase function;
`remove_phase()` drops a phase entirely.

## Build and Test

```bash
cd /home/kerrwang/repos/rocm-libraries/kernel-generator
.venv/bin/python -m pytest shared/kernel_generator/tests/gemm/ -q
```

## Benchmarking

Uses TensileLite's Tensile.sh harness for apples-to-apples comparison.

```bash
# Clean stale .s files first
rm -f /home/kerrwang/repos/rocm-libraries/GemmFromAnywhere/projects/hipblaslt/Tensile/CustomKernels/rocroller/Custom_*.s

# MXFP4 benchmark
HIP_VISIBLE_DEVICES=0 .venv/bin/python -m kernel_generator.gemm.bench_tensilelite \
  --tensile-dir /home/kerrwang/repos/rocm-libraries/GemmFromAnywhere/projects/hipblaslt \
  --dtype mxfp4 --sizes 4096x4096x4096 --device 0

# FP16 benchmark
HIP_VISIBLE_DEVICES=0 .venv/bin/python -m kernel_generator.gemm.bench_tensilelite \
  --tensile-dir /home/kerrwang/repos/rocm-libraries/GemmFromAnywhere/projects/hipblaslt \
  --dtype fp16 --sizes 4096x4096x4096 --device 0

# BF16 benchmark
HIP_VISIBLE_DEVICES=0 .venv/bin/python -m kernel_generator.gemm.bench_tensilelite \
  --tensile-dir /home/kerrwang/repos/rocm-libraries/GemmFromAnywhere/projects/hipblaslt \
  --dtype bf16 --sizes 4096x4096x4096 --device 0
```

Flags: `--streamk` enables StreamK. `--no-validate` skips validation.

## LDS Swizzle: DTLRotationSwizzle

Direct-to-LDS (DTL) writes have hardware-fixed sequential LDS addresses --
the kernel cannot choose where data lands in LDS. To avoid bank conflicts on
reads, DTLRotationSwizzle applies rotation to the **global read voffset**
instead of the LDS write address. This permutes which data ends up at which
LDS address, achieving conflict-free ds_read patterns.

Both FP16 and MXFP4 use this approach. The swizzle is selected automatically
by `auto_swizzle_dtl()` based on tile geometry.

## StreamK

Two-tile StreamK with tree reduction. StreamK is a partitioning strategy,
not a compute strategy -- the K-loop body is unchanged.

Key SGPRs:
- `s_k_tiles_init`: Must be saved before the K-loop (used by StreamK epilogue)
- `s_iter_current` / `s_iter_end`: Persistent loop bounds for work distribution
- `s_wg_idx_save`: Original WG index for workspace addressing

Workspace is indexed by WG index. Partial-K workgroups atomic-add partial
accumulators to workspace; the last partial WG for each tile writes the
final result to D.

Components:
- **StreamKPartitioner** (GPU): Loads per-WG iteration range, adjusts SRDs
- **StreamKEpilogue** (GPU): Conditional direct-store vs atomic-add path
- **compute_sk_params** (host): Computes per-WG ranges, allocates workspace

See `schedule/DESIGN_STREAMK.md` for full design.

## TensileLite Integration

Export path: `generate_custom_kernel()` in `export_tensilelite.py` produces a
`.s` file with `.amdgpu_metadata` compatible with TensileLite's custom kernel
loader. FP16, BF16, and MXFP4 use `tensilelite_abi=True` (no 16-byte header).

Benchmark path: `bench_tensilelite.py` automates:
1. Generate `.s` via `generate_custom_kernel()`
2. Write TensileLite benchmark YAML
3. Copy `.s` to `CustomKernels/` directory
4. Run `Tensile.sh` with the prebuilt `tensilelite-client`

YAML must set `HighPrecisionAccumulate: true` and `Batched: true` in the
ProblemType section to match kernel metadata.

## PGR Pipeline

PGR (Prefetch Global Read) depth controls software pipelining:

- **PGR=0**: No prefetch
- **PGR=1**: Produce-first (load before read). Ramp-up: 1 stage. Drain: 0 extra iterations.
- **PGR=2**: Consume-first (read before write). Ramp-up: 2 stages. Drain: 1 extra iteration.

Buffer placement rule: `loads_before_reads = (pgr < num_buffers)`.
When true, G stage goes before R stage; when false, G goes after R
(must consume before overwriting).

LDS uses double buffering with ADD-based toggle (negate step each iteration)
to support non-power-of-2 buffer sizes needed for scale data alongside
matrix data.

See `schedule/DESIGN_PGR.md` for full pipeline phase diagrams.

## Current Performance (4096x4096x4096)

**Validation Status:**
- FP16: PASSES at all sizes with random data
- BF16: PASSES at all sizes with random data (`mainloop_bf16()` with
  `tensilelite_abi`, GPU tests in `test_bf16.py`)
- MXFP4: PASSES at all sizes with random data + random scales via Python
  reference (scale pre-swizzle and SRD stride fixes applied). TensileLite
  Reference.cpp has a known scale layout bug; our kernel is proven correct.


- **FP16**: 545 TFLOPS (72% of TensileLite 256x256)
- **MXFP4**: 1231 TFLOPS with StreamK (72% of TensileLite 256x256)
- **Main gap**: Instruction scheduling -- MGRIPM-style DTL/scale load
  spreading across MFMA body not yet implemented

## Key Files for Common Tasks

**Adding a new data type:**
- `config.py` -- add one entry to `DTYPE_REGISTRY` (MFMA config, TensileLite
  metadata, element sizing). This replaces editing 6 separate files.
- `problem.py` -- add `DataType` enum variant if not already present

**Changing the K-loop schedule:**
- `schedule/graph_builder.py` -- modify op/dependency declarations
- `schedule/pipeline_scheduler.py` -- scheduling algorithm
- `schedule/pipeline_emitter.py` -- assembly emission from schedule
- `schedule/interleave.py` -- MFMA/DTL interleaving policies

**Modifying LDS swizzle:**
- `memory/swizzle.py` -- swizzle classes, `auto_swizzle_dtl()`
- `memory/lds_reader.py` -- swizzle-aware ds_read emission
- `memory/global_loader.py` -- DTLRotationSwizzle voffset application

**Fixing StreamK issues:**
- `schedule/pipeline.py` -- `StreamKPartitioner`, `_emit_persistent_loop`
- `schedule/tile_ops.py` -- tile decompose, SRD recompute, acc zeroing
- `schedule/DESIGN_STREAMK.md` -- design reference

**Exporting to TensileLite:**
- `export_tensilelite.py` -- `generate_custom_kernel()`, metadata generation
- `bench_tensilelite.py` -- end-to-end benchmark automation
