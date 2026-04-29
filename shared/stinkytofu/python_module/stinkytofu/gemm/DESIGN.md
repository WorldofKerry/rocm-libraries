# Kernel Generator Architecture Design

## Problem Statement

We need an architecture that:
1. **Automates** GEMM kernel generation (tiling, scheduling, register allocation, codegen)
2. Lets researchers **replace any piece** without understanding the whole pipeline
3. Handles the MFMA(n) + LR(n+1) + GR(n+2) interleaving that makes high-perf GEMM hard
4. Ties register lifetime to tile structure (auto-free when a partition/subtile is done)
5. Works for both "give me a fast kernel" and "I want to tweak the inner loop"

## Lessons from Existing Systems

| System | Strength we borrow | Weakness we avoid |
|--------|-------------------|-------------------|
| **MLIR/IREE** | Progressive tiling as transforms; SSA-based value lifetime; pipelining as loop annotation (`depth`, `order`) | Delegates instruction scheduling entirely to LLVM -- loses control over MFMA/LR/GR interleaving |
| **Triton** | User-facing simplicity (`num_stages`); compiler inserts LDS staging + barriers from dataflow | No control below the tile level; LLVM backend heuristics can destroy carefully planned schedules |
| **CK** | Calculated interleaving rates from tile geometry; `sched_group_barrier` for precise control | Hardcoded C++ control flow; no abstraction over pipeline structure; changing anything requires rewriting the pipeline class |
| **TensileLite** | Two-phase scheduling (abstract ops -> slot placement); VGPRTileAllocator with partition-scoped free; automatic NGLL/NLL derivation | String-based MT iteration tracking ("n+1"); hardcoded numSubIterK=2; no register pressure feedback; 1900 lines for scheduler alone |
| **Our current system** | Coordinate transforms for all addressing; tile tree with replaceable phases; GemmTiling as source of truth | Scheduling is imperative (emit instructions in order); no separation of "what" from "when"; VGPR allocation is static |

## Core Insight

The fundamental tension in GEMM codegen is:

> **High-level structure** (tiling, data flow, pipeline depth) determines **what** operations exist.
> **Low-level scheduling** (instruction interleaving, waitcnt placement, register lifetime) determines **performance**.
> These two concerns must be **separated** but **connected through a well-defined interface**.

Every system that achieves peak performance has some mechanism for interleaving non-MFMA
instructions between MFMAs. The question is where in the abstraction stack this happens.

## Proposed Architecture: Three Layers

```
+-----------------------------------------------------+
|  Layer 1: Tile Plan                                 |
|  "What to compute"                                  |
|  TileDim chains -> TileOps (MFMA, LdsRead, GLoad)  |
|  Coordinate transforms for all addressing           |
|  User writes: tile sizes, MFMA choice, pipeline     |
|  depth                                              |
+-----------------------------------------------------+
|  Layer 2: Schedule                                  |
|  "When to execute + register lifetime"              |
|  Dataflow DAG of TileOps with dep edges             |
|  Slot-based placer interleaves ops between MFMAs    |
|  VGPRPool with partition-scoped auto-free            |
|  User can: swap scheduling rules, adjust latencies  |
+-----------------------------------------------------+
|  Layer 3: Emission                                  |
|  "How to produce assembly"                          |
|  Each TileOp knows how to emit its asm              |
|  AsmContext with named registers                    |
|  User can: replace any emitter function             |
+-----------------------------------------------------+
```

### Layer 1: Tile Plan

**Purpose**: Describe the GEMM as a tree of tile operations. No assembly, no register
numbers, no scheduling decisions.

**Core type**: `TileOp` -- an abstract operation on tiles.

```python
class TileOp:
    """One operation in the GEMM dataflow."""
    kind: OpKind          # MFMA, LDS_READ, LDS_WRITE, GLOBAL_LOAD, BARRIER, WAIT, ...
    tile_coords: dict     # {dim_name: (level, index)} -- which tile this op touches
    iteration: int        # which K-iteration / pipeline stage this belongs to
    inputs: list[Value]   # logical values consumed (e.g., "A operand from LDS")
    outputs: list[Value]  # logical values produced (e.g., "A in VGPR for MFMA")

class Value:
    """A logical register value flowing between ops."""
    name: str             # e.g., "a_tile_mi2_ki0"
    reg_class: str        # "vgpr", "sgpr", "acc"
    count: int            # number of registers
    producer: TileOp      # op that creates this value
    consumers: list[TileOp]
    # Lifetime is IMPLICIT: value is live from producer to last consumer.
    # The scheduler uses this to determine when to free registers.
```

**How it's built**: The tile tree (TileDim chains) generates the full set of TileOps
for one K-tile iteration. The pipeline depth parameter (`num_stages`) determines
which ops reference which iteration.

```python
# User-facing API
plan = TilePlan.build(
    tile=GemmTiling.standard(wg_m=256, wg_n=256, unroll_k=64,
                             mfma=MfmaConfig.f16_16x16x32()),
    num_stages=2,    # PGR=2: load N+2 during compute N
    direct_to_lds=True,
)
# plan.ops is a flat list of TileOps with Value edges
# plan.partitions groups ops by partition (for VGPR recycling)
```

**Key property**: TileOps are pure data. They describe WHAT but not WHEN or HOW.
A researcher can:
- Add a custom op (e.g., a scaling step) by inserting a TileOp with the right Value edges
- Change tile sizes by changing the TileDim chains
- Switch MFMA variant by changing MfmaConfig

### Layer 2: Schedule

**Purpose**: Given a set of TileOps with dataflow edges, produce a linear instruction
order that maximizes MFMA utilization while respecting dependencies and register limits.

**Core types**:

```python
class Schedule:
    """A linear ordering of TileOps with register assignments."""
    slots: list[Slot]     # one Slot per MFMA interval
    reg_map: dict[Value, PhysReg]  # logical -> physical register mapping

class Slot:
    """One MFMA interval (16 cycles). Contains the MFMA + up to N side ops."""
    mfma: TileOp          # the MFMA that defines this interval
    side_ops: list[TileOp]  # ops placed in this interval's gap

class VGPRPool:
    """Partition-scoped VGPR allocator with auto-free."""
    def alloc(self, value: Value) -> PhysReg
    def free_partition(self, partition_id: int)  # free all values scoped to this partition
    # Peak tracking, alignment, free-list reuse
```

**How it works** (3 sub-steps):

1. **Dependency analysis**: Walk the TileOp graph, compute:
   - Data dependencies (Value producer -> consumer)
   - Sync dependencies (barrier placement, wait counts)
   - Anti-dependencies (register reuse after free)

2. **Register allocation**: Assign physical VGPRs to Values.
   - Partition-scoped: values created within a partition are freed when the partition ends
   - Double-buffered: LDS read operands use ping-pong buffers (current + prefetch)
   - Accumulators: permanent (live for entire kernel)
   - The allocator tracks peak pressure and can signal "too many VGPRs" to Layer 1

3. **Slot placement**: Place non-MFMA ops between MFMAs using rules.
   - MFMAs are the "spine" -- they define the timeline
   - Each interval has capacity for ~2 side ops (16-cycle gap, ~8 cycles per side op)
   - Rules (pluggable):
     - `max_ds_read_per_interval(1)` -- avoid LDS bank stalls
     - `min_gap_lr_to_wait(4)` -- hide LDS latency
     - `spread_global_loads()` -- distribute buffer_loads evenly
     - `no_m0_with_buffer_load()` -- hardware hazard
   - Paths: dependency chains (LR->wait->barrier->GR) are placed as units,
     either forward (GR paths: start early) or backward (wait paths: delay wait)

**NGLL/NLL derivation**: Automatic, by filtering TileOps:
- NGLL: remove ops with `iteration >= current + num_stages`
- NLL: remove all load ops, keep only compute + final waits

**Key property**: The schedule is deterministic given the TileOps and rules.
A researcher can:
- Change scheduling rules (e.g., allow 2 ds_reads per interval for experiments)
- Adjust latency estimates (e.g., different gap for a new GPU)
- Replace the entire scheduler with a custom one (just produce a `Schedule` from TileOps)

### Layer 3: Emission

**Purpose**: Given a Schedule with register assignments, emit assembly text.

**Core interface**: Each `OpKind` has an emitter function:

```python
# Registry of emitters -- one per OpKind
EMITTERS: dict[OpKind, Callable[[TileOp, AsmContext, Schedule], None]] = {
    OpKind.MFMA: emit_mfma,
    OpKind.LDS_READ: emit_lds_read,
    OpKind.GLOBAL_LOAD: emit_global_load,    # or emit_direct_to_lds
    OpKind.BARRIER: emit_barrier,
    OpKind.WAIT: emit_waitcnt,
    ...
}

def emit_kernel(schedule: Schedule, ctx: AsmContext):
    """Walk the schedule and emit each op."""
    emit_header(ctx)
    emit_prologue(ctx)     # kernarg load, thread indexing, address setup
    for slot in schedule.slots:
        for op in slot.all_ops():  # mfma first, then side ops
            EMITTERS[op.kind](op, ctx, schedule)
    emit_epilogue(ctx)     # global store, s_endpgm
    emit_descriptor(ctx)
```

**Key property**: Emitters are simple functions. They look up their register
assignments from the Schedule and emit 1-5 assembly instructions each.
A researcher can:
- Replace `emit_mfma` to use a different MFMA encoding
- Replace `emit_global_load` to switch between `global_load` and `buffer_load ... lds`
- Add a new emitter for a custom OpKind

## VGPR Allocation: Tied to Tile Structure

This is the key architectural question. The answer: **Values have scoped lifetimes
determined by the tile tree, and the allocator frees them at scope boundaries.**

### Lifetime Scopes

```
PERMANENT    Accumulators, address registers     Live for entire kernel
K_TILE       Global load buffers (non-DTL)       Live for one K-tile iteration
PARTITION    A/B operand tiles                   Live for one partition, freed after
PREFETCH     Prefetched A operands (next mi)     Live from ds_read to MFMA consumption
```

The partition scope is what enables large tiles (256x256) in limited VGPRs.
Within a partition (e.g., 2x2 subtiles), all A/B operand VGPRs are allocated.
When the partition finishes, they are freed and recycled for the next partition.

### How it connects to TileOps

Each `Value` carries a `scope` annotation derived from its position in the tile tree:
- A value produced by an LDS_READ at partition P gets `scope=("partition", P)`
- When the scheduler sees all consumers of that value are done (last MFMA using it),
  it marks the value as dead
- At partition boundary, VGPRPool.free_partition(P) releases all dead values

This is similar to MLIR SSA lifetime (value lives from def to last use) but
with explicit scope boundaries from the tile tree.

## How the MFMA + LR + GR Interleaving Works

The interleaving pattern emerges naturally from the three layers:

**Layer 1** produces TileOps with iteration tags:
```
MFMA(partition=0, ki=0)       iteration=N     # compute on current data
LDS_READ(partition=0, ki=1)   iteration=N     # read next K-step
LDS_READ(partition=1, ki=0)   iteration=N+1   # prefetch next partition data
GLOBAL_LOAD(tile=A, chunk=0)  iteration=N+2   # fetch data 2 iterations ahead
```

**Layer 2** places them in MFMA slots:
```
Slot 0:  MFMA[p0,k0]  |  LDS_READ_A[p0,k1]
Slot 1:  MFMA[p0,k0]  |  LDS_READ_B[p0,k1]
Slot 2:  MFMA[p0,k0]  |  GLOBAL_LOAD_A[chunk0]
Slot 3:  MFMA[p0,k0]  |  GLOBAL_LOAD_B[chunk0]
...
Slot 20: MFMA[p0,k1]  |  WAIT(lgkmcnt)
Slot 21: MFMA[p0,k1]  |  (empty -- wait was delayed as long as possible)
```

**Layer 3** emits:
```asm
v_mfma_f32_16x16x32_f16 acc[0:3], v[a0:a3], v[b0:b3], acc[0:3]
ds_read_b128 v[a4:a7], v[lds_rd_a] offset:64
v_mfma_f32_16x16x32_f16 acc[4:7], v[a0:a3], v[b4:b7], acc[4:7]
ds_read_b128 v[b8:b11], v[lds_rd_b] offset:128
v_mfma_f32_16x16x32_f16 acc[8:11], v[a0:a3], v[b8:b11], acc[8:11]
buffer_load_dwordx4 v[0], s[srd_a:srd_a+3], 0 offen offset:0, lds
...
```

## Researcher Customization: Replace Any Piece

### Level 1: Change tile config (easy)
```python
plan = TilePlan.build(
    tile=GemmTiling.standard(wg_m=128, wg_n=128, unroll_k=32,
                             mfma=MfmaConfig.f16_16x16x16()),
    num_stages=1,
)
```

### Level 2: Replace an emitter (medium)
```python
# Custom MFMA emitter that adds scaling
def my_scaled_mfma(op, ctx, schedule):
    emit_mfma(op, ctx, schedule)  # standard MFMA
    # Add scaling after each MFMA
    acc = schedule.reg_map[op.outputs[0]]
    ctx.inst("v_mul_f32", acc, acc, "s_scale")

kernel = GemmKernel.build(problem, optimized=True)
kernel.emitters[OpKind.MFMA] = my_scaled_mfma
```

### Level 3: Replace the inner loop (advanced)
```python
# Custom compute function for one partition
def my_partition_compute(partition_ops, ctx, schedule):
    """User-written compute for one partition.
    
    Gets: list of TileOps for this partition with register assignments.
    Must: emit assembly that consumes inputs and produces outputs.
    Can: reorder, add new ops, change instruction selection.
    Cannot: change register assignments (those are fixed by the scheduler).
    """
    for op in partition_ops:
        if op.kind == OpKind.MFMA:
            # Custom MFMA sequence
            ...
        elif op.kind == OpKind.LDS_READ:
            # Custom LDS read pattern
            ...

kernel.partition_compute = my_partition_compute
```

### Level 4: Replace the scheduler (expert)
```python
# Custom scheduler that uses a different interleaving strategy
class MyScheduler:
    def schedule(self, ops: list[TileOp], rules: SchedulingRules) -> Schedule:
        # Your own scheduling algorithm
        ...

kernel = GemmKernel.build(problem, scheduler=MyScheduler())
```

## Comparison to TensileLite SubtileBasedScheduler

### What we keep
- Two-phase separation (abstract ops -> instruction placement)
- VGPRTileAllocator with free-list reuse across partitions
- Slot-based placer with pluggable rules
- Automatic NGLL/NLL derivation by filtering

### What we improve
- **No string-based iteration tracking**. Use integer `iteration` field on TileOps.
  Pipeline depth is just `num_stages`, not "n+1"/"n+2" strings.
- **Values instead of VGPR tile maps**. Values flow between ops as edges in the
  dataflow graph. Register assignment is a separate step, not tangled with op construction.
- **Register pressure feedback**. The allocator can signal "peak VGPRs exceeded" and
  the planner can automatically reduce partition size or pipeline depth.
- **Separation of addressing from scheduling**. Coordinate transforms handle ALL address
  computation. The scheduler never computes offsets -- it just places ops.
- **Simpler emitters**. Each op emitter is ~10-20 lines, not a 50-parameter function
  call into a separate module.
- **General-purpose**. The TileOp/Value/Schedule types are not GEMM-specific. The same
  framework could schedule a convolution or attention kernel.

### What we skip (for now)
- Even/odd wave splitting (TensileLite `.if isOdd` macro) -- complex, marginal benefit
- MX scale double-buffering -- not needed for fp16
- gfx1250 TDM special-casing -- future work

## Implementation Phases

### Phase 1: Foundation (current sprint)
- Add `MfmaConfig.f16_16x16x32()` and update default tile to 256x256x64
- Get the existing K-loop working with the new MFMA (already parameterized)
- Measure: expect ~2x improvement from MFMA change alone

### Phase 2: TileOp + Schedule
- Define `TileOp`, `Value`, `Schedule` types
- Build `TilePlan.build()` that generates TileOps from TileDim chains
- Implement VGPRPool with partition-scoped lifetime
- Implement slot-based placer with basic rules
- Wire through: TilePlan -> Schedule -> Emission
- Measure: expect parity with Phase 1 perf, but cleaner architecture

### Phase 3: DirectToLDS + PGR=2
- Add `buffer_load ... lds` emitter to Layer 3
- Add `num_stages=2` support to Layer 1 (TileOps reference iteration N+2)
- Update scheduler to handle deeper pipeline
- Measure: expect 1.2-1.4x improvement over Phase 2

### Phase 4: Partition-scoped VGPR recycling
- Implement full partition scheduling with VGPR free/realloc
- Enable 256x256 tile with 504 VGPRs (248 regular + 256 acc)
- Measure: expect to approach hipBLASLt performance

## Concrete Types (Python)

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

class OpKind(Enum):
    MFMA = auto()
    LDS_READ = auto()
    LDS_WRITE = auto()
    GLOBAL_LOAD = auto()       # global_load -> VGPR
    DIRECT_TO_LDS = auto()     # buffer_load ... lds
    BARRIER = auto()
    WAIT_VMEM = auto()         # s_waitcnt vmcnt(N)
    WAIT_LDS = auto()          # s_waitcnt lgkmcnt(N)
    PTR_ADVANCE = auto()       # advance global load pointers
    LDS_BUFFER_SWAP = auto()   # toggle double-buffer offsets
    CUSTOM = auto()            # user-defined

@dataclass
class Value:
    name: str
    reg_class: str            # "vgpr", "sgpr", "acc"
    count: int
    scope: tuple              # ("permanent",) | ("partition", pid) | ("k_tile",)
    producer: Optional['TileOp'] = None
    consumers: list['TileOp'] = field(default_factory=list)
    physical_reg: Optional[int] = None  # set by allocator

@dataclass  
class TileOp:
    kind: OpKind
    tile_coords: dict = field(default_factory=dict)  # e.g. {"mi": 2, "ni": 3, "ki": 0}
    iteration: int = 0        # pipeline stage: 0=current, 1=next, 2=two-ahead
    partition_id: int = 0
    inputs: list[Value] = field(default_factory=list)
    outputs: list[Value] = field(default_factory=list)
    # Addressing info (from coordinate transforms)
    offset_transform: Optional['Embed'] = None
    static_offset: int = 0
    # Scheduling metadata (set by scheduler)
    slot: Optional[int] = None
    # Custom payload for user-defined ops
    payload: Optional[dict] = None

@dataclass
class Slot:
    index: int
    mfma: TileOp                          # the MFMA defining this slot
    side_ops: list[TileOp] = field(default_factory=list)
    capacity: int = 2                     # max side ops per slot

@dataclass
class Schedule:
    slots: list[Slot]
    prologue_ops: list[TileOp]            # pre-loop setup
    epilogue_ops: list[TileOp]            # post-loop store
    loop_control: list[TileOp]            # k-loop counter, branch
    values: list[Value]                   # all values in the schedule
    reg_map: dict[str, int] = field(default_factory=dict)  # value_name -> phys_reg

class SchedulingRules:
    """Pluggable rules for the slot-based placer."""
    max_ds_read_per_interval: int = 1
    min_gap_lr_to_wait: int = 4           # MFMA intervals between ds_read and waitcnt
    spread_global_loads: bool = True
    no_m0_with_buffer_load: bool = True   # hardware hazard

class VGPRPool:
    """Partition-scoped VGPR allocator."""
    def alloc(self, value: 'Value') -> int: ...
    def free_value(self, value: 'Value'): ...
    def free_partition(self, partition_id: int): ...
    @property
    def peak(self) -> int: ...
    @property  
    def current(self) -> int: ...
```

## Open Questions

1. **Should the slot placer be forward or backward?** TensileLite places GR paths
   forward and wait paths backward. CK calculates rates and emits sched_group_barriers.
   We could support both strategies via the pluggable rules.

2. **How to handle the K-loop boundary?** The mainloop body needs:
   - Compute on current data
   - Global loads for N+2 data (interleaved with compute)
   - Wait for N+1 data to arrive in LDS
   - LDS buffer swap
   
   The tricky part is that the "wait" should be as late as possible, but must happen
   before the next iteration's compute. The scheduler handles this by placing
   wait paths backward from the end.

3. **Register pressure feedback loop**: If the allocator detects peak VGPRs > 512,
   should it automatically:
   (a) Reduce partition size (more partitions, less live at once)?
   (b) Reduce pipeline depth (fewer in-flight loads)?
   (c) Report an error and let the user fix it?
   
   Proposal: (c) for now, with helpful error messages showing which values are
   live and suggesting which parameter to reduce.

4. **How do coordinate transforms connect to TileOps?** Each GLOBAL_LOAD and LDS_READ
   TileOp carries an `offset_transform` (Embed) that describes its address computation.
   The emitter calls `emit_affine()` with the transform + register bindings.
   This means addressing is fully declarative -- the scheduler never computes offsets.

5. **What about non-GEMM kernels?** The TileOp/Value/Schedule framework is general.
   A convolution kernel would have different TileOps (maybe im2col + MFMA), but the
   same scheduler and emission infrastructure. This is a future goal, not a current
   requirement.
