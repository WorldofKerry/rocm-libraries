# PGR=N Pipeline Architecture

Generic software-pipelined K-loop inspired by CUTLASS CuTeDSL
(`PipelineTmaUmma`) and classical modulo scheduling.

## Key Insight: CUTLASS vs AMD GCN

CUTLASS Blackwell uses **warp specialization**: separate TMA producer
and MMA consumer warps run independent loops, connected by barrier-
based `PipelineState` on shared circular buffers.

AMD GCN uses **single-warp pipelining**: all waves execute the same
loop, interleaving G (global load), R (LDS read), and M (MFMA)
stages within one loop body. Synchronization is `s_barrier` + `s_waitcnt`.

Despite different hardware, the **abstractions are the same**:

| CUTLASS concept            | AMD GCN mapping                              |
|----------------------------|----------------------------------------------|
| `PipelineState.index`      | LDS buffer index (XOR toggle for 2-buf)      |
| `PipelineState.phase`      | Not needed (single warp, implicit ordering)   |
| `PipelineState.advance()`  | `s_xor_b32 lds_wr, lds_wr, db_step`          |
| `producer.acquire()`       | `s_waitcnt vmcnt(0)` (buffer ready for write) |
| `producer.commit()`        | Implicit (DTL/buffer_load issued)             |
| `consumer.wait()`          | `s_barrier` (data visible in LDS)             |
| `consumer.release()`       | `v_xor_b32 lds_rd, lds_rd, db_step` (toggle)  |
| `num_stages`               | `num_lds_buffers` (2 = double buffer)         |
| `try_wait/try_acquire`     | Branch-based skip (`s_cbranch`)               |

## Pipeline Model

The K-loop is a pipeline of **stages** operating on tiles at different
offsets from the "current" tile:

```
Stage G: Global load   (tile offset = +PGR, writes LDS buffer)
Stage R: LDS read      (tile offset = 0,    reads LDS buffer)
Stage M: MFMA compute  (tile offset = 0,    uses VGPR from R)
```

Dependencies:
```
G(i) --[distance=1]--> R(i+1)    # R reads what G wrote last iter
R(i) --[distance=0]--> M(i)      # M uses R's output same iter
```

## Pipeline Phases

For PGR=N with T total tiles, the K-loop has three phases:

```
Phase        Active Stages   Iterations   Description
------------ --------------- ------------ ----------------------------
Ramp-up      G only          N            Fill pipeline (outside loop)
Steady-state G + R + M       T - N        All stages overlapped
Drain        R + M only      N - 1*       Pipeline empties
```

*Drain is implicit: the load skip condition `k_tiles > PGR-1`
naturally produces drain iterations where G is skipped.

The **inductive step** (loop body) is identical for steady-state
and drain -- only G's skip condition changes. This is the key to
a single code path for any PGR.

### PGR=1 (2 buffers, load-before-read)
```
Ramp:    G(0), wait, barrier
Loop[0]: G(1), barrier, R(0)+M(0), toggle
Loop[1]: G(2), barrier, R(1)+M(1), toggle
Loop[2]: G(3), barrier, R(2)+M(2), toggle
Loop[3]: [skip G], barrier, R(3)+M(3), toggle   <-- drain
```

### PGR=2 (2 buffers, read-before-write)
```
Ramp:    G(0), wait, barrier; G(1) into buf1
Loop[0]: barrier, R(0)+M(0), lgkmcnt(0), G(2), toggle
Loop[1]: barrier, R(1)+M(1), lgkmcnt(0), G(3), toggle
Loop[2]: barrier, R(2)+M(2), [skip G], toggle    <-- drain
Loop[3]: barrier, R(3)+M(3), [skip G], toggle    <-- drain
```

### PGR=2 (3 buffers, load-before-read)
```
Ramp:    G(0), wait, barrier; G(1) into buf1
Loop[0]: G(2), barrier, R(0)+M(0), advance  (buf rotation: 0->1->2->0)
Loop[1]: G(3), barrier, R(1)+M(1), advance
Loop[2]: [skip G], barrier, R(2)+M(2), advance   <-- drain
Loop[3]: [skip G], barrier, R(3)+M(3), advance   <-- drain
```

## Buffer Lifecycle Rule

The framework automatically derives G placement from PGR and buffer count:

```python
loads_before_reads = (pgr < num_buffers)
```

- `True`:  G goes BEFORE R (free buffer always available)
- `False`: G goes AFTER R  (must consume before overwriting)

This is the ONLY control flow difference. R+M scheduling is identical.

## Ramp-Up Detail

Ramp-up runs OUTSIDE the main loop, loading PGR tiles sequentially:

```python
for stage in range(pgr):
    if stage == 0:
        emit_loads()
        s_waitcnt vmcnt(0)    # wait for data
        s_barrier              # make visible
    else:
        if k_tiles <= stage: skip   # not enough tiles
        advance()
        toggle_write()
        emit_loads()           # prefetch, no wait
```

Each stage loads into a different LDS buffer. Stage 0 always waits
(data needed immediately for the first loop iteration). Stages 1+
are fire-and-forget prefetches.

## Drain Detail

Drain is NOT a separate code path. The same loop body runs, but the
load condition `k_tiles > PGR-1` evaluates false, causing the G stage
to be skipped via `s_cbranch`.

For PGR=1: drain = last 0 iterations with G skipped (just the final
iteration has `k_tiles == 0` which exits the loop).

For PGR=2: drain = last 1 iteration with G skipped.

General: drain = last `PGR-1` iterations.

## Steady-State Body (The Inductive Step)

### Load-before-read (`pgr < num_buffers`)
```asm
k_loop:
  ; Early B reads (overlap with arriving loads)
  ds_read early_B ...

  ; Decrement and check
  s_sub_u32 s_k_tiles, s_k_tiles, 1
  s_cmp_gt_u32 s_k_tiles, PGR-1
  s_cbranch_scc0 skip_G

  ; G stage: advance, toggle_write, load
  s_add_u32 srd_a, srd_a, stride
  s_xor_b32 lds_wr, lds_wr, db_step
  buffer_load / DTL ...

skip_G:
  s_barrier                ; consumer.wait()

  ; R+M body (from PipelineScheduler)
  ds_read ... ; v_mfma ...
  ...

  ; Suffix: vmcnt, toggle_read
  s_waitcnt vmcnt(0)       ; producer.acquire() for next iter
  v_xor_b32 lds_rd, lds_rd, db_step  ; consumer.release()
  s_barrier
  s_cbranch_scc1 k_loop
```

### Read-before-write (`pgr == num_buffers`)
```asm
k_loop:
  s_barrier                ; consumer.wait()

  ; Early B reads
  ds_read early_B ...

  ; R+M body (from PipelineScheduler)
  ds_read ... ; v_mfma ...
  ...

  ; Must finish all reads before overwriting
  s_waitcnt lgkmcnt(0)     ; drain LDS reads

  ; Decrement and check
  s_sub_u32 s_k_tiles, s_k_tiles, 1
  s_cmp_gt_u32 s_k_tiles, PGR-1
  s_cbranch_scc0 skip_G

  ; G stage (after reads complete)
  s_add_u32 srd_a, srd_a, stride
  s_xor_b32 lds_wr, lds_wr, db_step
  buffer_load / DTL ...

skip_G:
  s_waitcnt vmcnt(0)
  v_xor_b32 lds_rd, lds_rd, db_step
  s_barrier
  s_cbranch_scc1 k_loop
```

## Buffer Index Tracking

### 2 buffers (XOR toggle)
```python
# Write side: s_xor_b32 s_lds_wr, s_lds_wr, db_step
# Read side:  v_xor_b32 v_lds_rd, v_lds_rd, db_step
# db_step = (wg_m + wg_n) * unroll_k * elem_bytes
```

Maps to CUTLASS `PipelineState.advance()` which does:
```python
index += 1
if index == num_stages:
    index = 0
    phase ^= 1
```

For 2 buffers, XOR is equivalent: index toggles 0<->1.

### 3 buffers (modular rotation)
```python
# Write side: s_add_u32 s_lds_wr, s_lds_wr, db_step
#             s_cmp_ge_u32 s_lds_wr, total_lds
#             s_cmov_b32 s_lds_wr, 0
# Or use s_mod, or predicated subtract
```

This follows CUTLASS's `index = (index + 1) % num_stages` but in
scalar ALU.

## Extension: Scale Prefetch

MXFP4 has an additional scale loading stage:

```
Stage S: Scale load   (tile offset = +1, VMEM -> VGPR)
Dep: S(i) --[distance=1]--> M(i+1)  # scales ready 1 iter ahead
```

S has the same distance as G, so they're co-scheduled in the ramp-up
and share the same skip condition. The pipeline depth doesn't increase
unless S has longer latency (distance=2 would make min_pgr=2).

## Extension: StreamK

StreamK changes the **TilePartitioner** (work distribution), not the
**ComputePipeline** (K-loop body). The pipeline framework is
orthogonal:

```python
# Regular GEMM
pipeline = KernelPipeline(
    partitioner=GridPartitioner(),
    compute=compute_pipeline,
    epilogue=DirectEpilogue(),
)

# StreamK
pipeline = KernelPipeline(
    partitioner=StreamKPartitioner(num_cus=304),
    compute=compute_pipeline,  # SAME
    epilogue=ConditionalEpilogue(direct=..., atomic=...),
)
```

## PipelineState for AMD GCN

Inspired by CUTLASS, but simplified for the single-warp model:

```python
@dataclass
class PipelineState:
    """Circular buffer state for LDS double/triple buffering."""
    num_buffers: int
    write_offset: int   # current write buffer offset in bytes
    read_offset: int    # current read buffer offset in bytes
    buf_step: int       # bytes per buffer slice

    def advance_write(self) -> None:
        """Move write pointer to next buffer."""
        if self.num_buffers == 2:
            self.write_offset ^= self.buf_step   # XOR toggle
        else:
            self.write_offset += self.buf_step
            if self.write_offset >= self.num_buffers * self.buf_step:
                self.write_offset = 0

    def advance_read(self) -> None:
        """Move read pointer to next buffer (consumer.release)."""
        # Same logic as advance_write
        ...
```

This tracks the same state as CUTLASS's `PipelineState(index, phase)`
but uses byte offsets directly (matching our LDS addressing).

## Design Decisions

### Why not warp specialization on GCN?
GCN has no hardware warp specialization (no equivalent of CUDA's
named-barrier-based cooperative groups). All waves in a workgroup
execute the same code. Pipelining must be done by interleaving stages
within a single loop body, not by running producer/consumer loops
on different warps.

### Why branch-based over predicate-based?
Branch-based produces fewer instructions (one branch for G skip) vs
predicate-based (2-3 branches per iteration + extra counter register).
Both are correct for any PGR. Branch-based matches TensileLite's
approach and has lower overhead. However, predicate-based may be
easier to formally verify -- we keep both prototypes for comparison.

### Why "ramp-up" and "drain" instead of "prologue" and "epilogue"?
"Prologue" and "epilogue" are overloaded with kernel setup/store
phases. "Ramp-up" (fill pipeline) and "drain" (empty pipeline) are
unambiguous and match classical pipeline terminology.
