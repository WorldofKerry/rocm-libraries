# K-Loop Driver Approaches: Concrete Design

## Setup

All approaches share the same inputs:
- `stages`: G (global load), R (LDS read), M (MFMA)
- `deps`: G→R (distance=1), R→M (distance=0)
- `resources`: {"lds_buf": 2 buffers}
- Example: FP16 256x256, unroll_k=64, T=4 tiles

All approaches share the same building blocks:
- `KLoopGraph` + `KLoopScheduler` for intra-iteration R+M scheduling
- `GlobalLoader` for G stage emission
- `LDSReader` for R stage emission
- `MFMABlock` for M stage emission

The difference is HOW the loop structure is generated.

---

## Approach 1: Branch-Based (current + generalized)

Conditional branches skip inactive stages. The loop body is
the same for all iterations; branches gate the G stage.

```python
class BranchPipeline:
    def emit(self, ctx):
        # Ramp-up: PGR loads outside the loop
        for p in range(pgr):
            emit_G(tile=p)
            if p == 0: wait + barrier

        # Single loop body with conditional G
        label("k_loop")
        if loads_before_reads:
            k_tiles--
            if k_tiles > pgr - 1: emit_G(next)  # branch skips in drain
            barrier
            emit_R_M(current)                     # KLoopScheduler output
        else:
            barrier
            emit_R_M(current)
            lgkmcnt(0)
            k_tiles--
            if k_tiles > pgr - 1: emit_G(next)
        suffix: vmcnt, toggle_read
        barrier
        branch k_loop if k_tiles > 0
```

Assembly shape (PGR=1, T=4):
```asm
; ramp-up
buffer_load_dwordx4 ...  ; G(0)
s_waitcnt vmcnt(0)
s_barrier

; loop (4 iterations)
k_loop:
  s_sub_u32 s_k_tiles, s_k_tiles, 1
  s_cmp_lg_u32 s_k_tiles, 0
  s_cbranch_scc0 skip_G
  ; G(next): advance, toggle_write, load
  s_add_u32 s_srd_a, s_srd_a, stride
  s_xor_b32 s_lds_wr_a, s_lds_wr_a, db_step
  buffer_load_dwordx4 ...
skip_G:
  ; barrier (wait for G from previous iter)
  ; R+M: ds_reads interleaved with MFMAs (from KLoopScheduler)
  ds_read_b128 ...
  v_mfma_f32_16x16x32_f16 ...
  ...
  ; suffix
  s_waitcnt vmcnt(0)
  v_xor_b32 v_lds_rd_a, s_lds_db_step, v_lds_rd_a
  s_barrier
  s_cbranch_scc1 k_loop
```

**Pro**: Simple, matches existing code, minimal control flow.
**Con**: Same body for all iterations (can't optimize ramp/drain).

---

## Approach 2: Predicate-Based (MLIR-style)

Every iteration executes the SAME body. Each stage has a predicate
(exec mask or scalar flag) that activates/deactivates it.

```python
class PredicatePipeline:
    def emit(self, ctx):
        # No separate ramp-up. The loop runs T + PGR - 1 iterations.
        # Each stage checks its predicate.
        s_mov s_iter, 0
        label("k_loop")
          # G predicate: iter < T (tile = iter, active in first T iters)
          s_cmp_lt_u32 s_iter, T
          s_cbranch_scc0 skip_G
          emit_G(tile=s_iter)
        skip_G:
          # R predicate: iter >= PGR (tile = iter - PGR)
          s_cmp_ge_u32 s_iter, PGR
          s_cbranch_scc0 skip_R
          barrier
          emit_R(tile=s_iter - PGR)
        skip_R:
          # M predicate: iter >= PGR (same as R, distance=0)
          emit_M(tile=s_iter - PGR)  # same predicate as R
        skip_M:
          suffix
          s_add_u32 s_iter, s_iter, 1
          s_cmp_lt_u32 s_iter, T + PGR
          s_cbranch_scc1 k_loop
```

Assembly shape (PGR=1, T=4, total 5 iters):
```asm
s_mov_b32 s_iter, 0
k_loop:
  ; G: active when s_iter < 4
  s_cmp_lt_u32 s_iter, 4
  s_cbranch_scc0 skip_G
  buffer_load_dwordx4 ...
skip_G:
  ; R+M: active when s_iter >= 1
  s_cmp_ge_u32 s_iter, 1
  s_cbranch_scc0 skip_RM
  s_barrier
  ds_read_b128 ...
  v_mfma_f32_16x16x32_f16 ...
skip_RM:
  ; suffix (always)
  s_waitcnt vmcnt(0)
  toggle_read
  s_add_u32 s_iter, s_iter, 1
  s_cmp_lt_u32 s_iter, 5  ; T + PGR
  s_cbranch_scc1 k_loop
```

**Pro**: Uniform body, trivially correct for any PGR, clean model.
**Con**: Extra branches per iteration, wasted cycles in ramp/drain,
extra register for iteration counter, more total iterations.

---

## Approach 3: Acquire/Release (CUTLASS-style)

Explicit buffer state machine. Producer and consumer phases with
acquire/release barriers on named buffers.

```python
class AcquireReleasePipeline:
    def emit(self, ctx):
        # Ramp-up: producer fills buffers
        for p in range(pgr):
            buf = p % num_bufs
            acquire_write(buf)  # wait until buf is free
            emit_G(tile=p, buf=buf)
            commit(buf)         # signal buf is full

        # Main loop: consumer drives
        for i in range(T):
            buf = i % num_bufs
            wait_read(buf)      # wait until buf is full
            emit_R(tile=i, buf=buf)
            emit_M(tile=i)
            release(buf)        # signal buf is free

            # Producer (overlapped): fill next buffer
            next_tile = i + pgr
            if next_tile < T:
                next_buf = next_tile % num_bufs
                acquire_write(next_buf)
                emit_G(tile=next_tile, buf=next_buf)
                commit(next_buf)
```

On GPU, acquire/release map to barriers + buffer index tracking:
```asm
; acquire_write(buf): s_waitcnt vmcnt(0) if buf was read last iter
; commit(buf): implicit (load issued)
; wait_read(buf): s_barrier (all threads see the data)
; release(buf): implicit (done reading, toggle for next iter)
```

Assembly shape (PGR=1, T=4, 2 bufs):
```asm
; ramp-up
buffer_load_dwordx4 ...  ; G(0) -> buf0
s_waitcnt vmcnt(0)       ; commit(buf0)
s_barrier                ; wait_read(buf0)

k_loop:
  ; Consumer: R+M on current buf
  ds_read_b128 ...       ; R(i) from buf[i%2]
  v_mfma_f32 ...         ; M(i)
  ; release(buf): toggle_read

  ; Producer: G on next buf (if tiles remain)
  s_sub_u32 s_k_tiles, s_k_tiles, 1
  s_cbranch_scc0 skip_produce
  ; acquire_write(next_buf): implicit (just toggled)
  s_xor_b32 s_lds_wr, s_lds_wr, db_step
  buffer_load_dwordx4 ...  ; G(i+1)
skip_produce:
  s_waitcnt vmcnt(0)     ; commit
  v_xor_b32 v_rd, v_rd, db_step  ; toggle
  s_barrier              ; wait_read
  s_cbranch_scc1 k_loop
```

**Pro**: Explicit buffer lifecycle, naturally extends to N buffers,
matches GPU async copy semantics (cp.async in CUDA).
**Con**: More bookkeeping, buffer index register overhead,
harder to overlap producer/consumer within one iteration.

---

## Comparison Matrix

```
                    Branch    Predicate   Acquire/Release
Code complexity     Low       Medium      Medium
Extra registers     0         1 (iter)    1-2 (buf_idx)
Branches per iter   1         2-3         1
Total iterations    T         T+PGR-1     T
Ramp-up code        Separate  In-loop     Separate
Drain code          Implicit  In-loop     Implicit
PGR=N extension     Easy      Trivial     Easy
N-buffer extension  Medium    Medium      Natural
Ramp/drain optim.   Hard      Hard*       Medium
```

*MLIR can optimize ramp/drain by peeling + specializing, but
that creates separate code paths (losing the "uniform body" benefit).

## Recommendation

For implementation, I'd suggest trying:
1. **Branch-based** (already have it, just generalize PGR=N)
2. **MLIR-style predicate** (cleanest model, easy to verify correct)

Then compare the assembly output and performance. If they match,
keep whichever is simpler. The CUTLASS acquire/release model is
more relevant for async copy (cp.async) which we don't use on
GFX9 -- our DTL loads are synchronous vmcnt-tracked.
