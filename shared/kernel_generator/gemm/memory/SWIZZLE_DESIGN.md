# Generic LDS Swizzle Design

Auto-derive bank-conflict-free LDS access patterns from tile geometry,
data type, and memory architecture. Data-type-independent.

## Problem

16 lanes in a half-wave issue `ds_read_b128` (16B each) simultaneously.
Without swizzle, all lanes read from the same column offset within their
row, causing severe bank conflicts:

```
lane  0: row 0, col 0 -> bank 0-3
lane  1: row 1, col 0 -> bank 0-3   (128B row stride = 1 bank row)
lane  2: row 2, col 0 -> bank 0-3
...
lane 15: row 15, col 0 -> bank 0-3   -> 16-way conflict on banks 0-3
```

## Hardware: Banked Memory Model

### GFX950 (MI300)
```
32 banks, 4B stride, 1 access/bank/cycle
128B bank row = 32 * 4B
Access width = 16B (ds_read_b128) = 4 consecutive banks
```

### GFX1250 (MI400)
Nested: segments contain banks.
```
Level 0: 2 segments, 128B stride, 16 accesses/segment/cycle
Level 1: 32 banks, 4B stride, 1 access/bank/cycle
```
Bank conflicts only occur within the same segment. Segment conflicts
occur when > 16 accesses hit the same segment per cycle.

Both architectures are modeled by `BankedMemoryConfig` with a list of
`BankingLevel` entries. The `cycles()` method computes worst-case
across all levels.

## Key Parameters (Data-Type-Independent)

The swizzle depends on tile geometry, not data type directly:

```python
row_stride_bytes = unroll_k * elem_bytes   # LDS bytes per row
num_cols = row_stride_bytes // 16          # 16B columns per row
mfma_k                                     # K elements per MFMA
k_step = (mfma_k * elem_bytes) // 16      # columns between ki iters
rows_per_bank_row = 128 // row_stride_bytes  # rows per 128B bank cycle
```

For all current configs, `row_stride_bytes = 128` (one bank row per row):

| Data type | elem_bytes | unroll_k | row_stride | num_cols | mfma_k | k_step |
|-----------|-----------|----------|------------|---------|--------|--------|
| FP16      | 2         | 64       | 128        | 8       | 32     | 4      |
| BF16      | 2         | 64       | 128        | 8       | 32     | 4      |
| FP8       | 1         | 128      | 128        | 8       | 64     | 4      |
| MXFP4     | 0.5       | 256      | 128        | 8       | 128    | 4      |
| FP16 uk32 | 2         | 32       | 64         | 4       | 32     | 4      |
| FP32 uk16 | 4         | 16       | 64         | 4       | 16     | 4      |

All configs have `num_cols` = 4 or 8 and `k_step` = 4. The swizzle
formula is the same -- only the column count changes.

## XOR Swizzle: `col' = col ^ f(row)`

The XOR swizzle permutes the column index based on the row:

```python
def forward(row, col, num_cols):
    f = ((row >> shift_r) << shift_l) & (num_cols - 1)
    return col ^ f
```

### Auto-Derivation Algorithm

```python
def auto_derive_xor(layout, mem):
    """Exhaustive search over XOR params, return best."""
    best = (999, 0, 0)
    max_shift = int(log2(layout.num_cols)) + 2
    for sr in range(max_shift):
        for sl in range(max_shift):
            sw = XorSwizzle(sr, sl)
            c = sw.verify_all_ki(layout, mem)
            if c < best[0]:
                best = (c, sr, sl)
    return XorSwizzle(best[1], best[2]), best[0]
```

Results (exhaustive search):
```
num_cols=8, stride=128B: XOR(0,0) -> 2 cycles (optimal)
num_cols=4, stride=64B:  XOR(1,0) -> 2 cycles (optimal)
```

For `num_cols=8, XOR(0,0)`: `f(row) = row & 7`, so `col' = col ^ (row % 8)`:
```
lane  0: col'=0^0=0 -> banks 0-3
lane  1: col'=0^1=1 -> banks 4-7
lane  2: col'=0^2=2 -> banks 8-11
lane  3: col'=0^3=3 -> banks 12-15
lane  4: col'=0^4=4 -> banks 16-19
lane  5: col'=0^5=5 -> banks 20-23
lane  6: col'=0^6=6 -> banks 24-27
lane  7: col'=0^7=7 -> banks 28-31
lane  8: col'=0^0=0 -> banks 0-3   (repeats at lane 8)
...
```

Lanes 0-7 each hit unique bank groups (32 banks, 4 per lane = all 32).
Lanes 8-15 repeat -> 2-way conflict. This is optimal: 16 lanes * 4
banks = 64 accesses across 32 banks = ceil(64/32) = 2 cycles minimum.

### Theoretical Lower Bound

```
min_cycles = ceil(lanes_per_group * (access_width / bank_stride) / num_banks)
           = ceil(16 * 4 / 32) = 2
```

XOR(0,0) achieves this for all tested configs. The bound is tight.

### Why XOR Works Across All Data Types

The swizzle operates on 16B columns, not elements. Since all configs
use 128B row stride (= 8 columns), the XOR pattern is identical.
The data type only affects how many elements fit in 16B, not the
bank access pattern.

## Applying to Write and Read Paths

### Write Path (DTL/global load -> LDS)

DTL writes are per-thread with thread-specific offsets. The swizzle
modifies the thread's column offset:

```asm
; v_thread_col = which 16B column this thread writes
; v_thread_row = which LDS row this thread writes
v_xor_b32 v_swizzled_col, v_thread_col, v_thread_row  ; XOR(0,0)
v_and_b32 v_swizzled_col, v_swizzled_col, num_cols-1
; Use v_swizzled_col for LDS write offset
```

### Read Path (LDS -> VGPR via ds_read_b128)

Each lane reads from its `lane_row` at a `k_group` column. The swizzle
must match the write-side permutation:

```asm
; For each ki iteration:
;   read_col = (k_group ^ (lane_row & (num_cols-1)) + ki * k_step) % num_cols
v_xor_b32 v_base_col, v_k_group, v_lane_row  ; XOR(0,0)
v_and_b32 v_base_col, v_base_col, num_cols-1
v_lshlrev_b32 v_read_offset, 4, v_base_col   ; * 16
v_add_u32 v_read_addr, v_read_offset, v_row_base

; ki=1: add k_step columns (with wrap)
v_add_u32 v_tmp, v_base_col, k_step
v_and_b32 v_tmp, v_tmp, num_cols-1
v_lshlrev_b32 v_read_offset_ki1, 4, v_tmp
v_add_u32 v_read_addr_ki1, v_read_offset_ki1, v_row_base
```

## GFX1250 Segment Handling

GFX1250 has nested banking: segments contain banks. The `BankedMemoryConfig`
with two levels handles this automatically:

```python
LDS_GFX1250 = BankedMemoryConfig(
    levels=(
        BankingLevel("segment", num_units=2, stride=128, max_per_cycle=16),
        BankingLevel("bank", num_units=32, stride=4, max_per_cycle=1),
    ),
)
```

The same `verify_all_ki()` checks both levels. XOR(0,0) achieves
2 cycles on GFX1250 as well (no segment conflicts since 16B accesses
span only one segment; bank conflicts are resolved by the XOR).

If future architectures change the nesting (more segments, different
strides), only `BankedMemoryConfig` needs updating. The auto-derivation
algorithm finds the best swizzle for any configuration.

## Auto-Derive Entry Point

```python
def auto_swizzle(tile: TileConfig, mfma: MfmaConfig,
                 elem_bytes: float,
                 mem: BankedMemoryConfig = LDS_GFX950) -> Swizzle:
    """Return the best swizzle for the given tile + memory config."""
    layout = DataLayout.from_tile(tile, mfma, elem_bytes)

    # Try XOR family (cheap, no cross-lane ops)
    best_sw, best_cycles = auto_derive_xor(layout, mem)

    if best_cycles <= 2:
        return best_sw  # optimal or near-optimal

    # Fall back to rotation + cross-lane for harder cases
    rot = RotationSwizzle(use_cross_lane=True)
    rot_cycles = rot.verify_all_ki(layout, mem)
    if rot_cycles < best_cycles:
        return rot

    return best_sw
```

## Implementation Status

### Working
- `BankedMemoryConfig` with hierarchical levels (gfx950, gfx1250)
- `XorSwizzle`, `RotationSwizzle`, `ComposedSwizzle`
- `verify()` and `verify_all_ki()` conflict checkers
- `DataLayout.from_tile()` for data-type-independent geometry

### Broken (from prior session)
- Write/read address mismatch: `emit_write_swizzle` applies swizzle to
  DTL voffset but `emit_read_setup` produces inconsistent addresses.
  The swizzle verification passes in Python but the GPU produces wrong
  results. Root cause: the write path swizzles the column within
  `voffset` but the read path computes `lane_row` differently from
  `thread_row` (lane_row is derived from wave lane ID, thread_row
  is derived from workgroup thread ID).

### TODO
- Wire `auto_swizzle()` into the kernel builder
- Fix write/read address consistency
- Add `auto_derive_xor()` function
- Test on GPU with correctness verification
