# LDS Swizzle Design

Generic, composable bank-conflict avoidance for any banked memory.

## Problem

LDS has 32 banks, 4 bytes each. `ds_read_b128` reads 16 bytes (4 banks)
per lane. In a half-wave of 16 lanes, naive addressing puts all lanes on
the same 4 banks, causing 16-way conflicts. Optimal = 2 lanes per column
group (16 lanes across 8 column positions), yielding 2 cycles per batch.

## Memory Model

Banked memories are described as a **hierarchy of banking levels**, from
outermost to innermost. A stall occurs if any level is overloaded.

```python
@dataclass(frozen=True)
class BankingLevel:
    name: str
    num_units: int       # units at this level (e.g., 2 segments, 32 banks)
    stride: int          # byte address stride between consecutive units
    max_per_cycle: int   # max accesses served per unit per cycle

    def unit_of(self, byte_addr: int) -> int:
        return (byte_addr // self.stride) % self.num_units


@dataclass(frozen=True)
class BankedMemoryConfig:
    levels: tuple[BankingLevel, ...]
    access_width: int       # bytes per access (16 for ds_read_b128)
    lanes_per_group: int    # scheduling group (16 = half-wave)
```

### Architecture Presets

| Arch | Levels | Notes |
|------|--------|-------|
| gfx950 | `[banks(32, stride=4, max=1)]` | Flat 1D banking |
| gfx1250 | `[segments(2, stride=128, max=N), banks(32, stride=4, max=1)]` | Nested: segment then bank |

### Nested vs Flat

On gfx1250, LDS has segments AND banks. These are **nested**, not
independent dimensions:

```
LDS access
  +-- routed to Segment (each serves N accesses/cycle)
       +-- within segment, routed to Bank (1 access/bank/cycle)
```

A conflict at either level causes a stall. The swizzle must distribute
accesses evenly at ALL levels simultaneously.

Conflict detection walks the hierarchy:

```python
def cycles(self, byte_addrs: list[int]) -> int:
    """Worst-case cycles across all levels."""
    bank_width = self.levels[-1].stride
    worst = 1
    for level in self.levels:
        counts: dict[int, int] = {}
        for addr in byte_addrs:
            for b in range(self.access_width // bank_width):
                uid = level.unit_of(addr + b * bank_width)
                counts[uid] = counts.get(uid, 0) + 1
        if counts:
            busiest = max(counts.values())
            worst = max(worst, -(-busiest // level.max_per_cycle))
    return worst
```

This generalizes beyond LDS: register file banking, cache line conflicts,
or future memory structures are all hierarchical banking levels.

## Swizzle Abstraction

A `Swizzle` is a `(row, col) -> col'` bijection applied symmetrically to
write and read paths:

```python
class Swizzle(ABC):
    @abstractmethod
    def forward(self, row: int, col: int, num_cols: int) -> int:
        """Pure-Python column mapping for testing."""
        ...

    @abstractmethod
    def emit_setup(self, ctx, layout, mem) -> SwizzleState:
        """Emit GPU setup instructions. Returns allocated regs."""
        ...

    def verify(self, layout, mem) -> int:
        """Simulate all lanes, return worst-case cycles. 1 = optimal."""
        ...
```

`SwizzleState` holds registers allocated during setup:

```python
@dataclass
class SwizzleState:
    write_col_vreg: str           # swizzled thread_col for DTL writes
    read_base_vregs: list[str]    # per-ki precomputed LR base addresses
```

### Implementations

`IdentitySwizzle` -- passthrough, no permutation. Baseline for testing.

`XorSwizzle` -- `col' = col ^ f(row)` where `f(row) = ((row >> r) << l)
& mask`. No cross-lane ops. Achieves 4-way on gfx950 (not optimal).

`RotationSwizzle` -- rotation + `v_permlane16_swap_b32`. Achieves optimal
2-way on gfx950. See formula below.

`ComposedSwizzle` -- chains multiple patterns: `col' = s2(row, s1(row, col))`.
Useful when different banking levels need independent swizzle terms.

### Composability

```python
# Single swizzle for flat banking (gfx950):
swizzle = RotationSwizzle(use_cross_lane=True)

# Composed swizzle for nested banking (gfx1250):
swizzle = ComposedSwizzle(
    SegmentSwizzle(),    # distribute across segments
    BankSwizzle(),       # distribute across banks within segments
)

# Verify against any target -- same API:
assert swizzle.verify(layout, LDS_GFX950) <= 2
assert swizzle.verify(layout, LDS_GFX1250) <= 1
```

## Data Layout

Three derived values parameterize the swizzle:

```python
@dataclass(frozen=True)
class DataLayout:
    row_stride_bytes: int    # unroll_k * elem_bytes
    mfma_k: int              # K elements per MFMA
    mfma_m: int              # MFMA M dimension (16)
    elem_bytes: float        # 0.5 (mxfp4), 1 (fp8), 2 (fp16)
    wave_size: int           # 64

    num_cols: int      # row_stride_bytes // 16
    ki_count: int      # row_stride_bytes // (mfma_k * elem_bytes)
    k_step: int        # (mfma_k * elem_bytes) // 16
    k_groups: int      # wave_size // mfma_m
```

Coverage across data types:

| Config | elem | uk | row_stride | num_cols | k_step | ki |
|--------|------|----|------------|----------|--------|----|
| MXFP4 16x16x128 | 0.5B | 256 | 128 | 8 | 4 | 2 |
| FP8 16x16x64 | 1B | 128 | 128 | 8 | 4 | 2 |
| FP16 16x16x32 | 2B | 64 | 128 | 8 | 4 | 2 |
| FP16 16x16x32 | 2B | 32 | 64 | 4 | 4 | 1 |
| BF16 16x16x32 | 2B | 32 | 64 | 4 | 4 | 1 |

The swizzle formula and per-ki offset computation are independent of
macro tile size (wg_m, wg_n). They operate per-row within LDS.

## RotationSwizzle Formula

### Column Computation

```
Input:  lane_row (0..mfma_m-1), k_group (0..k_groups-1)
Output: swizzled column index (0..num_cols-1)

rows_per_bank_row = max(1, 128 / row_stride_bytes)
lds_row_id = lane_row / rows_per_bank_row
rotation   = (lds_row_id / 2) * 2
col        = (rotation + k_group) % num_cols
col        = permlane16_swap(col)          // exec = 0x33333333
```

### Why permlane16_swap Matters

Without it: all 16 lanes in a half-wave share the same k_group, so
the rotation only produces 4 unique column values (from 4 lds_row_id
groups of 4 lanes each). Result: 4-way bank conflict.

With it: column values from k_group=0 and k_group=1 are exchanged
between the two half-waves (lanes 0-15 and 16-31). Each half-wave
gets 8 unique column values. Result: optimal 2-way (theoretical
minimum for ds_read_b128 with 16 lanes and 8 column positions).

### Per-ki Offset VGPRs

Each MFMA K-iteration reads from a different K-offset within the
LDS row. Instead of computing the offset at runtime (XOR or add),
precompute one LR base VGPR per ki:

```
lr_offset[0] = swizzled_col * 16 + row_base
lr_offset[ki] = ((swizzled_col + ki * k_step) % num_cols) * 16 + row_base
```

Updated after each double-buffer toggle. Zero additional instructions
per ds_read in the K-loop.

### Write Side

DTL `buffer_load_dwordx4 ... lds` writes use the same swizzle applied
to the thread decomposition:

```
thread_row = tid / threads_per_row
thread_col = tid % threads_per_row
swizzled_col = same_rotation_formula(thread_row, thread_col)
v_dtl_off = thread_row * global_k_stride + swizzled_col * 16
```

The LDS write address is `m0 + lane_id * 16` (hardware-determined).
The swizzle controls which GLOBAL data each thread fetches, so data
arrives in the swizzled column order that the read side expects.

## Verification

`verify()` is pure Python -- testable without a GPU:

```python
def verify(self, layout, mem) -> int:
    worst = 1
    for k_group in range(layout.k_groups):
        addrs = []
        for lane_row in range(mem.lanes_per_group):
            col = self.forward(lane_row, k_group, layout.num_cols)
            addrs.append(lane_row * layout.row_stride_bytes
                         + col * mem.access_width)
        worst = max(worst, mem.cycles(addrs))
    return worst
```

Development workflow:
1. Implement `forward()` in pure Python
2. Run `verify()` against all target `BankedMemoryConfig` presets
3. Iterate until optimal for all targets
4. Implement `emit_setup()` to generate GPU assembly
5. Run GPU correctness tests

## Extensibility

**New data type:** Compute `DataLayout` from tile config. Existing
swizzle implementations work unchanged. Run `verify()` to confirm.

**New architecture:** Define `BankedMemoryConfig` preset. Run `verify()`
against existing swizzles. If conflict factor is suboptimal, implement a
new `Swizzle` subclass or compose existing ones.

**New memory type (register file, cache):** Define `BankedMemoryConfig`
with appropriate levels. The `Swizzle` interface is memory-agnostic.

**New swizzle pattern:** Subclass `Swizzle`, implement `forward()`, test
with `verify()`, then add `emit_setup()`. Use `ComposedSwizzle` to layer
with existing patterns.
