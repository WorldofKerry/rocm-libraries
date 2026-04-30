# MXFP4 Support Design

Concrete design for adding MXFP4 (4-bit MX floating point) GEMM support
to the kernel generator. Phased approach: Phase 1 uses constant scale=1
to validate MFMA instruction format and sub-byte data movement. Phase 2
adds the real scale loading pipeline.

Target: `v_mfma_scale_f32_16x16x128_f8f6f4` on gfx950 (MI355X).

---

## Phase 1: Constant Scale MVP

Goal: Generate a correct MXFP4 GEMM kernel with hardcoded scale=1.0
(E8M0 encoding `0x7F` in all 4 bytes = `0x7F7F7F7F`). No scale tensor
arguments, no scale loading, no scale LDS regions.

### 1.1 MfmaConfig Changes

**File:** `problem.py`, class `MfmaConfig` (line ~53)

Add a new factory method and extend `instruction_name`:

```python
@dataclass(frozen=True)
class MfmaConfig:
    m: int
    n: int
    k: int
    blocks: int
    input_type: str       # "f16", "bf16", "f8f6f4"
    acc_type: str         # "f32"
    a_vgprs: int = 0
    b_vgprs: int = 0
    acc_vgprs: int = 0
    element_bits: int = 16  # NEW: bits per element (16 for fp16, 4 for mxfp4)
    cbsz: int = 0          # NEW: format selector for A (4 = FP4)
    blgp: int = 0          # NEW: format selector for B (4 = FP4)
    is_mx: bool = False     # NEW: uses MX scale operands

    @property
    def element_bytes(self) -> float:
        """Bytes per element. 0.5 for 4-bit types."""
        return self.element_bits / 8

    @property
    def instruction_name(self) -> str:
        if self.is_mx:
            return f"v_mfma_scale_{self.acc_type}_{self.m}x{self.n}x{self.k}_{self.input_type}"
        return f"v_mfma_{self.acc_type}_{self.m}x{self.n}x{self.k}_{self.input_type}"

    @staticmethod
    def mxfp4_16x16x128() -> MfmaConfig:
        """v_mfma_scale_f32_16x16x128_f8f6f4: MXFP4 on gfx950.

        MI_K=128, 4 VGPRs per A/B operand (128 * 0.5B / 64 lanes = 1B/lane,
        but packed into 4 VGPRs due to instruction format). 4 acc VGPRs.
        """
        return MfmaConfig(
            m=16, n=16, k=128, blocks=1,
            input_type="f8f6f4", acc_type="f32",
            a_vgprs=4, b_vgprs=4, acc_vgprs=4,
            element_bits=4,
            cbsz=4, blgp=4,
            is_mx=True,
        )
```

**MFMA emission change:** The `ctx.inst()` call must include:
- 2 extra operands: `vScale_A`, `vScale_B` (both same constant VGPR)
- Modifiers appended to the instruction line: `cbsz:4 blgp:4`
- For constant scale, `op_sel` and `op_sel_hi` can be omitted (all zeros)

The instruction assembly looks like:
```asm
v_mfma_scale_f32_16x16x128_f8f6f4 acc[0:3], v[A:A+3], v[B:B+3], acc[0:3], vScale, vScale cbsz:4 blgp:4
```

This is 6 operands instead of the normal 4. The `ctx.inst()` method
already accepts variadic operands, so no infrastructure change needed.

### 1.2 DataType and GemmProblem Changes

**File:** `problem.py`, class `DataType` (line ~35)

```python
class DataType(Enum):
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    MXFP4 = "mxfp4"   # NEW
```

**File:** `problem.py`, class `GemmProblem` (line ~230)

```python
@property
def element_bytes(self) -> float:  # Change return type from int to float
    return {
        DataType.F16: 2,
        DataType.BF16: 2,
        DataType.F32: 4,
        DataType.MXFP4: 0.5,
    }[self.dtype]

@property
def element_bits(self) -> int:
    return {DataType.F16: 16, DataType.BF16: 16, DataType.F32: 32, DataType.MXFP4: 4}[self.dtype]
```

Note: `element_bytes` becomes `float` (0.5 for FP4). All callers that
multiply by `element_bytes` and expect integer byte counts must use
`int(...)` or handle fractional bytes via bit-level arithmetic. In
practice, tile dimensions are always multiples of 2 so the products
are always integer.

### 1.3 Sub-Byte Element Handling

The fundamental change: 4-bit elements mean 0.5 bytes per element.
Every address calculation that does `offset * element_bytes` needs
to handle this correctly.

**Affected sites and how they change:**

#### 1.3.1 LDS Layout

**Current fp16:** `lds_half = (wg_m + wg_n) * unroll_k * 2` bytes.

**MXFP4:** `lds_half = (wg_m + wg_n) * unroll_k * 0.5` bytes.

For a 128x128 tile with DepthU=256 (2 * MI_K):
- fp16: `(128+128) * 32 * 2 = 16384` bytes per half
- mxfp4: `(128+128) * 256 * 0.5 = 32768` bytes per half

The LDS size is larger for MXFP4 because DepthU=256 (vs 64 for fp16).

For the initial MVP, use a 128x128x256 tile (same as TensileLite's
test config). This gives:
- A in LDS: 128 * 256 * 0.5 = 16384 bytes
- B in LDS: 128 * 256 * 0.5 = 16384 bytes
- lds_half = 32768 bytes
- lds_total (double-buffered) = 65536 bytes

**Byte offset calculations:**

Row-major LDS: `byte_offset = row * row_stride_bytes + col_byte_offset`

For fp16: `col_byte_offset = col_elem * 2`
For mxfp4: `col_byte_offset = col_elem / 2` (or equivalently `col_elem >> 1`)

The `_a_off()` and `_b_off()` functions in `dtl_interleaved.py` compute:
```python
def _a_off(mi, ki, tile, mfma, elem):
    row_start = mi * mfma.m
    row_stride = tile.unroll_k * elem   # bytes per row
    return row_start * row_stride + ki * mfma.k * elem
```

For mxfp4 with `elem=0.5`:
- `row_stride = 256 * 0.5 = 128` bytes per row
- `ki * mfma.k * elem = ki * 128 * 0.5 = ki * 64` bytes per ki step
- These are all integers, so the math works.

For `ds_read_b128` (16 bytes = 4 VGPRs): reads 32 FP4 elements packed
into 4 dwords. This matches `mfma.a_vgprs = 4`.

#### 1.3.2 Global Memory Layout

A is row-major: A[m, k], stride = K elements = K/2 bytes.
B is column-major (trans_b): B[n, k], stride = K elements = K/2 bytes.

The SRD base calculation:
```python
wg_offset_bytes = wg_id * wg_tile * K * element_bytes
                = wg_id * 128 * K * 0.5
```

**Per-thread DTL offset:**
```python
# threads_per_row = unroll_k_bytes / 16  (each thread loads 16 bytes)
# For mxfp4: unroll_k_bytes = 256 * 0.5 = 128 bytes -> 8 threads/row
# For fp16: unroll_k_bytes = 64 * 2 = 128 bytes -> 8 threads/row
# Same number of threads_per_row! (both 128 bytes per row in LDS)

v_dtl_off = thread_row * K_bytes + thread_col_group * 16
```

The DTL `buffer_load_dwordx4` loads 16 bytes (32 FP4 elements) per lane
group. The calculation `threads_per_row = unroll_k // 8` must change to:
```python
unroll_k_bytes = int(tile.unroll_k * mfma.element_bytes)  # 128 for both fp16 and mxfp4
threads_per_row = unroll_k_bytes // 16  # 8 for both
```

Currently the code uses `threads_per_row = tile.unroll_k // 8` which
assumes 2 bytes/elem and 16 bytes/load. For mxfp4 with unroll_k=256,
this gives `256 // 8 = 32`, which is wrong. The fix:

```python
# Replace throughout dtl_interleaved.py and dtl_partitioned.py:
# OLD: threads_per_row = tile.unroll_k // 8
# NEW:
elem_bytes = mfma.element_bytes  # or problem.element_bytes
unroll_k_bytes = int(tile.unroll_k * elem_bytes)
threads_per_row = unroll_k_bytes // 16
```

#### 1.3.3 K-stride for SRD Advance

Current: `k_stride = tile.unroll_k * elem` (used to advance SRD each
K-loop iteration).

For mxfp4: `k_stride = 256 * 0.5 = 128` bytes. This is correct as-is
since `elem = element_bytes = 0.5`.

#### 1.3.4 ds_read Offsets

`ds_read_b128` reads 16 bytes = 4 VGPRs. For mxfp4, that's 32 elements.
The MFMA needs MI_K=128 elements per operand.

Operand layout for `v_mfma_scale_f32_16x16x128_f8f6f4`:
- A operand: 4 VGPRs (v[A:A+3]). Each VGPR holds 8 FP4 elements (4 bytes).
  Total: 32 FP4 elements per lane, 2048 across 64 lanes.
  Matrix is 16 x 128 = 2048 elements. Consistent.
- B operand: same shape (4 VGPRs).

So `ds_read_b128` (16 bytes) per lane reads exactly one MFMA operand's
worth of data. This is the same as fp16 (`ds_read_b128` reads 4 VGPRs =
8 fp16 elements, matching MI_K=32 / 4 = 8 per lane). The ds_read width
is consistent.

**Key insight:** The ds_read granularity is always `mfma.a_vgprs` dwords,
which is 4 for both fp16 and mxfp4. The offset calculation changes
because the LDS row stride is different (more K-elements packed into
the same number of bytes).

#### 1.3.5 LDS Read Address Computation

Current (fp16):
```python
# lane_row = lane_id % 16
# lane_k = (lane_id / 16) * k_per_group
# base = wave_m * m_per_wave + lane_row
# rd_a = base * unroll_k * elem + lane_k * elem + mi_offset + ki_offset
```

For mxfp4, `elem = 0.5`, so:
- `base * unroll_k * 0.5 = base * 128` bytes (for unroll_k=256)
- `lane_k`: For fp16 k_per_group = 32/4 = 8. For mxfp4 k_per_group = 128/4 = 32.
  But 32 FP4 elements = 16 bytes. The ds_read offset is in bytes.

The key change in the LDS read setup is using `mfma.element_bytes`
instead of `problem.element_bytes` for the shift, and adjusting
`k_per_group` calculation.

### 1.4 Register Allocation

**File:** `asm_emitter.py`, function `alloc_registers_dtl()` (line ~140)

New registers for MXFP4:

```python
# Constant scale VGPR (1 VGPR, initialized to 0x7F7F7F7F)
if tile.mfma.is_mx:
    ctx.alloc_vgpr_permanent(1, "v_mxscale")
```

In the setup phase, emit:
```asm
v_mov_b32 v_mxscale, 0x7F7F7F7F    // E8M0 scale = 1.0 (all bytes)
```

Accumulator count stays the same formula:
`acc_total = mr * nr * mfma.acc_vgprs = 4 * 4 * 4 = 64` (for 128x128 tile)

A/B operand VGPRs: `a_vgprs = b_vgprs = 4` (same as fp16 with K=32).

### 1.5 MFMA Emission

**File:** `dtl_partitioned.py`, the `_mk_mfma` closure (line ~85)

Current emission:
```python
ctx.inst(
    mfma.instruction_name,
    ctx.areg("acc_C", acc_off_, acc_per_),
    ctx.vreg(a_names[(buf_, ki_)], 0, av),
    ctx.vreg(b_names[(ni_, ki_)], 0, bv),
    ctx.areg("acc_C", acc_off_, acc_per_),
    comment=f"MFMA m{mi_}_n{ni_}_k{ki_}")
```

New emission for MXFP4:
```python
if mfma.is_mx:
    ctx.inst(
        mfma.instruction_name,
        ctx.areg("acc_C", acc_off_, acc_per_),
        ctx.vreg(a_names[(buf_, ki_)], 0, av),
        ctx.vreg(b_names[(ni_, ki_)], 0, bv),
        ctx.areg("acc_C", acc_off_, acc_per_),
        ctx.vreg("v_mxscale"),      # scale A
        ctx.vreg("v_mxscale"),      # scale B
        f"cbsz:{mfma.cbsz} blgp:{mfma.blgp}",  # modifiers
        comment=f"MFMA m{mi_}_n{ni_}_k{ki_}")
else:
    # existing fp16 path
    ctx.inst(...)
```

Note: The `cbsz:4 blgp:4` modifiers may need to go on the same line
as the last operand rather than as a separate operand, depending on
assembler syntax. Test with:
```asm
v_mfma_scale_f32_16x16x128_f8f6f4 acc[0:3], v[0:3], v[4:7], acc[0:3], v8, v8 cbsz:4 blgp:4
```

### 1.6 Tile Geometry

**MXFP4 tile config (MVP):**

```python
GemmTiling.high_perf(
    wg_m=128, wg_n=128, unroll_k=256,
    waves_m=2, waves_n=2,
    mfma=MfmaConfig.mxfp4_16x16x128(),
)
```

Derived values:
- `m_per_wave = 64`, `n_per_wave = 64`
- `mr = 64/16 = 4`, `nr = 64/16 = 4`
- `ki = 256/128 = 2`
- `total_mfma = 4 * 4 * 2 = 32` per wave per K-unroll

### 1.7 DTL Load Changes

**File:** `dtl_interleaved.py`, functions `_emit_dtl_loads_a/b` (line ~273)

The DTL `buffer_load_dwordx4` loads 16 bytes per lane. For mxfp4:
- 16 bytes = 32 FP4 elements per lane
- unroll_k = 256 -> 256 * 0.5 = 128 bytes per row
- threads_per_row = 128 / 16 = 8
- rows_per_load = 256 / 8 = 32 (for block_size=256)
- num_loads_a = 128 / 32 = 4
- num_loads_b = 128 / 32 = 4

vs fp16 (256x256x64):
- threads_per_row = 64*2/16 = 8
- rows_per_load = 256/8 = 32
- num_loads_a = 256/32 = 8

The `threads_per_row` calculation MUST change from:
```python
threads_per_row = tile.unroll_k // 8   # WRONG for mxfp4
```
to:
```python
unroll_k_bytes = int(tile.unroll_k * element_bytes)
threads_per_row = unroll_k_bytes // 16
```

This affects: `dtl_interleaved.py` (lines 56, 60, 186, 266, 279, 290,
303), `dtl_partitioned.py` (line 42), and `kernel_pipeline.py` (line
~136 in `emit()`).

### 1.8 Store Epilogue

The output is fp32 accumulators converted to the output data type. For
MXFP4 GEMM, the output is typically BF16 or F16 (not FP4). The store
path is controlled by `DestDataType`. For the MVP, output as fp16
(same as current path). The accumulator -> store conversion is unchanged.

### 1.9 Launcher Changes

**File:** `launcher.py`

The launcher allocates input matrices with `np.random.randn().astype(np.float16)`.
For mxfp4, input data needs to be packed as 4-bit values. NumPy doesn't
have a native FP4 type.

**MVP approach:** Generate random fp16 data, quantize to FP4 range, pack
into uint8 (2 elements per byte). The launcher needs:
- A helper to pack/quantize fp16 -> mxfp4 (or use random uint8 data and
  accept approximate correctness for now)
- Buffer allocation: `M * K / 2` bytes for A, `N * K / 2` bytes for B
- Reference computation for correctness: unpack FP4 -> fp32, multiply,
  compare with GPU output

For initial testing, use small values that are exactly representable in
FP4 to get exact matches.

### 1.10 File-by-File Change List (Phase 1)

#### `problem.py`
- `DataType`: Add `MXFP4 = "mxfp4"`
- `MfmaConfig`: Add fields `element_bits`, `cbsz`, `blgp`, `is_mx`.
  Add `element_bytes` property. Add `mxfp4_16x16x128()` factory.
  Update `instruction_name` for MX format.
- `GemmProblem.element_bytes`: Return `float`, add MXFP4 case (0.5).
  Add `element_bits` property.
- `GemmProblem.validate()`: Allow MXFP4 dtype with mxfp4 MFMA.

#### `tiling.py`
- `GemmTiling.high_perf()`: Accept optional `mfma` parameter for mxfp4.
  Already does -- no change needed.
- Potentially add `GemmTiling.mxfp4()` convenience constructor.

#### `asm_emitter.py`
- `alloc_registers_dtl()`: Allocate `v_mxscale` (1 VGPR) when
  `tile.mfma.is_mx`.

#### `dtl_interleaved.py`
- `_a_off()`, `_b_off()`: Change `elem` parameter handling. These
  already multiply by `elem` which will be 0.5. Cast result to `int`.
- `phase_dtl_interleaved_setup()`:
  - `threads_per_row`: Fix to use byte-based calculation.
  - `k_per_group`: Fix for MI_K=128.
  - LDS read address: adjust shift for sub-byte elements.
  - Add `v_mov_b32 v_mxscale, 0x7F7F7F7F` in setup.
- `_emit_dtl_loads_a/b()`: Fix `threads_per_row`, `rows_per_load`,
  `lds_data_per_load` calculations for sub-byte elements.

#### `dtl_partitioned.py`
- `phase_dtl_partitioned_k_loop()`:
  - Fix `threads_per_row` (line 42).
  - `lds_half`: Use `int(...)` for sub-byte products.
  - `_mk_mfma` closure: Branch on `mfma.is_mx` to emit MX instruction
    with scale operands and cbsz/blgp modifiers.

#### `kernel_pipeline.py`
- `GemmKernel.emit()`: Fix `lds_half` calculation for sub-byte elements.
  Use `int(tile.wg_m * tile.unroll_k * elem)` etc.

#### `launcher.py`
- Add MXFP4 data packing/unpacking helpers.
- Adjust buffer size calculations for 0.5 bytes/element.
- Reference computation with FP4 quantization.

#### `addressing.py`
- `_lds_offset_b`: `int(tile.wg_m * tile.unroll_k * self._elem)`.
- All `* self._elem` products: ensure integer result.

### 1.11 Test Plan (Phase 1)

1. **Assembly generation test:** Generate MXFP4 kernel, verify `v_mfma_scale`
   instruction appears with correct operands and modifiers.
2. **Small correctness test:** 16x16x128 (single MFMA), packed FP4 input
   with known values, verify output matches CPU reference.
3. **Tile correctness test:** 128x128x256, random quantized FP4 input,
   verify within tolerance of CPU reference.
4. **Performance test:** 4096x4096x4096 MXFP4 GEMM, compare with
   hipBLASLt's MXFP4 kernel.

### 1.12 Implementation Order

1. `MfmaConfig.mxfp4_16x16x128()` + `instruction_name` changes
2. `DataType.MXFP4` + `GemmProblem` changes
3. Fix `threads_per_row` everywhere (the critical sub-byte fix)
4. Register allocation (`v_mxscale`)
5. MFMA emission with scale operands
6. LDS size calculations
7. Launcher data packing
8. End-to-end correctness test

---

## Phase 2: Real Scale Loading (Architectural Notes)

Phase 2 adds the full scale loading pipeline. Only pursue this after
Phase 1 passes correctness tests.

### 2.1 Kernel Arguments

New kernel arguments (appended to existing 36 bytes):

| Offset | Size | Name | Description |
|--------|------|------|-------------|
| 36 | 8 | ptr_scale_A | Scale A tensor pointer |
| 44 | 8 | ptr_scale_B | Scale B tensor pointer |
| 52 | 4 | stride_scale_A | Scale A stride (bytes) |
| 56 | 4 | stride_scale_B | Scale B stride (bytes) |

Total kernarg size: 60 bytes (up from 36).

New SGPRs:
- `s_srd_scale_a[0:3]` (4 SGPRs) -- Scale A buffer resource descriptor
- `s_srd_scale_b[0:3]` (4 SGPRs) -- Scale B buffer resource descriptor
- `s_lds_wr_scale_a` (1 SGPR) -- LDS write base for scale A
- `s_lds_wr_scale_b` (1 SGPR) -- LDS write base for scale B

### 2.2 LDS Layout Extension

```
[ DataA | DataB | ScaleA | ScaleB ]  (double-buffered)
```

Scale region sizing (per half-buffer, for 128x128x256 tile):
- Scale A: `wg_m * (unroll_k / mx_block) * 1 byte`
  = 128 * (256/32) * 1 = 1024 bytes
- Scale B: same = 1024 bytes
- Total scale per half: 2048 bytes
- lds_half = data_half + scale_half = 32768 + 2048 = 34816 bytes
- lds_total = 69632 bytes

### 2.3 Scale Data Movement

Three stages, matching TensileLite:

**Stage 1: Global -> LDS (DTL)**
```asm
s_mov_b32 m0, s_lds_wr_scale_a
buffer_load_dwordx4 v_dtl_off_scale_a, s_srd_scale_a, 0, offen offset:0, lds
```
One B128 DTL load per wave suffices for most configs (1024 bytes / 4 waves
/ 16 bytes = 16 loads, but with 64 lanes per wave, each load covers
64*16 = 1024 bytes, so 1 load per wave).

**Stage 2: LDS -> VGPRs**
```asm
ds_read_b32 vScale_A, v_lds_rd_scale_a offset:X  // 4 bytes = 4 E8M0 scales
```
One `ds_read_b32` per scale group. Each VGPR holds 4 scale bytes,
serving 4 MFMA invocations via `op_sel`/`op_sel_hi` byte selection.

**Stage 3: VGPR -> MFMA**
```asm
v_mfma_scale_f32_16x16x128_f8f6f4 ..., vScale_A, vScale_B cbsz:4 blgp:4 op_sel:[X,Y] op_sel_hi:[Z,W]
```
Byte selection:
- `sAsel = (mma_row % 2) + 2 * (subIterK % 2)`
- `op_sel = [sAsel % 2, sBsel % 2]`
- `op_sel_hi = [(sAsel >> 1) % 2, (sBsel >> 1) % 2]`

### 2.4 Scale VGPR Management

Scale VGPRs are double-buffered to overlap MFMA reads with ds_read writes.

Per-buffer allocation: `ceil(localMMATileGrid[0] / 2)` per A and B.
For 128x128/2x2 WG: `ceil(4/2) = 2` per tensor, 4 total.
Double-buffered: 8 scale VGPRs total.

New VGPR allocations:
```python
ctx.alloc_vgpr_permanent(2, "v_scale_a_0")   # buffer 0
ctx.alloc_vgpr_permanent(2, "v_scale_a_1")   # buffer 1
ctx.alloc_vgpr_permanent(2, "v_scale_b_0")
ctx.alloc_vgpr_permanent(2, "v_scale_b_1")
```

### 2.5 Scheduler Integration

New `ModuleKind` values:
```python
class ModuleKind(Enum):
    ...
    SCALE_GR = auto()    # DTL load for scale data
    SCALE_LR = auto()    # ds_read for scale data
    SCALE_SWAP = auto()  # scale VGPR buffer swap
```

Scale loads are emitted at `subIterK==0` of each partition. The
`MainloopScheduler.build_modules()` adds SCALE_GR and SCALE_LR modules
alongside existing GR and LR modules.

Waitcnt tracking: scale loads use `lgkmcnt` (same counter as data
ds_reads). The scheduler must account for inflight scale ds_reads
when computing `lgkmcnt` values.

### 2.6 PartitionPlan Changes

The `VGPRTileAllocator` needs a third pool for scale VGPRs (or extend
A/B pools with scale tile IDs). Scale VGPR tiles are indexed differently:
`gid = subtileIdx // 2` (one scale VGPR covers 2 adjacent subtiles).

### 2.7 SRD Advance

Scale SRDs advance each K-loop iteration:
```asm
s_add_u32 s_srd_scale_a[0], s_srd_scale_a[0], scale_k_stride
s_addc_u32 s_srd_scale_a[1], s_srd_scale_a[1], 0
```

Where `scale_k_stride = unroll_k / mx_block * bpe_scale = 256/32*1 = 8`
bytes per K-unroll iteration.

### 2.8 Estimated Complexity

| Component | Lines | Risk |
|-----------|-------|------|
| MFMA with real scales (op_sel) | ~50 | Low |
| Scale kernel args + SRD setup | ~80 | Low |
| Scale LDS layout | ~50 | Low |
| Scale DTL loads | ~100 | Medium |
| Scale ds_reads | ~80 | Medium |
| Scale VGPR double-buffering | ~100 | Medium |
| Scheduler integration | ~200 | High |
| Launcher (scale tensor alloc) | ~80 | Low |
| **Total** | **~740** | |

The scheduler integration is highest risk because it requires correctly
interleaving scale loads with data loads and MFMAs, tracking two sets
of inflight counters, and coordinating scale VGPR buffer swaps with
data buffer swaps.
