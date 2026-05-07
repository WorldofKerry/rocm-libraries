# ASM Comparison: Our Kernel vs TensileLite Subtile (develop)

Both kernels: 256x256 macro tile, DepthU=256, MXFP4, gfx950.
Subtile kernel generated from develop branch TensileLite with `UseSubtileImpl=1, PGR=2, StreamK=3, DirectToLds=1`.

## Main Loop Metrics

| Metric | Ours | Subtile | Delta |
|---|---|---|---|
| MFMAs per loop body | 128 | 256 (2 copies) | 2x |
| K consumed per body | 256 | 512 (2 x DepthU) | 2x |
| DTL data loads (A+B) | 16 | 36 | -- |
| Scale loads from VMEM | **32** buffer_load_dword | **0** | critical |
| Scale DTL to LDS | 0 | 4 (2 per copy) | -- |
| Scale ds_read from LDS | 0 | 16 (8 per copy) | -- |
| ds_read for data | 32 | 64 (32 per copy) | -- |
| s_barrier | 1 | 4 (2 per copy) | -- |
| s_waitcnt | **26** | **8** | 3.25x |
| MFMA / mem-op ratio | **1.60** | **2.21** | 1.38x |

## Resource Usage

| Resource | Ours | Subtile |
|---|---|---|
| VGPRs | 448 | 512 |
| AGPRs | 256 | 256 |
| SGPRs | 84 | 99 |
| LDS | 128 KB | 144 KB (includes scale LDS) |
| Total instructions | 1,782 | 19,474 |

## Scheduling Pattern

| Aspect | Ours | Subtile |
|---|---|---|
| Schedule style | Semi-interleaved: bulk reads then MFMAs with fine lgkmcnt | Deeply interleaved: 1 ds_read per 1-2 MFMAs |
| Scale handling | Individual buffer_load_dword from VMEM, 1 VGPR/scale | DTL to LDS, ds_read_b32 packed, op_sel byte select |
| LDS swizzle | Yes (row rotation) | No (linear offsets) |
| PGR structure | PGR=2 single copy per body | PGR=2 double copy (C0+C1) per body |
| DirectToVgpr | No | No |
| DirectToLds | Yes (A+B data only) | Yes (A+B data + scales) |

## Root Causes of 2x Gap (ordered by impact)

1. **32 VMEM scale loads per loop** -- We issue 32 `buffer_load_dword` for scales from global memory every iteration. Subtile loads scales via DTL to LDS (2 loads/copy) then reads 8 packed dwords via `ds_read_b32`. op_sel packs 4 scale bytes per dword so 8 reads serve all 256 MFMAs. Our 32 VMEM loads contend with the 16 data DTL loads, nearly tripling VMEM traffic.

2. **Suboptimal scheduling / excessive waitcnts** -- 26 `s_waitcnt` vs 8. We issue all reads upfront then start MFMAs with per-MFMA lgkmcnt decrements, creating a long dependency chain. Subtile interleaves reads between MFMAs so latency is hidden under MFMA execution.

3. **No double-copy unrolling** -- Subtile processes 2 DepthU chunks per loop body (C0+C1), amortizing loop overhead and enabling better overlap between compute and loads. We process 1 DepthU per body.

4. **No op_sel scale packing** -- We use 1 VGPR per scale value (32 VGPRs). Subtile packs 4 bytes/VGPR and uses op_sel/op_sel_hi to select, needing only ~8 scale VGPRs.

## TODO: Optimizations to Close the Gap

### P0: Scale loading via DTL+LDS+op_sel
- [ ] Allocate LDS region for scales (separate from data, ~1KB per buffer half)
- [ ] Load pre-swizzled scales via DTL (`buffer_load_dwordx4 ... lds`) to scale LDS region
- [ ] Read packed scale dwords from LDS via `ds_read_b32`
- [ ] Use `op_sel:[a,b]` and `op_sel_hi:[k,k]` on v_mfma_scale to select scale bytes
- [ ] Remove all 32 `buffer_load_dword` scale loads from the loop
- [ ] Verify with `MXScaleFormat: 1` (pre-swizzled) through TensileLite client
- **Expected impact**: Eliminates 32 VMEM loads, replaces with 2 DTL + 8 LDS reads. Largest single win.

### P1: Interleave ds_reads between MFMAs
- [ ] Instead of bulk-issuing all ds_reads upfront, issue each ds_read 1-2 MFMAs before its consumer
- [ ] Remove per-MFMA lgkmcnt waits; use a few coarse waits at partition boundaries
- [ ] Target: max 4 consecutive MFMAs without a memory op
- [ ] Reduce s_waitcnt count from 26 to ~8
- **Expected impact**: Hides LDS latency under MFMA execution. Significant IPC improvement.

### P2: Double-copy unrolling (PGR=2 C0+C1)
- [ ] Unroll loop body to process 2 DepthU chunks: C0 consumes buffer 0 while loading buffer 1, C1 consumes buffer 1 while loading buffer 0
- [ ] Each copy has its own barrier pair and scale load section
- [ ] Loop counter decrements by 2 per body; exit when counter == 2
- [ ] Drain epilog handles last 2 iterations without global loads
- **Expected impact**: 2x compute per loop body, better load/compute overlap, less loop overhead.

### P3: op_sel scale packing
- [ ] Pack 4 scale bytes per VGPR: byte[0]=m_even/k_lo, byte[1]=m_odd/k_lo, byte[2]=m_even/k_hi, byte[3]=m_odd/k_hi
- [ ] Use op_sel to index into A scale groups (even/odd M rows)
- [ ] Use op_sel_hi to index K-slice (k0 vs k1)
- [ ] Reduce scale VGPRs from 32 to ~8
- **Note**: This is coupled with P0 (DTL+LDS scales). Implement together.

### P4: Scale DTL double-buffering
- [ ] Allocate double-buffered scale LDS regions (like subtile: base at 0x10000/0x11000, XOR swap with 73728-byte offset)
- [ ] Alternate scale LDS write targets between C0 and C1 copies
- [ ] Issue scale DTL loads early in compute phase, read scale data after barrier
- **Note**: Required for P0+P2 to work together correctly.

### P5: Reduce loop overhead
- [ ] Pre-compute all LDS read offsets in prolog (already done for data, extend to scales)
- [ ] Use double-buffer address toggle via XOR (like subtile: `s_xor_b32 sgprLocalWriteBaseAddr, sgprLocalWriteBaseAddr, sgprSwap`)
- [ ] Minimize SRD pointer arithmetic in the hot loop

## Observations

### What we do better
- **LDS swizzling**: We use row-rotation swizzle for zero bank conflicts. Subtile uses linear offsets (potential bank conflicts on ds_read).
- **Code size**: 1,782 instructions vs 19,474. Our kernel is 10x smaller, faster to compile, better I-cache behavior.
- **Register pressure**: 448 VGPRs vs 512. Lower occupancy pressure.

### What subtile does better
- **Scale handling**: DTL+LDS+op_sel is strictly superior to individual VMEM loads.
- **Scheduling**: Deep interleaving hides latency. Our bulk-load pattern leaves MFMAs stalled.
- **Loop structure**: Double-copy unrolling doubles compute per body with minimal overhead.

### Not a factor
- Both use DirectToLDS (not DirectToVgpr) for A and B data.
- Both use 256 AGPRs for accumulators.
- Neither uses LDS swizzle instructions (`ds_swizzle`); swizzling is in address computation.
- StreamK (subtile uses SK3) affects grid mapping, not the inner loop. Not a factor in per-tile performance.

## Performance Numbers (4096x4096x4096, GFA tensilelite-client)

| Kernel | Time (us) |
|---|---|
| TensileLite subtile (scaleA=1001) | 42.2 |
| AITER/rocRoller (scaleA=3) | 44.7 |
| **Our 256x256** | **91.2** |
| Our 128x128 | 152.6 |
