# gfxjit MXFP6 kernel -> hipBLASLt GFA: work in progress

## Layout analysis (done, conclusive)

hipBLASLt's `--swizzleA/--swizzleB` for FP6 (`HIPBLASLT_ORDER_COL256_R128_EXT`) and
`--scaleA/B 1018` (`HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT`) are implemented in
`projects/hipblaslt/clients/common/include/mx_tile128x4.hpp` and are byte-for-byte rocKE's
`pack_operand_c0c1` / `pack_scales_tile`. Both are also pinned as public API in
`library/include/hipblaslt/hipblaslt.h` (scale byte formula at line 394, order enum 104).

Per (256-row block, 128-K step) both hipBLASLt's image and the gfxjit kernel's private image are
24576 B with the *same* block index `(row/256)*(K/128) + kstep` and the *same* 24-byte lane
fragments. They differ only in placement inside the block, with tile `t` (0..15) and lane `l`:

    gfxjit private : 24 B contiguous at 3072*(t/2) + 48*l + 24*(t%2)
    hipBLASLt c0c1 : low 16 B at 16*(64t + l); high 8 B at 16384 + 8*(64t + l)

Scales, per (block, kstep), 1024 B:

    gfxjit private : byte 16*l + t
    hipBLASLt      : byte 512*(t/8) + 8*l + (t%8)

Both are pure permutations. Consequences:

* The direct-to-LDS stream is unchanged -- both layouts are block-contiguous with identical block
  ordering, so the 14 DTLs, LDS budget, barriers and grid all stay as they are.
* The scale read stays exactly one `ds_read_b64` per operand: a wave's 8 tiles (t = 8*wm + 0..7)
  are the 8 contiguous bytes at `512*wm + 8*l`. Only the address changes.
* The fragment read changes class: the private image allows 3 x `ds_read_b128` per tile PAIR
  (48 B contiguous per lane, 24 reads per K-step); c0c1 forces 1 x `ds_read_b128` (c0 -> 4 VGPRs)
  + 1 x `ds_read_b64` (c1 -> 2 VGPRs) per tile, 32 + 2 = 34 reads per K-step. This is exactly
  rocKE's own c0c1p -> c0c1 step; rocKE ships c0c1 in its fastest kernel (v4-v6).

Conclusion: consume hipBLASLt's layout. Do not change the host shuffle -- it is public API with a
byte-exact published formula and rocKE depends on it.

## Implementation state

`mxfp6_gfa.cpp` is the gfxjit kernel rewritten to consume c0c1 + scale layout 1018:

* host `swizzle_ab` emits c0c1, `swizzle_sc` emits the 1018 scale order;
* new per-stage c1 LDS address registers (`vA1addr_`/`vB1addr_`, v40-v45) alongside the existing
  c0 ones, bases in s52/s53 (NB: s36-s51 are the four buffer descriptors, s58/s59 are fine but
  s50/s51 are *not* -- they are the scale-B descriptor);
* `ds_units` issues b128(c0) + b64(c1) per tile into the 6 consecutive VGPRs the MFMA wants.

**It does not validate yet.** Status of the investigation:

Verified correct:
* host image <-> kernel read-address mapping, by `chk.cpp` (standalone, reimplements both sides
  and round-trips every (kstep, wm, tile, lane) fragment) -- reports OK;
* the generated addresses, by disassembly: c0 base `wm*8192 + lane*16` + `t*1024`, c1 base
  `16384 + wm*4096 + lane*8` + `t*512`, all three stage bases, all within the right LDS regions;
* the MFMA encodings are bit-identical to the working kernel's (`v[48:53]`, `v[96:101]`, ...);
* `buffer_load_dwordx4 ... lds` is verbatim (global image == LDS image), proved by feeding the
  known-good original kernel a dword-planar-permuted image: it fails, as it must under verbatim.

Ruled out:
* not a scale-layout problem (fails identically with the old scale layout, and with all scales
  pinned to 1.0);
* not the c1 read (fails identically with the c1 halves replaced by `v_mov 0`);
* not `ds_read_b128` VGPR alignment (fails identically with the b128 split into two b64);
* not a race (fails identically with `s_waitcnt lgkmcnt(0)` after every single operand read);
* not the opcode cache.

Symptom, from the structural probe (`mkprobe2.py`: A[row] = 1,2,3,4 by row%4, B all 1.0, scales
1.0; the same probe on the original kernel prints the exact expected 512/1024/1536/2048):
`C[m][0]` is a single constant (~18432) for every m, i.e. the A operand that reaches the MFMA is
the same for every row and every tile, and is far outside the data's value range, so the B operand
is garbage too. That is the behaviour of a read whose lane and tile components are not reaching
the LDS data -- but the addresses in the disassembly do carry both. The contradiction is not yet
resolved; the next step is a real LDS dump (store the operand VGPRs through the epilogue) rather
than more inference.
