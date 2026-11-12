# gfxjit MXFP6 kernel on hipBLASLt's operand layout

## Layout analysis

hipBLASLt's `--swizzleA/--swizzleB` for FP6 (`HIPBLASLT_ORDER_COL256_R128_EXT`) and
`--scaleA/B 1018` (`HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT`) are implemented in
`projects/hipblaslt/clients/common/include/mx_tile128x4.hpp` and are byte-for-byte rocKE's
`pack_operand_c0c1` / `pack_scales_tile`. Both are also pinned as public API in
`library/include/hipblaslt/hipblaslt.h` (scale byte formula at line 394, order enum 104), so the
host shuffle is not ours to change -- the kernel reads their layout.

Per (256-row block, 128-K step) both images are 24576 B with the *same* block index
`(row/256)*(K/128) + kstep` and the *same* 24-byte lane fragments. They differ only in placement
inside the block, with tile `t` (0..15) and lane `l`:

    gfxjit private : 24 B contiguous at 3072*(t/2) + 48*l + 24*(t%2)
    hipBLASLt c0c1 : low 16 B at 16*(64t + l); high 8 B at 16384 + 8*(64t + l)

Scales, per (block, kstep), 1024 B:

    gfxjit private : byte 16*l + t
    hipBLASLt      : byte 512*(t/8) + 8*l + (t%8)

Both are pure permutations, so:

* the direct-to-LDS stream is unchanged -- same 14 DTLs, same 153600 B of LDS, same barriers,
  same grid, because both layouts are block-contiguous with identical block ordering;
* the scale read stays exactly one `ds_read_b64` per operand: a wave's 8 tiles (t = 8*wm + 0..7)
  are the 8 contiguous bytes at `512*wm + 8*l`. Only the address changes;
* only the fragment read changes class: 24 `ds_read_b128` per K-step becomes 16 `b128` (c0 -> the
  low 4 of the operand's 6 VGPRs) + 16 `b64` (c1 -> the high 2), 34 LDS reads per K-step with the
  two scale reads. This is rocKE's own c0c1p -> c0c1 step, and rocKE ships c0c1 in its fastest
  kernel.

This two-region scheme is not new to this kernel: it is what `5ddfb36^` used before the paired
48-byte layout superseded it, and that commit's own measurements put the paired layout only ~20
TFLOP/s ahead. Measured here, the gap is smaller still (below).

## Measured, 4096x4096x65536, MI355X GPU 7

    paired private layout (baseline)  4332 4343 4366 4343 4370   median 4343
    c0c1 (this kernel)                4328 4333 4339 4354 4336 4334 4335   median 4336

About 0.16 % -- the hipBLASLt layout is essentially free. Correctness: PASS against the harness's
CPU reference (max rel err 0.0039, bf16 output rounding).

## What changed in the kernel

`mxfp6_gfa.cpp` against `Examples/mxfp6_gemm.cpp`:

* host `swizzle_ab` emits c0c1, `swizzle_sc` emits the 1018 scale order;
* `vA1addr_`/`vB1addr_` (v30-v35) hold the c1 plane address per stage alongside the c0 ones, with
  the wave bases in s58/s59; `vSAaddr_`/`vSBaddr_` move to v36/v39 and `vDummy_` to v44, which is
  the register map the two-region ancestor used;
* `ds_units` issues `b128`(c0) + `b64`(c1) per tile into the 6 consecutive VGPRs the MFMA wants.

Everything else -- XCD workgroup mapping, the walking-SRD direct-to-LDS fills, serpentine MFMA
order, the acquire protocol -- is HEAD's and is untouched.

## Two traps worth keeping

* **`fp6_pack32` ORs into its destination and never clears it.** Packing into a scratch
  `uint8_t frag[24]` requires `= {}`. Without it the swizzled image is garbage that still looks
  structured, and the failure presents as a layout bug: wrong-but-deterministic results, operands
  that decode outside the data's value range, and run-to-run variation from whatever the stack
  held. This cost the bulk of the debugging. The address algebra checked out the whole time
  precisely because the standalone checker zeroed its own buffer.
* **s36-s51 are the four buffer descriptors** (`make_desc(36/40/44/48, ...)`), so s50/s51 are the
  scale-B descriptor, not scratch. s52-s55 are free since the SRD walk replaced the soffsets.

## Next

Emit this as a standalone `.s` with a `custom.config` block, matching the GFA argument ABI
(`AddressD, AddressB, AddressA, AddressMXScaleB, AddressMXScaleA` -- B before A, unlike our
harness's c, a, b, sa, sb), drop it into `CustomKernels/`, and confirm by solution name that
hipblaslt-bench selects it.
