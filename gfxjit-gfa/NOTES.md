# gfxjit MXFP6 kernel dispatched through hipBLASLt's GFA path

Status: **working end to end.** Selected by name through `hipblaslt-bench --algo_method heuristic`,
passes hipBLASLt's own `--verify_strict`, and is faster than rocKE on the same path.

## Result

Selected solution, from `--print_kernel_info`:

    --Solution name:  gfxjit_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1

Correctness: `--verify_strict` exit code 0, norm_error 3.99e-06 against atol 1e-06 / rtol 0.01,
no failed checks.

TFLOP/s at 4096x4096x65536, GPU 7, same box and session, 7 runs each, `-i 20 -j 20`:

    gfxjit (this kernel)  4136 4166 4181 4184 4187 4211 4280   median 4184  max 4280
    rocKE                 3518 4056 4145 4148 4148 4179 4189   median 4148  max 4189

+0.9 % median, +2.2 % max over rocKE. (rocKE's two low runs are the bimodal contention artifact
documented in the standalone kernel's header; excluding them does not change its median.)

Both arms sit well below their standalone harness numbers -- ours 4336, rocKE's own package
reports 4529-4650 warm -- so that gap belongs to the hipblaslt-bench measurement path, not to this
integration. It is common to both arms, which is what makes the comparison above the meaningful one.

## The layout, and the asymmetry that bit

hipBLASLt's `--swizzleA/--swizzleB` for FP6 (`HIPBLASLT_ORDER_COL256_R128_EXT`) and `--scaleA/B
1018` (`HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT`) live in
`projects/hipblaslt/clients/common/include/mx_tile128x4.hpp` and are byte-for-byte rocKE's
`pack_operand_c0c1` / `pack_scales_tile`. Both are public API with published byte formulas
(`hipblaslt.h:394`, order enum 104), so the kernel reads their layout rather than changing it.

Operand image, per (256-row block, 128-K step), 24576 B, tile `t` (0..15) and lane `l`:

    c0: low 16 B at         16 * (64 t + l)
    c1: high 8 B at 16384 +  8 * (64 t + l)

Scale image, per (block, kstep), 1024 B: byte `512*(t/8) + 8*l + (t%8)`.

**The two images have OPPOSITE outer orderings, and this is easy to miss:**

    tiledGroupLowOffset : tile = (row / 256) * (K / 128) + kGroup / 4     <- row-block major
    shuffledScaleOffset : tile = (kBlock / 4) * (rows / 128) + row / 128  <- K-STEP major

So a 256-row block's 1 KiB of scales sits at `kstep * (rows/128) * 512 + block * 1024`, and the
scale descriptor's K stride is `rows * 4` bytes, not 1024. The operand descriptor keeps its
`block * ksteps * 24576` base and 24576 stride.

This cost a full build-and-measure cycle. It is invisible at one 256-row block, which is exactly
what the standalone harness validates by default, so it passed locally and failed in hipBLASLt
with norm_error 7.29. **Validate any layout change at a multi-block shape** (`-M 512 -N 512`), not
just the default. `-M 768` is not a usable check: a non-power-of-two block count breaks the XCD
workgroup mapping's mask, which fails identically on the unmodified kernel.

## The ABI

GFA binds by semantic, so `custom.config` declares whatever the kernel's own kernarg order is --
rocKE declares D *last*, the gfxjit reference declares it first. Ours is `(c, a, b, sa, sb)`, hence

    args: [ AddressD, AddressB, AddressA, AddressMXScaleB, AddressMXScaleA ]

The A/B exchange is required, not cosmetic: the kernel writes C row-major [M, N], which is the
column-major D hipBLASLt wants only if the two operands are swapped. rocKE does the same and says
so in its own config comment.

The kernel also has to use the user-SGPR setup both working GFA kernels use -- `user_sgpr_count 2`,
kernarg pointer at `s[0:1]`, workgroup id at `s2`, no hidden args. That is the `GFXJIT_GFA_ABI`
build flag in `mxfp6_gfa.cpp`; the default build keeps the HIP launch path's hidden args.

## Producing the .s

gfxjit has **no assembly emitter** -- it writes binary ELF only. `make_s.py` recovers the text with
llvm-objdump and pairs it with a hand-written `.amdhsa_kernel` block and the metadata. The
reconstruction is verified, not trusted:

* assembled and linked independently, then loaded into the standalone harness in place of gfxjit's
  own object: PASS against the CPU reference, same performance band;
* `.text` compared byte for byte against gfxjit's object: the only differences are the `vdata`
  field of the LDS-DMA `buffer_load`s, which llvm-objdump omits because the LDS form ignores it.
  Both proven GFA kernels use that same omitted form.

`Tensile.ValidateMetadata` cannot run outside a built tree (it imports `rocisa`, a build artifact);
`check_config.py` replicates its exact field checks and was run against the rocKE kernel as a
control. The build runs the real validator via `Toolchain.Assembly.validateCustomKernelMetadataAtBuild`.

## Selection

Dropping the `.s` into `CustomKernels/` is not sufficient -- the heuristic picks from a logic file.
rocKE is reached through

    Logic/asm_full/gfx950/gfx950_id75a3/Equality/gfx950_Cijk_Alik_Bljk_F6BS_MXAE8B32_MXBE8B32_STA_STB_H_UserArgs.yaml

whose `ExactLogic` maps `[4096, 4096, 1, 65536]` to a solution index. Ours is added as
`SolutionIndex: 2` and that entry repointed to it; rocKE stays as index 0, so changing that one
number switches the arms back for a like-for-like A/B.

## The gfxjit reference .s is not a usable model

`/home/kerrwang/gfxjit_..._preswz.s` reads the PRIVATE paired layout (`v_mul_u32_u24 v2, 48, v1`,
wave base `wm * 12288`), not c0c1, and its `custom.config` has no `SwizzleTensorA/B`, no
`MXScaleFormat`, and none of the `AssertSize`/`AssertStride`/`AssertAlphaValue` pins. It could not
be selected by the flags above and would compute garbage if it were. Model on rocKE's config.

## Reproducing

    cd <this worktree>/projects/hipblaslt
    python3 -m venv .venv && .venv/bin/pip install -r tensilelite/requirements.txt
    bash .../gfa/configure.sh && bash .../gfa/build.sh      # ~9 min at -j8
    bash .../gfa/verify.sh                                  # correctness
    bash .../gfa/perf.sh                                    # TFLOP/s, ours
    bash .../gfa/perf_rocke.sh                              # TFLOP/s, rocKE control

Standalone kernel: `gfxjit-gfa/mxfp6_gfa.cpp`, built with
`hipcc -std=c++23 -I<gfxjit> -DGFXJIT_USE_OPCODE_CACHE=1 -w`, add `-DGFXJIT_GFA_ABI` for the
code object the `.s` is generated from.

## Traps recorded

* `fp6_pack32` ORs into its destination and never clears it -- a scratch `uint8_t frag[24]` needs
  `= {}`. Without it the swizzled image is structured garbage and the failure imitates a layout bug.
* `s36-s51` are the four buffer descriptors, so `s50/s51` are the scale-B descriptor, not scratch.
* The operand and scale images have opposite outer orderings (above).
