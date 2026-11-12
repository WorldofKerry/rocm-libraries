// gfx950 MXFP6 (e2m3) GEMM with LDS staging, generated at runtime with gfxjit.
//
// Converted from the MXFP4 version; the structure is unchanged:
//
//   - 256x256x128 macrotile, 2x2 waves, MIWaveTile 8x8, wave64
//   - A/B/scales staged through LDS by direct-to-LDS buffer_load (m0 + "lds"
//     modifier), so there are no ds_write instructions at all
//   - multi-stage software pipeline with one s_barrier per K-step
//   - operands fed to the MFMA by 3 x ds_read_b128 per tile PAIR (12 VGPRs =
//     48 B = two 32-value fp6 fragments); the 8 per-tile scale bytes of a wave
//     arrive in a single ds_read_b64 and are selected per tile by op_sel
//   - f32 accumulation in AGPRs, bf16 output via v_cvt_pk_bf16_f32 +
//     global_store_short
//
// LDS budget per stage: A 24576 + B 24576 + scaleA 1024 + scaleB 1024
// = 51200 (0xc800), times 3 stages = 153600 B (fits the 160 KB limit).
//
// Swizzled layouts (built on the host, mirroring the LDS image exactly so a
// direct-to-LDS copy needs no address math):
//   A_sw[m_block][k_step] = 24576 B as [pair(8)][lane(64)][48B], a lane's 48 B
//                           holding tile 2p's 24 B then tile 2p+1's 24 B
//   S_sw[m_block][k_step] =  1024 B as [lane(64)][tile(16)] bytes
//
// Build:
//   hipcc -std=c++23 -I../gfxjit -DGFXJIT_USE_OPCODE_CACHE=1 mxfp6_gemm.cpp -o mxfp6
//
// MEASURED on one MI355X at 4096x4096x65536, bf16 out, random input data:
//
//   this kernel, MXFP6            4340 TFLOP/s   (4326-4361 over 8 runs)
//   gfxjit reference .s, MXFP6    4271
//   rocKE, MXFP6                  4141 median, 4185 max
//   hipBLASLt, MXFP4              3552
//   hipBLASLt, MXFP6              1220           (untuned fallback)
//
// Loop efficiency 82.6% against rocKE's 82.0% on the same ATT pipeline, with a
// 634,320-cycle main loop against its 639,276 for identical MFMA work.
// Caveats worth keeping with those numbers: the hipBLASLt and rocKE figures
// come from hipblaslt-bench and this one from the harness below, so the timing
// paths differ; and the MXFP4 comparison is cross-format, where FP4 moves two
// thirds of the bytes per FLOP that FP6 does.
//
// BENCHMARKING NOTE. At 4096x4096 the grid is exactly 16x16 = 256 workgroups on
// a 256-CU MI355X, one per CU, and the 8x8 wave tile pins occupancy to 1. The
// dispatch therefore has ZERO slack: if any single CU is momentarily occupied
// by another process, that workgroup cannot start until something retires, the
// dispatch needs a second wave, and wall time doubles. Results are bimodal
// rather than noisy -- a clean ~4260 band and a ~2150 band at almost exactly
// half, with nothing in between.
// Those 2x runs are a contention artifact and should be discarded, not
// averaged; averaging them is what makes this kernel look like it varies by
// 2900-4242. Note also that an idle-looking GPU is not a sufficient check:
// rocm-smi reported no other process on a device that still produced the mode
// in 9 of 30 runs. Thermal and power throttling were ruled out (43-59 C, peak
// sclk 2397 MHz), as was launch-loop descheduling (CPU pinning does not
// suppress it).

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "gfxjit.hpp"

// ----------------------------------------------------------------- geometry

// The 8x8 wave tile pins this to one workgroup per CU: 256 accumulator AGPRs
// and 153600 B of LDS leave no room for a second, so each SIMD runs a single
// wave and the software pipeline alone has to hide latency. That is a
// deliberate trade and it was measured. A 4x4 wave tile (128x128 macrotile)
// does reach 2 waves per SIMD and is worth +8% at equal tile size (2760 vs
// 2560 TFLOP/s), but the smaller macrotile doubles how many workgroups re-read
// each A and B block, and that costs 35%: 2760 against 4230 for this 8x8
// configuration. Occupancy is not the lever here; macrotile size is.

// Workgroup -> tile mapping. All 256 workgroups are resident at once, so only
// WHICH SET of tiles lands on each XCD matters, not the order. An XCD running
// an RB x CB patch streams RB A panels + CB B panels; with RB*CB = 32 fixed,
// min(RB+CB) = 12 at 4x8, against 18 for the naive linear map. That ratio is
// the whole effect and 4x8 is its optimum, so this lever is exhausted here.
constexpr int kXCDs = 8;
constexpr int kXCDRows = 4;  // RB
constexpr int kXCDCols = 8;  // CB

constexpr int kBM = 256, kBN = 256, kBK = 128;
constexpr int kMT = 8, kNT = 8;        // MFMA tiles per wave
constexpr int kWGM = 2, kWGN = 2;      // wave grid
constexpr int kThreads = kWGM * kWGN * 64;
constexpr int kTilesPerBlock = kBM / 16;              // 16
constexpr int kABytesPerStep = kBM * kBK * 6 / 8;     // 24576

// Operand image: hipBLASLt's HIPBLASLT_ORDER_COL256_R128_EXT ("c0c1"), so the
// stage image is byte-identical to what the library hands a GFA kernel and the
// direct-to-LDS copy stays a flat 24576 B stream. A fragment (tile t, lane l)
// is split across two regions of the stage:
//   c0: bytes 0..15  at        16 * (64 t + l)
//   c1: bytes 16..23 at 16384 + 8 * (64 t + l)
// so a lane reads one ds_read_b128 (c0 -> 4 VGPRs) + one ds_read_b64 (c1 -> 2
// VGPRs) per tile into the 6 consecutive VGPRs the MFMA wants: 34 ds_reads per
// K-step against 26 for the private [pair][lane][48 B] image this kernel used
// before. Both regions are conflict free (16- and 8-byte lane strides).
//
// Do NOT "simplify" this to a flat [tile][lane][24 B] image read as b128 + b64
// at +16. It validates, but the b128 lands 8-byte misaligned at a 24-byte lane
// stride: 6*lane mod 32 has gcd 2 with the bank count, so the 64 lanes cover
// only half the banks and the read serializes. That measured 1841 TFLOP/s.
constexpr int kC1Base = kBM * (kBK / 32) * 16;        // 16384
constexpr int kC0TileStride = 64 * 16;                // 1024
constexpr int kC1TileStride = 64 * 8;                 // 512
constexpr int kSBytesPerStep = kBM * (kBK / 32);      // 1024

// LDS offsets within one stage
constexpr int kLdsA = 0;
constexpr int kLdsB = kLdsA + kABytesPerStep;        // 0x6000
constexpr int kLdsSA = kLdsB + kABytesPerStep;       // 0xc000
constexpr int kLdsSB = kLdsSA + kSBytesPerStep;      // 0xc400
constexpr int kLdsStage = kLdsSB + kSBytesPerStep;   // 0xc800 = 51200

// Pipeline depth. 3 is the most that fits in LDS at fp6 operand width; the
// pre-barrier wait is vmcnt((kStages-2)*kLoadsPerStage) = vmcnt(14).
constexpr int kStages = 3;

// Per wave, per stage: A is 6144 B = 6 x buffer_load_dwordx4, same for B,
// plus one dword scale load each.
constexpr int kLoadsPerWaveAB = kABytesPerStep / 4 / 1024;  // 6

// The 6 A loads cannot share one m0 base: 6*1024 overflows the 12-bit
// instruction offset, so they are split into two groups of three.
constexpr int kLoadGroup = 3;
constexpr int kGroupBytes = kLoadGroup * 1024;  // 3072

struct GemmShape {
  int M, N, K;
  int k_steps() const { return K / kBK; }
  int blocks_m() const { return M / kBM; }
  int blocks_n() const { return N / kBN; }
};

static inline size_t ab_block_bytes(int ksteps) { return (size_t)ksteps * kABytesPerStep; }
static inline size_t s_block_bytes(int ksteps) { return (size_t)ksteps * kSBytesPerStep; }

// ---------------------------------------------------------------- generator

template <gfxjit::GFXArch arch>
class Mxfp6GemmLds : public gfxjit::GFXCodeGenerator<arch> {
public:
  GFXJIT_DECLARE_OPCODE_IN_SCOPE
  using typename gfxjit::GFXCodeGenerator<arch>::KerArg;
  using gfxjit::GFXCodeGenerator<arch>::alloc_label;
  using gfxjit::GFXCodeGenerator<arch>::get_arg_offset;
  using gfxjit::GFXCodeGenerator<arch>::label;
  using gfxjit::GFXCodeGenerator<arch>::m0_;
  using gfxjit::GFXCodeGenerator<arch>::null_;
  using gfxjit::GFXCodeGenerator<arch>::set_kernel_args;
  using gfxjit::GFXCodeGenerator<arch>::wcnt_;

  explicit Mxfp6GemmLds(const GemmShape &s)
      : gfxjit::GFXCodeGenerator<arch>(5, 0), s_(s) {
    using namespace gfxjit;
    using sgpr = gfxjit::operands::Sgpr;
    using vgpr = gfxjit::operands::Vgpr;
    using agpr = gfxjit::operands::Accvgpr;
    using gfxjit::operands::matrix_fmt;

    config_.use_hidden_args();
    config_.amdhsa_user_sgpr_private_segment_buffer(1);
    config_.amdhsa_user_sgpr_kernarg_segment_ptr(1);
    config_.uniform_work_group_size_ = 1;
    config_.amdhsa_group_segment_fixed_size(kStages * kLdsStage);

    set_kernel_args({
        KerArg(DataType::ptr),  // c  (bf16)
        KerArg(DataType::ptr),  // a  (swizzled)
        KerArg(DataType::ptr),  // b  (swizzled)
        KerArg(DataType::ptr),  // sa (swizzled)
        KerArg(DataType::ptr),  // sb (swizzled)
    });

    const int ksteps = s_.k_steps();

    // s[4:5] kernarg ptr, s6 workgroup_id_x
    s_load_dwordx2(sgpr(8).dw(2), sgpr(4).dw(2), get_arg_offset(0));   // c
    s_load_dwordx2(sgpr(10).dw(2), sgpr(4).dw(2), get_arg_offset(1));  // a
    s_load_dwordx2(sgpr(12).dw(2), sgpr(4).dw(2), get_arg_offset(2));  // b
    s_load_dwordx2(sgpr(14).dw(2), sgpr(4).dw(2), get_arg_offset(3));  // sa
    s_load_dwordx2(sgpr(16).dw(2), sgpr(4).dw(2), get_arg_offset(4));  // sb

    v_and_b32(vgpr(1), 63, vgpr(0));        // lane
    v_lshrrev_b32(vgpr(2), 6, vgpr(0));     // wave
    s_nop(4);
    v_readfirstlane_b32(sgpr(20), vgpr(2));
    s_nop(4);
    s_and_b32(sgpr(21), sgpr(20), kWGM - 1);   // wm
    s_lshr_b32(sgpr(22), sgpr(20), 1);         // wn

    // workgroup -> (block_m, block_n)
    emit_tile_of_wg();

    s_waitcnt(wcnt_.lgkmcnt(0));

    // ---- buffer descriptors (V#)
    const size_t ab_blk = ab_block_bytes(ksteps);
    const size_t s_blk = s_block_bytes(ksteps);
    // The DTL loads pass a literal 0 soffset: a buffer_load with an SGPR
    // soffset takes about 8 cycles to issue against the ~4 cycles between
    // MFMAs, so each one costs roughly 4 extra cycles. The descriptor base
    // walks K instead, advanced by SALUs that fit in the slots after an MFMA
    // issue. num_records therefore covers exactly one stage image, so the
    // valid window walks with the base.
    s_mul_i32(sgpr(30), sgpr(23), (int)ab_blk);
    make_desc(36, 10, 30, kABytesPerStep);
    s_mul_i32(sgpr(30), sgpr(24), (int)ab_blk);
    make_desc(40, 12, 30, kABytesPerStep);
    s_mul_i32(sgpr(30), sgpr(23), (int)s_blk);
    make_desc(44, 14, 30, kSBytesPerStep);
    s_mul_i32(sgpr(30), sgpr(24), (int)s_blk);
    make_desc(48, 16, 30, kSBytesPerStep);

    // per-wave LDS fill bases: A/B 6144 B per wave, scales 256 B per wave
    s_mul_i32(sgpr(32), sgpr(20), kABytesPerStep / 4);   // wave * 6144
    s_mul_i32(sgpr(33), sgpr(20), kSBytesPerStep / 4);   // wave * 256

    // ---- global voffsets for the direct-to-LDS copies (plain linear copy)
    v_lshlrev_b32(vgpr(3), 4, vgpr(1));                 // lane * 16
    v_add_u32(vgpr(3), sgpr(32), vgpr(3));              // + wave * 6144
    v_add_u32(vgpr(8), kGroupBytes, vgpr(3));           // second load group
    v_lshlrev_b32(vgpr(4), 2, vgpr(1));                 // lane * 4
    v_add_u32(vgpr(4), sgpr(33), vgpr(4));              // + wave * 256

    // ---- LDS read addresses, one set per stage
    // c0 plane: lane * 16, wave base wm * kMT * 1024
    v_lshlrev_b32(vgpr(5), 4, vgpr(1));                 // lane * 16
    s_mul_i32(sgpr(34), sgpr(21), kMT * kC0TileStride);    // wm * 8192
    s_mul_i32(sgpr(35), sgpr(22), kNT * kC0TileStride);
    // c1 plane: lane * 8, wave base kC1Base + wm * kMT * 512
    v_lshlrev_b32(vgpr(7), 3, vgpr(1));                 // lane * 8
    s_mul_i32(sgpr(52), sgpr(21), kMT * kC1TileStride);    // wm * 4096
    s_mul_i32(sgpr(53), sgpr(22), kNT * kC1TileStride);
    // scales: the wave's 8 tiles are the 8 contiguous bytes at 512*wm + 8*lane
    v_lshlrev_b32(vgpr(6), 3, vgpr(1));                 // lane * 8 (scales)
    s_mul_i32(sgpr(56), sgpr(21), kTilesPerBlock / 2 * 64);  // wm * 512
    s_mul_i32(sgpr(57), sgpr(22), kTilesPerBlock / 2 * 64);
    for (int st = 0; st < kStages; ++st) {
      const int base = st * kLdsStage;
      v_add_u32(vgpr(vAaddr_ + st), sgpr(34), vgpr(5));
      v_add_u32(vgpr(vAaddr_ + st), base + kLdsA, vgpr(vAaddr_ + st));
      v_add_u32(vgpr(vBaddr_ + st), sgpr(35), vgpr(5));
      v_add_u32(vgpr(vBaddr_ + st), base + kLdsB, vgpr(vBaddr_ + st));
      v_add_u32(vgpr(vA1addr_ + st), sgpr(52), vgpr(7));
      v_add_u32(vgpr(vA1addr_ + st), base + kLdsA + kC1Base, vgpr(vA1addr_ + st));
      v_add_u32(vgpr(vB1addr_ + st), sgpr(53), vgpr(7));
      v_add_u32(vgpr(vB1addr_ + st), base + kLdsB + kC1Base, vgpr(vB1addr_ + st));
      v_add_u32(vgpr(vSAaddr_ + st), sgpr(56), vgpr(6));
      v_add_u32(vgpr(vSAaddr_ + st), base + kLdsSA, vgpr(vSAaddr_ + st));
      v_add_u32(vgpr(vSBaddr_ + st), sgpr(57), vgpr(6));
      v_add_u32(vgpr(vSBaddr_ + st), base + kLdsSB, vgpr(vSBaddr_ + st));
    }

    for (int i = 0; i < kMT * kNT * 4; ++i) v_accvgpr_write_b32(agpr(i), 0);

    // One K-step's 34 ds_reads, as independently schedulable units, reading
    // LDS stage `st` into operand register set `r`. Each operand is 24 B:
    // b128 at the tile base, b64 at +16.
    auto ds_units = [&](int st, int r) {
      std::vector<std::function<void()>> u;
      u.push_back([=] { ds_read_b64(vgpr(vSA(r)).dw(2), vgpr(vSAaddr_ + st), 0); });
      u.push_back([=] { ds_read_b64(vgpr(vSB(r)).dw(2), vgpr(vSBaddr_ + st), 0); });
      // Per tile: b128 of the c0 plane into the low 4 of the operand's 6
      // VGPRs, b64 of the c1 plane into the high 2.
      for (int t = 0; t < kMT; ++t) {
        u.push_back([=] {
          ds_read_b128(vgpr(vA(r) + t * 6).dw(4), vgpr(vAaddr_ + st),
                       t * kC0TileStride);
        });
        u.push_back([=] {
          ds_read_b64(vgpr(vA(r) + t * 6 + 4).dw(2), vgpr(vA1addr_ + st),
                      t * kC1TileStride);
        });
        u.push_back([=] {
          ds_read_b128(vgpr(vB(r) + t * 6).dw(4), vgpr(vBaddr_ + st),
                       t * kC0TileStride);
        });
        u.push_back([=] {
          ds_read_b64(vgpr(vB(r) + t * 6 + 4).dw(2), vgpr(vB1addr_ + st),
                      t * kC1TileStride);
        });
      }
      return u;
    };

    // ---- prologue: fill all stages, then read stage 0 into register set 0
    for (int st = 0; st < kStages; ++st) fill_stage(st);
    s_waitcnt(wcnt_.vmcnt((kStages - 1) * kLoadsPerStage));
    s_barrier();
    for (auto &f : ds_units(0, 0)) f();

    // One K-step at pipeline phase `k`: LDS stage k%kStages, operand set k%2.
    // kStages is odd, so the phase only repeats every lcm(3,2)=6 K-steps;
    // the main loop is therefore unrolled by 6 and the remainder peeled.
    auto emit_kstep = [&](int k) {
      const int st = k % kStages, r = k & 1;
      const int nst = (k + 1) % kStages, nr = (k + 1) & 1;

      // Acquire. The vmcnt goes BEFORE the barrier and the lgkmcnt AFTER, as
      // three separate instructions: that defers the LDS drain past the
      // barrier so barrier arrival overlaps the still-outstanding ds_reads.
      //   vmcnt:      my fill of stage st+1 (issued a K-step ago) has landed,
      //               so after the barrier stage nst is readable
      //   lgkmcnt(0): my ds_reads of stage st are done, so stage st is free
      // Two further rearrangements of this acquire were measured and rejected,
      // both under 0.3% against ceilings that looked like 4% and 2.2%:
      //   - spreading the address-update SALUs further through the MFMA stream.
      //     They are already spread: the only SALU adjacencies left in the loop
      //     body are an m0 write and its own DTL load (which cannot separate),
      //     and the s_add_u32/s_addc_u32 descriptor pairs (which cannot, since
      //     everything around them clobbers SCC). The 4% SALU figure was never
      //     reachable because SALU was already hidden.
      //   - putting independent SALUs between the waits and the barrier. The
      //     barrier's ~28 cycles is wave synchronisation, not issue cost, so
      //     filling the slot ahead of it does not change when the last wave
      //     arrives.
      // lgkmcnt(0) is placed BEFORE the barrier, not after. Deferring it past
      // the barrier (as the reference does) lets a wave that has crossed the
      // barrier start refilling stage st while another wave's ds_reads of that
      // same stage are still outstanding. It is safe in practice only because
      // those reads were issued a full K-step earlier and the competing DTL
      // write must still traverse L2 to LDS, but it is not architecturally
      // guaranteed. It measured +8 TFLOP/s, inside the noise floor, so the
      // race is not worth carrying.
      s_waitcnt(wcnt_.vmcnt((kStages - 2) * kLoadsPerStage));
      s_waitcnt(wcnt_.lgkmcnt(0));
      s_barrier();

      // 48 memory units spread evenly through the 64 MFMAs: the 14 direct-to-
      // LDS loads refilling stage st first, then the 34 ds_reads of stage nst.
      // The 14 direct-to-LDS fills are spread evenly through the 34 ds_reads
      // rather than issued as a leading burst. Both orders were measured:
      // fills first 3665 TFLOP/s, ds_reads first 3711, round robin 4221. The
      // fills and the ds_reads contend for the same LDS write/read ports, so
      // bunching either one stalls the other.
      std::vector<std::function<void()>> ds = ds_units(nst, nr), fl = fill_units(st);

      // The 14 direct-to-LDS fills are merged proportionally into the 26
      // ds_reads and the single resulting stream is spread uniformly across
      // all 64 MFMAs. Fills and ds_reads contend for the same LDS ports, so
      // bunching either one stalls the other.
      //
      // The reference kernel instead front-loads its operand ds_reads into the
      // first ~24 of 64 MFMAs and spreads only the DTLs across the full step.
      // That was measured here and is worse at every front-load width tried:
      // 24 -> 4233, 38 -> 4235, 64 -> 4217, against 4267 for this uniform
      // merge. Emitting the two streams separately per MFMA slot rather than
      // merging them is worse again (4211). The difference from the reference
      // is most likely the 2x3 b128 per pair: an operand is not complete until
      // its third read lands, so spreading them is what keeps the LDS pipe fed.
      std::vector<std::function<void()>> units;
      size_t fi = 0;
      for (size_t i = 0; i < ds.size(); ++i) {
        while (fi * ds.size() < i * fl.size()) units.push_back(fl[fi++]);
        units.push_back(ds[i]);
      }
      while (fi < fl.size()) units.push_back(fl[fi++]);

      size_t next = 0;
      for (int t = 0; t < kMT * kNT; ++t) {
        const size_t due = ((size_t)t * units.size() + 63) / 64;
        while (next < due) units[next++]();
        // Serpentine: the inner direction flips on alternate outer rows, so
        // consecutive MFMAs share an operand at every row boundary instead of
        // jumping from (i,7) back to (i+1,0).
        const int i = t / kNT;
        const int j = (i & 1) ? kNT - 1 - t % kNT : t % kNT;
        const int acc = (i * kNT + j) * 4;
        // op_sel selects the tile's scale byte within the packed dword
        v_mfma_scale_f32_16x16x128_f8f6f4(
            agpr(acc).dw(4),
            vgpr(vA(r) + i * 6).dw(6).fmt(matrix_fmt::fp6).abid(1)
                .opsel(i & 1).opsel_hi((i >> 1) & 1),
            vgpr(vB(r) + j * 6).dw(6).fmt(matrix_fmt::fp6)
                .opsel(j & 1).opsel_hi((j >> 1) & 1),
            agpr(acc).dw(4).opsel_hi(0),
            vgpr(vSA(r) + i / 4), vgpr(vSB(r) + j / 4));
      }
      while (next < units.size()) units[next++]();
    };

    const int unroll = kStages * 2;  // 6
    const int iters = ksteps / unroll, tail = ksteps % unroll;
    if (iters > 0) {
      s_mov_b32(sgpr(31), iters);
      auto loop = alloc_label();
      label(loop);
      for (int k = 0; k < unroll; ++k) emit_kstep(k);
      s_sub_u32(sgpr(31), sgpr(31), 1);
      s_cmp_gt_i32(sgpr(31), 0);
      s_cbranch_scc1(loop);
    }
    // The loop leaves the pipeline phase at a multiple of 6, so the peeled
    // K-steps restart the phase sequence at 0.
    for (int k = 0; k < tail; ++k) emit_kstep(k);

    // ---------------------------------------------------------- epilogue
    s_waitcnt(wcnt_.lgkmcnt(0));
    for (int i = 0; i < 4; ++i) s_nop(15);

    // C is bf16, row-major. lane l, reg r -> row (l/16)*4+r, col l%16
    //
    // The 2-byte stores look wasteful and are not worth fixing; all three
    // routes were measured against this epilogue and none pays:
    //   - batching the v_accvgpr_reads to remove the s_nop:   0%
    //     The trace attributes 3,496 cycles of pure stall to that s_nop, 34%
    //     of a 10,240-cycle epilogue, and removing it changes nothing. The
    //     stall is real and simply not on the critical path.
    //   - widening to 32 x global_store_dwordx4 (8x fewer):  +0.35%, in noise
    //   - deleting the stores entirely:                      +1.7%
    // The last figure is not headroom. C is 4096x4096x2 = 33.5 MB that has to
    // reach memory, about 4 us of HBM traffic plus drain on a 513 us kernel.
    // That is the cost of producing the answer, and only the instruction-count
    // part of it is addressable -- which the 8x widening measured at 0.35%.
    // A lane's 4 accumulators are 4 consecutive ROWS of one column, N apart
    // here, so a wide store needs the output tile transposed (swap the MFMA
    // operands, carrying the scale op_sel mapping with it). Given 0.35%, don't.
    v_lshrrev_b32(vgpr(2), 4, vgpr(1));
    v_and_b32(vgpr(3), 15, vgpr(1));
    v_mul_u32_u24(vgpr(2), 4 * s_.N * 2, vgpr(2));
    v_lshlrev_b32(vgpr(3), 1, vgpr(3));
    v_add_u32(vgpr(7), vgpr(2), vgpr(3));

    // C += (block_m*256 + wm*128) * N * 2 + (block_n*256 + wn*128) * 2
    s_mul_i32(sgpr(30), sgpr(23), kBM * s_.N * 2);
    s_add_u32(sgpr(8), sgpr(8), sgpr(30));
    s_addc_u32(sgpr(9), sgpr(9), 0);
    s_mul_i32(sgpr(30), sgpr(21), kMT * 16 * s_.N * 2);
    s_add_u32(sgpr(8), sgpr(8), sgpr(30));
    s_addc_u32(sgpr(9), sgpr(9), 0);
    s_mul_i32(sgpr(30), sgpr(24), kBN * 2);
    s_add_u32(sgpr(8), sgpr(8), sgpr(30));
    s_addc_u32(sgpr(9), sgpr(9), 0);
    s_mul_i32(sgpr(30), sgpr(22), kNT * 16 * 2);
    s_add_u32(sgpr(8), sgpr(8), sgpr(30));
    s_addc_u32(sgpr(9), sgpr(9), 0);

    for (int i = 0; i < kMT; ++i)
      for (int j = 0; j < kNT; ++j) {
        const int acc = (i * kNT + j) * 4;
        for (int r = 0; r < 4; ++r) {
          v_add_u32(vgpr(10), (i * 16 + r) * s_.N * 2 + j * 16 * 2, vgpr(7));
          v_accvgpr_read_b32(vgpr(11), agpr(acc + r));
          s_nop(2);
          v_cvt_pk_bf16_f32(vgpr(12), vgpr(11), vgpr(11));
          global_store_short(vgpr(10), vgpr(12), sgpr(8).dw(2), 0);
        }
      }

    s_waitcnt(wcnt_.vmcnt(0));
    s_endpgm();
  }

  static constexpr int kLoadsPerStage = 2 * kLoadsPerWaveAB + 1 + 1;  // 14

private:
  static int log2i(int v) { int r = 0; while ((1 << r) < v) ++r; return r; }

  using sgpr = gfxjit::operands::Sgpr;
  using vgpr = gfxjit::operands::Vgpr;

  void emit_tile_of_wg() {
    const int bm = s_.blocks_m(), bn = s_.blocks_n();
    const int LM = log2i(bm), LN = log2i(bn);
    const int nx = log2i(kXCDs), rb = log2i(kXCDRows), cb = log2i(kXCDCols);
    // Falls back to the linear mapping when the grid cannot be cut into one
    // kXCDRows x kXCDCols patch per XCD.
    if (!pow2(bm) || !pow2(bn) || rb + cb + nx != LM + LN || rb > LM ||
        cb > LN || LN - cb > nx) {
      s_and_b32(sgpr(23), sgpr(6), bm - 1);
      s_lshr_b32(sgpr(24), sgpr(6), LM);
      return;
    }
    // block_n = (wg >> (nx+rb)) & (CB-1) | (wg & (blocks_n/CB - 1)) << cb
    s_lshr_b32(sgpr(24), sgpr(6), nx + rb);
    s_and_b32(sgpr(24), sgpr(24), kXCDCols - 1);
    s_and_b32(sgpr(25), sgpr(6), (bn >> cb) - 1);
    s_lshl_b32(sgpr(25), sgpr(25), cb);
    s_or_b32(sgpr(24), sgpr(24), sgpr(25));
    // block_m = (wg >> nx) & (RB-1) | (wg >> (LN-cb)) << rb
    s_lshr_b32(sgpr(23), sgpr(6), nx);
    s_and_b32(sgpr(23), sgpr(23), kXCDRows - 1);
    s_lshr_b32(sgpr(25), sgpr(6), LN - cb);
    s_and_b32(sgpr(25), sgpr(25), (1 << (nx - (LN - cb))) - 1);
    s_lshl_b32(sgpr(25), sgpr(25), rb);
    s_or_b32(sgpr(23), sgpr(23), sgpr(25));
  }

  static bool pow2(int v) { return v > 0 && !(v & (v - 1)); }

  // s[d:d+3] = buffer descriptor over s[ptr:ptr+1] + s[off], num_records bytes
  void make_desc(int d, int ptr, int off, int num_records) {
    s_add_u32(sgpr(d), sgpr(ptr), sgpr(off));
    s_addc_u32(sgpr(d + 1), sgpr(ptr + 1), 0);
    s_mov_b32(sgpr(d + 2), num_records);
    s_mov_b32(sgpr(d + 3), 0x20000);
  }

  // Operand register sets, ping-ponged between consecutive K-steps.
  // 8 tiles x 6 VGPRs = 48 registers per matrix per set.
  static constexpr int vA(int r) { return r ? 144 : 48; }
  static constexpr int vB(int r) { return r ? 192 : 96; }
  static constexpr int vSA(int r) { return 16 + r * 4; }
  static constexpr int vSB(int r) { return 18 + r * 4; }

  // One stage's 14 direct-to-LDS copies as schedulable units. A and B each
  // need 6 loads, split into two m0 groups of 3 because 6*1024 exceeds the
  // 12-bit instruction offset; the second group offsets both the LDS base
  // and the global voffset by 3072.
  std::vector<std::function<void()>> fill_units(int st) {
    const int base = st * kLdsStage;
    using gfxjit::operands::mbuf;
    std::vector<std::function<void()>> u;
    // Advance a descriptor's 64-bit base by one stage image, emitted right
    // after that operand's own loads so the four pairs land at four spread-out
    // points in the K-step rather than bunched at the end. s_add_u32 and
    // s_addc_u32 have to stay adjacent: the m0 updates clobber SCC.
    auto adv = [&](int sdesc, int step) {
      u.push_back([=, this] {
        s_add_u32(sgpr(sdesc), sgpr(sdesc), step);
        s_addc_u32(sgpr(sdesc + 1), sgpr(sdesc + 1), 0);
      });
    };
    auto ab = [&](int sdesc, int ldsoff) {
      for (int i = 0; i < kLoadsPerWaveAB; ++i) {
        const int g = i / kLoadGroup, gi = i % kLoadGroup;
        u.push_back([=, this] {
          if (gi == 0) s_add_u32(m0_, sgpr(32), ldsoff + g * kGroupBytes);
          buffer_load_dwordx4(vgpr(vDummy_).dw(4), vgpr(g ? 8 : 3), sgpr(sdesc).dw(4),
                              0, gi * 1024, {mbuf::offen, mbuf::lds});
        });
      }
      adv(sdesc, kABytesPerStep);
    };
    ab(36, base + kLdsA);
    ab(40, base + kLdsB);
    auto sc = [&](int sdesc, int ldsoff) {
      u.push_back([=, this] {
        s_add_u32(m0_, sgpr(33), ldsoff);
        buffer_load_dword(vgpr(vDummy_), vgpr(4), sgpr(sdesc).dw(4), 0, 0,
                          {mbuf::offen, mbuf::lds});
      });
      adv(sdesc, kSBytesPerStep);
    };
    sc(44, base + kLdsSA);
    sc(48, base + kLdsSB);
    return u;
  }

  // Issue one stage's direct-to-LDS copies; fill_units advances the bases.
  void fill_stage(int st) {
    for (auto &f : fill_units(st)) f();
  }

  GemmShape s_;
  // VGPR map: v0-v8 misc, v16-23 scales, v24-35 c0/scale LDS addresses,
  // v36-39 dummy, v40-45 c1 LDS addresses, operands v48-95 / v96-143 (set 0)
  // and v144-191 / v192-239 (set 1).
  static constexpr int vAaddr_ = 24, vBaddr_ = 27;   // c0 plane, per stage
  static constexpr int vSAaddr_ = 30, vSBaddr_ = 33;
  static constexpr int vDummy_ = 36;  // vdata is ignored by the LDS-DMA form,
                                      // but it is a 4-register tuple: v36-v39
  static constexpr int vA1addr_ = 40, vB1addr_ = 43;  // c1 plane, per stage
};

// --------------------------------------------------------------- host side

// MXFP6 e2m3: 1 sign bit, 2 exponent bits, 3 mantissa bits, exponent bias 1.
static inline float fp6_decode(uint8_t n) {
  const float sign = (n & 0x20) ? -1.f : 1.f;
  const int exp = (n >> 3) & 3, mant = n & 7;
  return sign * (exp == 0 ? mant / 8.f : (1.f + mant / 8.f) * ldexpf(1.f, exp - 1));
}
static inline float e8m0_decode(uint8_t e) { return e == 255 ? NAN : ldexpf(1.f, (int)e - 127); }
static inline float bf16_decode(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// 32 six-bit values, little-endian bit packed, into 24 bytes.
static inline void fp6_pack32(const uint8_t *src, uint8_t *dst) {
  for (int i = 0; i < 32; ++i) {
    const int bit = i * 6;
    const uint32_t v = src[i] & 0x3F;
    dst[bit >> 3] |= (uint8_t)(v << (bit & 7));
    if ((bit & 7) > 2) dst[(bit >> 3) + 1] |= (uint8_t)(v >> (8 - (bit & 7)));
  }
}
static inline uint8_t fp6_unpack(const uint8_t *src, int i) {
  const int bit = i * 6;
  uint32_t v = src[bit >> 3] >> (bit & 7);
  if ((bit & 7) > 2) v |= (uint32_t)src[(bit >> 3) + 1] << (8 - (bit & 7));
  return (uint8_t)(v & 0x3F);
}

static bool self_check() {
  uint8_t vals[32], buf[24] = {}, out[32];
  for (int i = 0; i < 32; ++i) vals[i] = (uint8_t)((i * 7 + 11) & 0x3F);
  fp6_pack32(vals, buf);
  for (int i = 0; i < 32; ++i) out[i] = fp6_unpack(buf, i);
  if (memcmp(vals, out, 32)) { printf("fp6 pack/unpack self-check FAILED\n"); return false; }
  // e2m3 magnitudes must be the 32-entry ladder 0 .. 7.5
  static const float want[32] = {
      0.f, .125f, .25f, .375f, .5f, .625f, .75f, .875f,
      1.f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
      2.f, 2.25f, 2.5f, 2.75f, 3.f, 3.25f, 3.5f, 3.75f,
      4.f, 4.5f, 5.f, 5.5f, 6.f, 6.5f, 7.f, 7.5f};
  for (int n = 0; n < 32; ++n)
    if (fp6_decode((uint8_t)n) != want[n] || fp6_decode((uint8_t)(n | 0x20)) != -want[n]) {
      printf("fp6 decode self-check FAILED at %d: %g vs %g\n", n, fp6_decode((uint8_t)n), want[n]);
      return false;
    }
  return true;
}

int main(int argc, char **argv) {
  int warmup = 20, iters = 200, M = 256, N = 256, K = 512;
  bool bench = false;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-iters") && i + 1 < argc) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-K") && i + 1 < argc) K = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-M") && i + 1 < argc) M = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-N") && i + 1 < argc) N = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-bench")) bench = true;
  }
  if (bench) { M = 4096; N = 4096; K = 65536; }
  if (!self_check()) return 1;
  GemmShape s{M, N, K};
  if (s.k_steps() < kStages) {
    printf("k_steps (%d) must be at least %d\n", s.k_steps(), kStages);
    return 1;
  }

  hipDeviceProp_t prop;
  (void)hipGetDeviceProperties(&prop, 0);
  prop.gcnArchName[6] = 0;
  if (std::string(prop.gcnArchName) != "gfx950") {
    printf("requires gfx950, found %s\n", prop.gcnArchName);
    return 1;
  }
  printf("device: %s\n", prop.name);
  printf("gemm: M=%d N=%d K=%d  macrotile=%dx%dx%d  waves=%dx%d  MIWaveTile=%dx%d\n",
         s.M, s.N, s.K, kBM, kBN, kBK, kWGM, kWGN, kMT, kNT);
  printf("stages=%d  LDS=%d bytes  k_steps=%d (%d x 6 + %d)\n", kStages,
         kStages * kLdsStage, s.k_steps(), s.k_steps() / 6, s.k_steps() % 6);

  const std::string ker_name = "mxfp6_gemm_lds";
  Mxfp6GemmLds<gfxjit::GFXArch::gfx950> gen(s);
  std::vector<uint8_t> co = gen.get_code_obj(ker_name);

  const int kblocks = s.K / 32, ksteps = s.k_steps();
  std::vector<uint8_t> An((size_t)s.M * s.K), Bn((size_t)s.N * s.K);
  std::vector<uint8_t> SA((size_t)s.M * kblocks), SB((size_t)s.N * kblocks);
  srand(7);
  for (auto &x : An) x = rand() & 0x3F;
  for (auto &x : Bn) x = rand() & 0x3F;
  for (auto &x : SA) x = 124 + (rand() % 7);
  for (auto &x : SB) x = 124 + (rand() % 7);

  // ---- swizzle: per (block, k_step) a 24576 B image [tile(16)][lane(64)][24B]
  auto swizzle_ab = [&](const std::vector<uint8_t> &src, int rows) {
    const int blocks = rows / kBM;
    std::vector<uint8_t> dst((size_t)blocks * ksteps * kABytesPerStep, 0);
    for (int b = 0; b < blocks; ++b)
      for (int ks = 0; ks < ksteps; ++ks)
        for (int t = 0; t < kTilesPerBlock; ++t)
          for (int lane = 0; lane < 64; ++lane) {
            // c0c1: the fragment's bytes 0..15 go to 16*(64t+lane) and bytes
            // 16..23 to kC1Base + 8*(64t+lane) of the 24576 B stage image.
            uint8_t *base = &dst[((size_t)b * ksteps + ks) * kABytesPerStep];
            const int o = 64 * t + lane;
            const int row = b * kBM + t * 16 + lane % 16;
            const int kbase = ks * kBK + (lane / 16) * 32;
            uint8_t frag[24];
            fp6_pack32(&src[(size_t)row * s.K + kbase], frag);
            memcpy(base + 16 * o, frag, 16);
            memcpy(base + kC1Base + 8 * o, frag + 16, 8);
          }
    return dst;
  };
  // ---- scales: per (block, k_step) a 1024 B image [lane(64)][tile(16)]
  auto swizzle_sc = [&](const std::vector<uint8_t> &src, int rows) {
    const int blocks = rows / kBM;
    std::vector<uint8_t> dst((size_t)blocks * ksteps * kSBytesPerStep, 0);
    for (int b = 0; b < blocks; ++b)
      for (int ks = 0; ks < ksteps; ++ks)
        for (int lane = 0; lane < 64; ++lane)
          for (int t = 0; t < kTilesPerBlock; ++t) {
            const int row = b * kBM + t * 16 + lane % 16;
            const int kblk = ks * 4 + lane / 16;
            // [t/8][lane(64)][t%8] bytes, matching scale type 1018
            dst[((size_t)b * ksteps + ks) * kSBytesPerStep +
                (size_t)(t / 8) * 512 + (size_t)lane * 8 + (t % 8)] =
                src[(size_t)row * kblocks + kblk];
          }
    return dst;
  };

  std::vector<uint8_t> Asw = swizzle_ab(An, s.M), Bsw = swizzle_ab(Bn, s.N);
  std::vector<uint8_t> SAsw = swizzle_sc(SA, s.M), SBsw = swizzle_sc(SB, s.N);

  // The descriptor base walks K, so the last kStages prefetches of the final
  // block run past the end of the data instead of being clamped by
  // num_records. Pad so they stay mapped; the values they fetch are never
  // consumed by an MFMA, since the last K-step's ds_reads have no MFMA after
  // them.
  const size_t kPadAB = (size_t)kStages * kABytesPerStep;
  const size_t kPadS = (size_t)kStages * kSBytesPerStep;
  void *d_a, *d_b, *d_sa, *d_sb, *d_c;
  (void)hipMalloc(&d_a, Asw.size() + kPadAB);
  (void)hipMalloc(&d_b, Bsw.size() + kPadAB);
  (void)hipMalloc(&d_sa, SAsw.size() + kPadS);
  (void)hipMalloc(&d_sb, SBsw.size() + kPadS);
  (void)hipMalloc(&d_c, (size_t)s.M * s.N * 2);
  (void)hipMemcpy(d_a, Asw.data(), Asw.size(), hipMemcpyHostToDevice);
  (void)hipMemcpy(d_b, Bsw.data(), Bsw.size(), hipMemcpyHostToDevice);
  (void)hipMemcpy(d_sa, SAsw.data(), SAsw.size(), hipMemcpyHostToDevice);
  (void)hipMemcpy(d_sb, SBsw.data(), SBsw.size(), hipMemcpyHostToDevice);
  (void)hipMemset(d_c, 0, (size_t)s.M * s.N * 2);

  hipModule_t module;
  if (hipModuleLoadData(&module, co.data()) != hipSuccess) {
    printf("hipModuleLoadData failed\n");
    return 1;
  }
  hipFunction_t kernel;
  if (hipModuleGetFunction(&kernel, module, ker_name.c_str()) != hipSuccess) {
    printf("hipModuleGetFunction failed\n");
    return 1;
  }

  size_t offset = 40;
  char args[256] = {};
  void *ptrs[5] = {d_c, d_a, d_b, d_sa, d_sb};
  memcpy(args, ptrs, sizeof(ptrs));
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &offset, HIP_LAUNCH_PARAM_END};
  const int grid = s.blocks_m() * s.blocks_n();
  auto launch = [&]() {
    return hipModuleLaunchKernel(kernel, grid, 1, 1, kThreads, 1, 1, 0, nullptr,
                                 nullptr, (void **)&config);
  };

  if (launch() != hipSuccess) { printf("launch failed\n"); return 1; }
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) { printf("kernel error: %s\n", hipGetErrorString(e)); return 1; }

  if (!bench) {
    std::vector<uint16_t> C((size_t)s.M * s.N);
    (void)hipMemcpy(C.data(), d_c, C.size() * 2, hipMemcpyDeviceToHost);
    double maxrel = 0;
    long bad = 0;
    std::vector<float> adec(s.K), bdec(s.K);
    for (int m = 0; m < s.M; ++m) {
      for (int k = 0; k < s.K; ++k)
        adec[k] = fp6_decode(An[(size_t)m * s.K + k]) *
                  e8m0_decode(SA[(size_t)m * kblocks + k / 32]);
      for (int n = 0; n < s.N; ++n) {
        for (int k = 0; k < s.K; ++k)
          bdec[k] = fp6_decode(Bn[(size_t)n * s.K + k]) *
                    e8m0_decode(SB[(size_t)n * kblocks + k / 32]);
        double ref = 0;
        for (int k = 0; k < s.K; ++k) ref += (double)adec[k] * bdec[k];
        const double got = bf16_decode(C[(size_t)m * s.N + n]);
        // bf16 has 8 mantissa bits; allow the rounding of the final store
        const double rel =
            fabs(ref) > 1e-30 ? fabs(got - ref) / fabs(ref) : fabs(got - ref);
        if (rel > 8e-3 && bad++ < 5)
          printf("mismatch m=%d n=%d ref=%g got=%g\n", m, n, ref, got);
        maxrel = fmax(maxrel, rel);
      }
    }
    printf("max rel err = %g (bf16 out), mismatches = %ld / %ld\n", maxrel, bad,
           (long)C.size());
    printf(bad == 0 ? "PASS\n" : "FAIL\n");
    if (bad) return 1;
  }

  for (int i = 0; i < warmup; ++i) (void)launch();
  (void)hipDeviceSynchronize();
  hipEvent_t t0, t1;
  (void)hipEventCreate(&t0);
  (void)hipEventCreate(&t1);
  (void)hipEventRecord(t0, nullptr);
  for (int i = 0; i < iters; ++i) (void)launch();
  (void)hipEventRecord(t1, nullptr);
  (void)hipEventSynchronize(t1);
  float ms_total = 0;
  (void)hipEventElapsedTime(&ms_total, t0, t1);
  const double ms = ms_total / iters;
  printf("  %.4f ms/iter over %d iters (after %d warmup)\n", ms, iters, warmup);
  printf("TFLOPS %.2f\n", (2.0 * s.M * s.N * s.K) / (ms * 1e-3) / 1e12);
  return 0;
}
