// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Host-side layouts of HIPBLASLT_ORDER_COL256_R128_EXT (FP6 data) and
// HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT (E8M0 scales), from a
// K-contiguous operand: transposed A or non-transposed B, one row per M (N) index.
//
// ============================ WORKTREE-LOCAL DIVERGENCE ============================
// The FP6 data packer here NO LONGER EMITS THE PUBLISHED HIPBLASLT_ORDER_COL256_R128_EXT
// BYTE FORMULA. It has been replaced in place with the "paired" layout (see
// tiledGroupOffset below) that the gfxjit MXFP6 GFA kernel consumes, so order 104 means
// something different in this worktree than it does in hipblaslt.h:394 and upstream.
//
// CONSEQUENCE: every other consumer of order 104 is BROKEN HERE, and broken SILENTLY --
// it gets a well-formed image with the bytes in the wrong places and computes wrong
// results at full speed. In particular rocKE's kernels expect the c0c1 layout and WILL
// FAIL --verify_strict in this tree. That is expected and is the positive control that
// this swap took effect; if rocKE still passes, this packer is not being used.
// For rocKE reference measurements use the untouched prebuilt tree at
// /home/kerrwang/repos/rocm-libraries/wt-mxfp6-rocke instead.
//
// The kernel that depends on this layout is
//   gfxjit_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_
//   4096x4096x65536_paired
// (CustomKernels/gfxjit/, reached via SolutionIndex 2 of
// Logic/asm_full/gfx950/gfx950_id75a3/Equality/
// gfx950_Cijk_Alik_Bljk_F6BS_MXAE8B32_MXBE8B32_STA_STB_H_UserArgs.yaml), generated from
// gfxjit-gfa/mxfp6_gfa.cpp. Its swizzle_ab lambda is the mirror of the packer below and
// the two must be changed together.
//
// The SCALE layout below is UNCHANGED and still matches the published formula.
// ==================================================================================

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx_tile128x4
{
    constexpr size_t kTileRows      = 256; // M (N) rows per data tile
    constexpr size_t kTileK         = 128; // K elements per data tile
    constexpr size_t kGroupK        = 32; // K elements per 24-byte group
    constexpr size_t kGroupBytes    = kGroupK * 6 / 8;
    constexpr size_t kTileBytes     = kTileRows * kTileK * 6 / 8;
    constexpr size_t kPairBytes     = 64 * 2 * kGroupBytes; // 3072, a sub-tile pair's slots
    constexpr size_t kScaleRows     = 128; // rows per scale tile
    constexpr size_t kScaleBlocks   = 4; // K blocks per scale tile

    inline void checkDataShape(size_t rows, size_t k)
    {
        if(rows % kTileRows != 0 || k % kTileK != 0)
            throw std::invalid_argument("HIPBLASLT_ORDER_COL256_R128_EXT needs rows % 256 == 0 "
                                        "and K % 128 == 0; got rows="
                                        + std::to_string(rows) + " K=" + std::to_string(k));
    }

    inline void checkScaleShape(size_t rows, size_t kBlocks)
    {
        if(rows % kScaleRows != 0 || kBlocks % kScaleBlocks != 0)
            throw std::invalid_argument("HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT needs "
                                        "rows % 128 == 0 and K blocks % 4 == 0; got rows="
                                        + std::to_string(rows)
                                        + " K blocks=" + std::to_string(kBlocks));
    }

    // Byte offset of the 24-byte group (row, k / 32), which is stored CONTIGUOUSLY.
    //
    // This is the paired layout, NOT the published order-104 formula: see the divergence
    // notice at the top of this file. A tile's 16-column sub-tiles are grouped in pairs,
    // and the two groups a lane holds sit back to back in one 48-byte slot, so the lane's
    // whole 48 bytes are fetched by three 16-byte loads. With t the sub-tile index and
    // `lane` the position within it, the group is at
    //     kPairBytes * (t / 2) + 48 * lane + 24 * (t % 2)
    // within the tile. The 48-byte lane stride is LDS-bank conflict free: 12*lane mod 32
    // cycles through 8 distinct 4-bank groups. Do not "simplify" this to a flat
    // [tile][lane][24 B] image -- that leaves 16-byte loads 8-byte misaligned at a
    // 24-byte stride and the read serializes (measured 1841 TFLOP/s against 4247).
    inline size_t tiledGroupOffset(size_t row, size_t kGroup, size_t kTotal)
    {
        size_t const kTiles = kTotal / kTileK;
        size_t const tile   = (row / kTileRows) * kTiles + kGroup / (kTileK / kGroupK);
        size_t const t      = (row % kTileRows) / 16;
        size_t const lane   = 16 * (kGroup % (kTileK / kGroupK)) + row % 16;
        return tile * kTileBytes + kPairBytes * (t / 2) + 2 * kGroupBytes * lane
               + kGroupBytes * (t % 2);
    }

    // dst and src hold rows * k * 6 / 8 bytes; src rows are the 6-bit LSB-first packed K
    // elements of each M (N) index.
    inline void tileFp6Data(uint8_t* dst, uint8_t const* src, size_t rows, size_t k)
    {
        checkDataShape(rows, k);
        size_t const rowBytes = k * 6 / 8;
        for(size_t r = 0; r < rows; ++r)
            for(size_t g = 0; g < k / kGroupK; ++g)
                std::memcpy(dst + tiledGroupOffset(r, g, k),
                            src + r * rowBytes + g * kGroupBytes,
                            kGroupBytes);
    }

    // Byte offset of the scale of row `row` and K block `kBlock`.
    inline size_t shuffledScaleOffset(size_t row, size_t kBlock, size_t rows)
    {
        size_t const tile = (kBlock / kScaleBlocks) * (rows / kScaleRows) + row / kScaleRows;
        return ((tile * 64 + 16 * (kBlock % kScaleBlocks) + row % 16) * 8)
               + (row % kScaleRows) / 16;
    }

    // src is [rows][kBlocks], K block fastest.
    inline std::vector<uint8_t>
        shuffleScales(std::vector<uint8_t> const& src, size_t rows, size_t kBlocks)
    {
        checkScaleShape(rows, kBlocks);
        if(src.size() != rows * kBlocks)
            throw std::invalid_argument("128x4 scale shuffle: expected "
                                        + std::to_string(rows * kBlocks) + " scales, got "
                                        + std::to_string(src.size()));
        std::vector<uint8_t> dst(src.size());
        for(size_t r = 0; r < rows; ++r)
            for(size_t kb = 0; kb < kBlocks; ++kb)
                dst[shuffledScaleOffset(r, kb, rows)] = src[r * kBlocks + kb];
        return dst;
    }
} // namespace mx_tile128x4
