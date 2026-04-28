#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Basic GEMM kernel generation example.

Demonstrates:
  1. Dry-run mode (no stinkytofu binary needed)
  2. Full generation with LogicalModule output
  3. Custom tile configurations
  4. FLOP and arithmetic-intensity analysis
  5. Overriding the MFMA emitter for hand-tuned code
"""
from __future__ import annotations

import sys


def dry_run_example():
    """Generate kernel metadata without needing stinkytofu binary."""
    from stinkytofu.gemm import (
        GemmProblem, TileConfig, MfmaConfig, generate_gemm_kernel,
    )

    print("=" * 60)
    print("Dry-run example (no stinkytofu binary needed)")
    print("=" * 60)

    problem = GemmProblem(m=4096, n=4096, k=4096)
    tile = TileConfig(
        wg_m=128, wg_n=128, unroll_k=32,
        waves_m=2, waves_n=2,
        mfma=MfmaConfig.f16_16x16x16(),
        vector_width=8,
    )

    result = generate_gemm_kernel(problem, tile, dry_run=True)
    print(result.summary())
    print()

    # Performance analysis
    print(f"Total FLOPs       : {problem.total_flops:,}")
    print(f"Arithmetic intensity: {problem.arithmetic_intensity:.1f} FLOPs/byte")
    print(f"Bytes read        : {problem.bytes_read:,}")
    print(f"Bytes written     : {problem.bytes_written:,}")
    print()

    # Tile decomposition verification
    grid_m, grid_n = problem.grid_dims(tile)
    k_tiles = problem.k // tile.unroll_k
    waves = tile.waves_m * tile.waves_n
    flops_check = (
        grid_m * grid_n * waves
        * tile.total_mfma_per_wave * k_tiles
        * tile.mfma.flops_per_instruction
    )
    print(f"FLOPs from tiles  : {flops_check:,}")
    print(f"Match             : {flops_check == problem.total_flops}")
    print()


def compare_tile_configs():
    """Compare multiple tile configurations for the same problem."""
    from stinkytofu.gemm import (
        GemmProblem, TileConfig, MfmaConfig, generate_gemm_kernel,
    )

    print("=" * 60)
    print("Tile configuration comparison")
    print("=" * 60)

    problem = GemmProblem(m=4096, n=4096, k=4096)

    configs = [
        ("16x16x16 MFMA, 128x128 WG", TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
        )),
        ("32x32x8 MFMA, 256x128 WG", TileConfig(
            wg_m=256, wg_n=128, unroll_k=32,
            waves_m=4, waves_n=2,
            mfma=MfmaConfig.f16_32x32x8(),
        )),
        ("16x16x16 MFMA, 64x64 WG", TileConfig(
            wg_m=64, wg_n=64, unroll_k=16,
            waves_m=1, waves_n=1,
            mfma=MfmaConfig.f16_16x16x16(),
        )),
    ]

    for label, tile in configs:
        result = generate_gemm_kernel(problem, tile, dry_run=True)
        grid_m, grid_n = problem.grid_dims(tile)
        print(f"\n--- {label} ---")
        print(f"  Workgroups     : {grid_m} x {grid_n} = {grid_m * grid_n}")
        print(f"  Threads/WG     : {tile.block_size}")
        print(f"  MFMAs/wave/unroll: {tile.total_mfma_per_wave}")
        print(f"  Acc registers  : {result.regs.acc_count}")
        print(f"  Total VGPRs    : {result.regs.vgpr_count}")
        print(f"  LDS bytes      : {result.mapping.lds_size_bytes}")
    print()


def custom_emitter_example():
    """Show how to override a specific codegen layer."""
    from stinkytofu.gemm import (
        GemmProblem, TileConfig, MfmaConfig, Emitter, generate_gemm_kernel,
    )

    print("=" * 60)
    print("Custom emitter override")
    print("=" * 60)

    class InterleavedMfmaEmitter(Emitter):
        """Interleave LDS reads with MFMAs for latency hiding.

        This is a simplified example -- a real implementation would
        schedule reads for k_iter+1 between MFMAs of k_iter.
        """
        def emit_mfma_block(self, module, k_iter):
            import stinkytofu as st
            mfma = self.tile.mfma
            acc_per = mfma.acc_vgprs

            for mi in range(self.tile.mfma_m_repeat):
                for ni in range(self.tile.mfma_n_repeat):
                    acc_off = (mi * self.tile.mfma_n_repeat + ni) * acc_per
                    # Insert a comment showing the interleave point
                    module.add(st.MFMA(
                        instType=mfma.input_type, accType=mfma.acc_type,
                        m=mfma.m, n=mfma.n, k=mfma.k,
                        blocks=mfma.blocks, mfma1k=False,
                        acc=self._acc("acc_C", acc_off, acc_per),
                        a=self._v("v_a", 0, mfma.a_vgprs),
                        b=self._v("v_b", 0, mfma.b_vgprs),
                        comment=f"interleaved MFMA m{mi}_n{ni} k{k_iter}",
                    ))

    problem = GemmProblem(m=128, n=128, k=32)
    tile = TileConfig(
        wg_m=128, wg_n=128, unroll_k=32,
        waves_m=2, waves_n=2,
        mfma=MfmaConfig.f16_16x16x16(),
    )

    try:
        result = generate_gemm_kernel(
            problem, tile, emitter_cls=InterleavedMfmaEmitter,
        )
        print("Generated kernel with custom MFMA emitter.")
        print(f"Name: {result.name}")
        print(f"Module has instructions: {result.module is not None}")
    except ImportError:
        print("stinkytofu not available; showing dry-run only.")
        result = generate_gemm_kernel(
            problem, tile, emitter_cls=InterleavedMfmaEmitter, dry_run=True,
        )
        print(f"Dry-run name: {result.name}")
        print(f"Emitter class: {result.codegen.emitter.__class__.__name__}")
    print()


def coordinate_transform_demo():
    """Demonstrate the coordinate transform system directly."""
    from stinkytofu.gemm.transforms import (
        Dim, Tile, Flatten, Pad, Embed, Xor, TileDescriptor, tile_hierarchy,
    )

    print("=" * 60)
    print("Coordinate transform demo")
    print("=" * 60)

    # Build a 3-level tiling for M=4096
    M = Dim("M", 4096)
    tiles = tile_hierarchy(M, [
        (128, "M_wg_id",   "M_wg"),      # workgroup tile
        (64,  "M_wave_id", "M_wave"),     # wave tile
        (16,  "M_mfma_id", "M_mfma"),    # MFMA tile
    ])

    print(f"Dimension: {M}")
    for i, t in enumerate(tiles):
        print(f"  Level {i}: {t}")
        print(f"    outer={t.outer}, inner={t.inner}")

    # Apply to a TileDescriptor
    desc = TileDescriptor("M_hierarchy", [M])
    for t in tiles:
        desc.add_transform(t)
    print(f"\nFinal descriptor: {desc}")
    print(f"Visible dims: {desc.visible_dims}")

    # Evaluate: wg_id=3, wave_id=1, mfma_id=2, mfma_elem=5
    # -> M = 3*128 + 1*64 + 2*16 + 5 = 384 + 64 + 32 + 5 = 485
    idx_outer = tiles[0].forward({"M_wg_id": 3, "M_wg": 0})  # partial
    print(f"\nExample: wg_id=3 -> M_offset = {3 * 128}")
    print(f"  + wave_id=1 -> +{1 * 64}")
    print(f"  + mfma_id=2 -> +{2 * 16}")
    print(f"  + mfma_elem=5 -> +5")
    print(f"  = M index {3*128 + 1*64 + 2*16 + 5}")

    # LDS bank-conflict avoidance with Xor
    print(f"\nXor transform for LDS:")
    xor = Xor(Dim("row", 64), Dim("col", 8), shift=2)
    print(f"  {xor}")
    result = xor.forward({"row": 5, "col": 12})
    print(f"  forward(row=5, col=12) -> {result}")
    print(f"  5 ^ (12 >> 2) = 5 ^ 3 = {5 ^ 3}")
    print()


if __name__ == "__main__":
    dry_run_example()
    compare_tile_configs()
    coordinate_transform_demo()

    # Only run stinkytofu-dependent examples if available
    try:
        import stinkytofu as _st
        if hasattr(_st, "LogicalModule"):
            custom_emitter_example()
        else:
            print("Skipping stinkytofu-dependent examples (C extension not built).\n")
    except ImportError:
        print("Skipping stinkytofu-dependent examples (not built).\n")
