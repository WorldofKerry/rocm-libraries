# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Assembly infrastructure: register allocation, header/descriptor, assembler.

This file contains only the kernel infrastructure that is independent
of specific phase implementations.  Phase functions live in phases.py.
Tree builders live in tiling.py and kernel_pipeline.py.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

from .context import AsmContext
from ..problem import GemmProblem, TileConfig, MfmaConfig
from ..tile.tree import TileLevel, TilePhase

__all__ = ["assemble_kernel"]


def assemble_kernel(asm_text: str, gpu_arch: str = "gfx950",
                    output_path: Optional[str] = None) -> str:
    """Assemble text into a .co code object."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".s", delete=False) as f:
        f.write(asm_text)
        s_path = f.name

    o_path = s_path.replace(".s", ".o")
    co_path = output_path or s_path.replace(".s", ".co")

    r = subprocess.run(
        ["amdclang++", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         f"-mcpu={gpu_arch}", "-c", s_path, "-o", o_path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Assembly failed:\n{r.stderr}")

    r = subprocess.run(
        ["amdclang++", "-target", "amdgcn-amd-amdhsa",
         f"-mcpu={gpu_arch}", "-o", co_path, o_path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Link failed:\n{r.stderr}")

    os.unlink(s_path)
    os.unlink(o_path)
    return co_path


# ===================================================================
# Register allocation
# ===================================================================

def alloc_registers(ctx: AsmContext, problem: GemmProblem,
                    tile: TileConfig) -> None:
    """Allocate all kernel registers using named bindings.

    ABI: s[0:1] = kernarg ptr, s2 = workgroup_id_x, s3 = workgroup_id_y
    """
    ctx.alloc_sgpr_permanent(2, "s_kernarg")
    ctx.alloc_sgpr_permanent(1, "s_wg_id_x")
    ctx.alloc_sgpr_permanent(1, "s_wg_id_y")

    ctx.alloc_sgpr_permanent(2, "s_ptr_A")
    ctx.alloc_sgpr_permanent(2, "s_ptr_B")
    ctx.alloc_sgpr_permanent(2, "s_ptr_D")
    ctx.alloc_sgpr_permanent(1, "s_M")
    ctx.alloc_sgpr_permanent(1, "s_N")
    ctx.alloc_sgpr_permanent(1, "s_K")
    ctx.alloc_sgpr_permanent(1, "s_k_tiles")
    ctx.alloc_sgpr_permanent(1, "s_tmp0")
    ctx.alloc_sgpr_permanent(1, "s_tmp1")

    ctx.alloc_vgpr_permanent(1, "v_tid")
    ctx.alloc_vgpr_permanent(1, "v_wave_id")
    ctx.alloc_vgpr_permanent(1, "v_lane_id")
    ctx.alloc_vgpr_permanent(1, "v_wave_m")
    ctx.alloc_vgpr_permanent(1, "v_wave_n")

    ctx.alloc_vgpr_permanent(1, "v_gload_row")
    ctx.alloc_vgpr_permanent(1, "v_gload_col")
    # Separate B load cluster coords (used when wg_m != wg_n)
    ctx.alloc_vgpr_permanent(1, "v_gload_row_b")
    ctx.alloc_vgpr_permanent(1, "v_gload_col_b")

    ctx.alloc_vgpr_permanent(1, "v_lds_wr_a")
    ctx.alloc_vgpr_permanent(1, "v_lds_wr_b")
    ctx.alloc_vgpr_permanent(1, "v_lds_rd_a")
    ctx.alloc_vgpr_permanent(1, "v_lds_rd_b")
    # Per-ki LDS read base VGPRs (ki=0 uses v_lds_rd_a/b directly)
    ki_count = tile.unroll_k // tile.mfma.k if tile.mfma.k > 0 else 1
    for ki in range(1, ki_count):
        ctx.alloc_vgpr_permanent(1, f"v_lds_rd_a_k{ki}")
        ctx.alloc_vgpr_permanent(1, f"v_lds_rd_b_k{ki}")

    ctx.alloc_vgpr_permanent(tile.mfma.a_vgprs, "v_a")
    ctx.alloc_vgpr_permanent(tile.mfma.b_vgprs, "v_b")

    elem = problem.element_bytes
    a_elems = (tile.wg_m * tile.unroll_k) // tile.block_size
    b_elems = (tile.wg_n * tile.unroll_k) // tile.block_size
    a_vgprs = max(1, int(a_elems * elem + 3) // 4)
    b_vgprs = max(1, int(b_elems * elem + 3) // 4)
    ctx.alloc_vgpr_permanent(a_vgprs, "v_gload_a")
    ctx.alloc_vgpr_permanent(b_vgprs, "v_gload_b")

    ctx.alloc_vgpr_permanent(2, "v_addr_a")
    ctx.alloc_vgpr_permanent(2, "v_addr_b")
    ctx.alloc_vgpr_permanent(2, "v_addr_d")

    ctx.alloc_vgpr_permanent(1, "v_store_tmp")
    ctx.alloc_vgpr_permanent(1, "v_tmp0")
    ctx.alloc_vgpr_permanent(1, "v_tmp1")
    ctx.alloc_vgpr_permanent(1, "v_tmp2")
    ctx.alloc_vgpr_permanent(1, "v_tmp3")
    ctx.alloc_vgpr_permanent(1, "v_tmp4")

    acc_total = tile.mfma_m_repeat * tile.mfma_n_repeat * tile.mfma.acc_vgprs
    ctx.alloc_acc_permanent(acc_total, "acc_C")


# ===================================================================
# Header and descriptor
# ===================================================================

def emit_header(ctx: AsmContext, kernel_name: str) -> None:
    """Emit the .text section header."""
    ctx.raw(f'.amdgcn_target "amdgcn-amd-amdhsa--gfx950"')
    ctx.raw(".text")
    ctx.raw(f".globl {kernel_name}")
    ctx.raw(".p2align 8")
    ctx.raw(f".type {kernel_name},@function")
    ctx.raw("")
    ctx.raw(f"{kernel_name}:")


def emit_descriptor(ctx: AsmContext, kernel_name: str,
                    lds_total: int, tile: TileConfig) -> None:
    """Emit the AMDHSA kernel descriptor and metadata."""
    accum_offset = (ctx._next["v"] + 3) & ~3
    sgpr_count = ctx._next["s"]
    acc_count = ctx._next["acc"]
    vgpr_count = accum_offset + acc_count
    # Kernarg size depends on ABI mode
    use_wave_abi = ctx._metadata.get('use_wave_abi', False)
    if use_wave_abi:
        kernarg_size = 104  # WaveGemmKernelArgs: 13 u64 fields = 104 bytes
    elif tile.mfma.is_mx:
        kernarg_size = 136  # TensileLite MXFP4 batched layout
    else:
        kernarg_size = 104  # TensileLite non-MX layout

    ctx.raw("")
    ctx.raw(".rodata")
    ctx.raw(".p2align 6")
    ctx.raw(f".amdhsa_kernel {kernel_name}")
    ctx.raw(f"    .amdhsa_group_segment_fixed_size {lds_total}")
    ctx.raw(f"    .amdhsa_private_segment_fixed_size 0")
    ctx.raw(f"    .amdhsa_kernarg_size {kernarg_size}")
    ctx.raw(f"    .amdhsa_user_sgpr_kernarg_segment_ptr 1")
    ctx.raw(f"    .amdhsa_system_sgpr_workgroup_id_x 1")
    ctx.raw(f"    .amdhsa_system_sgpr_workgroup_id_y 1")
    ctx.raw(f"    .amdhsa_system_vgpr_workitem_id 0")
    ctx.raw(f"    .amdhsa_next_free_vgpr {vgpr_count}")
    ctx.raw(f"    .amdhsa_next_free_sgpr {sgpr_count}")
    ctx.raw(f"    .amdhsa_accum_offset {accum_offset}")
    ctx.raw(f"    .amdhsa_float_denorm_mode_32 3")
    ctx.raw(f"    .amdhsa_float_denorm_mode_16_64 3")
    ctx.raw(f".end_amdhsa_kernel")
    ctx.raw("")
    ctx.raw(".amdgpu_metadata")
    ctx.raw("---")
    ctx.raw("amdhsa.version: [ 1, 2 ]")
    ctx.raw("amdhsa.kernels:")
    ctx.raw(f"  - .name:            {kernel_name}")
    ctx.raw(f"    .symbol:          {kernel_name}.kd")
    ctx.raw(f"    .sgpr_count:      {sgpr_count}")
    ctx.raw(f"    .vgpr_count:      {vgpr_count}")
    ctx.raw(f"    .agpr_count:      {acc_count}")
    ctx.raw(f"    .kernarg_segment_size: {kernarg_size}")
    ctx.raw(f"    .kernarg_segment_align: 8")
    ctx.raw(f"    .group_segment_fixed_size: {lds_total}")
    ctx.raw(f"    .private_segment_fixed_size: 0")
    ctx.raw(f"    .wavefront_size:  {tile.wave_size}")
    ctx.raw(f"    .max_flat_workgroup_size: {tile.block_size}")
    ctx.raw(f"    .args:")
    # Wave ABI kernarg layout (WaveGemmKernelArgs)
    if use_wave_abi:
        args = [
            (0, 8, "global_buffer", "ptr_a"),
            (8, 8, "global_buffer", "ptr_a_scale"),
            (16, 8, "global_buffer", "ptr_b"),
            (24, 8, "global_buffer", "ptr_b_scale"),
            (32, 8, "global_buffer", "ptr_c"),
            (40, 8, "by_value", "dim_m"),
            (48, 8, "by_value", "dim_n"),
            (56, 8, "by_value", "dim_k"),
            (64, 8, "by_value", "stride_a_dim0"),
            (72, 8, "by_value", "stride_a_scale_dim0"),
            (80, 8, "by_value", "stride_b_dim0"),
            (88, 8, "by_value", "stride_b_scale_dim0"),
            (96, 8, "by_value", "stride_c_dim0"),
        ]
    # TensileLite batched MXFP4 kernarg layout
    elif tile.mfma.is_mx:
        args = [
            (0, 4, "by_value", "Gemm info"),
            (4, 4, "by_value", "kernel info0"),
            (8, 4, "by_value", "kernel info1"),
            (12, 4, "by_value", "numWG"),
            (16, 4, "by_value", "SizesFree0"),
            (20, 4, "by_value", "SizesFree1"),
            (24, 4, "by_value", "SizesFree2"),
            (28, 4, "by_value", "SizesSum0"),
            (32, 8, "global_buffer", "D"),
            (40, 8, "global_buffer", "C"),
            (48, 8, "global_buffer", "A"),
            (56, 8, "global_buffer", "MXSA"),
            (64, 8, "global_buffer", "B"),
            (72, 8, "global_buffer", "MXSB"),
            (80, 4, "by_value", "strideD0"),
            (84, 4, "by_value", "strideD1"),
            (88, 4, "by_value", "strideC0"),
            (92, 4, "by_value", "strideC1"),
            (96, 4, "by_value", "strideA0"),
            (100, 4, "by_value", "strideA1"),
            (104, 4, "by_value", "strideMXSA0"),
            (108, 4, "by_value", "strideMXSA1"),
            (112, 4, "by_value", "strideB0"),
            (116, 4, "by_value", "strideB1"),
            (120, 4, "by_value", "strideMXSB0"),
            (124, 4, "by_value", "strideMXSB1"),
            (128, 4, "by_value", "alpha"),
            (132, 4, "by_value", "beta"),
        ]
    else:
        args = [
            (0, 4, "by_value", "Gemm info"),
            (4, 4, "by_value", "kernel info0"),
            (8, 4, "by_value", "kernel info1"),
            (12, 4, "by_value", "numWG"),
            (16, 4, "by_value", "SizesFree0"),
            (20, 4, "by_value", "SizesFree1"),
            (24, 4, "by_value", "SizesFree2"),
            (28, 4, "by_value", "SizesSum0"),
            (32, 8, "global_buffer", "D"),
            (40, 8, "global_buffer", "C"),
            (48, 8, "global_buffer", "A"),
            (56, 8, "global_buffer", "B"),
            (64, 4, "by_value", "strideD0"),
            (68, 4, "by_value", "strideD1"),
            (72, 4, "by_value", "strideC0"),
            (76, 4, "by_value", "strideC1"),
            (80, 4, "by_value", "strideA0"),
            (84, 4, "by_value", "strideA1"),
            (88, 4, "by_value", "strideB0"),
            (92, 4, "by_value", "strideB1"),
            (96, 4, "by_value", "alpha"),
            (100, 4, "by_value", "beta"),
        ]
    for off, sz, kind, name in args:
        ctx.raw(f"      - .name:           {name}")
        ctx.raw(f"        .offset:         {off}")
        ctx.raw(f"        .size:           {sz}")
        ctx.raw(f"        .value_kind:     {kind}")
        if kind == "global_buffer":
            ctx.raw(f"        .address_space:  generic")
    ctx.raw("...")
    ctx.raw(".end_amdgpu_metadata")


def alloc_registers_dtl(ctx: AsmContext, problem: GemmProblem,
                        tile: TileConfig) -> None:
    """Allocate registers for DirectToLDS kernel (no global load buffers).

    DTL replaces global_load + ds_write with buffer_load ... ,lds.
    This eliminates v_gload_a/b (16 VGPRs) and v_addr_a/b (4 VGPRs),
    replacing them with SRDs (8 SGPRs) and per-lane offsets (2 VGPRs).
    """
    # ABI registers
    ctx.alloc_sgpr_permanent(2, "s_kernarg")
    ctx.alloc_sgpr_permanent(1, "s_wg_id_x")
    ctx.alloc_sgpr_permanent(1, "s_wg_id_y")

    # Kernel args
    ctx.alloc_sgpr_permanent(2, "s_ptr_A")
    ctx.alloc_sgpr_permanent(2, "s_ptr_B")
    ctx.alloc_sgpr_permanent(2, "s_ptr_D")
    ctx.alloc_sgpr_permanent(1, "s_M")
    ctx.alloc_sgpr_permanent(1, "s_N")
    ctx.alloc_sgpr_permanent(1, "s_K")
    ctx.alloc_sgpr_permanent(1, "s_k_tiles")
    ctx.alloc_sgpr_permanent(1, "s_tmp0")
    ctx.alloc_sgpr_permanent(1, "s_tmp1")

    # DTL-specific SGPRs
    ctx.alloc_sgpr_permanent(4, "s_srd_a")   # Buffer resource descriptor A
    ctx.alloc_sgpr_permanent(4, "s_srd_b")   # Buffer resource descriptor B
    ctx.alloc_sgpr_permanent(1, "s_lds_wr_a_sg")  # LDS write base A (for m0)
    ctx.alloc_sgpr_permanent(1, "s_lds_wr_b_sg")  # LDS write base B (for m0)
    ctx.alloc_sgpr_permanent(1, "s_k_stride")     # K * elem bytes (SRD advance)
    ctx.alloc_sgpr_permanent(1, "s_soffset_a")    # Scalar offset for 2nd A load
    ctx.alloc_sgpr_permanent(1, "s_soffset_b")    # Scalar offset for 2nd B load

    # MX scale SGPRs (Phase 2: real scale loading)
    if tile.mfma.is_mx:
        ctx.alloc_sgpr_permanent(2, "s_ptr_scale_a")  # Scale A pointer from kernargs
        ctx.alloc_sgpr_permanent(2, "s_ptr_scale_b")  # Scale B pointer from kernargs
        ctx.alloc_sgpr_permanent(1, "s_stride_scale_a")  # Scale A stride (bytes)
        ctx.alloc_sgpr_permanent(1, "s_stride_scale_b")  # Scale B stride (bytes)
        ctx.alloc_sgpr_permanent(4, "s_srd_scale_a")  # Scale A buffer resource descriptor
        ctx.alloc_sgpr_permanent(4, "s_srd_scale_b")  # Scale B buffer resource descriptor
        ctx.alloc_sgpr_permanent(1, "s_lds_wr_scale_a_sg")  # LDS write base for scale A
        ctx.alloc_sgpr_permanent(1, "s_lds_wr_scale_b_sg")  # LDS write base for scale B

    # Standard VGPRs
    ctx.alloc_vgpr_permanent(1, "v_tid")
    ctx.alloc_vgpr_permanent(1, "v_wave_id")
    ctx.alloc_vgpr_permanent(1, "v_lane_id")
    ctx.alloc_vgpr_permanent(1, "v_wave_m")
    ctx.alloc_vgpr_permanent(1, "v_wave_n")

    # DTL per-lane offsets (replace gload_row/col and addr_a/b)
    ctx.alloc_vgpr_permanent(1, "v_dtl_off_a")  # Per-lane buffer offset for A
    ctx.alloc_vgpr_permanent(1, "v_dtl_off_b")  # Per-lane buffer offset for B

    # LDS read addresses (same as non-DTL)
    ctx.alloc_vgpr_permanent(1, "v_lds_rd_a")
    ctx.alloc_vgpr_permanent(1, "v_lds_rd_b")
    # Per-ki LDS read base VGPRs (ki=0 uses v_lds_rd_a/b directly)
    ki_count = tile.unroll_k // tile.mfma.k if tile.mfma.k > 0 else 1
    for ki in range(1, ki_count):
        ctx.alloc_vgpr_permanent(1, f"v_lds_rd_a_k{ki}")
        ctx.alloc_vgpr_permanent(1, f"v_lds_rd_b_k{ki}")

    # MFMA operands (same as non-DTL)
    ctx.alloc_vgpr_permanent(tile.mfma.a_vgprs, "v_a")
    ctx.alloc_vgpr_permanent(tile.mfma.b_vgprs, "v_b")

    # MX scale VGPRs
    if tile.mfma.is_mx:
        # Constant scale fallback (Phase 1, still used by dtl_interleaved)
        ctx.alloc_vgpr_permanent(1, "v_mxscale")
        # DTL offset VGPR for scale loads
        ctx.alloc_vgpr_permanent(1, "v_dtl_off_scale_a")
        ctx.alloc_vgpr_permanent(1, "v_dtl_off_scale_b")
        # LDS read addresses for scale data
        ctx.alloc_vgpr_permanent(1, "v_lds_rd_scale_a")
        ctx.alloc_vgpr_permanent(1, "v_lds_rd_scale_b")
        # Per-(mi,ki) and per-(ni,ki) scale VGPRs are allocated dynamically
        # in dtl_partitioned.py's phase function

    # Store registers
    ctx.alloc_vgpr_permanent(2, "v_addr_d")
    ctx.alloc_vgpr_permanent(1, "v_store_tmp")
    ctx.alloc_vgpr_permanent(1, "v_tmp0")
    ctx.alloc_vgpr_permanent(1, "v_tmp1")
    ctx.alloc_vgpr_permanent(1, "v_tmp2")
    ctx.alloc_vgpr_permanent(1, "v_tmp3")
    ctx.alloc_vgpr_permanent(1, "v_tmp4")

    # Accumulators
    acc_total = tile.mfma_m_repeat * tile.mfma_n_repeat * tile.mfma.acc_vgprs
    ctx.alloc_acc_permanent(acc_total, "acc_C")
