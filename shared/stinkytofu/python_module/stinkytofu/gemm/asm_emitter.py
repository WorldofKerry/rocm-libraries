# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Direct assembly text emitter for gfx950 GEMM kernels.

Generates a complete .s file that assembles with amdclang++ into a
.co code object runnable via hipModuleLoad.

No dependency on stinkytofu or rocisa -- emits raw assembly strings
using our TileConfig for register/LDS layout decisions. The kernel
structure matches what our TileLevel tree describes.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from .problem import GemmProblem, TileConfig, MfmaConfig

__all__ = ["AsmKernel", "emit_gemm_asm", "assemble_kernel"]


@dataclass
class AsmKernel:
    """A generated assembly kernel ready to assemble."""
    asm_text: str
    kernel_name: str
    problem: GemmProblem
    tile: TileConfig
    vgpr_count: int
    sgpr_count: int
    acc_count: int
    lds_bytes: int

    def save(self, path: str) -> str:
        with open(path, "w") as f:
            f.write(self.asm_text)
        return path

    def assemble(self, gpu_arch: str = "gfx950",
                 output_path: Optional[str] = None) -> str:
        """Assemble into a .co code object. Returns path to .co file."""
        return assemble_kernel(self.asm_text, gpu_arch, output_path)


def assemble_kernel(asm_text: str, gpu_arch: str = "gfx950",
                    output_path: Optional[str] = None) -> str:
    """Assemble text into a .co code object."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".s", delete=False) as f:
        f.write(asm_text)
        s_path = f.name

    o_path = s_path.replace(".s", ".o")
    co_path = output_path or s_path.replace(".s", ".co")

    # Assemble
    r = subprocess.run(
        ["amdclang++", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         f"-mcpu={gpu_arch}", "-c", s_path, "-o", o_path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Assembly failed:\n{r.stderr}")

    # Link
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


def emit_gemm_asm(
    problem: GemmProblem,
    tile: Optional[TileConfig] = None,
    kernel_name: str = "gemm_kernel",
) -> AsmKernel:
    """Generate a complete GEMM kernel as gfx950 assembly.

    The kernel implements: D[M,N] = A[M,K] @ B[K,N]  (fp16, alpha=1)

    A is row-major [M, K], B is row-major [K, N], D is row-major [M, N].

    Returns an AsmKernel with the full .s text.
    """
    if tile is None:
        tile = TileConfig()
    problem.validate(tile)

    t = tile
    p = problem
    mfma = t.mfma

    lines: List[str] = []
    def emit(s: str = ""):
        lines.append(s)
    def comment(s: str):
        lines.append(f"    // {s}")
    def inst(s: str, c: str = ""):
        if c:
            # Pad instruction to column 50 for comment alignment
            lines.append(f"    {s:<46s}// {c}")
        else:
            lines.append(f"    {s}")
    def label(s: str):
        lines.append(f"{s}:")

    # Register map
    V_TID = 0
    V_WAVE_ID = 1
    V_LANE_ID = 2
    V_WAVE_M = 3
    V_WAVE_N = 4
    V_TMP0 = 5
    V_TMP1 = 6
    V_LDS_WR_A = 7
    V_LDS_WR_B = 8
    V_LDS_RD_A = 9
    V_LDS_RD_B = 10
    V_A = 12  # MFMA A operand: v[12:13]
    V_B = 14  # MFMA B operand: v[14:15]
    V_GLOAD_A = 16  # v[16:23] = 8 VGPRs for A tile
    V_GLOAD_B = 24  # v[24:31] = 8 VGPRs for B tile
    V_STORE_TMP = 32
    V_ADDR_D = 34  # v[34:35] = 64-bit D address
    VGPR_COUNT = 36

    ACC_BASE = 0
    acc_per = mfma.acc_vgprs
    ACC_COUNT = t.mfma_m_repeat * t.mfma_n_repeat * acc_per

    # SGPRs: s[0:1] = kernarg ptr, s[2:3] = workgroup_id packed
    S_KARG_LO = 0
    S_KARG_HI = 1
    S_A_LO = 4; S_A_HI = 5  # loaded from kernarg
    S_B_LO = 6; S_B_HI = 7
    S_D_LO = 8; S_D_HI = 9
    S_M = 10; S_N = 11; S_K = 12
    S_KTILES = 13
    S_TMP0 = 14; S_TMP1 = 15
    SGPR_COUNT = 16

    # LDS layout
    lds_a_elems = t.wg_m * t.unroll_k
    lds_a_bytes = lds_a_elems * p.element_bytes
    lds_b_offset = lds_a_bytes
    lds_total = lds_a_bytes + t.wg_n * t.unroll_k * p.element_bytes

    cluster_k = t.vector_width  # 8
    cluster_m = t.block_size // cluster_k

    # ==== Header ====
    emit(f'.amdgcn_target "amdgcn-amd-amdhsa--gfx950"')
    emit(".text")
    emit(f".globl {kernel_name}")
    emit(".p2align 8")
    emit(f".type {kernel_name},@function")
    emit()
    emit(f"{kernel_name}:")

    # ==== Load kernel args ====
    comment("Load kernel arguments from kernarg segment")
    inst(f"s_load_dwordx2 s[{S_A_LO}:{S_A_HI}], s[{S_KARG_LO}:{S_KARG_HI}], 0", "A ptr")
    inst(f"s_load_dwordx2 s[{S_B_LO}:{S_B_HI}], s[{S_KARG_LO}:{S_KARG_HI}], 8", "B ptr")
    inst(f"s_load_dwordx2 s[{S_D_LO}:{S_D_HI}], s[{S_KARG_LO}:{S_KARG_HI}], 16", "D ptr")
    inst(f"s_load_dword s{S_M}, s[{S_KARG_LO}:{S_KARG_HI}], 24", "M")
    inst(f"s_load_dword s{S_N}, s[{S_KARG_LO}:{S_KARG_HI}], 28", "N")
    inst(f"s_load_dword s{S_K}, s[{S_KARG_LO}:{S_KARG_HI}], 32", "K")
    inst("s_waitcnt lgkmcnt(0)", "wait for kernarg loads")
    emit()

    # ==== Thread/wave indexing ====
    comment("Thread indexing")
    inst(f"v_lshrrev_b32 v{V_WAVE_ID}, 6, v{V_TID}", "wave_id = tid >> 6")
    inst(f"v_and_b32 v{V_LANE_ID}, 63, v{V_TID}", "lane_id = tid & 63")
    inst(f"v_lshrrev_b32 v{V_WAVE_M}, 1, v{V_WAVE_ID}", "wave_m = wave_id >> 1")
    inst(f"v_and_b32 v{V_WAVE_N}, 1, v{V_WAVE_ID}", "wave_n = wave_id & 1")
    emit()

    # ==== Compute LDS write offsets ====
    comment(f"Global load cluster: {cluster_m} x {cluster_k}")
    log2_ck = int(math.log2(cluster_k))
    inst(f"v_lshrrev_b32 v{V_TMP0}, {log2_ck}, v{V_TID}",
         f"tid / {cluster_k}")
    inst(f"v_and_b32 v{V_TMP0}, {cluster_m - 1}, v{V_TMP0}",
         f"thread_row = (tid/{cluster_k}) % {cluster_m}")
    inst(f"v_and_b32 v{V_TMP1}, {cluster_k - 1}, v{V_TID}",
         f"thread_col = tid % {cluster_k}")
    emit()

    comment("LDS write offset A = (row * unroll_k + col) * elem_bytes")
    inst(f"v_mul_lo_u32 v{V_LDS_WR_A}, {t.unroll_k}, v{V_TMP0}",
         "row * unroll_k")
    inst(f"v_add_u32 v{V_LDS_WR_A}, v{V_LDS_WR_A}, v{V_TMP1}",
         "+ col")
    inst(f"v_lshlrev_b32 v{V_LDS_WR_A}, {int(math.log2(p.element_bytes))}, v{V_LDS_WR_A}",
         f"* {p.element_bytes}")
    emit()

    comment("LDS write offset B = lds_b_offset + (row * wg_n + col) * elem_bytes")
    inst(f"s_mov_b32 s{S_TMP0}, {t.wg_n}", "wg_n")
    inst(f"v_mul_lo_u32 v{V_LDS_WR_B}, s{S_TMP0}, v{V_TMP0}",
         "row * wg_n")
    inst(f"v_add_u32 v{V_LDS_WR_B}, v{V_LDS_WR_B}, v{V_TMP1}",
         "+ col")
    inst(f"v_lshlrev_b32 v{V_LDS_WR_B}, {int(math.log2(p.element_bytes))}, v{V_LDS_WR_B}",
         f"* {p.element_bytes}")
    inst(f"s_mov_b32 s{S_TMP0}, {lds_b_offset}", "lds_b_offset")
    inst(f"v_add_u32 v{V_LDS_WR_B}, s{S_TMP0}, v{V_LDS_WR_B}",
         f"+ {lds_b_offset}")
    emit()

    # ==== Compute LDS read offsets ====
    comment("LDS read offset A: (wave_m * m_per_wave + lane_row) * unroll_k * elem")
    inst(f"v_and_b32 v{V_TMP0}, {mfma.m - 1}, v{V_LANE_ID}",
         f"lane_row = lane_id % {mfma.m}")
    inst(f"v_mul_lo_u32 v{V_LDS_RD_A}, {t.m_per_wave}, v{V_WAVE_M}",
         "wave_m * m_per_wave")
    inst(f"v_add_u32 v{V_LDS_RD_A}, v{V_LDS_RD_A}, v{V_TMP0}",
         "+ lane_row")
    inst(f"v_mul_lo_u32 v{V_LDS_RD_A}, {t.unroll_k * p.element_bytes}, v{V_LDS_RD_A}",
         f"* {t.unroll_k * p.element_bytes}")
    emit()

    comment("LDS read offset B: lds_b_offset + (wave_n * n_per_wave + lane_row) * elem")
    inst(f"v_mul_lo_u32 v{V_LDS_RD_B}, {t.n_per_wave}, v{V_WAVE_N}",
         "wave_n * n_per_wave")
    inst(f"v_add_u32 v{V_LDS_RD_B}, v{V_LDS_RD_B}, v{V_TMP0}",
         "+ lane_row")
    inst(f"v_mul_lo_u32 v{V_LDS_RD_B}, {p.element_bytes}, v{V_LDS_RD_B}",
         f"* {p.element_bytes}")
    inst(f"s_mov_b32 s{S_TMP0}, {lds_b_offset}", "lds_b_offset")
    inst(f"v_add_u32 v{V_LDS_RD_B}, s{S_TMP0}, v{V_LDS_RD_B}",
         f"+ {lds_b_offset}")
    emit()

    # ==== Init accumulators ====
    comment(f"Init {ACC_COUNT} accumulator registers")
    for i in range(ACC_COUNT):
        inst(f"v_accvgpr_write acc{ACC_BASE + i}, 0")
    emit()

    # ==== K-tile loop ====
    comment("K-tile loop setup")
    inst(f"s_lshr_b32 s{S_KTILES}, s{S_K}, {int(math.log2(t.unroll_k))}",
         f"k_tiles = K / {t.unroll_k}")
    emit()

    label("k_loop")
    emit()

    # Global load A: use flat_load since we have 64-bit addresses
    # For simplicity, compute flat address:
    #   addr_A = A_ptr + (wg_m * blockIdx.x + thread_row) * K * 2 + thread_col * 2
    # This needs workgroup ID which comes in s2 for dim0
    # For now: use buffer_load with SRD = {A_ptr, size, stride, 0}
    # Actually for v1: skip the global load and just do LDS read + MFMA to verify the structure works

    comment("TODO: global load A, B (using flat_load_dwordx4)")
    comment("For now: LDS is assumed pre-filled (testing structure only)")
    emit()

    # LDS read + MFMA inner loop
    for ki in range(t.k_iterations):
        comment(f"--- K iteration {ki} ---")
        # LDS read A operand
        k_byte_off = ki * mfma.k * p.element_bytes
        for r in range(mfma.a_vgprs):
            off = k_byte_off + r * 4
            inst(f"ds_read_b32 v{V_A + r}, v{V_LDS_RD_A} offset:{off}",
                 f"LDS read A[{r}] k={ki}")

        # LDS read B operand
        for r in range(mfma.b_vgprs):
            off = k_byte_off + r * 4
            inst(f"ds_read_b32 v{V_B + r}, v{V_LDS_RD_B} offset:{off}",
                 f"LDS read B[{r}] k={ki}")

        inst("s_waitcnt lgkmcnt(0)", "wait LDS reads")
        emit()

        # MFMA grid
        for mi in range(t.mfma_m_repeat):
            for ni in range(t.mfma_n_repeat):
                acc_off = ACC_BASE + (mi * t.mfma_n_repeat + ni) * acc_per
                inst(
                    f"v_mfma_f32_16x16x16_f16 "
                    f"acc[{acc_off}:{acc_off + acc_per - 1}], "
                    f"v[{V_A}:{V_A + mfma.a_vgprs - 1}], "
                    f"v[{V_B}:{V_B + mfma.b_vgprs - 1}], "
                    f"acc[{acc_off}:{acc_off + acc_per - 1}]",
                    f"mfma m{mi}_n{ni} k{ki}",
                )
        emit()

    # Loop back
    inst(f"s_sub_u32 s{S_KTILES}, s{S_KTILES}, 1", "k_tiles--")
    inst(f"s_cbranch_scc0 k_loop", "loop if k_tiles != 0")
    emit()

    # ==== Epilogue ====
    comment("Epilogue: store accumulators to D")
    comment("TODO: compute D addresses and store")
    comment("For now: just end")
    inst("s_endpgm")
    emit()

    # ==== Kernel descriptor ====
    emit(".rodata")
    emit(".p2align 6")
    emit(f".amdhsa_kernel {kernel_name}")
    emit(f"    .amdhsa_group_segment_fixed_size {lds_total}")
    emit(f"    .amdhsa_private_segment_fixed_size 0")
    emit(f"    .amdhsa_kernarg_size 64")
    emit(f"    .amdhsa_user_sgpr_kernarg_segment_ptr 1")
    emit(f"    .amdhsa_system_vgpr_workitem_id 0")
    emit(f"    .amdhsa_next_free_vgpr {VGPR_COUNT}")
    emit(f"    .amdhsa_next_free_sgpr {SGPR_COUNT}")
    emit(f"    .amdhsa_accum_offset {VGPR_COUNT}")
    emit(f"    .amdhsa_float_denorm_mode_32 3")
    emit(f"    .amdhsa_float_denorm_mode_16_64 3")
    emit(f".end_amdhsa_kernel")

    asm_text = "\n".join(lines) + "\n"

    return AsmKernel(
        asm_text=asm_text,
        kernel_name=kernel_name,
        problem=problem,
        tile=tile,
        vgpr_count=VGPR_COUNT,
        sgpr_count=SGPR_COUNT,
        acc_count=ACC_COUNT,
        lds_bytes=lds_total,
    )
