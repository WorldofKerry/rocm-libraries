# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM kernel codegen using rocisa as the assembly backend.

Generates a complete GEMM kernel as a rocisa Module. str(module) gives
assembly text ready for assembling with amdclang++.
"""
from __future__ import annotations

import sys
import math
from typing import Optional

# rocisa lives in the tensilelite tree
_ROCISA_PATH = None
for p in sys.path:
    if "tensilelite" in p:
        _ROCISA_PATH = p
        break
if _ROCISA_PATH is None:
    import os
    candidate = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "..",
        "projects", "hipblaslt", "tensilelite",
    )
    if os.path.isdir(candidate):
        sys.path.insert(0, os.path.abspath(candidate))

from rocisa.code import Module, Label
from rocisa.instruction import (
    MFMAInstruction, VMovB32, VAddU32, VAndB32, VLShiftRightB32,
    VLShiftLeftB32, VMulLOU32, SMovB32, SMulI32, SAddU32, SSubU32,
    SBarrier, DSLoadB128, DSLoadB32, DSStoreB128, DSStoreB32,
    BufferLoadB128, SWaitCnt, VAccvgprWrite, VAccvgprReadB32,
    VCvtF32toF16, FlatStoreB32, SCBranchSCC0, SCmpEQU32,
)
from rocisa.container import vgpr, sgpr, accvgpr, DSModifiers, MUBUFModifiers
from rocisa.enum import InstType

from .problem import GemmProblem, TileConfig, MfmaConfig

__all__ = ["generate_rocisa_kernel"]


def generate_rocisa_kernel(
    problem: GemmProblem,
    tile: Optional[TileConfig] = None,
) -> Module:
    """Generate a complete GEMM kernel as a rocisa Module.

    The kernel implements::

        D[M,N] = A[M,K] @ B[K,N]   (fp16, no alpha/beta for v1)

    Structure:
        prologue:   thread/wave indexing, compute addresses
        init_acc:   zero accumulators
        k_loop:     for each K-tile:
                      global_load A, B -> VGPRs
                      wait vmcnt
                      lds_write A, B -> LDS
                      wait lgkmcnt + barrier
                      for ki in k_iterations:
                        lds_read A, B operands
                        wait lgkmcnt
                        4x4 MFMA grid
        epilogue:   acc -> vgpr -> flat_store D

    Returns:
        rocisa Module (call ``str(module)`` for assembly text)
    """
    if tile is None:
        tile = TileConfig()
    problem.validate(tile)

    p = problem
    t = tile
    mfma = t.mfma

    m = Module(f"gemm_f16_{t.wg_m}x{t.wg_n}x{t.unroll_k}")

    # Register assignments (manual, like TensileLite)
    # v0 = thread_id (ABI)
    V_TID       = 0
    V_WAVE_ID   = 1
    V_LANE_ID   = 2
    V_WAVE_M    = 3
    V_WAVE_N    = 4
    V_GLOAD_OFF = 5    # global load vgpr offset
    V_LDS_WR_A  = 6    # LDS write address A
    V_LDS_WR_B  = 7    # LDS write address B
    V_LDS_RD_A  = 8    # LDS read address A
    V_LDS_RD_B  = 9    # LDS read address B
    V_GLOAD_A   = 10   # 8 VGPRs: v[10:17]
    V_GLOAD_B   = 18   # 8 VGPRs: v[18:25]
    V_A         = 26   # 2 VGPRs: MFMA A operand v[26:27]
    V_B         = 28   # 2 VGPRs: MFMA B operand v[28:29]
    V_STORE_TMP = 30
    V_ADDR_D_LO = 31
    V_ADDR_D_HI = 32
    V_TMP0      = 33
    V_TMP1      = 34

    # SGPRs: kernel arguments (loaded by runtime before kernel launch)
    # s[0:1] = A ptr, s[2:3] = B ptr, s[4:5] = D ptr
    # s6 = M, s7 = N, s8 = K
    S_A_LO = 0; S_A_HI = 1
    S_B_LO = 2; S_B_HI = 3
    S_D_LO = 4; S_D_HI = 5
    S_M = 6; S_N = 7; S_K = 8
    S_KTILES = 9
    S_TMP0 = 10; S_TMP1 = 11; S_TMP2 = 12

    # Accumulator regs: 4x4 tiles * 4 VGPRs = 64
    ACC_BASE = 0
    acc_per = mfma.acc_vgprs  # 4

    # LDS layout
    # A: [0, wg_m * unroll_k * 2)  bytes
    # B: [wg_m * unroll_k * 2, ...)
    lds_a_bytes = t.wg_m * t.unroll_k * p.element_bytes
    lds_b_offset = lds_a_bytes

    # ================= PROLOGUE =================
    m.add(Label("prologue", ""))

    # wave_id = tid >> 6, lane_id = tid & 63
    m.add(VLShiftRightB32(dst=vgpr(V_WAVE_ID), shiftHex=vgpr(V_TID), src=6,
                          comment="wave_id = tid >> 6"))
    m.add(VAndB32(dst=vgpr(V_LANE_ID), src0=vgpr(V_TID), src1=63,
                  comment="lane_id = tid & 63"))

    # wave_m = wave_id >> 1, wave_n = wave_id & 1
    m.add(VLShiftRightB32(dst=vgpr(V_WAVE_M), shiftHex=vgpr(V_WAVE_ID), src=1,
                          comment="wave_m = wave_id >> 1"))
    m.add(VAndB32(dst=vgpr(V_WAVE_N), src0=vgpr(V_WAVE_ID), src1=1,
                  comment="wave_n = wave_id & 1"))

    # Compute global load cluster mapping for A[M_wg, K_unroll]
    # Each thread loads: elems_per_thread = (wg_m * unroll_k) / block_size
    # Thread cluster: cluster_k = 8 (vector_width for fp16), cluster_m = blocksize/8
    cluster_k = t.vector_width  # 8
    cluster_m = t.block_size // cluster_k  # 32

    # thread_row = (tid / cluster_k) % cluster_m
    # thread_col = tid % cluster_k
    m.add(VLShiftRightB32(dst=vgpr(V_TMP0), shiftHex=vgpr(V_TID),
                          src=int(math.log2(cluster_k)),
                          comment=f"tid / {cluster_k}"))
    m.add(VAndB32(dst=vgpr(V_TMP0), src0=vgpr(V_TMP0), src1=cluster_m - 1,
                  comment=f"thread_row = (tid/{cluster_k}) % {cluster_m}"))
    m.add(VAndB32(dst=vgpr(V_TMP1), src0=vgpr(V_TID), src1=cluster_k - 1,
                  comment=f"thread_col = tid % {cluster_k}"))

    # LDS write offset A = (thread_row * unroll_k + thread_col) * elem_bytes
    m.add(VMulLOU32(dst=vgpr(V_LDS_WR_A), src0=vgpr(V_TMP0),
                    src1=t.unroll_k, comment="thread_row * unroll_k"))
    m.add(VAddU32(dst=vgpr(V_LDS_WR_A), src0=vgpr(V_LDS_WR_A),
                  src1=vgpr(V_TMP1), comment="+ thread_col"))
    m.add(VLShiftLeftB32(dst=vgpr(V_LDS_WR_A), shiftHex=vgpr(V_LDS_WR_A),
                         src=int(math.log2(p.element_bytes)),
                         comment=f"* {p.element_bytes} (bytes)"))

    # LDS write offset B = lds_b_offset + (thread_row * wg_n + thread_col) * elem_bytes
    m.add(VMulLOU32(dst=vgpr(V_LDS_WR_B), src0=vgpr(V_TMP0),
                    src1=t.wg_n, comment="thread_row * wg_n"))
    m.add(VAddU32(dst=vgpr(V_LDS_WR_B), src0=vgpr(V_LDS_WR_B),
                  src1=vgpr(V_TMP1), comment="+ thread_col"))
    m.add(VLShiftLeftB32(dst=vgpr(V_LDS_WR_B), shiftHex=vgpr(V_LDS_WR_B),
                         src=int(math.log2(p.element_bytes)),
                         comment=f"* {p.element_bytes}"))
    m.add(VAddU32(dst=vgpr(V_LDS_WR_B), src0=vgpr(V_LDS_WR_B),
                  src1=lds_b_offset, comment=f"+ lds_b_offset ({lds_b_offset})"))

    # LDS read offset A = (wave_m * m_per_wave + lane_row) * unroll_k * elem_bytes
    # lane_row = lane_id % mfma_m (for 16x16x16: lane_row = lane_id % 16)
    m.add(VAndB32(dst=vgpr(V_TMP0), src0=vgpr(V_LANE_ID), src1=mfma.m - 1,
                  comment=f"lane_row = lane_id % {mfma.m}"))
    m.add(VMulLOU32(dst=vgpr(V_LDS_RD_A), src0=vgpr(V_WAVE_M),
                    src1=t.m_per_wave, comment="wave_m * m_per_wave"))
    m.add(VAddU32(dst=vgpr(V_LDS_RD_A), src0=vgpr(V_LDS_RD_A),
                  src1=vgpr(V_TMP0), comment="+ lane_row"))
    m.add(VMulLOU32(dst=vgpr(V_LDS_RD_A), src0=vgpr(V_LDS_RD_A),
                    src1=t.unroll_k * p.element_bytes,
                    comment=f"* unroll_k * elem_bytes"))

    # LDS read offset B = lds_b_offset + (wave_n * n_per_wave + lane_row) * ??? 
    # For B stored as [K, N]: lds_rd_b = lds_b_offset + lane_col * N_wg * elem + ...
    # Simplified: base + (wave_n * n_per_wave + lane_row) * elem
    m.add(VMulLOU32(dst=vgpr(V_LDS_RD_B), src0=vgpr(V_WAVE_N),
                    src1=t.n_per_wave, comment="wave_n * n_per_wave"))
    m.add(VAddU32(dst=vgpr(V_LDS_RD_B), src0=vgpr(V_LDS_RD_B),
                  src1=vgpr(V_TMP0), comment="+ lane_row"))
    m.add(VMulLOU32(dst=vgpr(V_LDS_RD_B), src0=vgpr(V_LDS_RD_B),
                    src1=p.element_bytes, comment=f"* elem_bytes"))
    m.add(VAddU32(dst=vgpr(V_LDS_RD_B), src0=vgpr(V_LDS_RD_B),
                  src1=lds_b_offset, comment=f"+ lds_b_offset"))

    # ================= INIT ACCUMULATORS =================
    m.add(Label("init_acc", ""))
    total_acc = t.mfma_m_repeat * t.mfma_n_repeat * acc_per
    for i in range(total_acc):
        m.add(VAccvgprWrite(dst=accvgpr(ACC_BASE + i), src=0,
                            comment=f"acc[{i}]=0"))

    # ================= K TILE LOOP =================
    # s_ktiles = K / unroll_k
    m.add(Label("k_loop_setup", ""))
    m.add(SMovB32(dst=sgpr(S_KTILES), src=p.k // t.unroll_k,
                  comment=f"k_tiles = {p.k // t.unroll_k}"))

    m.add(Label("k_loop", ""))

    # -- Global load A, B into VGPRs --
    mubuf = MUBUFModifiers(offen=True, offset12=0)
    for i in range(2):  # 2 x buffer_load_b128 = 8 VGPRs for A
        m.add(BufferLoadB128(
            dst=vgpr(V_GLOAD_A + i*4, 4),
            vaddr=vgpr(V_GLOAD_OFF), saddr=sgpr(S_A_LO, 4), soffset=0,
            mubuf=mubuf,
            comment=f"gload A chunk {i}",
        ))
    for i in range(2):  # 2 x buffer_load_b128 for B
        m.add(BufferLoadB128(
            dst=vgpr(V_GLOAD_B + i*4, 4),
            vaddr=vgpr(V_GLOAD_OFF), saddr=sgpr(S_B_LO, 4), soffset=0,
            mubuf=mubuf,
            comment=f"gload B chunk {i}",
        ))

    # Wait for global loads
    m.add(SWaitCnt(vlcnt=0, comment="wait global loads"))

    # -- LDS write --
    ds = DSModifiers(offset=0)
    for i in range(2):
        m.add(DSStoreB128(
            dstAddr=vgpr(V_LDS_WR_A), src=vgpr(V_GLOAD_A + i*4, 4),
            ds=ds, comment=f"lds write A[{i}]",
        ))
    for i in range(2):
        m.add(DSStoreB128(
            dstAddr=vgpr(V_LDS_WR_B), src=vgpr(V_GLOAD_B + i*4, 4),
            ds=ds, comment=f"lds write B[{i}]",
        ))

    m.add(SWaitCnt(dscnt=0, comment="wait lds writes"))
    m.add(SBarrier(comment="sync workgroup"))

    # -- Inner K loop: LDS read + MFMA --
    for ki in range(t.k_iterations):
        # LDS read A operand (2 VGPRs = 4 x f16)
        for r in range(mfma.a_vgprs):
            ds_off = DSModifiers(offset=ki * mfma.k * p.element_bytes + r * 4)
            m.add(DSLoadB32(
                dst=vgpr(V_A + r), src=vgpr(V_LDS_RD_A),
                ds=ds_off, comment=f"lds read A[{r}] k={ki}",
            ))
        # LDS read B operand
        for r in range(mfma.b_vgprs):
            ds_off = DSModifiers(offset=ki * mfma.k * p.element_bytes + r * 4)
            m.add(DSLoadB32(
                dst=vgpr(V_B + r), src=vgpr(V_LDS_RD_B),
                ds=ds_off, comment=f"lds read B[{r}] k={ki}",
            ))

        m.add(SWaitCnt(dscnt=0, comment="wait lds reads"))

        # 4x4 MFMA grid
        for mi in range(t.mfma_m_repeat):
            for ni in range(t.mfma_n_repeat):
                acc_off = ACC_BASE + (mi * t.mfma_n_repeat + ni) * acc_per
                m.add(MFMAInstruction(
                    instType=InstType.INST_F16, accType=InstType.INST_F32,
                    variant=[mfma.m, mfma.n, mfma.k, mfma.blocks],
                    mfma1k=False,
                    acc=accvgpr(acc_off, acc_per),
                    a=vgpr(V_A, mfma.a_vgprs),
                    b=vgpr(V_B, mfma.b_vgprs),
                    comment=f"mfma m{mi}_n{ni} k{ki}",
                ))

    # -- Loop back --
    m.add(SSubU32(dst=sgpr(S_KTILES), src0=sgpr(S_KTILES), src1=1,
                  comment="k_tiles--"))
    m.add(SCBranchSCC0(labelName="k_loop", comment="loop if k_tiles != 0"))

    # ================= EPILOGUE =================
    m.add(Label("epilogue", ""))
    n_tiles = t.mfma_m_repeat * t.mfma_n_repeat
    for t_idx in range(n_tiles):
        for r in range(acc_per):
            acc_off = ACC_BASE + t_idx * acc_per + r
            m.add(VAccvgprReadB32(
                dst=vgpr(V_STORE_TMP), src=accvgpr(acc_off),
                comment=f"acc->vgpr tile{t_idx}[{r}]",
            ))
            m.add(VCvtF32toF16(
                dst=vgpr(V_STORE_TMP), src=vgpr(V_STORE_TMP),
                comment="f32->f16",
            ))
            m.add(FlatStoreB32(
                src=vgpr(V_STORE_TMP), vaddr=vgpr(V_ADDR_D_LO, 2),
                comment=f"store D[{t_idx},{r}]",
            ))

    m.add(Label("kernel_end", ""))

    return m
