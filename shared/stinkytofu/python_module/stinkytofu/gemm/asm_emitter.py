# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM kernel assembly emitter using the tile-tree infrastructure.

Generates a complete .s file for gfx950 using:
- ``AsmContext`` for register allocation (named bindings, scoped lifetimes)
- ``TileLevel`` tree + ``walk_tile_tree`` for the MFMA loop structure
- ``AddressComputer`` formulas for offset computation

No dependency on stinkytofu or rocisa -- emits raw assembly strings.
Assembles with amdclang++ into a .co code object runnable via hipModuleLoad.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from .asm_context import AsmContext
from .addressing import AddressComputer
from .problem import GemmProblem, TileConfig, MfmaConfig, DataType
from .tile import TileLevel, build_gemm_tile_tree, walk_tile_tree
from .asm_transforms import emit_affine, GemmLayouts
from .transforms import Embed, Dim

__all__ = ["AsmKernel", "emit_gemm_asm", "assemble_kernel"]


@dataclass
class AsmKernel:
    """A generated assembly kernel ready to assemble."""
    asm_text: str
    kernel_name: str
    problem: GemmProblem
    tile: TileConfig
    ctx: AsmContext
    lds_bytes: int

    @property
    def vgpr_count(self) -> int:
        return self.ctx._next["v"]

    @property
    def sgpr_count(self) -> int:
        return self.ctx._next["s"]

    @property
    def acc_count(self) -> int:
        return self.ctx._next["acc"]

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
# Register allocation using AsmContext (named bindings)
# ===================================================================

def _alloc_registers(ctx: AsmContext, problem: GemmProblem,
                     tile: TileConfig) -> None:
    """Allocate all kernel registers using named bindings.

    ABI: s[0:1] = kernarg ptr, s2 = workgroup_id_x, s3 = workgroup_id_y
    """
    ctx.alloc_sgpr_permanent(2, "s_kernarg")   # s[0:1]
    ctx.alloc_sgpr_permanent(1, "s_wg_id_x")   # s2
    ctx.alloc_sgpr_permanent(1, "s_wg_id_y")   # s3

    # Kernel arguments loaded from kernarg segment
    ctx.alloc_sgpr_permanent(2, "s_ptr_A")
    ctx.alloc_sgpr_permanent(2, "s_ptr_B")
    ctx.alloc_sgpr_permanent(2, "s_ptr_D")
    ctx.alloc_sgpr_permanent(1, "s_M")
    ctx.alloc_sgpr_permanent(1, "s_N")
    ctx.alloc_sgpr_permanent(1, "s_K")
    ctx.alloc_sgpr_permanent(1, "s_k_tiles")
    ctx.alloc_sgpr_permanent(1, "s_tmp0")
    ctx.alloc_sgpr_permanent(1, "s_tmp1")

    # Thread indexing
    ctx.alloc_vgpr_permanent(1, "v_tid")       # workitem ID (from HW v0)
    ctx.alloc_vgpr_permanent(1, "v_wave_id")
    ctx.alloc_vgpr_permanent(1, "v_lane_id")
    ctx.alloc_vgpr_permanent(1, "v_wave_m")
    ctx.alloc_vgpr_permanent(1, "v_wave_n")

    # Global load thread cluster coords
    ctx.alloc_vgpr_permanent(1, "v_gload_row")
    ctx.alloc_vgpr_permanent(1, "v_gload_col")

    # LDS write/read address registers
    ctx.alloc_vgpr_permanent(1, "v_lds_wr_a")
    ctx.alloc_vgpr_permanent(1, "v_lds_wr_b")
    ctx.alloc_vgpr_permanent(1, "v_lds_rd_a")
    ctx.alloc_vgpr_permanent(1, "v_lds_rd_b")

    # MFMA operand registers
    ctx.alloc_vgpr_permanent(tile.mfma.a_vgprs, "v_a")
    ctx.alloc_vgpr_permanent(tile.mfma.b_vgprs, "v_b")

    # Global load data buffers
    elem = problem.element_bytes
    a_elems = (tile.wg_m * tile.unroll_k) // tile.block_size
    b_elems = (tile.wg_n * tile.unroll_k) // tile.block_size
    a_vgprs = max(1, (a_elems * elem + 3) // 4)
    b_vgprs = max(1, (b_elems * elem + 3) // 4)
    ctx.alloc_vgpr_permanent(a_vgprs, "v_gload_a")
    ctx.alloc_vgpr_permanent(b_vgprs, "v_gload_b")

    # 64-bit global addresses
    ctx.alloc_vgpr_permanent(2, "v_addr_a")
    ctx.alloc_vgpr_permanent(2, "v_addr_b")
    ctx.alloc_vgpr_permanent(2, "v_addr_d")

    # Temporaries
    ctx.alloc_vgpr_permanent(1, "v_store_tmp")
    ctx.alloc_vgpr_permanent(1, "v_tmp0")
    ctx.alloc_vgpr_permanent(1, "v_tmp1")

    # Accumulators
    acc_total = tile.mfma_m_repeat * tile.mfma_n_repeat * tile.mfma.acc_vgprs
    ctx.alloc_acc_permanent(acc_total, "acc_C")


# ===================================================================
# Prologue: header, kernarg load, thread indexing, address setup
# ===================================================================

def _emit_header(ctx: AsmContext, kernel_name: str) -> None:
    ctx.raw(f'.amdgcn_target "amdgcn-amd-amdhsa--gfx950"')
    ctx.raw(".text")
    ctx.raw(f".globl {kernel_name}")
    ctx.raw(".p2align 8")
    ctx.raw(f".type {kernel_name},@function")
    ctx.raw("")
    ctx.raw(f"{kernel_name}:")


def _emit_load_kernargs(ctx: AsmContext) -> None:
    """Load kernel arguments from the kernarg segment."""
    ctx.comment("Load kernel arguments")
    karg = ctx.sreg("s_kernarg")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_A"), karg, "0",
             comment="A ptr")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_B"), karg, "8",
             comment="B ptr")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_D"), karg, "16",
             comment="D ptr")
    ctx.inst("s_load_dword", ctx.sreg("s_M"), karg, "24", comment="M")
    ctx.inst("s_load_dword", ctx.sreg("s_N"), karg, "28", comment="N")
    ctx.inst("s_load_dword", ctx.sreg("s_K"), karg, "32", comment="K")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait for kernarg loads")
    ctx.raw("")


def _emit_thread_indexing(ctx: AsmContext, tile: TileConfig) -> None:
    """Compute wave_id, lane_id, wave_m, wave_n from thread ID."""
    ctx.comment("Thread indexing")
    log2_ws = int(math.log2(tile.wave_size))
    ctx.v_lshr(ctx.vreg("v_wave_id"), ctx.vreg("v_tid"), log2_ws,
               comment=f"wave_id = tid >> {log2_ws}")
    ctx.v_and(ctx.vreg("v_lane_id"), ctx.vreg("v_tid"), tile.wave_size - 1,
              comment=f"lane_id = tid & {tile.wave_size - 1}")
    if tile.waves_n > 1:
        log2_wn = int(math.log2(tile.waves_n))
        ctx.v_lshr(ctx.vreg("v_wave_m"), ctx.vreg("v_wave_id"), log2_wn,
                   comment=f"wave_m = wave_id >> {log2_wn}")
        ctx.v_and(ctx.vreg("v_wave_n"), ctx.vreg("v_wave_id"),
                  tile.waves_n - 1,
                  comment=f"wave_n = wave_id & {tile.waves_n - 1}")
    else:
        ctx.v_mov(ctx.vreg("v_wave_m"), ctx.vreg("v_wave_id"),
                  comment="wave_m = wave_id (waves_n=1)")
        ctx.v_mov(ctx.vreg("v_wave_n"), "0", comment="wave_n = 0")
    ctx.raw("")


def _emit_global_load_cluster(ctx: AsmContext, tile: TileConfig,
                              problem: GemmProblem) -> None:
    """Compute global-load thread cluster coordinates (row, col).

    Each thread loads ``elems_per_thread`` contiguous elements along K.
    Threads are split: row selects M-position, col selects K-start.

      row = tid / k_groups       (covers wg_m rows)
      col = (tid % k_groups) * contiguous_k   (K-start offset)
    """
    elem = problem.element_bytes
    elems_per_thread = (tile.wg_m * tile.unroll_k) // tile.block_size
    contiguous_k = min(elems_per_thread, tile.unroll_k)
    k_groups = max(1, tile.unroll_k // contiguous_k)
    m_coverage = tile.block_size // k_groups

    ctx.comment(f"Global-load cluster: {m_coverage} rows x {k_groups} "
                f"K-groups ({contiguous_k} elems each)")
    if k_groups == 1:
        # All threads cover different rows, col = 0
        ctx.v_mov(ctx.vreg("v_gload_row"), ctx.vreg("v_tid"),
                  comment="row = tid (k_groups=1)")
        ctx.v_mov(ctx.vreg("v_gload_col"), "0", comment="col = 0")
    else:
        log2_kg = int(math.log2(k_groups))
        ctx.v_lshr(ctx.vreg("v_gload_row"), ctx.vreg("v_tid"), log2_kg,
                   comment=f"row = tid >> {log2_kg}")
        ctx.v_and(ctx.vreg("v_gload_col"), ctx.vreg("v_tid"), k_groups - 1,
                  comment=f"tid % {k_groups}")
        if contiguous_k > 1:
            log2_ck = int(math.log2(contiguous_k))
            ctx.v_lshl(ctx.vreg("v_gload_col"), ctx.vreg("v_gload_col"),
                       log2_ck,
                       comment=f"* {contiguous_k} -> k_start")
    ctx.raw("")


def _emit_lds_write_addrs(ctx: AsmContext, problem: GemmProblem,
                          tile: TileConfig,
                          layouts: GemmLayouts) -> None:
    """Compute LDS write offsets for A and B using coordinate transforms."""
    bindings = {
        "row": ctx.vreg("v_gload_row"),
        "col": ctx.vreg("v_gload_col"),
    }

    ctx.comment(f"LDS write A: {layouts.lds_a}")
    emit_affine(ctx, layouts.lds_a, bindings,
                result=ctx.vreg("v_lds_wr_a"),
                scale=layouts.elem_bytes,
                comment="lds_wr_a = (row * unroll_k + col) * elem")
    ctx.raw("")

    ctx.comment(f"LDS write B: {layouts.lds_b} + offset {layouts.lds_b_offset}")
    emit_affine(ctx, layouts.lds_b, bindings,
                result=ctx.vreg("v_lds_wr_b"),
                scale=layouts.elem_bytes,
                base=str(layouts.lds_b_offset),
                comment="lds_wr_b = lds_b_offset + (row * unroll_k + col) * elem")
    ctx.raw("")


def _emit_lds_read_addrs(ctx: AsmContext, problem: GemmProblem,
                         tile: TileConfig, layouts: GemmLayouts) -> None:
    """Compute initial LDS read offsets for A and B using transforms.

    The MFMA lane mapping determines which elements each lane reads:
      row = lane_id % mfma_m
      k   = (lane_id / mfma_m) * k_per_group
    These are combined with the wave position via the LDS layout Embed.
    """
    mfma = tile.mfma
    elem = problem.element_bytes
    k_per_group = mfma.k // (tile.wave_size // mfma.m)

    ctx.comment("MFMA lane mapping: lane_row, lane_k")
    ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
              comment=f"lane_row = lane_id % {mfma.m}")
    ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
               int(math.log2(mfma.m)),
               comment=f"lane_id / {mfma.m}")
    ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_tmp1"),
               int(math.log2(k_per_group)),
               comment=f"* {k_per_group} -> lane_k_offset")
    ctx.raw("")

    # Build the LDS read row: wave_m * m_per_wave + lane_row
    # Then use LDS A Embed: offset = row * unroll_k + lane_k
    # This is a 2-step affine: first compute row, then embed
    lds_rd_a_embed = Embed(
        [Dim("a_row", tile.wg_m), Dim("a_k", tile.unroll_k)],
        Dim("lds_rd_a", tile.wg_m * tile.unroll_k),
        [tile.unroll_k, 1],
    )

    # Compute a_row = wave_m * m_per_wave + lane_row into v_lds_rd_a
    ctx.comment(f"LDS read A: {lds_rd_a_embed}")
    ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(tile.m_per_wave),
              ctx.vreg("v_wave_m"), comment=f"wave_m * {tile.m_per_wave}")
    ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
              ctx.vreg("v_tmp0"), comment="+ lane_row")

    emit_affine(ctx, lds_rd_a_embed,
                bindings={"a_row": ctx.vreg("v_lds_rd_a"),
                           "a_k": ctx.vreg("v_tmp1")},
                result=ctx.vreg("v_lds_rd_a"),
                scale=elem,
                comment="lds_rd_a = (row * unroll_k + lane_k) * elem")
    ctx.raw("")

    # Same for B
    lds_rd_b_embed = Embed(
        [Dim("b_row", tile.wg_n), Dim("b_k", tile.unroll_k)],
        Dim("lds_rd_b", tile.wg_n * tile.unroll_k),
        [tile.unroll_k, 1],
    )

    ctx.comment(f"LDS read B: {lds_rd_b_embed} + lds_b_offset")
    ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(tile.n_per_wave),
              ctx.vreg("v_wave_n"), comment=f"wave_n * {tile.n_per_wave}")
    ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
              ctx.vreg("v_tmp0"), comment="+ lane_row")

    emit_affine(ctx, lds_rd_b_embed,
                bindings={"b_row": ctx.vreg("v_lds_rd_b"),
                           "b_k": ctx.vreg("v_tmp1")},
                result=ctx.vreg("v_lds_rd_b"),
                scale=elem,
                base=str(layouts.lds_b_offset),
                comment="lds_rd_b = lds_b_off + (row * unroll_k + lane_k) * elem")
    ctx.raw("")


def _emit_global_addr_a(ctx: AsmContext, problem: GemmProblem,
                        tile: TileConfig, layouts: GemmLayouts) -> None:
    """Compute 64-bit global address for A using transforms.

    A is row-major [M, K]: addr = A_ptr + (wg_base_m + row) * K + col.
    K is a dynamic coefficient (from kernarg s_K).
    """
    ctx.comment(f"Global address A: {layouts.global_a_row_major}")
    # Compute global row = wg_id_x * wg_m + thread_row
    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"), str(tile.wg_m),
              comment=f"wg_id_x * {tile.wg_m}")
    ctx.v_add(ctx.vreg("v_tmp0"), ctx.sreg("s_tmp0"),
              ctx.vreg("v_gload_row"), comment="+ thread_row -> global_m")
    # offset = global_m * K + col (K is dynamic, use s_K register)
    ctx.v_mul(ctx.vreg("v_tmp0"), ctx.sreg("s_K"),
              ctx.vreg("v_tmp0"), comment="global_m * K (dynamic coeff)")
    ctx.v_add(ctx.vreg("v_tmp0"), ctx.vreg("v_tmp0"),
              ctx.vreg("v_gload_col"), comment="+ col")
    ctx.v_lshl(ctx.vreg("v_tmp0"), ctx.vreg("v_tmp0"),
               int(math.log2(layouts.elem_bytes)),
               comment=f"* {layouts.elem_bytes} (bytes)")
    # 64-bit: v_addr_a = s_ptr_A + byte_offset
    ctx.inst("v_add_co_u32", ctx.vreg("v_addr_a", 0, 1), "vcc",
             ctx.sreg("s_ptr_A", 0, 1), ctx.vreg("v_tmp0"),
             comment="addr_A_lo + carry out")
    ctx.v_mov(ctx.vreg("v_tmp1"), ctx.sreg("s_ptr_A", 1, 1),
              comment="move A_hi to VGPR (const bus)")
    ctx.inst("v_addc_co_u32", ctx.vreg("v_addr_a", 1, 1), "vcc",
             ctx.vreg("v_tmp1"), "0", "vcc",
             comment="addr_A_hi + carry in")
    ctx.raw("")


def _emit_global_addr_b(ctx: AsmContext, problem: GemmProblem,
                        tile: TileConfig, layouts: GemmLayouts) -> None:
    """Compute 64-bit global address for B using transforms.

    B is [N, K] row-major (trans_b): addr = B_ptr + (wg_base_n + row) * K + col.
    """
    ctx.comment(f"Global address B: {layouts.global_b_row_major}")
    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_y"), str(tile.wg_n),
              comment=f"wg_id_y * {tile.wg_n}")
    ctx.v_add(ctx.vreg("v_tmp0"), ctx.sreg("s_tmp0"),
              ctx.vreg("v_gload_row"), comment="+ thread_row -> global_n")
    ctx.v_mul(ctx.vreg("v_tmp0"), ctx.sreg("s_K"),
              ctx.vreg("v_tmp0"), comment="global_n * K (dynamic coeff)")
    ctx.v_add(ctx.vreg("v_tmp0"), ctx.vreg("v_tmp0"),
              ctx.vreg("v_gload_col"), comment="+ col")
    ctx.v_lshl(ctx.vreg("v_tmp0"), ctx.vreg("v_tmp0"),
               int(math.log2(layouts.elem_bytes)),
               comment=f"* {layouts.elem_bytes} (bytes)")
    ctx.inst("v_add_co_u32", ctx.vreg("v_addr_b", 0, 1), "vcc",
             ctx.sreg("s_ptr_B", 0, 1), ctx.vreg("v_tmp0"),
             comment="addr_B_lo + carry out")
    ctx.v_mov(ctx.vreg("v_tmp1"), ctx.sreg("s_ptr_B", 1, 1),
              comment="move B_hi to VGPR (const bus)")
    ctx.inst("v_addc_co_u32", ctx.vreg("v_addr_b", 1, 1), "vcc",
             ctx.vreg("v_tmp1"), "0", "vcc",
             comment="addr_B_hi + carry in")
    ctx.raw("")


def _emit_global_load(ctx: AsmContext, problem: GemmProblem,
                      tile: TileConfig) -> None:
    """Emit flat_load for A and B tiles."""
    a_load = ctx.get("v_gload_a")
    b_load = ctx.get("v_gload_b")
    addr_a = ctx.vreg("v_addr_a", 0, 2)
    addr_b = ctx.vreg("v_addr_b", 0, 2)

    ctx.comment("Global load A tile")
    for i in range(0, a_load.count, 4):
        cnt = min(4, a_load.count - i)
        width = {4: "dwordx4", 2: "dwordx2", 1: "dword"}[cnt]
        dst = ctx.vreg("v_gload_a", i, cnt)
        if i > 0:
            ctx.inst(f"global_load_{width}", dst, addr_a,
                     f"off offset:{i * 4}", comment=f"load A[{i}:{i+cnt}]")
        else:
            ctx.inst(f"global_load_{width}", dst, addr_a,
                     "off", comment=f"load A[0:{cnt}]")

    ctx.comment("Global load B tile")
    for i in range(0, b_load.count, 4):
        cnt = min(4, b_load.count - i)
        width = {4: "dwordx4", 2: "dwordx2", 1: "dword"}[cnt]
        dst = ctx.vreg("v_gload_b", i, cnt)
        if i > 0:
            ctx.inst(f"global_load_{width}", dst, addr_b,
                     f"off offset:{i * 4}", comment=f"load B[{i}:{i+cnt}]")
        else:
            ctx.inst(f"global_load_{width}", dst, addr_b,
                     "off", comment=f"load B[0:{cnt}]")

    ctx.s_waitcnt("0", comment="wait for ALL (flat loads use flat counter on gfx940+)")
    ctx.raw("")


def _emit_lds_write(ctx: AsmContext, tile: TileConfig) -> None:
    """Write loaded A/B data into LDS."""
    a_load = ctx.get("v_gload_a")
    b_load = ctx.get("v_gload_b")

    ctx.comment("LDS write A")
    for i in range(0, a_load.count, 4):
        cnt = min(4, a_load.count - i)
        src = ctx.vreg("v_gload_a", i, cnt)
        ctx.ds_write(ctx.vreg("v_lds_wr_a"), src, offset=i * 4,
                     width=cnt, comment=f"LDS write A[{i}:{i+cnt}]")

    ctx.comment("LDS write B")
    for i in range(0, b_load.count, 4):
        cnt = min(4, b_load.count - i)
        src = ctx.vreg("v_gload_b", i, cnt)
        ctx.ds_write(ctx.vreg("v_lds_wr_b"), src, offset=i * 4,
                     width=cnt, comment=f"LDS write B[{i}:{i+cnt}]")

    ctx.s_waitcnt("lgkmcnt(0)", comment="wait for LDS writes")
    ctx.s_barrier(comment="sync workgroup after LDS fill")
    ctx.raw("")


def _emit_init_acc(ctx: AsmContext, tile: TileConfig) -> None:
    """Zero-initialize accumulator registers."""
    acc_total = (tile.mfma_m_repeat * tile.mfma_n_repeat
                 * tile.mfma.acc_vgprs)
    ctx.comment(f"Init {acc_total} accumulators to zero")
    for i in range(acc_total):
        ctx.inst("v_accvgpr_write_b32", ctx.areg("acc_C", i, 1), "0")
    ctx.raw("")


# ===================================================================
# K-loop body: LDS read + MFMA (driven by tile tree via walk_tile_tree)
# ===================================================================

def _mfma_visitor(level: TileLevel, ctx: AsmContext) -> None:
    """Tile-tree visitor: emits LDS reads + MFMAs at the mfma leaf level."""
    if level.name == "wave":
        return  # structural level, no direct code

    if level.name != "mfma":
        return

    # Current iteration indices from the tree walk
    mi = ctx.indices.get("wave.mi", 0)
    ni = ctx.indices.get("wave.ni", 0)
    ki = ctx.indices.get("wave.ki", 0)

    tile = ctx._metadata["tile"]
    problem = ctx._metadata["problem"]
    mfma = tile.mfma
    elem = problem.element_bytes

    # Compute full LDS address for A at runtime
    mi_byte_off = mi * mfma.m * tile.unroll_k * elem
    k_byte_off = ki * mfma.k * elem
    total_a_off = mi_byte_off + k_byte_off
    if total_a_off > 0:
        ctx.v_add(ctx.vreg("v_tmp0"), str(total_a_off),
                  ctx.vreg("v_lds_rd_a"),
                  comment=f"lds_rd_a + mi={mi} ki={ki} off={total_a_off}")
        a_addr = ctx.vreg("v_tmp0")
    else:
        a_addr = ctx.vreg("v_lds_rd_a")
    for r in range(mfma.a_vgprs):
        ctx.ds_read(ctx.vreg("v_a", r, 1), a_addr,
                    offset=r * 4, width=1,
                    comment=f"LDS read A[{r}] mi={mi} ki={ki}")

    # Compute full LDS address for B at runtime
    ni_byte_off = ni * mfma.n * tile.unroll_k * elem
    total_b_off = ni_byte_off + k_byte_off
    if total_b_off > 0:
        ctx.v_add(ctx.vreg("v_tmp1"), str(total_b_off),
                  ctx.vreg("v_lds_rd_b"),
                  comment=f"lds_rd_b + ni={ni} ki={ki} off={total_b_off}")
        b_addr = ctx.vreg("v_tmp1")
    else:
        b_addr = ctx.vreg("v_lds_rd_b")
    for r in range(mfma.b_vgprs):
        ctx.ds_read(ctx.vreg("v_b", r, 1), b_addr,
                    offset=r * 4, width=1,
                    comment=f"LDS read B[{r}] ni={ni} ki={ki}")

    ctx.s_waitcnt("lgkmcnt(0)", comment="wait LDS reads")

    # MFMA instruction
    acc_per = mfma.acc_vgprs
    acc_off = (mi * tile.mfma_n_repeat + ni) * acc_per
    ctx.inst(
        f"v_mfma_f32_{mfma.m}x{mfma.n}x{mfma.k}_f16",
        ctx.areg("acc_C", acc_off, acc_per),
        ctx.vreg("v_a", 0, mfma.a_vgprs),
        ctx.vreg("v_b", 0, mfma.b_vgprs),
        ctx.areg("acc_C", acc_off, acc_per),
        comment=f"mfma m{mi}_n{ni} k{ki}",
    )
    ctx.raw("")


# ===================================================================
# Epilogue: store accumulators to global memory D
# ===================================================================

def _emit_store_d(ctx: AsmContext, problem: GemmProblem,
                  tile: TileConfig) -> None:
    """Read accumulators, convert f32->f16, store to D.

    For v_mfma_f32_16x16x16_f16, accumulator layout per MFMA tile:
      4 acc VGPRs, 64 lanes, each lane computes 4 output elements.
      acc[i] at lane l: row = l % 16, col = i * 4 + l / 16
    """
    mfma = tile.mfma
    acc_per = mfma.acc_vgprs
    elem = problem.element_bytes

    ctx.comment("Epilogue: store D")
    # MFMA 16x16x16 output mapping:
    #   d_m = (lane_id/16)*4 + acc_index  (v_tmp1 = lane_m_base)
    #   d_n = lane_id % 16                (v_tmp0 = lane_n)
    ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
              comment="lane_n = lane_id % 16 (output N-col)")
    ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
               int(math.log2(mfma.m)),
               comment="lane_id / 16")
    ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_tmp1"), 2,
               comment="* 4 -> lane_m_base (output M-row base)")

    for mi in range(tile.mfma_m_repeat):
        for ni in range(tile.mfma_n_repeat):
            acc_base = (mi * tile.mfma_n_repeat + ni) * acc_per
            for ai in range(acc_per):
                # Read accumulator to VGPR
                ctx.inst("v_accvgpr_read_b32",
                         ctx.vreg("v_store_tmp"),
                         ctx.areg("acc_C", acc_base + ai, 1),
                         comment=f"acc[{acc_base+ai}]")
                # Convert f32 -> f16
                ctx.inst("v_cvt_f16_f32_e32",
                         ctx.vreg("v_store_tmp"),
                         ctx.vreg("v_store_tmp"),
                         comment="f32 -> f16")

                # MFMA output: d_m = lane_m_base + ai, d_n = lane_n
                # row = wg_base_m + wave_m*m_per_wave + mi*mfma_m
                #       + lane_m_base(v_tmp1) + ai
                row_imm = mi * mfma.m + ai
                ctx.v_mul(ctx.vreg("v_addr_d", 0, 1),
                          str(tile.m_per_wave), ctx.vreg("v_wave_m"),
                          comment=f"wave_m * {tile.m_per_wave}")
                ctx.v_add(ctx.vreg("v_addr_d", 0, 1),
                          ctx.vreg("v_addr_d", 0, 1),
                          ctx.vreg("v_tmp1"),
                          comment="+ lane_m_base")
                if row_imm:
                    ctx.v_add(ctx.vreg("v_addr_d", 0, 1),
                              str(row_imm),
                              ctx.vreg("v_addr_d", 0, 1),
                              comment=f"+ mi*mfma_m+ai ({row_imm})")
                ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"),
                          str(tile.wg_m),
                          comment=f"wg_id_x * {tile.wg_m}")
                ctx.v_add(ctx.vreg("v_addr_d", 0, 1),
                          ctx.sreg("s_tmp0"),
                          ctx.vreg("v_addr_d", 0, 1),
                          comment="+ wg_base_m")
                # row * N
                ctx.v_mul(ctx.vreg("v_addr_d", 0, 1),
                          ctx.sreg("s_N"),
                          ctx.vreg("v_addr_d", 0, 1),
                          comment="* N")

                # col = wg_base_n + wave_n*n_per_wave + ni*mfma_n
                #       + lane_n(v_tmp0)
                col_imm = ni * mfma.n
                ctx.v_mul(ctx.vreg("v_addr_d", 1, 1),
                          str(tile.n_per_wave), ctx.vreg("v_wave_n"),
                          comment=f"wave_n * {tile.n_per_wave}")
                ctx.v_add(ctx.vreg("v_addr_d", 1, 1),
                          ctx.vreg("v_addr_d", 1, 1),
                          ctx.vreg("v_tmp0"),
                          comment="+ lane_n")
                if col_imm:
                    ctx.v_add(ctx.vreg("v_addr_d", 1, 1),
                              str(col_imm),
                              ctx.vreg("v_addr_d", 1, 1),
                              comment=f"+ ni*mfma_n ({col_imm})")
                ctx.s_mul(ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_y"),
                          str(tile.wg_n),
                          comment=f"wg_id_y * {tile.wg_n}")
                ctx.v_add(ctx.vreg("v_addr_d", 1, 1),
                          ctx.sreg("s_tmp1"),
                          ctx.vreg("v_addr_d", 1, 1),
                          comment="+ wg_base_n")

                # D offset = row * N + col (N is dynamic)
                # v_addr_d[0] has row*N, v_addr_d[1] has col
                ctx.v_add(ctx.vreg("v_addr_d", 0, 1),
                          ctx.vreg("v_addr_d", 0, 1),
                          ctx.vreg("v_addr_d", 1, 1),
                          comment="row*N + col (Embed transform)")
                ctx.v_lshl(ctx.vreg("v_addr_d", 0, 1),
                           ctx.vreg("v_addr_d", 0, 1),
                           int(math.log2(elem)),
                           comment=f"* {elem} (bytes)")
                # 64-bit: D_ptr + byte_offset
                ctx.v_add(ctx.vreg("v_addr_d", 0, 1),
                          ctx.sreg("s_ptr_D", 0, 1),
                          ctx.vreg("v_addr_d", 0, 1),
                          comment="D_lo + offset")
                ctx.v_mov(ctx.vreg("v_addr_d", 1, 1),
                          ctx.sreg("s_ptr_D", 1, 1),
                          comment="D_hi")

                # Store f16
                addr = ctx.vreg("v_addr_d", 0, 2)
                ctx.inst("global_store_short", addr,
                         ctx.vreg("v_store_tmp"), "off",
                         comment=f"store D m{mi}_n{ni}_a{ai}")

    ctx.s_waitcnt("vmcnt(0)", comment="wait for stores")
    ctx.raw("")


# ===================================================================
# Kernel descriptor
# ===================================================================

def _emit_descriptor(ctx: AsmContext, kernel_name: str,
                     lds_total: int, tile: TileConfig) -> None:
    accum_offset = (ctx._next["v"] + 3) & ~3  # must be 4-aligned
    sgpr_count = ctx._next["s"]
    acc_count = ctx._next["acc"]
    # gfx940+: unified VGPR/AGPR file. next_free_vgpr must include
    # both regular VGPRs and accumulator VGPRs.
    vgpr_count = accum_offset + acc_count
    ctx.raw("")
    ctx.raw(".rodata")
    ctx.raw(".p2align 6")
    ctx.raw(f".amdhsa_kernel {kernel_name}")
    ctx.raw(f"    .amdhsa_group_segment_fixed_size {lds_total}")
    ctx.raw(f"    .amdhsa_private_segment_fixed_size 0")
    ctx.raw(f"    .amdhsa_kernarg_size 64")
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
    # AMDHSA metadata note -- required for hipModuleLoad
    ctx.raw(".amdgpu_metadata")
    ctx.raw("---")
    ctx.raw("amdhsa.version: [ 1, 2 ]")
    ctx.raw("amdhsa.kernels:")
    ctx.raw(f"  - .name:            {kernel_name}")
    ctx.raw(f"    .symbol:          {kernel_name}.kd")
    ctx.raw(f"    .sgpr_count:      {sgpr_count}")
    ctx.raw(f"    .vgpr_count:      {vgpr_count}")
    ctx.raw(f"    .agpr_count:      {acc_count}")
    ctx.raw(f"    .kernarg_segment_size: 64")
    ctx.raw(f"    .kernarg_segment_align: 8")
    ctx.raw(f"    .group_segment_fixed_size: {lds_total}")
    ctx.raw(f"    .private_segment_fixed_size: 0")
    ctx.raw(f"    .wavefront_size:  {tile.wave_size}")
    ctx.raw(f"    .max_flat_workgroup_size: {tile.block_size}")
    ctx.raw(f"    .args:")
    ctx.raw(f"      - .address_space:  global")
    ctx.raw(f"        .offset:         0")
    ctx.raw(f"        .size:           8")
    ctx.raw(f"        .value_kind:     global_buffer")
    ctx.raw(f"      - .address_space:  global")
    ctx.raw(f"        .offset:         8")
    ctx.raw(f"        .size:           8")
    ctx.raw(f"        .value_kind:     global_buffer")
    ctx.raw(f"      - .address_space:  global")
    ctx.raw(f"        .offset:         16")
    ctx.raw(f"        .size:           8")
    ctx.raw(f"        .value_kind:     global_buffer")
    ctx.raw(f"      - .offset:         24")
    ctx.raw(f"        .size:           4")
    ctx.raw(f"        .value_kind:     by_value")
    ctx.raw(f"      - .offset:         28")
    ctx.raw(f"        .size:           4")
    ctx.raw(f"        .value_kind:     by_value")
    ctx.raw(f"      - .offset:         32")
    ctx.raw(f"        .size:           4")
    ctx.raw(f"        .value_kind:     by_value")
    ctx.raw("...")
    ctx.raw(".end_amdgpu_metadata")


# ===================================================================
# Main entry point
# ===================================================================

def emit_gemm_asm(
    problem: GemmProblem,
    tile: Optional[TileConfig] = None,
    kernel_name: str = "gemm_kernel",
    tile_tree: Optional[TileLevel] = None,
) -> AsmKernel:
    """Generate a complete GEMM kernel as gfx950 assembly.

    Uses the tile-tree infrastructure for register allocation,
    loop structure, and address math.

    The kernel implements: D[M,N] = A[M,K] @ B[K,N]  (fp16, alpha=1)
    A is row-major [M, K], B is [N, K] (trans_b), D is row-major [M, N].
    """
    if tile is None:
        tile = TileConfig()
    problem.validate(tile)

    # Build tile tree for the MFMA loop structure
    if tile_tree is None:
        tile_tree = build_gemm_tile_tree(
            wg_m=tile.wg_m, wg_n=tile.wg_n, unroll_k=tile.unroll_k,
            waves_m=tile.waves_m, waves_n=tile.waves_n,
            mfma_m=tile.mfma.m, mfma_n=tile.mfma.n, mfma_k=tile.mfma.k,
        )
    tile_tree.validate()

    elem = problem.element_bytes
    layouts = GemmLayouts.build(problem, tile)
    lds_a_bytes = tile.wg_m * tile.unroll_k * elem
    lds_b_offset = lds_a_bytes
    lds_total = lds_a_bytes + tile.wg_n * tile.unroll_k * elem

    # Create AsmContext and allocate all registers via named bindings
    ctx = AsmContext()
    ctx._metadata = {"tile": tile, "problem": problem, "layouts": layouts}
    _alloc_registers(ctx, problem, tile)

    # Emit the full kernel
    _emit_header(ctx, kernel_name)
    _emit_load_kernargs(ctx)
    _emit_thread_indexing(ctx, tile)
    _emit_global_load_cluster(ctx, tile, problem)
    _emit_lds_write_addrs(ctx, problem, tile, layouts)
    _emit_lds_read_addrs(ctx, problem, tile, layouts)
    _emit_init_acc(ctx, tile)
    _emit_global_addr_a(ctx, problem, tile, layouts)
    _emit_global_addr_b(ctx, problem, tile, layouts)

    # K-tile loop
    ctx.comment("K-tile loop setup")
    log2_uk = int(math.log2(tile.unroll_k))
    ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
               comment=f"k_tiles = K / {tile.unroll_k}")
    ctx.raw("")
    ctx.label("k_loop")
    ctx.raw("")

    # Global load + LDS write for this K-tile
    _emit_global_load(ctx, problem, tile)
    _emit_lds_write(ctx, tile)

    # Inner loop: LDS read + MFMA (driven by tile tree)
    walk_tile_tree(tile_tree, ctx, _mfma_visitor)

    # Advance global addresses for next K-tile
    k_stride_bytes = tile.unroll_k * elem
    ctx.comment("Advance A, B pointers by unroll_k")
    ctx.inst("v_add_co_u32", ctx.vreg("v_addr_a", 0, 1), "vcc",
             str(k_stride_bytes), ctx.vreg("v_addr_a", 0, 1),
             comment=f"A += {k_stride_bytes}")
    ctx.inst("v_addc_co_u32", ctx.vreg("v_addr_a", 1, 1), "vcc",
             ctx.vreg("v_addr_a", 1, 1), "0", "vcc",
             comment="carry")
    ctx.inst("v_add_co_u32", ctx.vreg("v_addr_b", 0, 1), "vcc",
             str(k_stride_bytes), ctx.vreg("v_addr_b", 0, 1),
             comment=f"B += {k_stride_bytes}")
    ctx.inst("v_addc_co_u32", ctx.vreg("v_addr_b", 1, 1), "vcc",
             ctx.vreg("v_addr_b", 1, 1), "0", "vcc",
             comment="carry")
    ctx.raw("")

    # Barrier before next iteration's LDS write
    ctx.s_barrier(comment="sync before next K-tile LDS write")

    # Loop control
    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="SCC = (k_tiles != 0)")
    ctx.inst("s_cbranch_scc1", "k_loop",
             comment="branch if k_tiles > 0")
    ctx.raw("")

    # Epilogue: store D
    _emit_store_d(ctx, problem, tile)

    ctx.inst("s_endpgm", comment="end of kernel")

    # Kernel descriptor
    _emit_descriptor(ctx, kernel_name, lds_total, tile)

    return AsmKernel(
        asm_text=ctx.asm_text(),
        kernel_name=kernel_name,
        problem=problem,
        tile=tile,
        ctx=ctx,
        lds_bytes=lds_total,
    )
