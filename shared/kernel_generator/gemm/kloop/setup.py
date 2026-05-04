"""DTL + 3-barrier + interleaved K-loop for 256x256x64 tile.

Matches TensileLite's architecture:
- 128 MFMAs per K-loop iteration (8x8x2 = mr*nr*ki)
- 16 buffer_load_dwordx4 ... ,lds (DTL): 8 A + 8 B
- 32 ds_read_b128: 16 B + 16 A (A double-buffered)
- 3 barriers per iteration
- XOR-based LDS double-buffer toggle
- No ds_write instructions (DTL bypasses VGPRs)

K-loop structure (128 MFMA slots):
  Phase 1 (mfma 0-19):  Compute X0 + ds_read A for X1
  BARRIER 1 (mfma ~20): Wait lgkmcnt(0), barrier -> safe to DTL write A
  Phase 2 (mfma 21-50): Compute X0 + DTL A loads + ds_read B for X1
  BARRIER 2 (mfma ~51): Wait lgkmcnt(0), barrier -> safe to DTL write B
  Phase 3 (mfma 52-91): Compute X0/X1 + DTL B loads + remaining A loads
  vmcnt(N) at mfma ~91:  Wait for enough DTL loads to land
  BARRIER 3 (mfma ~92): Barrier -> safe to ds_read from new buffer
  Phase 4 (mfma 93-127): Compute X1 + ds_read A,B from new buffer
"""
from __future__ import annotations

import math

from ..emit.context import AsmContext
from ..emit.layouts import GemmLayouts
from ..problem import GemmProblem, MfmaConfig, TileConfig
from ..tile.tree import TileLevel, TilePhase

__all__ = ["phase_mx_scale_setup", "phase_dtl_interleaved_setup",
           "phase_wave_abi_setup"]


def _tile(ctx: AsmContext) -> TileConfig: return ctx._metadata["tile"]
def _problem(ctx: AsmContext) -> GemmProblem: return ctx._metadata["problem"]
def _layouts(ctx: AsmContext) -> GemmLayouts: return ctx._metadata["layouts"]


def _a_off(mi: int, ki: int, tile: TileConfig, mfma: MfmaConfig, elem: float) -> int:
    """LDS byte offset for A operand at (mi, ki).

    With paired-row swizzle: ki is handled by per-ki base VGPRs,
    and row offset uses paired layout (lds_row * eff_stride).
    Without swizzle: standard row*stride + ki*k_elems*elem + padding.
    """
    row_start = mi * mfma.m
    row_stride = int(tile.unroll_k * elem)
    swz = tile.resolved_swizzle(elem)
    if swz is not None and hasattr(swz, 'pair_factor'):
        # Paired-row swizzle: mi offset is handled by
        # emit_recompute_a_for_mi() which fully recomputes the read
        # base. Return 0 so ds_read uses the recomputed base directly.
        return 0
    if getattr(tile, 'lds_swizzle', False):
        return int(row_start * row_stride)
    pad_bytes = tile.lds_pad
    threads_per_row = int(tile.unroll_k * elem) // 16
    rows_per_load = (tile.waves_m * tile.waves_n * tile.wave_size) // threads_per_row
    lines_crossed = row_start // rows_per_load
    return int(row_start * row_stride + lines_crossed * pad_bytes + ki * mfma.k * elem)


def _b_off(ni: int, ki: int, tile: TileConfig, mfma: MfmaConfig, elem: float) -> int:
    """LDS byte offset for B operand at (ni, ki)."""
    row_start = ni * mfma.n
    row_stride = int(tile.unroll_k * elem)
    swz = tile.resolved_swizzle(elem)
    if swz is not None and hasattr(swz, 'pair_factor'):
        # Paired-row swizzle: ni offset handled by recomputation. Return 0.
        return 0
    if getattr(tile, 'lds_swizzle', False):
        return int(row_start * row_stride)
    pad_bytes = tile.lds_pad
    threads_per_row = int(tile.unroll_k * elem) // 16
    rows_per_load = (tile.waves_m * tile.waves_n * tile.wave_size) // threads_per_row
    lines_crossed = row_start // rows_per_load
    return int(row_start * row_stride + lines_crossed * pad_bytes + ki * mfma.k * elem)


def phase_wave_abi_setup(level: TileLevel, ctx: AsmContext) -> None:
    """Setup for WaveGemmKernelArgs ABI (rocRoller custom kernel path).

    WaveGemmKernelArgs layout (104 bytes, all u64):
        0:  ptr_a              -- kernel's A (hipBLASLt's B, swapped)
        8:  ptr_a_scale        -- kernel's ScaleA
       16:  ptr_b              -- kernel's B (hipBLASLt's A, swapped)
       24:  ptr_b_scale        -- kernel's ScaleB
       32:  ptr_c              -- D output
       40:  m                  -- kernel's M (u64)
       48:  n                  -- kernel's N (u64)
       56:  k                  -- K (u64)
       64:  stride_a_dim0      -- A row stride in bytes (u64)
       72:  stride_a_scale_dim0-- k/32 (u64)
       80:  stride_b_dim0      -- B row stride in bytes (u64)
       88:  stride_b_scale_dim0-- k/32 (u64)
       96:  stride_c_dim0      -- D column stride (u64)

    Grid is 2D: grid.x = tilesM, grid.y = tilesN.
    No 1D decomposition needed.
    """
    tile = _tile(ctx)
    problem = _problem(ctx)
    elem = problem.element_bytes
    mfma = tile.mfma
    layouts = _layouts(ctx)

    ctx.comment("=== Wave ABI Setup (rocRoller custom kernel) ===")
    ctx._metadata["use_wave_abi"] = True

    # Load kernargs -- all fields are u64, load dwordx2 and use low 32 bits
    karg = ctx.sreg("s_kernarg")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_A"), karg, "0",
             comment="ptr_a (kernel A)")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_scale_a"), karg, "8",
             comment="ptr_a_scale")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_B"), karg, "16",
             comment="ptr_b (kernel B)")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_scale_b"), karg, "24",
             comment="ptr_b_scale")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_D"), karg, "32",
             comment="ptr_c (D output)")
    # M, N, K are u64 -- load low 32 bits via s_load_dword (little-endian)
    ctx.inst("s_load_dword", ctx.sreg("s_M"), karg, "40",
             comment="M = low dword of m (u64)")
    ctx.inst("s_load_dword", ctx.sreg("s_N"), karg, "48",
             comment="N = low dword of n (u64)")
    ctx.inst("s_load_dword", ctx.sreg("s_K"), karg, "56",
             comment="K = low dword of k (u64)")
    # Scale strides (u64, load low 32 bits)
    ctx.inst("s_load_dword", ctx.sreg("s_stride_scale_a"), karg, "72",
             comment="stride_a_scale_dim0 (low 32)")
    ctx.inst("s_load_dword", ctx.sreg("s_stride_scale_b"), karg, "88",
             comment="stride_b_scale_dim0 (low 32)")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait kernargs")
    ctx.raw("")

    # 2D grid: s_wg_id_x = tile M index, s_wg_id_y = tile N index (no decomp)

    # Compute K stride in bytes: K * element_bytes
    if elem >= 1:
        ctx.s_lshl(ctx.sreg("s_k_stride"), ctx.sreg("s_K"),
                   int(math.log2(elem)), comment=f"s_k_stride = K * {elem}")
    else:
        ctx.s_lshr(ctx.sreg("s_k_stride"), ctx.sreg("s_K"),
                   int(math.log2(1.0 / elem)), comment=f"s_k_stride = K * {elem}")

    # Thread indexing
    log2_ws = int(math.log2(tile.wave_size))
    ctx.v_lshr(ctx.vreg("v_wave_id"), ctx.vreg("v_tid"), log2_ws,
               comment=f"wave_id = tid >> {log2_ws}")
    ctx.v_and(ctx.vreg("v_lane_id"), ctx.vreg("v_tid"), tile.wave_size - 1,
              comment=f"lane_id = tid & {tile.wave_size - 1}")
    log2_wn = int(math.log2(tile.waves_n)) if tile.waves_n > 1 else 0
    if tile.waves_n > 1:
        ctx.v_lshr(ctx.vreg("v_wave_m"), ctx.vreg("v_wave_id"), log2_wn,
                   comment=f"wave_m = wave_id >> {log2_wn}")
        ctx.v_and(ctx.vreg("v_wave_n"), ctx.vreg("v_wave_id"),
                  tile.waves_n - 1, comment=f"wave_n = wave_id & {tile.waves_n - 1}")
    else:
        ctx.v_mov(ctx.vreg("v_wave_m"), ctx.vreg("v_wave_id"), comment="wave_m")
        ctx.v_mov(ctx.vreg("v_wave_n"), "0", comment="wave_n = 0")
    ctx.raw("")

    # DTL per-lane offset
    threads_per_row = int(tile.unroll_k * elem) // 16
    log2_tpr = int(math.log2(threads_per_row))
    ctx.comment(f"DTL offset: {threads_per_row} threads/row")
    ctx.v_lshr(ctx.vreg("v_tmp0"), ctx.vreg("v_tid"), log2_tpr,
               comment="thread_row")
    ctx.v_and(ctx.vreg("v_tmp1"), ctx.vreg("v_tid"), threads_per_row - 1,
              comment="thread_col_group")
    ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_tmp1"), 4,
               comment="* 16 -> col_bytes")

    # Compute swizzled LDS write offset (for BufferLoader path)
    # v_tmp0 = thread_row, v_tmp1 = col_bytes (= thread_col * 16)
    swz = tile.resolved_swizzle(elem)
    if swz is not None and hasattr(swz, 'pair_factor'):
        from ..memory.swizzle import DataLayout as SwzLayout, LDS_GFX950
        swz_layout = SwzLayout(row_stride_bytes=int(tile.unroll_k * elem),
                               mfma_k=mfma.k, mfma_m=mfma.m,
                               elem_bytes=elem, wave_size=tile.wave_size)
        pf = swz.pair_factor
        ec = swz.effective_cols
        eff_stride = ec * 16

        ctx.comment(f"Swizzled LDS write offset (pair={pf}, eff_cols={ec})")
        # thread_col in 16B column units (v_tmp1 is already col_bytes)
        ctx.v_lshr(ctx.vreg("v_tmp4"), ctx.vreg("v_tmp1"), 4,
                   comment="thread_col = col_bytes / 16")
        # Apply write swizzle: computes swizzled_col from (thread_row, thread_col)
        swz.emit_write_swizzle(ctx, swz_layout, LDS_GFX950,
                               ctx.vreg("v_tmp0"), ctx.vreg("v_tmp4"),
                               ctx.vreg("v_tmp4"))
        # lds_row = thread_row / pair_factor
        ctx.v_lshr(ctx.vreg("v_tmp3"), ctx.vreg("v_tmp0"),
                   int(math.log2(pf)), comment=f"lds_row = row / {pf}")
        # v_lds_wr_swz = lds_row * eff_stride + swizzled_col * 16
        ctx.alloc_vgpr_permanent(1, "v_lds_wr_swz")
        ctx.v_mul(ctx.vreg("v_lds_wr_swz"), str(eff_stride),
                  ctx.vreg("v_tmp3"), comment=f"lds_row * {eff_stride}")
        ctx.v_lshl(ctx.vreg("v_tmp4"), ctx.vreg("v_tmp4"), 4,
                   comment="swizzled_col * 16")
        ctx.v_add(ctx.vreg("v_lds_wr_swz"), ctx.vreg("v_lds_wr_swz"),
                  ctx.vreg("v_tmp4"), comment="+ swizzled_col_bytes")
        ctx.raw("")

    # LDS write offset = thread_row * unroll_k * elem + col_bytes
    # This uses the LDS row stride (unroll_k*elem), NOT the global stride (K*elem)
    row_stride_lds = int(tile.unroll_k * elem)
    ctx.alloc_vgpr_permanent(1, "v_lds_wr_off")
    ctx.v_mul(ctx.vreg("v_lds_wr_off"), str(row_stride_lds),
              ctx.vreg("v_tmp0"), comment=f"row * {row_stride_lds} (LDS stride)")
    ctx.v_add(ctx.vreg("v_lds_wr_off"), ctx.vreg("v_lds_wr_off"),
              ctx.vreg("v_tmp1"), comment="+ col_bytes -> v_lds_wr_off")
    ctx.raw("")

    # Double-buffer write offset for BufferLoader (starts at 0)
    ctx.alloc_sgpr_permanent(1, "s_buf_wr_db")
    ctx.s_mov(ctx.sreg("s_buf_wr_db"), "0",
              comment="buf_wr_db = 0 (buffer 0)")
    ctx.raw("")

    # DTL voffset = thread_row * K * elem + col_bytes (global stride)
    ctx.inst("v_mul_lo_u32", ctx.vreg("v_dtl_off_a"),
             ctx.sreg("s_k_stride"), ctx.vreg("v_tmp0"), comment="row * K*elem")
    ctx.v_add(ctx.vreg("v_dtl_off_a"), ctx.vreg("v_dtl_off_a"),
              ctx.vreg("v_tmp1"), comment="+ col_bytes")
    ctx.v_mov(ctx.vreg("v_dtl_off_b"), ctx.vreg("v_dtl_off_a"),
              comment="B offset = same")
    ctx.raw("")

    # SRD A
    ctx.comment("SRD A")
    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"), str(tile.wg_m),
              comment=f"wg_id * {tile.wg_m}")
    ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
             ctx.sreg("s_k_stride"), comment="* K*elem")
    ctx.inst("s_add_u32", ctx.sreg("s_srd_a", 0, 1),
             ctx.sreg("s_ptr_A", 0, 1), ctx.sreg("s_tmp0"), comment="SRD_A lo")
    ctx.inst("s_addc_u32", ctx.sreg("s_srd_a", 1, 1),
             ctx.sreg("s_ptr_A", 1, 1), "0", comment="SRD_A hi")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_a", 2, 1), "0xFFFFFFFF", comment="limit")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_a", 3, 1), "0x20000", comment="flags")
    ctx.raw("")

    # SRD B
    ctx.comment("SRD B")
    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_y"), str(tile.wg_n),
              comment=f"wg_id * {tile.wg_n}")
    ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
             ctx.sreg("s_k_stride"), comment="* K*elem")
    ctx.inst("s_add_u32", ctx.sreg("s_srd_b", 0, 1),
             ctx.sreg("s_ptr_B", 0, 1), ctx.sreg("s_tmp0"), comment="SRD_B lo")
    ctx.inst("s_addc_u32", ctx.sreg("s_srd_b", 1, 1),
             ctx.sreg("s_ptr_B", 1, 1), "0", comment="SRD_B hi")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_b", 2, 1), "0xFFFFFFFF", comment="limit")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_b", 3, 1), "0x20000", comment="flags")
    ctx.raw("")

    # Scalar offsets for multi-line DTL loads
    rows_per_load = tile.block_size // threads_per_row
    ctx.comment(f"Scalar offset for DTL lines ({rows_per_load} rows/load)")
    ctx.s_mul(ctx.sreg("s_soffset_a"), ctx.sreg("s_k_stride"),
              str(rows_per_load), comment=f"soffset = {rows_per_load} * K*elem")
    ctx.s_mov(ctx.sreg("s_soffset_b"), ctx.sreg("s_soffset_a"), comment="same")
    ctx.raw("")

    # LDS write base for DTL
    ctx.comment("LDS write base for DTL")
    dtl_row_stride = int(tile.unroll_k * elem)
    ctx.v_mul(ctx.vreg("v_tmp0"), str(dtl_row_stride),
              ctx.vreg("v_tmp0"), comment=f"row * {dtl_row_stride}")
    ctx.v_add(ctx.vreg("v_tmp0"), ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
              comment="+ col_bytes -> per-thread LDS offset")
    ctx.inst("v_readfirstlane_b32", ctx.sreg("s_lds_wr_a_sg"),
             ctx.vreg("v_tmp0"), comment="LDS write base A")
    if tile.lds_pad > 0:
        threads_per_row_ = int(tile.unroll_k * elem) // 16
        rows_per_load_ = tile.block_size // threads_per_row_
        num_loads_a_ = tile.wg_m // rows_per_load_
        dtl_lds_b_offset = int(tile.wg_m * tile.unroll_k * elem) + num_loads_a_ * tile.lds_pad
    else:
        dtl_lds_b_offset = layouts.lds_b_offset
    ctx.inst("s_add_u32", ctx.sreg("s_lds_wr_b_sg"),
             ctx.sreg("s_lds_wr_a_sg"), str(dtl_lds_b_offset),
             comment=f"LDS write base B = A + {dtl_lds_b_offset}")
    ctx.raw("")

    # LDS read addresses
    k_per_group = mfma.k // (tile.wave_size // mfma.m)
    threads_per_row_rd = int(tile.unroll_k * elem) // 16
    row_stride_bytes = int(tile.unroll_k * elem)
    ctx.comment("LDS read addresses")
    ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
              comment=f"lane_row = lane_id % {mfma.m}")

    swz = tile.resolved_swizzle(elem)
    if swz is not None and hasattr(swz, 'pair_factor'):
        # Paired-row swizzle: use actual M-row (not lane_row) for rotation
        from ..memory.swizzle import DataLayout as SwzLayout, LDS_GFX950
        swz_layout = SwzLayout(row_stride_bytes=row_stride_bytes,
                               mfma_k=mfma.k, mfma_m=mfma.m,
                               elem_bytes=elem, wave_size=tile.wave_size)
        pf = swz.pair_factor
        ec = swz.effective_cols
        eff_stride = ec * 16
        ki_count = swz_layout.ki_count

        ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
                   int(math.log2(mfma.m)), comment=f"k_group = lane_id / {mfma.m}")

        # A: m_row = wave_m * m_per_wave + lane_row
        ctx.comment("LDS read A (paired-row swizzle)")
        ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(tile.m_per_wave),
                  ctx.vreg("v_wave_m"), comment=f"wave_m * {tile.m_per_wave}")
        ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
                  ctx.vreg("v_tmp0"), comment="+ lane_row -> m_row_a")
        # row_base = (m_row / pair_factor) * eff_stride
        ctx.v_lshr(ctx.vreg("v_tmp2"), ctx.vreg("v_lds_rd_a"),
                   int(math.log2(pf)), comment=f"lds_row = m_row / {pf}")
        ctx.v_mul(ctx.vreg("v_tmp2"), str(eff_stride), ctx.vreg("v_tmp2"),
                  comment=f"row_base = lds_row * {eff_stride}")
        # emit_read_setup: v_lds_rd_a = m_row, v_tmp1 = k_group, v_tmp2 = row_base
        a_out = [ctx.vreg("v_lds_rd_a")]
        for ki in range(1, ki_count):
            vname = f"v_lds_rd_a_k{ki}"
            if not ctx.has(vname):
                ctx.alloc_vgpr_permanent(1, vname)
            a_out.append(ctx.vreg(vname))
        swz.emit_read_setup(ctx, swz_layout, LDS_GFX950,
                            ctx.vreg("v_lds_rd_a"), ctx.vreg("v_tmp1"),
                            ctx.vreg("v_tmp2"), a_out)
        ctx.raw("")

        # B: n_row = wave_n * n_per_wave + lane_row
        ctx.comment("LDS read B (paired-row swizzle)")
        ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
                  comment=f"lane_row = lane_id % {mfma.m}")
        ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(tile.n_per_wave),
                  ctx.vreg("v_wave_n"), comment=f"wave_n * {tile.n_per_wave}")
        ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
                  ctx.vreg("v_tmp0"), comment="+ lane_row -> n_row_b")
        # row_base = (n_row / pf) * eff_stride + lds_b_offset
        lds_b_off = int((tile.wg_m + tile.wg_n) * tile.unroll_k * elem) // 2
        ctx.v_lshr(ctx.vreg("v_tmp2"), ctx.vreg("v_lds_rd_b"),
                   int(math.log2(pf)), comment=f"lds_row = n_row / {pf}")
        ctx.v_mul(ctx.vreg("v_tmp2"), str(eff_stride), ctx.vreg("v_tmp2"),
                  comment=f"row_base = lds_row * {eff_stride}")
        ctx.s_mov(ctx.sreg("s_tmp0"), str(lds_b_off), comment=f"lds_b_off={lds_b_off}")
        ctx.v_add(ctx.vreg("v_tmp2"), ctx.vreg("v_tmp2"), ctx.sreg("s_tmp0"),
                  comment=f"+ lds_b_offset={lds_b_off}")
        b_out = [ctx.vreg("v_lds_rd_b")]
        for ki in range(1, ki_count):
            vname = f"v_lds_rd_b_k{ki}"
            if not ctx.has(vname):
                ctx.alloc_vgpr_permanent(1, vname)
            b_out.append(ctx.vreg(vname))
        swz.emit_read_setup(ctx, swz_layout, LDS_GFX950,
                            ctx.vreg("v_lds_rd_b"), ctx.vreg("v_tmp1"),
                            ctx.vreg("v_tmp2"), b_out)
    elif swz is not None:
        # Non-paired swizzle (legacy path)
        from ..memory.swizzle import DataLayout as SwzLayout, LDS_GFX950
        swz_layout = SwzLayout(row_stride_bytes=row_stride_bytes,
                               mfma_k=mfma.k, mfma_m=mfma.m,
                               elem_bytes=elem, wave_size=tile.wave_size)
        ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
                   int(math.log2(mfma.m)), comment=f"k_group = lane_id / {mfma.m}")
        ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(row_stride_bytes),
                  ctx.vreg("v_tmp0"), comment=f"lane_row * {row_stride_bytes}")
        ki_count = swz_layout.ki_count
        a_out = [ctx.vreg("v_lds_rd_a")] + [ctx.vreg(f"v_lds_rd_a_k{ki}") for ki in range(1, ki_count)]
        swz.emit_read_setup(ctx, swz_layout, LDS_GFX950,
                            ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
                            ctx.vreg("v_lds_rd_a"), a_out)
        ctx.raw("")
        ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
                  comment=f"lane_row (re-derive)")
        ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(row_stride_bytes),
                  ctx.vreg("v_tmp0"), comment=f"lane_row * {row_stride_bytes}")
        b_out = [ctx.vreg("v_lds_rd_b")] + [ctx.vreg(f"v_lds_rd_b_k{ki}") for ki in range(1, ki_count)]
        swz.emit_read_setup(ctx, swz_layout, LDS_GFX950,
                            ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
                            ctx.vreg("v_lds_rd_b"), b_out)
    else:
        ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
                   int(math.log2(mfma.m)), comment=f"lane_id / {mfma.m}")
        ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_tmp1"),
                   int(math.log2(k_per_group)), comment=f"* {k_per_group}")

        # LDS read A
        ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(tile.m_per_wave),
                  ctx.vreg("v_wave_m"), comment=f"wave_m * {tile.m_per_wave}")
        ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
                  ctx.vreg("v_tmp0"), comment="+ lane_row")
        ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(tile.unroll_k),
                  ctx.vreg("v_lds_rd_a"), comment=f"* {tile.unroll_k}")
        if tile.lds_pad > 0:
            tpr = int(tile.unroll_k * elem) // 16
            rpl = tile.block_size // tpr
            wave_lines = tile.m_per_wave // rpl
            if wave_lines > 0:
                ctx.v_mul(ctx.vreg("v_tmp0"), str(wave_lines * tile.lds_pad),
                          ctx.vreg("v_wave_m"),
                          comment=f"wave pad = wave_m * {wave_lines * tile.lds_pad}")
                ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
                          ctx.vreg("v_tmp0"), comment="+ wave padding")
        ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
                  ctx.vreg("v_tmp1"), comment="+ lane_k")
        if elem >= 1:
            ctx.v_lshl(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
                       int(math.log2(elem)), comment=f"* {elem}")
        else:
            ctx.v_lshr(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
                       int(math.log2(1.0 / elem)), comment=f"* {elem} (sub-byte)")
        ctx.raw("")

        # LDS read B
        ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
                  comment=f"lane_row = lane_id % {mfma.m} (re-derive)")
        ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(tile.n_per_wave),
                  ctx.vreg("v_wave_n"), comment=f"wave_n * {tile.n_per_wave}")
        ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
                  ctx.vreg("v_tmp0"), comment="+ lane_row")
        ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(tile.unroll_k),
                  ctx.vreg("v_lds_rd_b"), comment=f"* {tile.unroll_k}")
        ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
                  ctx.vreg("v_tmp1"), comment="+ lane_k")
        if elem >= 1:
            ctx.v_lshl(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
                       int(math.log2(elem)), comment=f"* {elem}")
        else:
            ctx.v_lshr(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
                       int(math.log2(1.0 / elem)), comment=f"* {elem} (sub-byte)")
        if tile.lds_pad > 0:
            tpr_b = int(tile.unroll_k * elem) // 16
            rpl_b = tile.block_size // tpr_b
            wave_lines_b = tile.n_per_wave // rpl_b
            if wave_lines_b > 0:
                ctx.v_mul(ctx.vreg("v_tmp0"), str(wave_lines_b * tile.lds_pad),
                          ctx.vreg("v_wave_n"),
                          comment=f"wave pad = wave_n * {wave_lines_b * tile.lds_pad}")
                ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
                          ctx.vreg("v_tmp0"), comment="+ wave padding (bytes)")
    if tile.lds_pad > 0:
        threads_per_row__ = int(tile.unroll_k * elem) // 16
        rows_per_load__ = tile.block_size // threads_per_row__
        num_loads_a__ = tile.wg_m // rows_per_load__
        dtl_b_off = int(tile.wg_m * tile.unroll_k * elem) + num_loads_a__ * tile.lds_pad
    else:
        dtl_b_off = layouts.lds_b_offset
    # Paired-row swizzle already added lds_b_offset in the swizzle block
    swz_check = tile.resolved_swizzle(elem)
    if swz_check is None or not hasattr(swz_check, 'pair_factor'):
        ctx.v_add(ctx.vreg("v_lds_rd_b"), str(dtl_b_off),
                  ctx.vreg("v_lds_rd_b"), comment=f"+ lds_b_offset({dtl_b_off})")
    ctx.raw("")

    # Init accumulators
    acc_total = tile.mfma_m_repeat * tile.mfma_n_repeat * tile.mfma.acc_vgprs
    ctx.comment(f"Init {acc_total} accumulators")
    for i in range(acc_total):
        ctx.inst("v_accvgpr_write_b32", ctx.areg("acc_C", i, 1), "0")
    ctx.raw("")

    # Init MX constant scale VGPR
    if mfma.is_mx:
        ctx.comment("Init MX constant scale = 1.0 (E8M0 0x7F)")
        ctx.v_mov(ctx.vreg("v_mxscale"), "0x7F7F7F7F",
                  comment="scale = 1.0 for all byte lanes")
        ctx.raw("")


def phase_dtl_interleaved_setup(level: TileLevel, ctx: AsmContext) -> None:
    """Setup SRDs, offsets, LDS read addrs, accumulators for DTL kernel."""
    tile = _tile(ctx)
    problem = _problem(ctx)
    elem = problem.element_bytes
    mfma = tile.mfma
    layouts = _layouts(ctx)

    ctx.comment("=== DTL Interleaved Setup ===")

    # TensileLite kernarg layout (batched MXFP4, KernArgsVersion >= 1):
    #   0-15: header (Gemm info, kernel info0/1, numWG) -- ignored
    #   16: M, 20: N, 24: batch, 28: K
    #   32: D, 40: C, 48: A, 56: MXSA, 64: B, 72: MXSB
    #   80+: strides, alpha, beta
    karg = ctx.sreg("s_kernarg")
    ctx.inst("s_load_dword", ctx.sreg("s_M"), karg, "16", comment="M")
    ctx.inst("s_load_dword", ctx.sreg("s_N"), karg, "20", comment="N")
    ctx.inst("s_load_dword", ctx.sreg("s_K"), karg, "28", comment="K")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_D"), karg, "32", comment="D ptr")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_A"), karg, "48", comment="A ptr")
    # B ptr offset: 64 for MX (MXSA at 56), 56 for non-MX (no MXSA)
    b_offset = "64" if mfma.is_mx else "56"
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_B"), karg, b_offset,
             comment="B ptr")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait kernargs")
    ctx.raw("")

    # 1D WG decomposition (KernArgsVersion >= 1: grid flattened to 1D)
    if ctx._metadata.get("use_1d_grid", False):
        import math as _math
        _log2_mt = int(_math.log2(tile.wg_m))

        # WorkGroupMappingXCC: remap WG serial for L2 locality across XCCs
        wgmxcc = ctx._metadata.get("wg_mapping_xcc", 1)
        if wgmxcc > 1:
            _log2_xcc = int(_math.log2(wgmxcc))
            ctx.comment(f"WorkGroupMappingXCC={wgmxcc}: remap for L2 locality")
            # Load numWG from kernarg offset 12
            ctx.alloc_sgpr_permanent(1, "s_numWG")
            ctx.inst("s_load_dword", ctx.sreg("s_numWG"), ctx.sreg("s_kernarg"),
                     "12", comment="numWG (total workgroups)")
            ctx.s_waitcnt("lgkmcnt(0)", comment="wait numWG")
            # Interleave: new = (old >> K) + (old & (WGMXCC-1)) * (numWG >> K)
            ctx.inst("s_lshr_b32", ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"),
                     str(_log2_xcc), comment=f"old_wg / {wgmxcc}")
            ctx.inst("s_and_b32", ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_x"),
                     str(wgmxcc - 1), comment=f"old_wg % {wgmxcc} (XCC lane)")
            ctx.inst("s_lshr_b32", ctx.sreg("s_numWG"), ctx.sreg("s_numWG"),
                     str(_log2_xcc), comment=f"numWG / {wgmxcc}")
            ctx.inst("s_mul_i32", ctx.sreg("s_tmp1"), ctx.sreg("s_tmp1"),
                     ctx.sreg("s_numWG"), comment="XCC_lane * (numWG / WGMXCC)")
            ctx.inst("s_add_u32", ctx.sreg("s_wg_id_x"), ctx.sreg("s_tmp0"),
                     ctx.sreg("s_tmp1"), comment="remapped WG serial")
            ctx.raw("")

        # 1D WG decomposition: tile_m = serial % numWG_m, tile_n = serial / numWG_m
        ctx.s_mov(ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_x"),
                  comment="save wg_serial")
        # numWG_m = (M + MT_M - 1) >> log2(MT_M)
        ctx.inst("s_add_u32", ctx.sreg("s_tmp0"), ctx.sreg("s_M"),
                 str(tile.wg_m - 1), comment=f"M + {tile.wg_m - 1}")
        ctx.inst("s_lshr_b32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                 str(_log2_mt), comment=f"numWG_m = ceil(M/{tile.wg_m})")
        # Division: tile_n = serial / numWG_m, tile_m = serial % numWG_m
        # Use s_ff1 to detect if numWG_m is power-of-2 and use shift
        ctx.inst("s_sub_u32", ctx.sreg("s_wg_id_x"),
                 ctx.sreg("s_tmp0"), "1", comment="numWG_m - 1")
        ctx.inst("s_and_b32", ctx.sreg("s_wg_id_y"),
                 ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"),
                 comment="numWG_m & (numWG_m-1) == 0 if power-of-2")
        # For now assume power-of-2 (all our tiles/problems satisfy this)
        ctx.inst("s_ff1_i32_b32", ctx.sreg("s_wg_id_y"),
                 ctx.sreg("s_tmp0"), comment="log2(numWG_m)")
        ctx.inst("s_lshr_b32", ctx.sreg("s_wg_id_y"),
                 ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_y"),
                 comment="tile_n = serial >> log2(numWG_m)")  # s3 = tile_n  (s_wg_id_y)
        # tile_m = serial & (numWG_m - 1)
        ctx.inst("s_sub_u32", ctx.sreg("s_wg_id_x"),
                 ctx.sreg("s_tmp0"), "1", comment="numWG_m - 1")
        ctx.inst("s_and_b32", ctx.sreg("s_wg_id_x"),
                 ctx.sreg("s_tmp1"), ctx.sreg("s_wg_id_x"),
                 comment="tile_m = serial & (numWG_m - 1)")



    # Thread indexing
    log2_ws = int(math.log2(tile.wave_size))
    ctx.v_lshr(ctx.vreg("v_wave_id"), ctx.vreg("v_tid"), log2_ws,
               comment=f"wave_id = tid >> {log2_ws}")
    ctx.v_and(ctx.vreg("v_lane_id"), ctx.vreg("v_tid"), tile.wave_size - 1,
              comment=f"lane_id = tid & {tile.wave_size - 1}")
    log2_wn = int(math.log2(tile.waves_n)) if tile.waves_n > 1 else 0
    if tile.waves_n > 1:
        ctx.v_lshr(ctx.vreg("v_wave_m"), ctx.vreg("v_wave_id"), log2_wn,
                   comment=f"wave_m = wave_id >> {log2_wn}")
        ctx.v_and(ctx.vreg("v_wave_n"), ctx.vreg("v_wave_id"),
                  tile.waves_n - 1, comment=f"wave_n = wave_id & {tile.waves_n - 1}")
    else:
        ctx.v_mov(ctx.vreg("v_wave_m"), ctx.vreg("v_wave_id"), comment="wave_m")
        ctx.v_mov(ctx.vreg("v_wave_n"), "0", comment="wave_n = 0")
    ctx.raw("")

    # DTL per-lane offset
    threads_per_row = int(tile.unroll_k * elem) // 16
    log2_tpr = int(math.log2(threads_per_row))
    ctx.comment(f"DTL offset: {threads_per_row} threads/row")
    ctx.v_lshr(ctx.vreg("v_tmp0"), ctx.vreg("v_tid"), log2_tpr,
               comment="thread_row")
    ctx.v_and(ctx.vreg("v_tmp1"), ctx.vreg("v_tid"), threads_per_row - 1,
              comment="thread_col_group")
    # Note: swizzle is NOT applied to DTL offsets because DTL uses the
    # same offset for both global read and LDS write. Swizzling would
    # read from wrong global addresses. Swizzle is handled by the
    # BufferLoader path instead (global_load -> VGPR -> ds_write).
    ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_tmp1"), 4,
               comment="* 16 -> col_bytes")

    # K stride: K * element_bytes
    if elem >= 1:
        ctx.s_lshl(ctx.sreg("s_k_stride"), ctx.sreg("s_K"),
                   int(math.log2(elem)), comment=f"s_k_stride = K * {elem}")
    else:
        # Sub-byte: elem=0.5 means K / 2
        ctx.s_lshr(ctx.sreg("s_k_stride"), ctx.sreg("s_K"),
                   int(math.log2(1.0 / elem)), comment=f"s_k_stride = K * {elem}")

    # Compute swizzled LDS write offset (for BufferLoader path)
    # v_tmp0 = thread_row, v_tmp1 = col_bytes (= thread_col * 16)
    swz = tile.resolved_swizzle(elem)
    if swz is not None and hasattr(swz, 'pair_factor'):
        from ..memory.swizzle import DataLayout as SwzLayout, LDS_GFX950
        swz_layout = SwzLayout(row_stride_bytes=int(tile.unroll_k * elem),
                               mfma_k=mfma.k, mfma_m=mfma.m,
                               elem_bytes=elem, wave_size=tile.wave_size)
        pf = swz.pair_factor
        ec = swz.effective_cols
        eff_stride = ec * 16

        ctx.comment(f"Swizzled LDS write offset (pair={pf}, eff_cols={ec})")
        # thread_col in 16B column units (v_tmp1 is already col_bytes)
        ctx.v_lshr(ctx.vreg("v_tmp4"), ctx.vreg("v_tmp1"), 4,
                   comment="thread_col = col_bytes / 16")
        # Apply write swizzle: computes swizzled_col from (thread_row, thread_col)
        swz.emit_write_swizzle(ctx, swz_layout, LDS_GFX950,
                               ctx.vreg("v_tmp0"), ctx.vreg("v_tmp4"),
                               ctx.vreg("v_tmp4"))
        # lds_row = thread_row / pair_factor
        ctx.v_lshr(ctx.vreg("v_tmp3"), ctx.vreg("v_tmp0"),
                   int(math.log2(pf)), comment=f"lds_row = row / {pf}")
        # v_lds_wr_swz = lds_row * eff_stride + swizzled_col * 16
        ctx.alloc_vgpr_permanent(1, "v_lds_wr_swz")
        ctx.v_mul(ctx.vreg("v_lds_wr_swz"), str(eff_stride),
                  ctx.vreg("v_tmp3"), comment=f"lds_row * {eff_stride}")
        ctx.v_lshl(ctx.vreg("v_tmp4"), ctx.vreg("v_tmp4"), 4,
                   comment="swizzled_col * 16")
        ctx.v_add(ctx.vreg("v_lds_wr_swz"), ctx.vreg("v_lds_wr_swz"),
                  ctx.vreg("v_tmp4"), comment="+ swizzled_col_bytes")
        ctx.raw("")

    # LDS write offset = thread_row * unroll_k * elem + col_bytes
    # This uses the LDS row stride (unroll_k*elem), NOT the global stride (K*elem)
    row_stride_lds = int(tile.unroll_k * elem)
    ctx.alloc_vgpr_permanent(1, "v_lds_wr_off")
    ctx.v_mul(ctx.vreg("v_lds_wr_off"), str(row_stride_lds),
              ctx.vreg("v_tmp0"), comment=f"row * {row_stride_lds} (LDS stride)")
    ctx.v_add(ctx.vreg("v_lds_wr_off"), ctx.vreg("v_lds_wr_off"),
              ctx.vreg("v_tmp1"), comment="+ col_bytes -> v_lds_wr_off")
    ctx.raw("")

    # Double-buffer write offset for BufferLoader (starts at 0)
    ctx.alloc_sgpr_permanent(1, "s_buf_wr_db")
    ctx.s_mov(ctx.sreg("s_buf_wr_db"), "0",
              comment="buf_wr_db = 0 (buffer 0)")
    ctx.raw("")

    # DTL voffset = thread_row * K * elem + col_bytes (global stride)
    ctx.inst("v_mul_lo_u32", ctx.vreg("v_dtl_off_a"),
             ctx.sreg("s_k_stride"), ctx.vreg("v_tmp0"), comment="row * K*elem")
    ctx.v_add(ctx.vreg("v_dtl_off_a"), ctx.vreg("v_dtl_off_a"),
              ctx.vreg("v_tmp1"), comment="+ col_bytes")
    ctx.v_mov(ctx.vreg("v_dtl_off_b"), ctx.vreg("v_dtl_off_a"),
              comment="B offset = same")
    ctx.raw("")

    # SRD A
    ctx.comment("SRD A")
    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"), str(tile.wg_m),
              comment=f"wg_id * {tile.wg_m}")
    ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
             ctx.sreg("s_k_stride"), comment="* K*elem")
    ctx.inst("s_add_u32", ctx.sreg("s_srd_a", 0, 1),
             ctx.sreg("s_ptr_A", 0, 1), ctx.sreg("s_tmp0"), comment="SRD_A lo")
    ctx.inst("s_addc_u32", ctx.sreg("s_srd_a", 1, 1),
             ctx.sreg("s_ptr_A", 1, 1), "0", comment="SRD_A hi")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_a", 2, 1), "0xFFFFFFFF", comment="limit")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_a", 3, 1), "0x20000", comment="flags")
    ctx.raw("")

    # SRD B
    ctx.comment("SRD B")
    ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_y"), str(tile.wg_n),
              comment=f"wg_id * {tile.wg_n}")
    ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
             ctx.sreg("s_k_stride"), comment="* K*elem")
    ctx.inst("s_add_u32", ctx.sreg("s_srd_b", 0, 1),
             ctx.sreg("s_ptr_B", 0, 1), ctx.sreg("s_tmp0"), comment="SRD_B lo")
    ctx.inst("s_addc_u32", ctx.sreg("s_srd_b", 1, 1),
             ctx.sreg("s_ptr_B", 1, 1), "0", comment="SRD_B hi")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_b", 2, 1), "0xFFFFFFFF", comment="limit")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_b", 3, 1), "0x20000", comment="flags")
    ctx.raw("")

    # Scalar offsets for multi-line DTL loads
    rows_per_load = tile.block_size // threads_per_row
    ctx.comment(f"Scalar offset for DTL lines ({rows_per_load} rows/load)")
    ctx.s_mul(ctx.sreg("s_soffset_a"), ctx.sreg("s_k_stride"),
              str(rows_per_load), comment=f"soffset = {rows_per_load} * K*elem")
    ctx.s_mov(ctx.sreg("s_soffset_b"), ctx.sreg("s_soffset_a"), comment="same")
    ctx.raw("")

    # LDS write base for DTL (SGPR, wave-uniform)
    pad_e = tile.lds_pad // elem if tile.lds_pad > 0 else 0
    lds_row_stride = tile.unroll_k + pad_e
    ctx.comment("LDS write base for DTL")
    # DTL writes contiguously (m0 + lane_id*16), so row stride = unroll_k * elem
    # NOT lds_row_stride (which includes per-row padding for non-DTL).
    # The per-load-line padding is handled by m0 increments.
    dtl_row_stride = int(tile.unroll_k * elem)  # 128 bytes for uk=64 fp16
    ctx.v_mul(ctx.vreg("v_tmp0"), str(dtl_row_stride),
              ctx.vreg("v_tmp0"), comment=f"row * {dtl_row_stride}")
    ctx.v_add(ctx.vreg("v_tmp0"), ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
              comment="+ col_bytes -> per-thread LDS offset")
    ctx.inst("v_readfirstlane_b32", ctx.sreg("s_lds_wr_a_sg"),
             ctx.vreg("v_tmp0"), comment="LDS write base A")
    # Compute DTL-specific lds_b_offset (with per-load-line padding)
    if tile.lds_pad > 0:
        threads_per_row_ = int(tile.unroll_k * elem) // 16
        rows_per_load_ = tile.block_size // threads_per_row_
        num_loads_a_ = tile.wg_m // rows_per_load_
        dtl_lds_b_offset = int(tile.wg_m * tile.unroll_k * elem) + num_loads_a_ * tile.lds_pad
    else:
        dtl_lds_b_offset = layouts.lds_b_offset
    ctx.inst("s_add_u32", ctx.sreg("s_lds_wr_b_sg"),
             ctx.sreg("s_lds_wr_a_sg"), str(dtl_lds_b_offset),
             comment=f"LDS write base B = A + {dtl_lds_b_offset}")
    ctx.raw("")

    # LDS read addresses
    k_per_group = mfma.k // (tile.wave_size // mfma.m)
    threads_per_row_rd = int(tile.unroll_k * elem) // 16
    row_stride_bytes = int(tile.unroll_k * elem)
    ctx.comment("LDS read addresses")
    ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
              comment=f"lane_row = lane_id % {mfma.m}")

    swz = tile.resolved_swizzle(elem)
    if swz is not None and hasattr(swz, 'pair_factor'):
        # Paired-row swizzle: use m_row for rotation, paired row_base
        from ..memory.swizzle import DataLayout as SwzLayout, LDS_GFX950
        swz_layout = SwzLayout(row_stride_bytes=row_stride_bytes,
                               mfma_k=mfma.k, mfma_m=mfma.m,
                               elem_bytes=elem, wave_size=tile.wave_size)
        pf = swz.pair_factor
        ec = swz.effective_cols
        eff_stride = ec * 16
        ki_count = swz_layout.ki_count

        ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
                   int(math.log2(mfma.m)), comment=f"k_group = lane_id / {mfma.m}")

        # A: m_row = wave_m * m_per_wave + lane_row
        ctx.comment("LDS read A (paired-row swizzle)")
        ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(tile.m_per_wave),
                  ctx.vreg("v_wave_m"), comment=f"wave_m * {tile.m_per_wave}")
        ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
                  ctx.vreg("v_tmp0"), comment="+ lane_row -> m_row_a")
        # row_base = (m_row / pf) * eff_stride
        ctx.v_lshr(ctx.vreg("v_tmp2"), ctx.vreg("v_lds_rd_a"),
                   int(math.log2(pf)), comment=f"lds_row = m_row / {pf}")
        ctx.v_mul(ctx.vreg("v_tmp2"), str(eff_stride), ctx.vreg("v_tmp2"),
                  comment=f"row_base = lds_row * {eff_stride}")
        a_out = [ctx.vreg("v_lds_rd_a")]
        for ki in range(1, ki_count):
            vname = f"v_lds_rd_a_k{ki}"
            if not ctx.has(vname):
                ctx.alloc_vgpr_permanent(1, vname)
            a_out.append(ctx.vreg(vname))
        swz.emit_read_setup(ctx, swz_layout, LDS_GFX950,
                            ctx.vreg("v_lds_rd_a"), ctx.vreg("v_tmp1"),
                            ctx.vreg("v_tmp2"), a_out)

        # Save m_row_base and k_group for per-mi recomputation
        ctx.alloc_vgpr_permanent(1, "v_m_row_base_a")
        ctx.v_mul(ctx.vreg("v_m_row_base_a"), str(tile.m_per_wave),
                  ctx.vreg("v_wave_m"), comment=f"m_row_base = wave_m * {tile.m_per_wave}")
        ctx.v_add(ctx.vreg("v_m_row_base_a"), ctx.vreg("v_m_row_base_a"),
                  ctx.vreg("v_tmp0"), comment="+ lane_row (persistent)")
        ctx.alloc_vgpr_permanent(1, "v_k_group_a")
        ctx.v_mov(ctx.vreg("v_k_group_a"), ctx.vreg("v_tmp1"),
                  comment="k_group (persistent)")
        ctx.raw("")

        # B: n_row = wave_n * n_per_wave + lane_row
        ctx.comment("LDS read B (paired-row swizzle)")
        ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
                  comment=f"lane_row = lane_id % {mfma.m}")
        ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(tile.n_per_wave),
                  ctx.vreg("v_wave_n"), comment=f"wave_n * {tile.n_per_wave}")
        ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
                  ctx.vreg("v_tmp0"), comment="+ lane_row -> n_row_b")
        lds_b_off = dtl_lds_b_offset
        ctx.v_lshr(ctx.vreg("v_tmp2"), ctx.vreg("v_lds_rd_b"),
                   int(math.log2(pf)), comment=f"lds_row = n_row / {pf}")
        ctx.v_mul(ctx.vreg("v_tmp2"), str(eff_stride), ctx.vreg("v_tmp2"),
                  comment=f"row_base = lds_row * {eff_stride}")
        ctx.s_mov(ctx.sreg("s_tmp0"), str(lds_b_off), comment=f"lds_b_off={lds_b_off}")
        ctx.v_add(ctx.vreg("v_tmp2"), ctx.vreg("v_tmp2"), ctx.sreg("s_tmp0"),
                  comment=f"+ lds_b_offset={lds_b_off}")
        b_out = [ctx.vreg("v_lds_rd_b")]
        for ki in range(1, ki_count):
            vname = f"v_lds_rd_b_k{ki}"
            if not ctx.has(vname):
                ctx.alloc_vgpr_permanent(1, vname)
            b_out.append(ctx.vreg(vname))
        swz.emit_read_setup(ctx, swz_layout, LDS_GFX950,
                            ctx.vreg("v_lds_rd_b"), ctx.vreg("v_tmp1"),
                            ctx.vreg("v_tmp2"), b_out)

        # Save n_row_base for per-ni recomputation
        ctx.alloc_vgpr_permanent(1, "v_n_row_base_b")
        ctx.v_mul(ctx.vreg("v_n_row_base_b"), str(tile.n_per_wave),
                  ctx.vreg("v_wave_n"), comment=f"n_row_base = wave_n * {tile.n_per_wave}")
        ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
                  comment="lane_row (re-derive for save)")
        ctx.v_add(ctx.vreg("v_n_row_base_b"), ctx.vreg("v_n_row_base_b"),
                  ctx.vreg("v_tmp0"), comment="+ lane_row (persistent)")
        ctx.raw("")
    elif swz is not None:
        # Legacy swizzle (non-paired)
        from ..memory.swizzle import DataLayout as SwzLayout, LDS_GFX950
        swz_layout = SwzLayout(row_stride_bytes=row_stride_bytes,
                               mfma_k=mfma.k, mfma_m=mfma.m,
                               elem_bytes=elem, wave_size=tile.wave_size)
        ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
                   int(math.log2(mfma.m)), comment=f"k_group = lane_id / {mfma.m}")
        ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(row_stride_bytes),
                  ctx.vreg("v_tmp0"), comment=f"lane_row * {row_stride_bytes}")
        ki_count = swz_layout.ki_count
        a_out = [ctx.vreg("v_lds_rd_a")] + [ctx.vreg(f"v_lds_rd_a_k{ki}") for ki in range(1, ki_count)]
        swz.emit_read_setup(ctx, swz_layout, LDS_GFX950,
                            ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
                            ctx.vreg("v_lds_rd_a"), a_out)
        ctx.raw("")
        ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
                  comment=f"lane_row (re-derive)")
        ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(row_stride_bytes),
                  ctx.vreg("v_tmp0"), comment=f"lane_row * {row_stride_bytes}")
        b_out = [ctx.vreg("v_lds_rd_b")] + [ctx.vreg(f"v_lds_rd_b_k{ki}") for ki in range(1, ki_count)]
        swz.emit_read_setup(ctx, swz_layout, LDS_GFX950,
                            ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
                            ctx.vreg("v_lds_rd_b"), b_out)
    else:
        ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
                   int(math.log2(mfma.m)), comment=f"lane_id / {mfma.m}")
        ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_tmp1"),
                   int(math.log2(k_per_group)), comment=f"* {k_per_group}")

        # LDS read A
        ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(tile.m_per_wave),
                  ctx.vreg("v_wave_m"), comment=f"wave_m * {tile.m_per_wave}")
        ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
                  ctx.vreg("v_tmp0"), comment="+ lane_row")
        ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(tile.unroll_k),
                  ctx.vreg("v_lds_rd_a"), comment=f"* {tile.unroll_k}")
        if tile.lds_pad > 0:
            tpr = int(tile.unroll_k * elem) // 16
            rpl = tile.block_size // tpr
            wave_lines = tile.m_per_wave // rpl
            if wave_lines > 0:
                ctx.v_mul(ctx.vreg("v_tmp0"), str(wave_lines * tile.lds_pad),
                          ctx.vreg("v_wave_m"),
                          comment=f"wave pad = wave_m * {wave_lines * tile.lds_pad}")
                ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
                          ctx.vreg("v_tmp0"), comment="+ wave padding")
        ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
                  ctx.vreg("v_tmp1"), comment="+ lane_k")
        if elem >= 1:
            ctx.v_lshl(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
                       int(math.log2(elem)), comment=f"* {elem}")
        else:
            ctx.v_lshr(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
                       int(math.log2(1.0 / elem)), comment=f"* {elem} (sub-byte)")
        ctx.raw("")

        # LDS read B
        ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
                  comment=f"lane_row = lane_id % {mfma.m} (re-derive)")
        ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(tile.n_per_wave),
                  ctx.vreg("v_wave_n"), comment=f"wave_n * {tile.n_per_wave}")
        ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
                  ctx.vreg("v_tmp0"), comment="+ lane_row")
        ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(tile.unroll_k),
                  ctx.vreg("v_lds_rd_b"), comment=f"* {tile.unroll_k}")
        ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
                  ctx.vreg("v_tmp1"), comment="+ lane_k")
        if elem >= 1:
            ctx.v_lshl(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
                       int(math.log2(elem)), comment=f"* {elem}")
        else:
            ctx.v_lshr(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
                       int(math.log2(1.0 / elem)), comment=f"* {elem} (sub-byte)")
        if tile.lds_pad > 0:
            tpr_b = int(tile.unroll_k * elem) // 16
            rpl_b = tile.block_size // tpr_b
            wave_lines_b = tile.n_per_wave // rpl_b
            if wave_lines_b > 0:
                ctx.v_mul(ctx.vreg("v_tmp0"), str(wave_lines_b * tile.lds_pad),
                          ctx.vreg("v_wave_n"),
                          comment=f"wave pad = wave_n * {wave_lines_b * tile.lds_pad}")
                ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
                          ctx.vreg("v_tmp0"), comment="+ wave padding (bytes)")
    if tile.lds_pad > 0:
        threads_per_row__ = int(tile.unroll_k * elem) // 16
        rows_per_load__ = tile.block_size // threads_per_row__
        num_loads_a__ = tile.wg_m // rows_per_load__
        dtl_b_off = int(tile.wg_m * tile.unroll_k * elem) + num_loads_a__ * tile.lds_pad
    else:
        dtl_b_off = layouts.lds_b_offset
    # Paired-row swizzle already added lds_b_offset in the swizzle block
    swz_check = tile.resolved_swizzle(elem)
    if swz_check is None or not hasattr(swz_check, 'pair_factor'):
        ctx.v_add(ctx.vreg("v_lds_rd_b"), str(dtl_b_off),
                  ctx.vreg("v_lds_rd_b"), comment=f"+ lds_b_offset({dtl_b_off})")
    ctx.raw("")

    # Init accumulators
    acc_total = tile.mfma_m_repeat * tile.mfma_n_repeat * tile.mfma.acc_vgprs
    ctx.comment(f"Init {acc_total} accumulators")
    for i in range(acc_total):
        ctx.inst("v_accvgpr_write_b32", ctx.areg("acc_C", i, 1), "0")
    ctx.raw("")

    # Init MX constant scale VGPR (E8M0 scale=1.0 in all 4 bytes)
    if mfma.is_mx:
        ctx.comment("Init MX constant scale = 1.0 (E8M0 0x7F)")
        ctx.v_mov(ctx.vreg("v_mxscale"), "0x7F7F7F7F",
                  comment="scale = 1.0 for all byte lanes")
        ctx.raw("")


def _emit_dtl_loads_a(ctx: AsmContext, tile: TileConfig, problem: GemmProblem, num_loads: int) -> None:
    """Issue DTL loads for A matrix."""
    elem = problem.element_bytes
    threads_per_row = int(tile.unroll_k * elem) // 16
    rows_per_load = tile.block_size // threads_per_row
    lds_data_per_load = int(rows_per_load * tile.unroll_k * elem)
    lds_stride = lds_data_per_load + tile.lds_pad  # add padding per load line

    ctx.inst("s_mov_b32", "m0", ctx.sreg("s_lds_wr_a_sg"), comment="m0 = LDS base A")
    has_precomputed_a = ctx.has(f"s_dtl_soff_a1") if num_loads > 1 else False
    if not has_precomputed_a:
        ctx.s_mov(ctx.sreg("s_tmp0"), "0", comment="cumulative soffset A")
    for i in range(num_loads):
        if has_precomputed_a:
            soff = "0" if i == 0 else ctx.sreg(f"s_dtl_soff_a{i}")
        else:
            soff = ctx.sreg("s_tmp0")
        ctx.inst("buffer_load_dwordx4",
                 ctx.vreg("v_dtl_off_a"), ctx.sreg("s_srd_a", 0, 4),
                 soff, "offen offset:0, lds",
                 comment=f"DTL A[{i}]")
        if i < num_loads - 1:
            ctx.inst("s_add_u32", "m0", "m0", str(lds_stride),
                     comment=f"m0 += {lds_stride}")
            if not has_precomputed_a:
                ctx.inst("s_add_u32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                         ctx.sreg("s_soffset_a"), comment="soffset += stride")


def _emit_dtl_loads_b(ctx: AsmContext, tile: TileConfig, problem: GemmProblem, num_loads: int) -> None:
    """Issue DTL loads for B matrix."""
    elem = problem.element_bytes
    threads_per_row = int(tile.unroll_k * elem) // 16
    rows_per_load = tile.block_size // threads_per_row
    lds_data_per_load = int(rows_per_load * tile.unroll_k * elem)
    lds_stride = lds_data_per_load + tile.lds_pad

    ctx.inst("s_mov_b32", "m0", ctx.sreg("s_lds_wr_b_sg"), comment="m0 = LDS base B")
    has_precomputed_b = ctx.has(f"s_dtl_soff_b1") if num_loads > 1 else False
    if not has_precomputed_b:
        ctx.s_mov(ctx.sreg("s_tmp0"), "0", comment="cumulative soffset B")
    for i in range(num_loads):
        if has_precomputed_b:
            soff = "0" if i == 0 else ctx.sreg(f"s_dtl_soff_b{i}")
        else:
            soff = ctx.sreg("s_tmp0")
        ctx.inst("buffer_load_dwordx4",
                 ctx.vreg("v_dtl_off_b"), ctx.sreg("s_srd_b", 0, 4),
                 soff, "offen offset:0, lds",
                 comment=f"DTL B[{i}]")
        if i < num_loads - 1:
            ctx.inst("s_add_u32", "m0", "m0", str(lds_stride),
                     comment=f"m0 += {lds_stride}")
            if not has_precomputed_b:
                ctx.inst("s_add_u32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                         ctx.sreg("s_soffset_b"), comment="soffset += stride")



# ---------------------------------------------------------------------------
# MX scale setup (moved from partitioned.py)
# ---------------------------------------------------------------------------




def phase_mx_scale_setup(level: TileLevel, ctx: AsmContext) -> None:
    """Set up scale SRDs for direct VGPR loading (no LDS).

    Scale data is small enough to load directly from global memory
    into VGPRs, bypassing LDS entirely. This keeps the full 64KB LDS
    budget for data A/B.

    Each MFMA tile needs 4 scale bytes (one per K-block of 32 elements).
    The scale is per-MFMA-tile (shared across all lanes), loaded via
    buffer_load_dword into a VGPR, then broadcast.
    """
    tile = ctx._metadata["tile"]
    mfma = tile.mfma
    use_dtl = ctx._metadata.get("use_dtl", True)
    use_real_scales = ctx._metadata.get("use_real_scales", False)
    use_swizzled_scales = ctx._metadata.get("swizzled_scales", False) and use_real_scales
    if not mfma.is_mx or not use_real_scales:
        return

    mx_block = mfma.mx_block
    scale_k_cols = tile.unroll_k // mx_block

    ctx.comment("=== MX Scale Setup (direct VGPR, no LDS) ===")

    # Wave ABI already loads scale ptrs/strides in setup phase
    if not ctx._metadata.get("use_wave_abi", False):
        # TensileLite kernarg offsets: MXSA@56, MXSB@72, strides@104,120
        karg = ctx.sreg("s_kernarg")
        ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_scale_a"), karg, "56",
                 comment="scale A ptr (MXSA)")
        ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_scale_b"), karg, "72",
                 comment="scale B ptr (MXSB)")
        ctx.inst("s_load_dword", ctx.sreg("s_stride_scale_a"), karg, "104",
                 comment="strideMXSA0")
        ctx.inst("s_load_dword", ctx.sreg("s_stride_scale_b"), karg, "120",
                 comment="strideMXSB0")
        ctx.s_waitcnt("lgkmcnt(0)", comment="wait scale kernargs")
        ctx.raw("")

    # Scale SRD A
    ctx.comment("Scale SRD A")
    if use_swizzled_scales:
        # Swizzled: offset = wg_id * (wg_m / 32) * stride
        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"),
                  str(tile.wg_m // 32),
                  comment=f"wg_id_x * {tile.wg_m // 32}")
    else:
        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_x"), str(tile.wg_m),
                  comment=f"wg_id_x * {tile.wg_m}")
    ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
             ctx.sreg("s_stride_scale_a"), comment="* stride_scale_a")
    ctx.inst("s_add_u32", ctx.sreg("s_srd_scale_a", 0, 1),
             ctx.sreg("s_ptr_scale_a", 0, 1), ctx.sreg("s_tmp0"),
             comment="SRD_scaleA lo")
    ctx.inst("s_addc_u32", ctx.sreg("s_srd_scale_a", 1, 1),
             ctx.sreg("s_ptr_scale_a", 1, 1), "0",
             comment="SRD_scaleA hi")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_scale_a", 2, 1),
             "0xFFFFFFFF", comment="limit")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_scale_a", 3, 1),
             "0x20000", comment="flags")
    ctx.raw("")

    # Scale SRD B
    ctx.comment("Scale SRD B")
    if use_swizzled_scales:
        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_y"),
                  str(tile.wg_n // 32),
                  comment=f"wg_id_y * {tile.wg_n // 32}")
    else:
        ctx.s_mul(ctx.sreg("s_tmp0"), ctx.sreg("s_wg_id_y"), str(tile.wg_n),
                  comment=f"wg_id_y * {tile.wg_n}")
    ctx.inst("s_mul_i32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
             ctx.sreg("s_stride_scale_b"), comment="* stride_scale_b")
    ctx.inst("s_add_u32", ctx.sreg("s_srd_scale_b", 0, 1),
             ctx.sreg("s_ptr_scale_b", 0, 1), ctx.sreg("s_tmp0"),
             comment="SRD_scaleB lo")
    ctx.inst("s_addc_u32", ctx.sreg("s_srd_scale_b", 1, 1),
             ctx.sreg("s_ptr_scale_b", 1, 1), "0",
             comment="SRD_scaleB hi")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_scale_b", 2, 1),
             "0xFFFFFFFF", comment="limit")
    ctx.inst("s_mov_b32", ctx.sreg("s_srd_scale_b", 3, 1),
             "0x20000", comment="flags")
    ctx.raw("")

    # Allocate scale soffset SGPRs for swizzled mode
    if ctx._metadata.get("use_1d_grid", False) or ctx._metadata.get("use_wave_abi", False):
        ctx.alloc_sgpr_permanent(1, "s_scale_soff_a0")
        ctx.alloc_sgpr_permanent(1, "s_scale_soff_a1")
        ctx.alloc_sgpr_permanent(1, "s_scale_soff_b0")
        ctx.alloc_sgpr_permanent(1, "s_scale_soff_b1")

    use_swizzled_scales = ctx._metadata.get("swizzled_scales", False)

    if use_swizzled_scales:
        # Pre-swizzled scale layout (AITER e8m0_shuffle):
        # Per-lane voffset = (lane_id % 16) * 4
        # Per-group soffset = (wave_m * 2 + group) * 256
        # The 4 bytes at each position contain scales for:
        #   byte[0]: (mi_lo, ki_lo), byte[1]: (mi_hi, ki_lo)
        #   byte[2]: (mi_lo, ki_hi), byte[3]: (mi_hi, ki_hi)
        ctx.comment("Scale swizzled voffset: lane_id * 4")
        ctx.v_lshl(ctx.vreg("v_dtl_off_scale_a"),
                   ctx.vreg("v_lane_id"), 2,
                   comment="lane_id * 4 -> swizzled scale voffset")
        ctx.inst("v_mov_b32", ctx.vreg("v_dtl_off_scale_b"),
                 ctx.vreg("v_dtl_off_scale_a"),
                 comment="scaleB voffset = same")
        ctx.raw("")

        # Compute SGPR soffsets for each mi-group and ni-group
        # group0: (wave_m * 2) * 256, group1: (wave_m * 2 + 1) * 256
        ctx.comment("Scale group soffsets")
        ctx.v_lshl(ctx.vreg("v_tmp0"), ctx.vreg("v_wave_m"), 1,
                   comment="wave_m * 2")
        ctx.inst("v_readfirstlane_b32", ctx.sreg("s_tmp0"),
                 ctx.vreg("v_tmp0"), comment="wave_m * 2 -> SGPR")
        ctx.s_lshl(ctx.sreg("s_scale_soff_a0"), ctx.sreg("s_tmp0"), 8,
                   comment="group0 soffset = wave_m*2 * 256")
        ctx.inst("s_add_u32", ctx.sreg("s_scale_soff_a1"),
                 ctx.sreg("s_scale_soff_a0"), "256",
                 comment="group1 soffset = (wave_m*2+1) * 256")
        # Same for B with wave_n
        ctx.v_lshl(ctx.vreg("v_tmp0"), ctx.vreg("v_wave_n"), 1,
                   comment="wave_n * 2")
        ctx.inst("v_readfirstlane_b32", ctx.sreg("s_tmp0"),
                 ctx.vreg("v_tmp0"), comment="wave_n * 2 -> SGPR")
        ctx.s_lshl(ctx.sreg("s_scale_soff_b0"), ctx.sreg("s_tmp0"), 8,
                   comment="group0 soffset B = wave_n*2 * 256")
        ctx.inst("s_add_u32", ctx.sreg("s_scale_soff_b1"),
                 ctx.sreg("s_scale_soff_b0"), "256",
                 comment="group1 soffset B = (wave_n*2+1) * 256")
        ctx.raw("")
    else:
        # Linear scale addressing (standalone path)
        ctx.comment("Scale A wave-level voffset")
        ctx.v_mul(ctx.vreg("v_tmp0"),
                  str(tile.m_per_wave), ctx.vreg("v_wave_m"),
                  comment=f"wave_m * {tile.m_per_wave}")
        ctx.inst("v_mul_lo_u32", ctx.vreg("v_dtl_off_scale_a"),
                 ctx.sreg("s_stride_scale_a"), ctx.vreg("v_tmp0"),
                 comment="wave_m_base * stride_scale_a -> voffset_scale_a")
        ctx.raw("")

        ctx.comment("Scale B wave-level voffset")
        ctx.v_mul(ctx.vreg("v_tmp0"),
                  str(tile.n_per_wave), ctx.vreg("v_wave_n"),
                  comment=f"wave_n * {tile.n_per_wave}")
        ctx.inst("v_mul_lo_u32", ctx.vreg("v_dtl_off_scale_b"),
                 ctx.sreg("s_stride_scale_b"), ctx.vreg("v_tmp0"),
                 comment="wave_n_base * stride_scale_b -> voffset_scale_b")
        ctx.raw("")

    # LDS scale write bases (for LDSScaleLoader / DTL scale path)
    use_lds_scales = ctx._metadata.get("use_lds_scales", False)
    if use_lds_scales:
        lds_data_half = ctx._metadata.get("lds_data_half", 0)
        mx_block = tile.mfma.mx_block
        scale_a_lds_size = max(tile.wg_m * (tile.unroll_k // mx_block), 4096)

        # Scale LDS is AFTER both data buffers (data uses power-of-2 DB)
        lds_data_total = lds_data_half * 2  # both data buffers
        scale_buf0_a = lds_data_total
        scale_buf0_b = lds_data_total + scale_a_lds_size
        lds_scale_half = ctx._metadata.get("lds_scale_half", 0)
        scale_buf1_a = lds_data_total + lds_scale_half
        scale_buf1_b = scale_buf1_a + scale_a_lds_size

        ctx.comment("Scale LDS write bases (after both data buffers)")
        ctx.alloc_sgpr_permanent(1, "s_lds_wr_scale_a")
        ctx.alloc_sgpr_permanent(1, "s_lds_wr_scale_b")
        ctx.s_mov(ctx.sreg("s_lds_wr_scale_a"),
                  str(scale_buf0_a),
                  comment=f"scale A LDS write base = {scale_buf0_a}")
        ctx.s_mov(ctx.sreg("s_lds_wr_scale_b"),
                  str(scale_buf0_b),
                  comment=f"scale B LDS write base = {scale_buf0_b}")

        # Scale DB swap mask: XOR toggles between buf0 and buf1
        scale_swap = scale_buf0_a ^ scale_buf1_a
        ctx.alloc_sgpr_permanent(1, "s_scale_db_swap")
        ctx.s_mov(ctx.sreg("s_scale_db_swap"), str(scale_swap),
                  comment=f"scale DB swap mask = {scale_swap:#x}")
        ctx.raw("")


