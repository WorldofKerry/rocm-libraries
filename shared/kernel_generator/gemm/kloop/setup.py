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
from ..tile.tree import TilePhase

__all__ = ["phase_dtl_interleaved_k_loop", "DTL_INTERLEAVED_PROLOGUE_PHASES",
           "phase_wave_abi_setup", "WAVE_ABI_PROLOGUE_PHASES"]


def _tile(ctx: AsmContext) -> TileConfig: return ctx._metadata["tile"]
def _problem(ctx: AsmContext) -> GemmProblem: return ctx._metadata["problem"]
def _layouts(ctx: AsmContext) -> GemmLayouts: return ctx._metadata["layouts"]


def _a_off(mi: int, ki: int, tile: TileConfig, mfma: MfmaConfig, elem: float) -> int:
    """LDS byte offset for A operand at (mi, ki).

    When lds_swizzle is enabled the ki contribution is handled by
    base-register selection (v_lds_rd_a for ki=0, XOR toggle for ki>0),
    so this returns only the mi row offset.
    """
    row_start = mi * mfma.m
    row_stride = int(tile.unroll_k * elem)
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

    # DTL voffset = thread_row * K * elem + col_bytes
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
    if swz is not None:
        from ..memory.swizzle import DataLayout as SwzLayout, LDS_GFX950
        swz_layout = SwzLayout(row_stride_bytes=row_stride_bytes,
                               mfma_k=mfma.k, mfma_m=mfma.m,
                               elem_bytes=elem, wave_size=tile.wave_size)
        ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
                   int(math.log2(mfma.m)), comment=f"k_group = lane_id / {mfma.m}")

        # Row base A
        ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(tile.m_per_wave),
                  ctx.vreg("v_wave_m"), comment=f"wave_m * {tile.m_per_wave}")
        ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
                  ctx.vreg("v_tmp0"), comment="+ lane_row")
        ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(row_stride_bytes),
                  ctx.vreg("v_lds_rd_a"), comment=f"* {row_stride_bytes} -> row_base_a")
        ki_count = swz_layout.ki_count
        a_out = [ctx.vreg("v_lds_rd_a")] + [ctx.vreg(f"v_lds_rd_a_k{ki}") for ki in range(1, ki_count)]
        swz.emit_read_setup(ctx, swz_layout, LDS_GFX950,
                            ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
                            ctx.vreg("v_lds_rd_a"), a_out)
        ctx.raw("")

        # Row base B
        ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
                  comment=f"lane_row = lane_id % {mfma.m} (re-derive)")
        ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(tile.n_per_wave),
                  ctx.vreg("v_wave_n"), comment=f"wave_n * {tile.n_per_wave}")
        ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
                  ctx.vreg("v_tmp0"), comment="+ lane_row")
        ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(row_stride_bytes),
                  ctx.vreg("v_lds_rd_b"), comment=f"* {row_stride_bytes} -> row_base_b")
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
        # 1D WG decomposition using pure scalar integer math
        # numWG_m = ceil(M / MT_M), tile_n = serial / numWG_m, tile_m = serial % numWG_m
        import math as _math
        _log2_mt = int(_math.log2(tile.wg_m))
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
    swz = tile.resolved_swizzle(elem)
    if swz is not None:
        from ..memory.swizzle import DataLayout as SwzLayout
        swz_layout = SwzLayout(row_stride_bytes=int(tile.unroll_k * elem),
                               mfma_k=mfma.k, mfma_m=mfma.m,
                               elem_bytes=elem, wave_size=tile.wave_size)
        from ..memory.swizzle import LDS_GFX950
        swz.emit_write_swizzle(ctx, swz_layout, LDS_GFX950,
                               ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
                               ctx.vreg("v_tmp1"))
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

    # DTL voffset = thread_row * K * elem + col_bytes
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
    if swz is not None:
        from ..memory.swizzle import DataLayout as SwzLayout, LDS_GFX950
        swz_layout = SwzLayout(row_stride_bytes=row_stride_bytes,
                               mfma_k=mfma.k, mfma_m=mfma.m,
                               elem_bytes=elem, wave_size=tile.wave_size)
        ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_lane_id"),
                   int(math.log2(mfma.m)), comment=f"k_group = lane_id / {mfma.m}")

        # Compute row_base for A
        ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(tile.m_per_wave),
                  ctx.vreg("v_wave_m"), comment=f"wave_m * {tile.m_per_wave}")
        ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.vreg("v_lds_rd_a"),
                  ctx.vreg("v_tmp0"), comment="+ lane_row")
        ctx.v_mul(ctx.vreg("v_lds_rd_a"), str(row_stride_bytes),
                  ctx.vreg("v_lds_rd_a"), comment=f"* {row_stride_bytes} -> row_base_a")
        # Emit swizzled read setup for A (ki=0 into v_lds_rd_a, ki>0 into v_lds_rd_a_k{ki})
        ki_count = swz_layout.ki_count
        a_out = [ctx.vreg("v_lds_rd_a")] + [ctx.vreg(f"v_lds_rd_a_k{ki}") for ki in range(1, ki_count)]
        swz.emit_read_setup(ctx, swz_layout, LDS_GFX950,
                            ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
                            ctx.vreg("v_lds_rd_a"), a_out)
        ctx.raw("")

        # Compute row_base for B
        ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), mfma.m - 1,
                  comment=f"lane_row = lane_id % {mfma.m} (re-derive)")
        ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(tile.n_per_wave),
                  ctx.vreg("v_wave_n"), comment=f"wave_n * {tile.n_per_wave}")
        ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.vreg("v_lds_rd_b"),
                  ctx.vreg("v_tmp0"), comment="+ lane_row")
        ctx.v_mul(ctx.vreg("v_lds_rd_b"), str(row_stride_bytes),
                  ctx.vreg("v_lds_rd_b"), comment=f"* {row_stride_bytes} -> row_base_b")
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


def phase_dtl_interleaved_k_loop(level: TileLevel, ctx: AsmContext) -> None:
    """DTL + interleaved K-loop: DTL loads and toggle between MFMAs.

    128 MFMAs split into phases:
      Phase A (mi 0-3): compute X0 half + issue DTL loads between MFMAs
      Phase B (mi 4):   vmcnt for DTL, barrier, toggle read addrs
      Phase C (mi 4-7): compute X1 half + ds_read next iter from new buffer
    """
    tile = _tile(ctx)
    problem = _problem(ctx)
    elem = problem.element_bytes
    mfma = tile.mfma
    layouts = _layouts(ctx)

    mr = tile.mfma_m_repeat   # 8
    nr = tile.mfma_n_repeat   # 8
    ki_count = tile.k_iterations  # 2
    av = mfma.a_vgprs   # 4
    bv = mfma.b_vgprs   # 4

    pad_e = tile.lds_pad // elem if tile.lds_pad > 0 else 0
    # For DTL: per-load-line padding (not per-row)
    if tile.lds_pad > 0:
        threads_per_row_l = int(tile.unroll_k * elem) // 16
        rows_per_load_l = tile.block_size // threads_per_row_l
        num_loads_a_l = tile.wg_m // rows_per_load_l
        num_loads_b_l = tile.wg_n // rows_per_load_l
        lds_a_half = int(tile.wg_m * tile.unroll_k * elem) + num_loads_a_l * tile.lds_pad
        lds_b_half = int(tile.wg_n * tile.unroll_k * elem) + num_loads_b_l * tile.lds_pad
        lds_half = lds_a_half + lds_b_half
    else:
        lds_half = int((tile.wg_m + tile.wg_n) * tile.unroll_k * elem)
    k_stride = int(tile.unroll_k * elem)
    log2_uk = int(math.log2(tile.unroll_k))

    threads_per_row = int(tile.unroll_k * elem) // 16
    rows_per_load = tile.block_size // threads_per_row
    num_loads_a = tile.wg_m // rows_per_load  # 8
    num_loads_b = tile.wg_n // rows_per_load  # 8

    ctx.alloc_sgpr_permanent(1, "s_lds_db_step")

    ctx.comment("=== DTL Interleaved K-loop ===")
    ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
               comment=f"k_tiles = K / {tile.unroll_k}")
    ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_half),
              comment=f"DB step = {lds_half}")
    ctx.raw("")

    # Prologue: DTL load first tile, wait, barrier
    ctx.comment("Prologue: DTL tile 0")
    _emit_dtl_loads_a(ctx, tile, problem, num_loads_a)
    _emit_dtl_loads_b(ctx, tile, problem, num_loads_b)
    ctx.s_waitcnt("vmcnt(0)", comment="wait DTL")
    ctx.s_barrier(comment="sync")
    ctx.raw("")

    # Allocate operand registers
    b_names = {}
    for ni in range(nr):
        for ki in range(ki_count):
            name = f"v_b_s{ni}k{ki}"
            if not ctx.has(name):
                ctx.alloc_vgpr_permanent(bv, name)
            b_names[(ni, ki)] = name

    a_names = {}
    for buf in range(2):
        for ki in range(ki_count):
            name = f"v_a_b{buf}k{ki}"
            if not ctx.has(name):
                ctx.alloc_vgpr_permanent(av, name)
            a_names[(buf, ki)] = name

    # === Main loop ===
    ctx.label("k_loop")
    ctx.raw("")

    # --- K-tile counter and conditional DTL load setup ---
    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="more tiles?")
    ctx.inst("s_cbranch_scc0", "dtl_skip_all",
             comment="skip DTL on last iter")

    # Advance SRDs for next tile
    for srd in ["s_srd_a", "s_srd_b"]:
        ctx.inst("s_add_u32", ctx.sreg(srd, 0, 1),
                 ctx.sreg(srd, 0, 1), str(k_stride), comment=f"{srd} += {k_stride}")
        ctx.inst("s_addc_u32", ctx.sreg(srd, 1, 1),
                 ctx.sreg(srd, 1, 1), "0", comment="carry")

    # Toggle write addresses for DTL
    ctx.inst("s_add_u32", ctx.sreg("s_lds_wr_a_sg"),
             ctx.sreg("s_lds_wr_a_sg"), ctx.sreg("s_lds_db_step"),
             comment="wr_a += db")
    ctx.inst("s_add_u32", ctx.sreg("s_lds_wr_b_sg"),
             ctx.sreg("s_lds_wr_b_sg"), ctx.sreg("s_lds_db_step"),
             comment="wr_b += db")

    # Issue all DTL loads
    _emit_dtl_loads_a(ctx, tile, problem, num_loads_a)
    _emit_dtl_loads_b(ctx, tile, problem, num_loads_b)
    ctx.raw("")

    ctx.label("dtl_skip_all")
    ctx.raw("")

    # --- Preamble: B[ki=0] + A[m0,ki=0], lgkmcnt, then B[ki=1] interleaved ---
    ctx.comment("Preamble: B[ki=0] + A[m0]")
    # Issue ki=0 B reads and A[m0,k0]
    for ni in range(nr):
        ctx.ds_read(ctx.vreg(b_names[(ni, 0)], 0, bv),
                    ctx.vreg("v_lds_rd_b"),
                    offset=_b_off(ni, 0, tile, mfma, elem),
                    width=bv, comment=f"LR B n{ni}k0")

    cur_a = 0
    ctx.ds_read(ctx.vreg(a_names[(cur_a, 0)], 0, av),
                ctx.vreg("v_lds_rd_a"),
                offset=_a_off(0, 0, tile, mfma, elem),
                width=av, comment=f"LR A m0k0 b{cur_a}")

    # Issue ki=1 B reads (will be interleaved with mi=0 ki=0 MFMAs)
    for ni in range(nr):
        ctx.ds_read(ctx.vreg(b_names[(ni, 1)], 0, bv),
                    ctx.vreg("v_lds_rd_b"),
                    offset=_b_off(ni, 1, tile, mfma, elem),
                    width=bv, comment=f"LR B n{ni}k1")

    ctx.ds_read(ctx.vreg(a_names[(cur_a, 1)], 0, av),
                ctx.vreg("v_lds_rd_a"),
                offset=_a_off(0, 1, tile, mfma, elem),
                width=av, comment=f"LR A m0k1 b{cur_a}")

    # Wait for ki=0 reads (we need them for the first 8 MFMAs)
    # There are 18 reads total, ki=0 was issued first (9 reads)
    # lgkmcnt(9) means wait until 9 are left = first 9 are done
    ctx.s_waitcnt("lgkmcnt(9)", comment="wait B[ki=0] + A[m0,k0]")
    ctx.raw("")

    # --- 128 MFMAs with interleaved ops ---
    for mi in range(mr):
        has_pf = mi < mr - 1
        is_last = mi == mr - 1
        if has_pf:
            next_a = 1 - cur_a

        mfma_idx = 0
        for ki in range(ki_count):
            # Wait for ki=1 data before first ki=1 MFMA of mi=0
            if mi == 0 and ki == 1:
                ctx.s_waitcnt("lgkmcnt(0)", comment="wait B[ki=1] + A[m0,k1]")
            for ni in range(nr):
                # A-prefetch: spread ds_reads at slots 2 and 10
                # (earlier = more time for data to arrive before lgkmcnt)
                if has_pf and mfma_idx == 2:
                    ctx.ds_read(ctx.vreg(a_names[(next_a, 0)], 0, av),
                                ctx.vreg("v_lds_rd_a"),
                                offset=_a_off(mi + 1, 0, tile, mfma, elem),
                                width=av, comment=f"LR A m{mi+1}k0 b{next_a}")
                elif has_pf and mfma_idx == 10:
                    ctx.ds_read(ctx.vreg(a_names[(next_a, 1)], 0, av),
                                ctx.vreg("v_lds_rd_a"),
                                offset=_a_off(mi + 1, 1, tile, mfma, elem),
                                width=av, comment=f"LR A m{mi+1}k1 b{next_a}")

                # Suffix ops in last mi group
                if is_last and mfma_idx == 8:
                    ctx.s_waitcnt("vmcnt(0)", comment="wait DTL")
                elif is_last and mfma_idx == 10:
                    ctx.v_add(ctx.vreg("v_lds_rd_a"), ctx.sreg("s_lds_db_step"),
                              ctx.vreg("v_lds_rd_a"), comment="rd_a += db")
                elif is_last and mfma_idx == 11:
                    ctx.v_add(ctx.vreg("v_lds_rd_b"), ctx.sreg("s_lds_db_step"),
                              ctx.vreg("v_lds_rd_b"), comment="rd_b += db")
                elif is_last and mfma_idx == 12:
                    ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"), "0",
                             ctx.sreg("s_lds_db_step"), comment="negate db")

                acc_per = mfma.acc_vgprs
                acc_off = (mi * nr + ni) * acc_per
                if mfma.is_mx:
                    ctx.inst(mfma.instruction_name,
                             ctx.areg("acc_C", acc_off, acc_per),
                             ctx.vreg(a_names[(cur_a, ki)], 0, av),
                             ctx.vreg(b_names[(ni, ki)], 0, bv),
                             ctx.areg("acc_C", acc_off, acc_per),
                             ctx.vreg("v_mxscale"),
                             ctx.vreg("v_mxscale"),
                             f"cbsz:{mfma.cbsz} blgp:{mfma.blgp}",
                             comment=f"MFMA m{mi}_n{ni}_k{ki}")
                else:
                    ctx.inst(mfma.instruction_name,
                             ctx.areg("acc_C", acc_off, acc_per),
                             ctx.vreg(a_names[(cur_a, ki)], 0, av),
                             ctx.vreg(b_names[(ni, ki)], 0, bv),
                             ctx.areg("acc_C", acc_off, acc_per),
                             comment=f"MFMA m{mi}_n{ni}_k{ki}")
                mfma_idx += 1

        if has_pf:
            ctx.s_waitcnt("lgkmcnt(0)", comment=f"wait A[{mi+1}]")
            cur_a = next_a
        ctx.raw("")

    # Suffix: only barrier remains (vmcnt + toggle moved into last mi group)
    ctx.s_barrier(comment="sync")

    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0", comment="more?")
    ctx.inst("s_cbranch_scc1", "k_loop", comment="loop")
    ctx.raw("")


DTL_INTERLEAVED_PROLOGUE_PHASES = [
    TilePhase("dtl_interleaved_setup", phase_dtl_interleaved_setup),
    TilePhase("dtl_interleaved_k_loop", phase_dtl_interleaved_k_loop),
]


WAVE_ABI_PROLOGUE_PHASES = [
    TilePhase("wave_abi_setup", phase_wave_abi_setup),
    TilePhase("dtl_interleaved_k_loop", phase_dtl_interleaved_k_loop),
]
