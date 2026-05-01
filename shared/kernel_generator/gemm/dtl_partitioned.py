"""Partition-based DTL K-loop using MainloopScheduler + SlotPlacer.

The macrotile is divided into partitions. Within each partition, mi values
are processed with ping-pong A buffers. The SlotPlacer decides where to
interleave A-prefetch ds_reads between MFMAs.

Flow:
  1. PartitionPlan.from_tiling() -- derive partition structure
  2. MainloopScheduler.build_modules() -- create MFMA/LR emit closures
  3. SlotPlacer -- interleave LR ops between MFMAs
  4. Emit: preamble (manual) + scheduled body + suffix (manual)
"""
from __future__ import annotations

import math

from .asm_context import AsmContext
from .problem import GemmProblem, TileConfig
from .tile import TilePhase
from .partition_plan import PartitionPlan, Partition
from .mainloop_scheduler import MainloopScheduler, ScheduleModule, ModuleKind
from .slot_placer import SlotPlacer, PlacedOp, Path, SchedulingRules
from .dtl_interleaved import (
    phase_dtl_interleaved_setup,
    _emit_dtl_loads_a, _emit_dtl_loads_b,
    _a_off, _b_off, _swizzle_xor_bytes,
)

__all__ = ["phase_dtl_partitioned_k_loop", "DTL_PARTITIONED_PROLOGUE_PHASES"]


# ---------------------------------------------------------------------------
# MX scale helpers
# ---------------------------------------------------------------------------

def _scale_lds_offset_a(tile):
    """LDS byte offset where ScaleA region starts (within one half-buffer)."""
    elem = tile.mfma.element_bytes
    data_a = int(tile.wg_m * tile.unroll_k * elem)
    data_b = int(tile.wg_n * tile.unroll_k * elem)
    return data_a + data_b


def _scale_lds_offset_b(tile):
    """LDS byte offset where ScaleB region starts (within one half-buffer)."""
    mx_block = tile.mfma.mx_block
    scale_a_bytes = tile.wg_m * (tile.unroll_k // mx_block)
    return _scale_lds_offset_a(tile) + scale_a_bytes


def _scale_region_bytes(tile, dim_size):
    """Bytes for one scale tensor region (e.g. ScaleA or ScaleB) per half."""
    mx_block = tile.mfma.mx_block
    return dim_size * (tile.unroll_k // mx_block)


def phase_mx_scale_setup(level, ctx):
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
    use_real_scales = ctx._metadata.get("use_real_scales", False)
    use_swizzled_scales = (ctx._metadata.get("use_1d_grid", False) or ctx._metadata.get("use_wave_abi", False)) and use_real_scales
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

    use_swizzled_scales = ctx._metadata.get("use_1d_grid", False) or ctx._metadata.get("use_wave_abi", False)

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



def _emit_swizzled_ds_read(ctx, dst, base_reg, offset, ki, tile, mfma, elem,
                           width=4, comment=""):
    """Emit ds_read with LDS swizzle support.
    
    For ki=0, reads directly from base_reg.
    For ki>0 with lds_swizzle, uses precomputed v_lds_rd_a_k1 / v_lds_rd_b_k1
    (set up once per iteration) instead of recomputing XOR each time.
    """
    if getattr(tile, 'lds_swizzle', False) and ki > 0:
        # Use precomputed swizzled base register
        if base_reg == ctx.vreg("v_lds_rd_a"):
            swz_reg = ctx.vreg("v_lds_rd_a_k1")
        elif base_reg == ctx.vreg("v_lds_rd_b"):
            swz_reg = ctx.vreg("v_lds_rd_b_k1")
        else:
            # Fallback: compute inline
            xor_bytes = _swizzle_xor_bytes(tile, mfma, elem)
            ctx.inst("v_xor_b32", ctx.vreg("v_tmp4"),
                     base_reg, str(xor_bytes),
                     comment=f"swizzle ki={ki}: XOR {xor_bytes}")
            swz_reg = ctx.vreg("v_tmp4")
        ctx.ds_read(dst, swz_reg, offset=offset,
                    width=width, comment=comment)
    else:
        ctx.ds_read(dst, base_reg, offset=offset,
                    width=width, comment=comment)

def phase_dtl_partitioned_k_loop(level, ctx):
    """DTL K-loop with partition-based scheduling."""
    tile = ctx._metadata["tile"]
    problem = ctx._metadata["problem"]
    elem = problem.element_bytes
    mfma = tile.mfma

    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    ki_count = tile.k_iterations
    av = mfma.a_vgprs
    bv = mfma.b_vgprs

    # LDS half-buffer size: data only (scales loaded directly to VGPRs)
    lds_data_half = int((tile.wg_m + tile.wg_n) * tile.unroll_k * elem)
    lds_half = lds_data_half
    mx_block = mfma.mx_block
    use_real_scales = ctx._metadata.get("use_real_scales", False)
    use_swizzled_scales = (ctx._metadata.get("use_1d_grid", False) or ctx._metadata.get("use_wave_abi", False)) and use_real_scales

    k_stride = int(tile.unroll_k * elem)
    log2_uk = int(math.log2(tile.unroll_k))
    unroll_k_bytes = int(tile.unroll_k * mfma.element_bytes)
    threads_per_row = unroll_k_bytes // 16
    rows_per_load = tile.block_size // threads_per_row
    num_loads_a = tile.wg_m // rows_per_load
    num_loads_b = tile.wg_n // rows_per_load

    # Scale SRD advance per K-loop
    if use_real_scales:
        if use_swizzled_scales:
            # Swizzled layout: d3 stride = 256 bytes per K-unroll
            scale_k_stride = 256
        else:
            # Linear: unroll_k / mx_block bytes per row
            scale_k_stride = tile.unroll_k // mx_block
    else:
        scale_k_stride = 0

    partition_m = 4
    mfmas_per_mi = nr * ki_count  # 16
    num_subtiles = mr // partition_m

    # ---- Registers ----
    ctx.alloc_sgpr_permanent(1, "s_lds_db_step")

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

    # Scale VGPRs: allocation depends on scale format
    use_swizzled_scales = (ctx._metadata.get("use_1d_grid", False) or ctx._metadata.get("use_wave_abi", False)) and use_real_scales
    scale_a_names = {}
    scale_b_names = {}
    if use_real_scales:
        if use_swizzled_scales:
            # Swizzled: 2 VGPRs for A (group0, group1), 2 for B
            # Each VGPR has 4 bytes: (mi_lo,ki_lo), (mi_hi,ki_lo), (mi_lo,ki_hi), (mi_hi,ki_hi)
            # op_sel selects mi pair, op_sel_hi selects ki pair
            for g in range(2):  # group0 = mi 0,1; group1 = mi 2,3
                name = f"v_scale_a_g{g}"
                if not ctx.has(name):
                    ctx.alloc_vgpr_permanent(1, name)
                for mi_in_g in range(2):
                    mi_ = g * 2 + mi_in_g
                    if mi_ < mr:
                        for ki in range(ki_count):
                            scale_a_names[(mi_, ki)] = name
            for g in range(2):
                name = f"v_scale_b_g{g}"
                if not ctx.has(name):
                    ctx.alloc_vgpr_permanent(1, name)
                for ni_in_g in range(2):
                    ni_ = g * 2 + ni_in_g
                    if ni_ < nr:
                        for ki in range(ki_count):
                            scale_b_names[(ni_, ki)] = name
        else:
            # Linear: one VGPR per mi (A) and per ni (B)
            for mi in range(mr):
                name = f"v_scale_a_m{mi}"
                if not ctx.has(name):
                    ctx.alloc_vgpr_permanent(1, name)
                for ki in range(ki_count):
                    scale_a_names[(mi, ki)] = name
            for ni in range(nr):
                name = f"v_scale_b_n{ni}"
                if not ctx.has(name):
                    ctx.alloc_vgpr_permanent(1, name)
                for ki in range(ki_count):
                    scale_b_names[(ni, ki)] = name

    # ---- K-loop setup ----
    ctx.comment("=== DTL Partitioned K-loop ===")
    ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
               comment=f"k_tiles = K / {tile.unroll_k}")
    ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_half),
              comment=f"DB step = {lds_half}")
    ctx.raw("")

    # Precompute scale soffsets into SGPRs to avoid s_mul in K-loop
    if use_real_scales and not use_swizzled_scales:
        ctx.comment("Precompute scale soffsets")
        for mi_ in range(1, mr):
            soff_name = f"s_soff_sa_{mi_}"
            ctx.alloc_sgpr_permanent(1, soff_name)
            ctx.s_mul(ctx.sreg(soff_name), ctx.sreg("s_stride_scale_a"),
                      str(mi_ * mfma.m),
                      comment=f"soff_a[{mi_}] = stride * {mi_ * mfma.m}")
        for ni_ in range(1, nr):
            soff_name = f"s_soff_sb_{ni_}"
            ctx.alloc_sgpr_permanent(1, soff_name)
            ctx.s_mul(ctx.sreg(soff_name), ctx.sreg("s_stride_scale_b"),
                      str(ni_ * mfma.n),
                      comment=f"soff_b[{ni_}] = stride * {ni_ * mfma.n}")
        ctx.raw("")

    # Prologue: DTL load first tile
    # Precompute DTL soffsets: soff[i] = i * soffset_stride
    ctx.comment("Precompute DTL soffsets")
    for i in range(1, num_loads_a):
        name = f"s_dtl_soff_a{i}"
        ctx.alloc_sgpr_permanent(1, name)
        ctx.s_mul(ctx.sreg(name), ctx.sreg("s_soffset_a"), str(i),
                  comment=f"dtl_soff_a[{i}] = {i} * soffset_a")
    for i in range(1, num_loads_b):
        name = f"s_dtl_soff_b{i}"
        ctx.alloc_sgpr_permanent(1, name)
        ctx.s_mul(ctx.sreg(name), ctx.sreg("s_soffset_b"), str(i),
                  comment=f"dtl_soff_b[{i}] = {i} * soffset_b")
    ctx.raw("")

    ctx.comment("Prologue: DTL tile 0")
    _emit_dtl_loads_a(ctx, tile, problem, num_loads_a)
    _emit_dtl_loads_b(ctx, tile, problem, num_loads_b)
    if use_real_scales:
        if use_swizzled_scales:
            # Swizzled scale loads: 2 groups for A, 2 for B
            ctx.comment("Load swizzled scales A (2 groups)")
            ctx.inst("buffer_load_dword", ctx.vreg("v_scale_a_g0"),
                     ctx.vreg("v_dtl_off_scale_a"),
                     ctx.sreg("s_srd_scale_a", 0, 4),
                     ctx.sreg("s_scale_soff_a0"), "offen",
                     comment="scaleA group0 (mi=0,1)")
            ctx.inst("buffer_load_dword", ctx.vreg("v_scale_a_g1"),
                     ctx.vreg("v_dtl_off_scale_a"),
                     ctx.sreg("s_srd_scale_a", 0, 4),
                     ctx.sreg("s_scale_soff_a1"), "offen",
                     comment="scaleA group1 (mi=2,3)")
            ctx.comment("Load swizzled scales B (2 groups)")
            ctx.inst("buffer_load_dword", ctx.vreg("v_scale_b_g0"),
                     ctx.vreg("v_dtl_off_scale_b"),
                     ctx.sreg("s_srd_scale_b", 0, 4),
                     ctx.sreg("s_scale_soff_b0"), "offen",
                     comment="scaleB group0 (ni=0,1)")
            ctx.inst("buffer_load_dword", ctx.vreg("v_scale_b_g1"),
                     ctx.vreg("v_dtl_off_scale_b"),
                     ctx.sreg("s_srd_scale_b", 0, 4),
                     ctx.sreg("s_scale_soff_b1"), "offen",
                     comment="scaleB group1 (ni=2,3)")
        else:
            # Linear scale loads: subtile 0's scale_a + all scale_b
            ctx.comment("Load scales A subtile 0 (direct VGPR)")
            for mi_ in range(partition_m):  # only subtile 0
                if mi_ == 0:
                    soff = "0"
                else:
                    soff = ctx.sreg(f"s_soff_sa_{mi_}")
                ctx.inst("buffer_load_dword",
                         ctx.vreg(f"v_scale_a_m{mi_}"),
                         ctx.vreg("v_dtl_off_scale_a"),
                         ctx.sreg("s_srd_scale_a", 0, 4),
                         soff, "offen", comment=f"scale A mi={mi_}")
            ctx.comment("Load scales B (direct VGPR)")
            for ni_ in range(nr):
                if ni_ == 0:
                    soff = "0"
                else:
                    soff = ctx.sreg(f"s_soff_sb_{ni_}")
                ctx.inst("buffer_load_dword",
                         ctx.vreg(f"v_scale_b_n{ni_}"),
                         ctx.vreg("v_dtl_off_scale_b"),
                         ctx.sreg("s_srd_scale_b", 0, 4),
                         soff, "offen", comment=f"scale B ni={ni_}")
    if use_real_scales and not use_swizzled_scales:
        # Scale loads (newest vmem ops) can stay in-flight past barrier;
        # only DTL loads (oldest) must complete for LDS coherency.
        num_scale_loads = partition_m + nr  # scale_a subtile0 + all scale_b
        ctx.s_waitcnt(f"vmcnt({num_scale_loads})",
                      comment=f"wait DTL (leave {num_scale_loads} scale loads in-flight)")
    else:
        ctx.s_waitcnt("vmcnt(0)", comment="wait DTL loads")
    ctx.s_barrier(comment="sync")
    ctx.raw("")

    # ================================================================
    # Build MFMA + LR schedule via SlotPlacer
    # ================================================================

    # Helper to compute scale LDS read offset for A or B
    def _scale_rd_off_a(mi, ki):
        """ds_read_u8 offset for A scale at (mi, ki).

        Scale LDS layout: [wg_m, scale_k_cols], row-major, 1 byte/elem.
        v_lds_rd_scale_a already points to (wave_m_row + lane_row) * scale_k_cols + base.
        Offset = mi * mfma.m * scale_k_cols + ki * (mfma.k // mx_block)
        """
        return mi * mfma.m * (tile.unroll_k // mx_block) + ki * (mfma.k // mx_block)

    def _scale_rd_off_b(ni, ki):
        """ds_read_u8 offset for B scale at (ni, ki)."""
        return ni * mfma.n * (tile.unroll_k // mx_block) + ki * (mfma.k // mx_block)

    all_mfma_ops = []
    for mi in range(mr):
        cur_buf = mi % 2
        for ki in range(ki_count):
            for ni in range(nr):
                acc_per = mfma.acc_vgprs
                acc_off = (mi * nr + ni) * acc_per

                def _mk_mfma(mi_=mi, ni_=ni, ki_=ki, buf_=cur_buf,
                              acc_off_=acc_off, acc_per_=acc_per):
                    def emit():
                        if use_real_scales and (mi_, ki_) in scale_a_names:
                            sa_name = scale_a_names[(mi_, ki_)]
                            sb_name = scale_b_names[(ni_, ki_)]
                            if use_swizzled_scales:
                                # op_sel:[mi%2, ni%2] selects M/N pair within group
                                # op_sel_hi:[ki, ki] selects K pair
                                a_sel = mi_ % 2
                                b_sel = ni_ % 2
                                hi_a = ki_
                                hi_b = ki_
                                ctx.inst(
                                    mfma.instruction_name,
                                    ctx.areg("acc_C", acc_off_, acc_per_),
                                    ctx.vreg(a_names[(buf_, ki_)], 0, av),
                                    ctx.vreg(b_names[(ni_, ki_)], 0, bv),
                                    ctx.areg("acc_C", acc_off_, acc_per_),
                                    ctx.vreg(sa_name),
                                    ctx.vreg(sb_name),
                                    f"op_sel:[{a_sel},{b_sel}] op_sel_hi:[{hi_a},{hi_b}]"
                                    f" cbsz:{mfma.cbsz} blgp:{mfma.blgp}",
                                    comment=f"MFMA m{mi_}_n{ni_}_k{ki_}")
                            else:
                                ctx.inst(
                                    mfma.instruction_name,
                                    ctx.areg("acc_C", acc_off_, acc_per_),
                                    ctx.vreg(a_names[(buf_, ki_)], 0, av),
                                    ctx.vreg(b_names[(ni_, ki_)], 0, bv),
                                    ctx.areg("acc_C", acc_off_, acc_per_),
                                    ctx.vreg(sa_name),
                                    ctx.vreg(sb_name),
                                    f"cbsz:{mfma.cbsz} blgp:{mfma.blgp}",
                                    comment=f"MFMA m{mi_}_n{ni_}_k{ki_}")
                        elif mfma.is_mx:
                            # Phase 1 fallback: constant scale
                            ctx.inst(
                                mfma.instruction_name,
                                ctx.areg("acc_C", acc_off_, acc_per_),
                                ctx.vreg(a_names[(buf_, ki_)], 0, av),
                                ctx.vreg(b_names[(ni_, ki_)], 0, bv),
                                ctx.areg("acc_C", acc_off_, acc_per_),
                                ctx.vreg("v_mxscale"),
                                ctx.vreg("v_mxscale"),
                                f"cbsz:{mfma.cbsz} blgp:{mfma.blgp}",
                                comment=f"MFMA m{mi_}_n{ni_}_k{ki_}")
                        else:
                            ctx.inst(
                                mfma.instruction_name,
                                ctx.areg("acc_C", acc_off_, acc_per_),
                                ctx.vreg(a_names[(buf_, ki_)], 0, av),
                                ctx.vreg(b_names[(ni_, ki_)], 0, bv),
                                ctx.areg("acc_C", acc_off_, acc_per_),
                                comment=f"MFMA m{mi_}_n{ni_}_k{ki_}")
                    return emit

                # Track which scale registers this MFMA reads for lifetime analysis
                mfma_reads = ()
                if use_real_scales and (mi, ki) in scale_a_names:
                    mfma_reads = (scale_a_names[(mi, ki)], scale_b_names[(ni, ki)])
                all_mfma_ops.append(PlacedOp(
                    emit_fn=_mk_mfma(), op_type="mfma",
                    comment=f"m{mi}_n{ni}_k{ki}",
                    reads_regs=mfma_reads))

    # LR paths: A-prefetch for mi+1, placed within mi's MFMA range
    lr_paths = []
    for mi in range(mr - 1):
        next_buf = (mi + 1) % 2
        path_ops = []
        for ki in range(ki_count):
            def _mk_lr(mi_=mi + 1, ki_=ki, buf_=next_buf):
                def emit():
                    _emit_swizzled_ds_read(
                        ctx, ctx.vreg(a_names[(buf_, ki_)], 0, av),
                        ctx.vreg("v_lds_rd_a"),
                        offset=_a_off(mi_, ki_, tile, mfma, elem),
                        ki=ki_, tile=tile, mfma=mfma, elem=elem,
                        width=av,
                        comment=f"LR A m{mi_}k{ki_} b{buf_}")
                return emit
            path_ops.append(PlacedOp(
                emit_fn=_mk_lr(), op_type="ds_read",
                comment=f"A m{mi+1}k{ki}"))
        lr_paths.append(Path(ops=path_ops, reverse=False, module_id=mi))

    # Scale A prefetch: load subtile N+1 during subtile N
    scale_prefetch_paths = []
    if use_real_scales and not use_swizzled_scales:
        for st in range(num_subtiles - 1):  # subtile 0,1,2 prefetch for 1,2,3
            path_ops = []
            for mi_in_st in range(partition_m):
                target_mi = (st + 1) * partition_m + mi_in_st
                def _mk_scale_load(mi_=target_mi):
                    def emit():
                        soff = "0" if mi_ == 0 else ctx.sreg(f"s_soff_sa_{mi_}")
                        ctx.inst("buffer_load_dword",
                                 ctx.vreg(f"v_scale_a_m{mi_}"),
                                 ctx.vreg("v_dtl_off_scale_a"),
                                 ctx.sreg("s_srd_scale_a", 0, 4),
                                 soff, "offen", comment=f"scale A mi={mi_} (prefetch)")
                    return emit
                path_ops.append(PlacedOp(
                    emit_fn=_mk_scale_load(), op_type="buffer_load",
                    comment=f"scale_a m{target_mi}"))
            scale_prefetch_paths.append(Path(ops=path_ops, reverse=False, module_id=100 + st))

    # Suffix ops for the last mi group (placed backward)
    # When cross-iteration prefetch is active, the suffix vmcnt must
    # leave the prefetch loads in-flight (they complete by next iter).
    _pf_inflight = (partition_m + nr) if (use_real_scales and not use_swizzled_scales) else 0
    suffix_ops = [
        PlacedOp(emit_fn=lambda n=_pf_inflight: ctx.s_waitcnt(
                 f"vmcnt({n})", comment=f"wait DTL (leave {n} prefetch in-flight)"),
                 op_type="wait", comment="vmcnt"),
        PlacedOp(emit_fn=lambda: (
                 ctx.v_add(ctx.vreg("v_lds_rd_a"),
                 ctx.sreg("s_lds_db_step"), ctx.vreg("v_lds_rd_a"),
                 comment="rd_a += db"),
                 ctx.inst("v_xor_b32", ctx.vreg("v_lds_rd_a_k1"),
                 ctx.vreg("v_lds_rd_a"),
                 str(_swizzle_xor_bytes(tile, mfma, elem)),
                 comment="rd_a_k1 = rd_a ^ swz") if getattr(tile, 'lds_swizzle', False) and ki_count > 1 else None
                 ), op_type="salu", comment="toggle_a"),
        PlacedOp(emit_fn=lambda: (
                 ctx.v_add(ctx.vreg("v_lds_rd_b"),
                 ctx.sreg("s_lds_db_step"), ctx.vreg("v_lds_rd_b"),
                 comment="rd_b += db"),
                 ctx.inst("v_xor_b32", ctx.vreg("v_lds_rd_b_k1"),
                 ctx.vreg("v_lds_rd_b"),
                 str(_swizzle_xor_bytes(tile, mfma, elem)),
                 comment="rd_b_k1 = rd_b ^ swz") if getattr(tile, 'lds_swizzle', False) and ki_count > 1 else None
                 ), op_type="salu", comment="toggle_b"),
    ]
    # Toggle scale LDS read addresses if MX
    if use_real_scales:
        suffix_ops.append(PlacedOp(
            emit_fn=lambda: ctx.v_add(ctx.vreg("v_lds_rd_scale_a"),
                ctx.sreg("s_lds_db_step"), ctx.vreg("v_lds_rd_scale_a"),
                comment="rd_scale_a += db"),
            op_type="salu", comment="toggle_scale_a"))
        suffix_ops.append(PlacedOp(
            emit_fn=lambda: ctx.v_add(ctx.vreg("v_lds_rd_scale_b"),
                ctx.sreg("s_lds_db_step"), ctx.vreg("v_lds_rd_scale_b"),
                comment="rd_scale_b += db"),
            op_type="salu", comment="toggle_scale_b"))
    suffix_ops.append(PlacedOp(
        emit_fn=lambda: ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"),
            "0", ctx.sreg("s_lds_db_step"), comment="negate db"),
        op_type="salu", comment="negate"))
    suffix_path = Path(ops=suffix_ops, reverse=True, module_id=99)

    # SlotPlacer: constrained LR placement + backward suffix
    rules = SchedulingRules(
        total_slots=(len(all_mfma_ops) - 1) * 2,
        min_ds_read_gap=4)

    placer = SlotPlacer(
        mfmas=all_mfma_ops,
        validators=[rules.one_ds_read_per_interval],
        on_place=rules.track_placement)

    # Place each LR path within its mi's MFMA interval range
    for mi, path in enumerate(lr_paths):
        mi_start_slot = mi * mfmas_per_mi * 2
        mi_end_slot = (mi + 1) * mfmas_per_mi * 2
        range_size = mi_end_slot - mi_start_slot
        # Place ki=0 read early, ki=1 read later (both within mi's range).
        # Clamp slot_b so it doesn't exceed the range (fixes 4x4 tile configs).
        slot_a = mi_start_slot + min(4, range_size - 2)
        slot_b = mi_start_slot + min(20, range_size - 2)
        for i, op in enumerate(path.ops):
            target = slot_a if i == 0 else slot_b
            placed = False
            for s in range(target, mi_end_slot):
                if placer._can_place(s, op):
                    placer._slots[s].append(op)
                    if placer._on_place:
                        placer._on_place(placer, s, op)
                    placed = True
                    break
            if not placed:
                placer.leftovers.append(op)

    # Place scale_a prefetch paths within preceding subtile's MFMA range
    if use_real_scales and not use_swizzled_scales:
        mfmas_per_subtile = partition_m * mfmas_per_mi
        for st, path in enumerate(scale_prefetch_paths):
            # Place in subtile st's MFMA interval range
            st_start_slot = st * mfmas_per_subtile * 2
            st_end_slot = (st + 1) * mfmas_per_subtile * 2
            # Place early in the subtile for max latency hiding
            for i, op in enumerate(path.ops):
                target = st_start_slot + 2 + i * 4
                placed = False
                for s in range(target, st_end_slot):
                    if placer._can_place(s, op):
                        placer._slots[s].append(op)
                        if placer._on_place:
                            placer._on_place(placer, s, op)
                        placed = True
                        break
                if not placed:
                    placer.leftovers.append(op)

    # Place suffix backward
    placer.place_path(suffix_path)

    # Place cross-iteration scale prefetch (software pipelining)
    # Loads next K-iteration's scale data during current iteration's
    # last MFMAs, after each register's last consumer.
    if use_real_scales and not use_swizzled_scales:
        from .data_stream import compute_register_last_use, PrefetchOp, \
            build_prefetch_path, place_prefetch_path

        # Collect all scale register names
        scale_reg_names = []
        for mi_ in range(partition_m):
            scale_reg_names.append(f"v_scale_a_m{mi_}")
        for ni_ in range(nr):
            scale_reg_names.append(f"v_scale_b_n{ni_}")

        last_use = compute_register_last_use(all_mfma_ops, scale_reg_names)

        # Build prefetch load ops
        prefetch_loads = []
        for mi_ in range(partition_m):
            reg = f"v_scale_a_m{mi_}"
            def _mk_pf_a(m=mi_):
                def emit():
                    soff = "0" if m == 0 else ctx.sreg(f"s_soff_sa_{m}")
                    ctx.inst("buffer_load_dword",
                             ctx.vreg(f"v_scale_a_m{m}"),
                             ctx.vreg("v_dtl_off_scale_a"),
                             ctx.sreg("s_srd_scale_a", 0, 4),
                             soff, "offen", comment=f"scale A mi={m} (next K)")
                return emit
            prefetch_loads.append(PrefetchOp(
                reg_name=reg, emit_fn=_mk_pf_a(),
                earliest_slot=last_use.get(reg, 0)))

        for ni_ in range(nr):
            reg = f"v_scale_b_n{ni_}"
            def _mk_pf_b(n=ni_):
                def emit():
                    soff = "0" if n == 0 else ctx.sreg(f"s_soff_sb_{n}")
                    ctx.inst("buffer_load_dword",
                             ctx.vreg(f"v_scale_b_n{n}"),
                             ctx.vreg("v_dtl_off_scale_b"),
                             ctx.sreg("s_srd_scale_b", 0, 4),
                             soff, "offen", comment=f"scale B ni={n} (next K)")
                return emit
            prefetch_loads.append(PrefetchOp(
                reg_name=reg, emit_fn=_mk_pf_b(),
                earliest_slot=last_use.get(reg, 0)))

        # SRD advancement function
        def _srd_advance():
            for srd_name in ["s_srd_scale_a", "s_srd_scale_b"]:
                ctx.inst("s_add_u32", ctx.sreg(srd_name, 0, 1),
                         ctx.sreg(srd_name, 0, 1), str(scale_k_stride),
                         comment=f"{srd_name} += {scale_k_stride} (next K)")
                ctx.inst("s_addc_u32", ctx.sreg(srd_name, 1, 1),
                         ctx.sreg(srd_name, 1, 1), "0", comment="carry")

        pf_path = build_prefetch_path(
            all_mfma_ops, prefetch_loads,
            srd_advance_fn=_srd_advance)
        place_prefetch_path(placer, pf_path, all_mfma_ops, prefetch_loads)

    schedule = placer.build()

    # ================================================================
    # Emit K-loop
    # ================================================================
    ctx.label("k_loop")
    ctx.raw("")

    # DTL prefix
    ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
              comment="k_tiles--")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
             comment="more tiles?")
    ctx.inst("s_cbranch_scc0", "dtl_skip_all",
             comment="skip DTL on last iter")

    for srd in ["s_srd_a", "s_srd_b"]:
        ctx.inst("s_add_u32", ctx.sreg(srd, 0, 1),
                 ctx.sreg(srd, 0, 1), str(k_stride),
                 comment=f"{srd} += {k_stride}")
        ctx.inst("s_addc_u32", ctx.sreg(srd, 1, 1),
                 ctx.sreg(srd, 1, 1), "0", comment="carry")

    # Advance scale SRDs
    if use_real_scales and scale_k_stride > 0 and use_swizzled_scales:
        for srd in ["s_srd_scale_a", "s_srd_scale_b"]:
            ctx.inst("s_add_u32", ctx.sreg(srd, 0, 1),
                     ctx.sreg(srd, 0, 1), str(scale_k_stride),
                     comment=f"{srd} += {scale_k_stride}")
            ctx.inst("s_addc_u32", ctx.sreg(srd, 1, 1),
                     ctx.sreg(srd, 1, 1), "0", comment="carry")

    ctx.inst("s_add_u32", ctx.sreg("s_lds_wr_a_sg"),
             ctx.sreg("s_lds_wr_a_sg"), ctx.sreg("s_lds_db_step"),
             comment="wr_a += db")
    ctx.inst("s_add_u32", ctx.sreg("s_lds_wr_b_sg"),
             ctx.sreg("s_lds_wr_b_sg"), ctx.sreg("s_lds_db_step"),
             comment="wr_b += db")

    # Toggle scale LDS write bases


    _emit_dtl_loads_a(ctx, tile, problem, num_loads_a)
    _emit_dtl_loads_b(ctx, tile, problem, num_loads_b)
    if use_real_scales:
        if use_swizzled_scales:
            # Swizzled scale loads: 2 groups for A, 2 for B
            ctx.comment("Load swizzled scales A (2 groups)")
            ctx.inst("buffer_load_dword", ctx.vreg("v_scale_a_g0"),
                     ctx.vreg("v_dtl_off_scale_a"),
                     ctx.sreg("s_srd_scale_a", 0, 4),
                     ctx.sreg("s_scale_soff_a0"), "offen",
                     comment="scaleA group0 (mi=0,1)")
            ctx.inst("buffer_load_dword", ctx.vreg("v_scale_a_g1"),
                     ctx.vreg("v_dtl_off_scale_a"),
                     ctx.sreg("s_srd_scale_a", 0, 4),
                     ctx.sreg("s_scale_soff_a1"), "offen",
                     comment="scaleA group1 (mi=2,3)")
            ctx.comment("Load swizzled scales B (2 groups)")
            ctx.inst("buffer_load_dword", ctx.vreg("v_scale_b_g0"),
                     ctx.vreg("v_dtl_off_scale_b"),
                     ctx.sreg("s_srd_scale_b", 0, 4),
                     ctx.sreg("s_scale_soff_b0"), "offen",
                     comment="scaleB group0 (ni=0,1)")
            ctx.inst("buffer_load_dword", ctx.vreg("v_scale_b_g1"),
                     ctx.vreg("v_dtl_off_scale_b"),
                     ctx.sreg("s_srd_scale_b", 0, 4),
                     ctx.sreg("s_scale_soff_b1"), "offen",
                     comment="scaleB group1 (ni=2,3)")
        else:
            # Linear scale loads: subtile 0's scale_a + all scale_b
            # Scale loads handled by cross-iteration prefetch
            pass
        # No waitcnt here: scale loads overlap with barrier + LDS reads
    ctx.raw("")

    ctx.label("dtl_skip_all")
    # No barrier needed here: DTL writes go to the OTHER buffer,
    # preamble reads come from the CURRENT buffer. The suffix barrier
    # from the previous iteration already synced all waves.
    # On first iteration, the prologue barrier handles the sync.
    ctx.raw("")

    # Preamble: scale reads + B + A[m0]
    ctx.comment("Preamble: scales + B + A[m0]")


    for ni in range(nr):
        ctx.ds_read(ctx.vreg(b_names[(ni, 0)], 0, bv),
                    ctx.vreg("v_lds_rd_b"),
                    offset=_b_off(ni, 0, tile, mfma, elem),
                    width=bv, comment=f"LR B n{ni}k0")
    ctx.ds_read(ctx.vreg(a_names[(0, 0)], 0, av),
                ctx.vreg("v_lds_rd_a"),
                offset=_a_off(0, 0, tile, mfma, elem),
                width=av, comment="LR A m0k0 b0")
    preamble_inflight = nr + 1  # B[ki=0] reads + A[m0,k0]
    if ki_count > 1:
        for ni in range(nr):
            _emit_swizzled_ds_read(
                ctx, ctx.vreg(b_names[(ni, 1)], 0, bv),
                ctx.vreg("v_lds_rd_b"),
                offset=_b_off(ni, 1, tile, mfma, elem),
                ki=1, tile=tile, mfma=mfma, elem=elem,
                width=bv, comment=f"LR B n{ni}k1")
        _emit_swizzled_ds_read(
            ctx, ctx.vreg(a_names[(0, 1)], 0, av),
            ctx.vreg("v_lds_rd_a"),
            offset=_a_off(0, 1, tile, mfma, elem),
            ki=1, tile=tile, mfma=mfma, elem=elem,
            width=av, comment="LR A m0k1 b0")
        preamble_inflight += nr + 1  # B[ki=1] + A[m0,k1]
    # Wait for first batch (B[k0] + A[m0,k0]) to be ready.
    # Remaining reads (ki=1 batch) can stay outstanding.
    first_batch = nr + 1
    remaining = preamble_inflight - first_batch
    wait_cnt = min(remaining, 15)  # lgkmcnt max is 15 on gfx9
    ctx.s_waitcnt(f"lgkmcnt({wait_cnt})", comment="wait B[ki=0] + A[m0,k0]")
    if use_real_scales and not use_swizzled_scales:
        # DTL loads for next iter (newest) can stay in-flight;
        # only scale loads (oldest) must complete for MFMA operands.
        num_dtl = num_loads_a + num_loads_b
        ctx.s_waitcnt(f"vmcnt({num_dtl})",
                      comment=f"wait scales (leave {num_dtl} DTL in-flight)")
    elif use_real_scales:
        ctx.s_waitcnt("vmcnt(0)", comment="wait scale VGPR loads")
    ctx.raw("")

    # Precompute swizzled LDS read bases for ki=1
    if getattr(tile, 'lds_swizzle', False) and ki_count > 1:
        xor_val = _swizzle_xor_bytes(tile, mfma, elem)
        ctx.inst("v_xor_b32", ctx.vreg("v_lds_rd_a_k1"),
                 ctx.vreg("v_lds_rd_a"), str(xor_val),
                 comment=f"precompute rd_a_k1 = rd_a ^ {xor_val}")
        ctx.inst("v_xor_b32", ctx.vreg("v_lds_rd_b_k1"),
                 ctx.vreg("v_lds_rd_b"), str(xor_val),
                 comment=f"precompute rd_b_k1 = rd_b ^ {xor_val}")
    ctx.raw("")

    # ---- Emit scheduled body ----
    inflight_lgkm = preamble_inflight
    mfma_count = 0

    for side_ops, mfma_op in schedule.intervals:
        # Wait for B[ki=1] before mi=0 ki=1
        if mfma_count == nr and inflight_lgkm > 0:
            ctx.s_waitcnt("lgkmcnt(0)", comment="wait B[ki=1] + A[m0,k1]")
            inflight_lgkm = 0

        # Wait for A prefetch at each mi boundary
        if mfma_count > 0 and mfma_count % mfmas_per_mi == 0 and inflight_lgkm > 0:
            ctx.s_waitcnt("lgkmcnt(0)",
                          comment=f"wait A[m{mfma_count // mfmas_per_mi}]")
            inflight_lgkm = 0

        # Wait for prefetched scale_a at subtile boundaries
        if use_real_scales and not use_swizzled_scales:
            mfmas_per_subtile = partition_m * mfmas_per_mi
            if mfma_count > 0 and mfma_count % mfmas_per_subtile == 0:
                st_idx = mfma_count // mfmas_per_subtile
                if st_idx < num_subtiles:
                    # DTL loads (newest) can stay in-flight;
                    # only scale prefetch loads (oldest) need to complete.
                    num_dtl_ = num_loads_a + num_loads_b
                    ctx.s_waitcnt(f"vmcnt({num_dtl_})",
                                  comment=f"wait scale_a subtile {st_idx} (leave DTL)")

        # Partition boundary comments
        if mfma_count % (partition_m * mfmas_per_mi) == 0:
            ctx.comment(f"--- Partition {mfma_count // (partition_m * mfmas_per_mi)} ---")

        for op in side_ops:
            if op.emit_fn:
                op.emit_fn()
            if op.op_type == "ds_read":
                inflight_lgkm += 1

        if mfma_op and mfma_op.emit_fn:
            mfma_op.emit_fn()
            mfma_count += 1

    # Emit leftovers (suffix ops that couldn't be placed)
    for op in schedule.leftovers:
        if op.emit_fn:
            op.emit_fn()

    # Postamble
    ctx.s_barrier(comment="sync")
    ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0", comment="more?")
    if use_real_scales and not use_swizzled_scales and scale_k_stride > 0:
        # On last iter (scc=0), skip to end; otherwise loop back.
        # Scale prefetch loads are in-flight from MFMAs above.
        ctx.inst("s_cbranch_scc0", "k_loop_end", comment="exit if last")
        ctx.inst("s_branch", "k_loop", comment="loop back")
        ctx.label("k_loop_end")
    else:
        ctx.inst("s_cbranch_scc1", "k_loop", comment="loop")
    ctx.raw("")


DTL_PARTITIONED_PROLOGUE_PHASES = [
    TilePhase("dtl_interleaved_setup", phase_dtl_interleaved_setup),
    TilePhase("mx_scale_setup", phase_mx_scale_setup),
    TilePhase("dtl_partitioned_k_loop", phase_dtl_partitioned_k_loop),
]
