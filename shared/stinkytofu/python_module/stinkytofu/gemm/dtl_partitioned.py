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
    _a_off, _b_off,
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


def _emit_dtl_loads_scale(ctx, tile, srd_name, lds_wr_sg_name,
                          dtl_off_name, dim_size):
    """Issue one buffer_load_dword DTL load for a scale tensor.

    Scale tile for one half-buffer:
      dim_size * (unroll_k / mx_block) bytes (e.g. 128*8 = 1024 bytes).

    With 256 threads * 4 bytes/thread = 1024 bytes, exactly one
    buffer_load_dword instruction fills the scale LDS region.
    """
    ctx.inst("s_mov_b32", "m0", ctx.sreg(lds_wr_sg_name),
             comment=f"m0 = LDS base {srd_name}")
    ctx.inst("buffer_load_dword",
             ctx.vreg(dtl_off_name), ctx.sreg(srd_name, 0, 4),
             "0", "offen offset:0, lds",
             comment=f"DTL scale {srd_name}")


def phase_mx_scale_setup(level, ctx):
    """Set up scale SRDs, DTL offsets, and LDS read addresses.

    Runs after phase_dtl_interleaved_setup which already loaded the
    base kernargs (ptrs, M, N, K) and set up data SRDs.
    """
    tile = ctx._metadata["tile"]
    mfma = tile.mfma
    use_real_scales = ctx._metadata.get("use_real_scales", False)
    if not mfma.is_mx or not use_real_scales:
        return

    mx_block = mfma.mx_block
    scale_k_cols = tile.unroll_k // mx_block  # scale columns per unroll (8)

    ctx.comment("=== MX Scale Setup (Phase 2) ===")

    # Load scale kernargs (offsets 36-59)
    karg = ctx.sreg("s_kernarg")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_scale_a"), karg, "36",
             comment="scale A ptr")
    ctx.inst("s_load_dwordx2", ctx.sreg("s_ptr_scale_b"), karg, "44",
             comment="scale B ptr")
    ctx.inst("s_load_dword", ctx.sreg("s_stride_scale_a"), karg, "52",
             comment="scale A stride")
    ctx.inst("s_load_dword", ctx.sreg("s_stride_scale_b"), karg, "56",
             comment="scale B stride")
    ctx.s_waitcnt("lgkmcnt(0)", comment="wait scale kernargs")
    ctx.raw("")

    # Scale DTL per-thread offset
    # Scale tile: [dim_size, scale_k_cols] row-major, 1 byte per element
    # Using buffer_load_dword (4 bytes/thread), 256 threads * 4 = 1024 bytes
    scale_threads_per_row = scale_k_cols // 4  # 8/4 = 2
    log2_stpr = int(math.log2(scale_threads_per_row)) if scale_threads_per_row > 1 else 0
    ctx.comment(f"Scale DTL offset: {scale_threads_per_row} threads/row, "
                f"{scale_k_cols} cols")
    if scale_threads_per_row > 1:
        ctx.v_lshr(ctx.vreg("v_tmp0"), ctx.vreg("v_tid"), log2_stpr,
                   comment="scale_thread_row = tid >> log2(tpr)")
        ctx.v_and(ctx.vreg("v_tmp1"), ctx.vreg("v_tid"),
                  scale_threads_per_row - 1,
                  comment="scale_thread_col_grp")
    else:
        ctx.v_mov(ctx.vreg("v_tmp0"), ctx.vreg("v_tid"),
                  comment="scale_thread_row = tid")
        ctx.v_mov(ctx.vreg("v_tmp1"), "0", comment="scale_thread_col_grp = 0")
    ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_tmp1"), 2,
               comment="* 4 -> col_bytes")

    # Scale A DTL voffset = thread_row * stride_scale_a + col_bytes
    ctx.inst("v_mul_lo_u32", ctx.vreg("v_dtl_off_scale_a"),
             ctx.sreg("s_stride_scale_a"), ctx.vreg("v_tmp0"),
             comment="row * stride_scale_a")
    ctx.v_add(ctx.vreg("v_dtl_off_scale_a"),
              ctx.vreg("v_dtl_off_scale_a"), ctx.vreg("v_tmp1"),
              comment="+ col_bytes")

    # Scale B DTL voffset = thread_row * stride_scale_b + col_bytes
    ctx.inst("v_mul_lo_u32", ctx.vreg("v_dtl_off_scale_b"),
             ctx.sreg("s_stride_scale_b"), ctx.vreg("v_tmp0"),
             comment="row * stride_scale_b")
    ctx.v_add(ctx.vreg("v_dtl_off_scale_b"),
              ctx.vreg("v_dtl_off_scale_b"), ctx.vreg("v_tmp1"),
              comment="+ col_bytes")
    ctx.raw("")

    # Scale SRD A: base = ptr_scale_a + wg_id_x * wg_m * stride_scale_a
    ctx.comment("Scale SRD A")
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

    # Scale SRD B: base = ptr_scale_b + wg_id_y * wg_n * stride_scale_b
    ctx.comment("Scale SRD B")
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

    # Scale LDS write bases (offset into LDS for scale regions)
    scale_a_lds_off = _scale_lds_offset_a(tile)
    scale_b_lds_off = _scale_lds_offset_b(tile)
    ctx.comment("Scale LDS write bases")
    ctx.s_mov(ctx.sreg("s_lds_wr_scale_a_sg"), str(scale_a_lds_off),
              comment=f"scale A LDS base = {scale_a_lds_off}")
    ctx.s_mov(ctx.sreg("s_lds_wr_scale_b_sg"), str(scale_b_lds_off),
              comment=f"scale B LDS base = {scale_b_lds_off}")
    ctx.raw("")

    # Scale LDS read address per lane
    # Scale in LDS: [dim, scale_k_cols] row-major, 1 byte per element
    # Lane L reads row = wave_m + L%16, col = L/16 (for A)
    # v_lds_rd_scale_a = scale_a_lds_off + (wave_m_start + L%16) * scale_k_cols
    ctx.comment("Scale LDS read addresses")
    # lane_row = lane_id % 16
    ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_lane_id"), 15,
              comment="lane_row = lane_id % 16")
    # A: base_row = wave_m * m_per_wave + lane_row
    ctx.v_mul(ctx.vreg("v_lds_rd_scale_a"), str(tile.m_per_wave),
              ctx.vreg("v_wave_m"),
              comment=f"wave_m * {tile.m_per_wave}")
    ctx.v_add(ctx.vreg("v_lds_rd_scale_a"),
              ctx.vreg("v_lds_rd_scale_a"), ctx.vreg("v_tmp0"),
              comment="+ lane_row")
    ctx.v_mul(ctx.vreg("v_lds_rd_scale_a"), str(scale_k_cols),
              ctx.vreg("v_lds_rd_scale_a"),
              comment=f"* {scale_k_cols} (bytes/row)")
    ctx.v_add(ctx.vreg("v_lds_rd_scale_a"),
              str(scale_a_lds_off), ctx.vreg("v_lds_rd_scale_a"),
              comment=f"+ scale_a_lds_base ({scale_a_lds_off})")
    ctx.raw("")

    # B: base_row = wave_n * n_per_wave + lane_row
    ctx.v_mul(ctx.vreg("v_lds_rd_scale_b"), str(tile.n_per_wave),
              ctx.vreg("v_wave_n"),
              comment=f"wave_n * {tile.n_per_wave}")
    ctx.v_add(ctx.vreg("v_lds_rd_scale_b"),
              ctx.vreg("v_lds_rd_scale_b"), ctx.vreg("v_tmp0"),
              comment="+ lane_row")
    ctx.v_mul(ctx.vreg("v_lds_rd_scale_b"), str(scale_k_cols),
              ctx.vreg("v_lds_rd_scale_b"),
              comment=f"* {scale_k_cols} (bytes/row)")
    ctx.v_add(ctx.vreg("v_lds_rd_scale_b"),
              str(scale_b_lds_off), ctx.vreg("v_lds_rd_scale_b"),
              comment=f"+ scale_b_lds_base ({scale_b_lds_off})")
    ctx.raw("")


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

    # LDS half-buffer size: data + scale regions
    lds_data_half = int((tile.wg_m + tile.wg_n) * tile.unroll_k * elem)
    lds_scale_half = 0
    mx_block = mfma.mx_block
    use_real_scales = ctx._metadata.get("use_real_scales", False)
    if use_real_scales:
        scale_k_cols = tile.unroll_k // mx_block
        lds_scale_half = (tile.wg_m + tile.wg_n) * scale_k_cols
    lds_half = lds_data_half + lds_scale_half

    k_stride = int(tile.unroll_k * elem)
    log2_uk = int(math.log2(tile.unroll_k))
    unroll_k_bytes = int(tile.unroll_k * mfma.element_bytes)
    threads_per_row = unroll_k_bytes // 16
    rows_per_load = tile.block_size // threads_per_row
    num_loads_a = tile.wg_m // rows_per_load
    num_loads_b = tile.wg_n // rows_per_load

    # Scale SRD advance per K-loop: unroll_k / mx_block bytes
    scale_k_stride = tile.unroll_k // mx_block if (use_real_scales) else 0

    partition_m = 2
    mfmas_per_mi = nr * ki_count  # 16

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

    # Scale VGPRs: one per (mi, ki) for A and per (ni, ki) for B
    # Each holds 4 bytes = 4 E8M0 K-block scales for one MFMA invocation
    # Loaded via ds_read_u8 (1 byte per lane); op_sel selects byte 0
    scale_a_names = {}
    scale_b_names = {}
    if use_real_scales:
        for mi in range(mr):
            for ki in range(ki_count):
                name = f"v_scale_a_m{mi}k{ki}"
                if not ctx.has(name):
                    ctx.alloc_vgpr_permanent(1, name)
                scale_a_names[(mi, ki)] = name
        for ni in range(nr):
            for ki in range(ki_count):
                name = f"v_scale_b_n{ni}k{ki}"
                if not ctx.has(name):
                    ctx.alloc_vgpr_permanent(1, name)
                scale_b_names[(ni, ki)] = name

    # ---- K-loop setup ----
    ctx.comment("=== DTL Partitioned K-loop ===")
    ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
               comment=f"k_tiles = K / {tile.unroll_k}")
    ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_half),
              comment=f"DB step = {lds_half}")
    ctx.raw("")

    # Prologue: DTL load first tile
    ctx.comment("Prologue: DTL tile 0")
    _emit_dtl_loads_a(ctx, tile, problem, num_loads_a)
    _emit_dtl_loads_b(ctx, tile, problem, num_loads_b)
    if use_real_scales:
        _emit_dtl_loads_scale(ctx, tile, "s_srd_scale_a",
                              "s_lds_wr_scale_a_sg", "v_dtl_off_scale_a",
                              tile.wg_m)
        _emit_dtl_loads_scale(ctx, tile, "s_srd_scale_b",
                              "s_lds_wr_scale_b_sg", "v_dtl_off_scale_b",
                              tile.wg_n)
    ctx.s_waitcnt("vmcnt(0)", comment="wait DTL")
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
                            # Phase 2: real scale VGPRs with op_sel
                            sa_name = scale_a_names[(mi_, ki_)]
                            sb_name = scale_b_names[(ni_, ki_)]
                            # Scale in byte 0 of VGPR; op_sel defaults to 0
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

                all_mfma_ops.append(PlacedOp(
                    emit_fn=_mk_mfma(), op_type="mfma",
                    comment=f"m{mi}_n{ni}_k{ki}"))

    # LR paths: A-prefetch for mi+1, placed within mi's MFMA range
    lr_paths = []
    for mi in range(mr - 1):
        next_buf = (mi + 1) % 2
        path_ops = []
        for ki in range(ki_count):
            def _mk_lr(mi_=mi + 1, ki_=ki, buf_=next_buf):
                def emit():
                    ctx.ds_read(ctx.vreg(a_names[(buf_, ki_)], 0, av),
                                ctx.vreg("v_lds_rd_a"),
                                offset=_a_off(mi_, ki_, tile, mfma, elem),
                                width=av,
                                comment=f"LR A m{mi_}k{ki_} b{buf_}")
                return emit
            path_ops.append(PlacedOp(
                emit_fn=_mk_lr(), op_type="ds_read",
                comment=f"A m{mi+1}k{ki}"))
        lr_paths.append(Path(ops=path_ops, reverse=False, module_id=mi))

    # Suffix ops for the last mi group (placed backward)
    suffix_ops = [
        PlacedOp(emit_fn=lambda: ctx.s_waitcnt("vmcnt(0)", comment="wait DTL"),
                 op_type="wait", comment="vmcnt"),
        PlacedOp(emit_fn=lambda: ctx.v_add(ctx.vreg("v_lds_rd_a"),
                 ctx.sreg("s_lds_db_step"), ctx.vreg("v_lds_rd_a"),
                 comment="rd_a += db"), op_type="salu", comment="toggle_a"),
        PlacedOp(emit_fn=lambda: ctx.v_add(ctx.vreg("v_lds_rd_b"),
                 ctx.sreg("s_lds_db_step"), ctx.vreg("v_lds_rd_b"),
                 comment="rd_b += db"), op_type="salu", comment="toggle_b"),
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
        slot_a = mi_start_slot + 4
        slot_b = mi_start_slot + 20
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

    # Place suffix backward
    placer.place_path(suffix_path)

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
    if use_real_scales and scale_k_stride > 0:
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
    if use_real_scales:
        ctx.inst("s_add_u32", ctx.sreg("s_lds_wr_scale_a_sg"),
                 ctx.sreg("s_lds_wr_scale_a_sg"), ctx.sreg("s_lds_db_step"),
                 comment="wr_scale_a += db")
        ctx.inst("s_add_u32", ctx.sreg("s_lds_wr_scale_b_sg"),
                 ctx.sreg("s_lds_wr_scale_b_sg"), ctx.sreg("s_lds_db_step"),
                 comment="wr_scale_b += db")

    _emit_dtl_loads_a(ctx, tile, problem, num_loads_a)
    _emit_dtl_loads_b(ctx, tile, problem, num_loads_b)
    if use_real_scales:
        _emit_dtl_loads_scale(ctx, tile, "s_srd_scale_a",
                              "s_lds_wr_scale_a_sg", "v_dtl_off_scale_a",
                              tile.wg_m)
        _emit_dtl_loads_scale(ctx, tile, "s_srd_scale_b",
                              "s_lds_wr_scale_b_sg", "v_dtl_off_scale_b",
                              tile.wg_n)
    ctx.raw("")

    ctx.label("dtl_skip_all")
    ctx.raw("")

    # Preamble: scale reads + B + A[m0]
    ctx.comment("Preamble: scales + B + A[m0]")

    # Load all scale VGPRs from LDS (ds_read_u8, 1 byte per lane)
    if use_real_scales:
        ctx.comment("Scale LDS reads (A + B)")
        for mi in range(mr):
            for ki in range(ki_count):
                off = _scale_rd_off_a(mi, ki)
                if off:
                    ctx.inst("ds_read_u8", ctx.vreg(scale_a_names[(mi, ki)]),
                             ctx.vreg("v_lds_rd_scale_a"),
                             f"offset:{off}",
                             comment=f"scale A m{mi}k{ki}")
                else:
                    ctx.inst("ds_read_u8", ctx.vreg(scale_a_names[(mi, ki)]),
                             ctx.vreg("v_lds_rd_scale_a"),
                             comment=f"scale A m{mi}k{ki}")
        for ni in range(nr):
            for ki in range(ki_count):
                off = _scale_rd_off_b(ni, ki)
                if off:
                    ctx.inst("ds_read_u8", ctx.vreg(scale_b_names[(ni, ki)]),
                             ctx.vreg("v_lds_rd_scale_b"),
                             f"offset:{off}",
                             comment=f"scale B n{ni}k{ki}")
                else:
                    ctx.inst("ds_read_u8", ctx.vreg(scale_b_names[(ni, ki)]),
                             ctx.vreg("v_lds_rd_scale_b"),
                             comment=f"scale B n{ni}k{ki}")
        ctx.s_waitcnt("lgkmcnt(0)", comment="wait scale LDS reads")
        ctx.raw("")

    for ni in range(nr):
        ctx.ds_read(ctx.vreg(b_names[(ni, 0)], 0, bv),
                    ctx.vreg("v_lds_rd_b"),
                    offset=_b_off(ni, 0, tile, mfma, elem),
                    width=bv, comment=f"LR B n{ni}k0")
    ctx.ds_read(ctx.vreg(a_names[(0, 0)], 0, av),
                ctx.vreg("v_lds_rd_a"),
                offset=_a_off(0, 0, tile, mfma, elem),
                width=av, comment="LR A m0k0 b0")
    for ni in range(nr):
        ctx.ds_read(ctx.vreg(b_names[(ni, 1)], 0, bv),
                    ctx.vreg("v_lds_rd_b"),
                    offset=_b_off(ni, 1, tile, mfma, elem),
                    width=bv, comment=f"LR B n{ni}k1")
    ctx.ds_read(ctx.vreg(a_names[(0, 1)], 0, av),
                ctx.vreg("v_lds_rd_a"),
                offset=_a_off(0, 1, tile, mfma, elem),
                width=av, comment="LR A m0k1 b0")
    ctx.s_waitcnt("lgkmcnt(9)", comment="wait B[ki=0] + A[m0,k0]")
    ctx.raw("")

    # ---- Emit scheduled body ----
    inflight_lgkm = 9
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
    ctx.inst("s_cbranch_scc1", "k_loop", comment="loop")
    ctx.raw("")


DTL_PARTITIONED_PROLOGUE_PHASES = [
    TilePhase("dtl_interleaved_setup", phase_dtl_interleaved_setup),
    TilePhase("mx_scale_setup", phase_mx_scale_setup),
    TilePhase("dtl_partitioned_k_loop", phase_dtl_partitioned_k_loop),
]
