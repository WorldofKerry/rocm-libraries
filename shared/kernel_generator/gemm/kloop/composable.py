"""Composable K-loop built from modular building blocks.

Users create a K-loop by choosing:
  - GlobalLoader: DTLLoader or BufferLoader
  - Swizzle: RotationSwizzle, XorSwizzle, or IdentitySwizzle
  - ScaleLoader: VMEMScaleLoader or NullScaleLoader
  - SchedulingRules: hand-tunable SlotPlacer parameters

Example usage::

    from kernel_generator.gemm.memory.global_loader import DTLLoader
    from kernel_generator.gemm.memory.lds_reader import LDSReader
    from kernel_generator.gemm.memory.scale_loader import VMEMScaleLoader
    from kernel_generator.gemm.memory.swizzle import RotationSwizzle

    def my_kloop(level, ctx):
        tile = ctx._metadata["tile"]
        problem = ctx._metadata["problem"]

        loader = DTLLoader(ctx, tile, problem)
        reader = LDSReader(ctx, tile, problem, RotationSwizzle())
        scales = VMEMScaleLoader(ctx, tile)

        kloop = ComposableKLoop(ctx, tile, problem, loader, reader, scales)
        kloop.emit()
"""
from __future__ import annotations

import math

from ..problem import TileConfig, GemmProblem
from ..tile.tree import TilePhase
from ..schedule.slot_placer import SlotPlacer, PlacedOp, Path, SchedulingRules
from ..memory.global_loader import GlobalLoader, DTLLoader, BufferLoader
from ..memory.lds_reader import LDSReader

__all__ = ["ComposableKLoop", "composable_kloop_phase"]


class ComposableKLoop:
    """K-loop assembled from pluggable building blocks.

    Structure:
        prologue: first tile load + barrier
        loop:
            early B[ki=0] reads (overlap with loads)
            conditional next-tile load (advance + toggle + loads)
            sync
            preamble A reads
            scheduled MFMA body (with interleaved LR + suffix ops)
            barrier + branch
    """

    def __init__(self, ctx, tile, problem, loader, reader,
                 scale_loader=None, partition_m=4,
                 scheduling_rules=None):
        self.ctx = ctx
        self.tile = tile
        self.problem = problem
        self.loader = loader
        self.reader = reader
        self.scale_loader = scale_loader
        self.partition_m = partition_m
        self.scheduling_rules = scheduling_rules

        self.elem = problem.element_bytes
        self.mfma = tile.mfma
        self.mr = tile.mfma_m_repeat
        self.nr = tile.mfma_n_repeat
        self.ki_count = tile.k_iterations
        self.mfmas_per_mi = self.nr * self.ki_count

        lds_data_half = int((tile.wg_m + tile.wg_n) * tile.unroll_k * self.elem)
        self.lds_half = lds_data_half

    def emit(self):
        """Emit the complete K-loop."""
        ctx = self.ctx
        tile = self.tile
        log2_uk = int(math.log2(tile.unroll_k))

        # DB step register
        ctx.alloc_sgpr_permanent(1, "s_lds_db_step")

        ctx.comment("=== Composable K-loop ===")
        ctx.s_lshr(ctx.sreg("s_k_tiles"), ctx.sreg("s_K"), log2_uk,
                   comment=f"k_tiles = K / {tile.unroll_k}")
        ctx.s_mov(ctx.sreg("s_lds_db_step"), str(self.lds_half),
                  comment=f"DB step = {self.lds_half}")
        ctx.raw("")

        # Precompute offsets
        self.loader.precompute_soffsets()
        if self.scale_loader:
            self.scale_loader.precompute_soffsets()

        # Prologue: DTL first, then scales.
        # DTL loads must be issued BEFORE scale loads so that vmcnt(N)
        # drains DTL while leaving N scale loads in-flight.
        ctx.comment("Prologue: load tile 0")
        self.loader.emit_loads()
        if self.scale_loader:
            self.scale_loader.emit_initial_loads(self.partition_m)
            extra_vmcnt = self.scale_loader.num_initial_inflight(
                self.partition_m)
            ctx.s_waitcnt(f"vmcnt({extra_vmcnt})",
                          comment=f"wait DTL (leave {extra_vmcnt} scale loads)")
        else:
            ctx.s_waitcnt("vmcnt(0)", comment="wait DTL loads")
        ctx.s_barrier(comment="sync first tile")
        ctx.raw("")

        # Build schedule before entering loop
        all_mfma_ops = self._build_mfma_ops()
        lr_paths = self._build_lr_paths()
        suffix_path = self._build_suffix_path()
        schedule = self._run_slot_placer(all_mfma_ops, lr_paths, suffix_path)

        # ============== K-loop body ==============
        ctx.label("k_loop")
        ctx.raw("")

        # Early B[ki=0] reads (overlap with DTL / global loads)
        ctx.comment("Early B reads (overlap with loads)")
        for ni in range(self.nr):
            self.reader.emit_read_b(ni, 0)

        # Conditional load of next tile
        ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
                  comment="k_tiles--")
        ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
                 comment="more tiles?")
        ctx.inst("s_cbranch_scc0", "load_skip_all",
                 comment="skip loads on last iter")

        self.loader.advance()
        self.loader.toggle_write()
        self.loader.emit_loads()

        if self.scale_loader:
            # Swizzled mode: advance + reload every iteration.
            # Linear mode: advance is handled by cross-iter prefetch path.
            if not self.scale_loader.has_cross_iter_prefetch:
                self.scale_loader.advance()
            self.scale_loader.emit_loop_loads()

        ctx.raw("")
        ctx.label("load_skip_all")

        self.loader.emit_sync()
        ctx.raw("")

        # Preamble: A[m0,k0], B[ki=1], A[m0,k1]
        ctx.comment("Preamble: A[m0] + B ki=1")
        self.reader.emit_read_a(mi=0, ki=0, buf=0)
        preamble_inflight = self.nr + 1  # B[ki=0] already issued + A[m0,k0]
        if self.ki_count > 1:
            for ni in range(self.nr):
                self.reader.emit_read_b(ni, 1)
            self.reader.emit_read_a(mi=0, ki=1, buf=0)
            preamble_inflight += self.nr + 1

        first_batch = self.nr + 1
        remaining = preamble_inflight - first_batch
        wait_cnt = min(remaining, 15)
        ctx.s_waitcnt(f"lgkmcnt({wait_cnt})",
                      comment="wait B[ki=0] + A[m0,k0]")

        if self.scale_loader:
            self.scale_loader.emit_scale_wait(self.loader)
        ctx.raw("")

        # Recompute per-ki LDS read bases
        self.reader.emit_recompute_ki_bases()
        ctx.raw("")

        # Emit scheduled MFMA body
        self._emit_scheduled_body(schedule, preamble_inflight)

        # Postamble
        ctx.s_barrier(comment="sync")
        has_pf = (self.scale_loader
                  and self.scale_loader.has_cross_iter_prefetch)
        ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
                 comment="more?")
        if has_pf:
            ctx.inst("s_cbranch_scc0", "k_loop_end",
                     comment="exit if last")
            ctx.inst("s_branch", "k_loop", comment="loop back")
            ctx.label("k_loop_end")
        else:
            ctx.inst("s_cbranch_scc1", "k_loop", comment="loop")
        ctx.raw("")

    # ------------------------------------------------------------------
    # Schedule construction
    # ------------------------------------------------------------------

    def _build_mfma_ops(self):
        """Build PlacedOps for all MFMAs."""
        ctx = self.ctx
        mfma = self.mfma
        reader = self.reader
        sl = self.scale_loader

        ops = []
        for mi in range(self.mr):
            cur_buf = mi % 2
            for ki in range(self.ki_count):
                for ni in range(self.nr):
                    acc_per = mfma.acc_vgprs
                    acc_off = (mi * self.nr + ni) * acc_per

                    def _mk(mi_=mi, ni_=ni, ki_=ki, buf_=cur_buf,
                            aoff=acc_off, aper=acc_per):
                        def emit():
                            a_reg = ctx.vreg(
                                reader.a_names[(buf_, ki_)], 0, reader.av)
                            b_reg = ctx.vreg(
                                reader.b_names[(ni_, ki_)], 0, reader.bv)
                            acc = ctx.areg("acc_C", aoff, aper)

                            if sl and sl.has_scales:
                                sl.emit_mfma(ctx, mfma, acc, a_reg,
                                             b_reg, mi_, ni_, ki_)
                            elif mfma.is_mx:
                                ctx.inst(
                                    mfma.instruction_name, acc,
                                    a_reg, b_reg, acc,
                                    ctx.vreg("v_mxscale"),
                                    ctx.vreg("v_mxscale"),
                                    f"cbsz:{mfma.cbsz} blgp:{mfma.blgp}",
                                    comment=f"MFMA m{mi_}_n{ni_}_k{ki_}")
                            else:
                                ctx.inst(
                                    mfma.instruction_name, acc,
                                    a_reg, b_reg, acc,
                                    comment=f"MFMA m{mi_}_n{ni_}_k{ki_}")
                        return emit

                    reads = ()
                    if sl and sl.has_scales:
                        sa = sl.scale_names_a.get((mi, ki))
                        sb = sl.scale_names_b.get((ni, ki))
                        if sa and sb:
                            reads = (sa, sb)

                    ops.append(PlacedOp(
                        emit_fn=_mk(), op_type="mfma",
                        comment=f"m{mi}_n{ni}_k{ki}",
                        reads_regs=reads))
        return ops

    def _build_lr_paths(self):
        """Build LR (ds_read) paths for A-prefetch."""
        paths = []
        for mi in range(self.mr - 1):
            ops = self.reader.make_lr_ops(mi)
            if ops:
                paths.append((mi, Path(ops=ops, reverse=False,
                                       module_id=mi)))
        return paths

    def _build_suffix_path(self):
        """Build suffix ops: vmcnt wait + toggle + negate."""
        ctx = self.ctx
        pf_inflight = 0
        if self.scale_loader and self.scale_loader.has_cross_iter_prefetch:
            pf_inflight = self.scale_loader.cross_iter_inflight(
                self.partition_m, self.nr)

        ops = [
            PlacedOp(
                emit_fn=lambda n=pf_inflight: ctx.s_waitcnt(
                    f"vmcnt({n})",
                    comment=f"wait loads (leave {n} prefetch in-flight)"),
                op_type="wait", comment="vmcnt"),
        ]
        ops.extend(self.reader.make_suffix_ops())
        return Path(ops=ops, reverse=True, module_id=99)

    def _run_slot_placer(self, all_mfma_ops, lr_paths, suffix_path):
        """Schedule ops between MFMAs via SlotPlacer."""
        rules = self.scheduling_rules or SchedulingRules(
            total_slots=(len(all_mfma_ops) - 1) * 2,
            min_ds_read_gap=4)

        placer = SlotPlacer(
            mfmas=all_mfma_ops,
            validators=[rules.one_ds_read_per_interval],
            on_place=rules.track_placement)

        # Place LR paths within each mi's MFMA range
        for mi, path in lr_paths:
            mi_start = mi * self.mfmas_per_mi * 2
            mi_end = (mi + 1) * self.mfmas_per_mi * 2
            rng = mi_end - mi_start
            slot_a = mi_start + min(4, rng - 2)
            slot_b = mi_start + min(20, rng - 2)
            for i, op in enumerate(path.ops):
                target = slot_a if i == 0 else slot_b
                placed = False
                for s in range(target, mi_end):
                    if placer._can_place(s, op):
                        placer._slots[s].append(op)
                        if placer._on_place:
                            placer._on_place(placer, s, op)
                        placed = True
                        break
                if not placed:
                    placer.leftovers.append(op)

        # Scale subtile prefetch
        if self.scale_loader:
            num_subtiles = self.mr // self.partition_m
            scale_paths = self.scale_loader.make_subtile_prefetch_paths(
                self.partition_m, num_subtiles)
            mfmas_per_st = self.partition_m * self.mfmas_per_mi
            for st, path in enumerate(scale_paths):
                st_start = st * mfmas_per_st * 2
                st_end = (st + 1) * mfmas_per_st * 2
                for i, op in enumerate(path.ops):
                    target = st_start + 2 + i * 4
                    placed = False
                    for s in range(target, st_end):
                        if placer._can_place(s, op):
                            placer._slots[s].append(op)
                            if placer._on_place:
                                placer._on_place(placer, s, op)
                            placed = True
                            break
                    if not placed:
                        placer.leftovers.append(op)

        # Suffix backward
        placer.place_path(suffix_path)

        # Cross-iteration scale prefetch
        if self.scale_loader and self.scale_loader.has_cross_iter_prefetch:
            self.scale_loader.place_cross_iter_prefetch(
                placer, all_mfma_ops, self.partition_m, self.nr)

        return placer.build()

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def _emit_scheduled_body(self, schedule, preamble_inflight):
        """Walk the schedule and emit MFMA + side ops with waitcnts."""
        ctx = self.ctx
        inflight_lgkm = preamble_inflight
        mfma_count = 0
        mfmas_per_mi = self.mfmas_per_mi
        partition_m = self.partition_m

        for side_ops, mfma_op in schedule.intervals:
            # Wait for B[ki=1] before mi=0 ki=1
            if mfma_count == self.nr and inflight_lgkm > 0:
                ctx.s_waitcnt("lgkmcnt(0)",
                              comment="wait B[ki=1] + A[m0,k1]")
                inflight_lgkm = 0

            # Wait for A prefetch at each mi boundary
            if (mfma_count > 0 and mfma_count % mfmas_per_mi == 0
                    and inflight_lgkm > 0):
                ctx.s_waitcnt("lgkmcnt(0)",
                              comment=f"wait A[m{mfma_count // mfmas_per_mi}]")
                inflight_lgkm = 0

            # Scale subtile boundary wait
            if self.scale_loader:
                mps = partition_m * mfmas_per_mi
                n_st = self.mr // partition_m
                if mfma_count > 0 and mfma_count % mps == 0:
                    st_idx = mfma_count // mps
                    if st_idx < n_st:
                        self.scale_loader.emit_subtile_wait(
                            self.loader, st_idx)

            # Partition comment
            if mfma_count % (partition_m * mfmas_per_mi) == 0:
                ctx.comment(
                    f"--- Partition "
                    f"{mfma_count // (partition_m * mfmas_per_mi)} ---")

            for op in side_ops:
                if op.emit_fn:
                    op.emit_fn()
                if op.op_type == "ds_read":
                    inflight_lgkm += 1

            if mfma_op and mfma_op.emit_fn:
                mfma_op.emit_fn()
                mfma_count += 1

        for op in schedule.leftovers:
            if op.emit_fn:
                op.emit_fn()


def composable_kloop_phase(level, ctx):
    """Phase function: auto-configured composable K-loop.

    Reads config from ctx._metadata:
        loader_cls: DTLLoader or BufferLoader (default: from use_dtl)
        swizzle: Swizzle instance (default: tile.resolved_swizzle)
        partition_m: int (default: 4)
    """
    tile = ctx._metadata["tile"]
    problem = ctx._metadata["problem"]
    use_dtl = ctx._metadata.get("use_dtl", True)

    loader_cls = ctx._metadata.get("loader_cls",
                                   DTLLoader if use_dtl else BufferLoader)
    loader = loader_cls(ctx, tile, problem)

    swizzle = ctx._metadata.get("swizzle", None)
    reader = LDSReader(ctx, tile, problem, swizzle=swizzle)

    scale_loader = None
    use_real_scales = ctx._metadata.get("use_real_scales", False)
    if use_real_scales and tile.mfma.is_mx:
        from ..memory.scale_loader import VMEMScaleLoader
        swizzled = ctx._metadata.get("swizzled_scales", False)
        scale_loader = VMEMScaleLoader(ctx, tile, swizzled=swizzled)

    partition_m = ctx._metadata.get("partition_m", 4)
    kloop = ComposableKLoop(ctx, tile, problem, loader, reader,
                            scale_loader, partition_m=partition_m)
    kloop.emit()
