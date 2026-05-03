"""K-loop pipeline drivers: multiple approaches for comparison.

Each driver takes the same inputs (loader, reader, scheduler output,
pgr, num_buffers) and emits the K-loop via AsmContext. The assembly
output can be diffed to compare approaches.

Usage:
    driver = BranchDriver(pgr=1, num_buffers=2)
    driver.emit_kloop(ctx, schedule, loader, reader, scale_loader)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..emit.context import AsmContext
from ..memory.global_loader import GlobalLoader
from ..memory.lds_reader import LDSReader
from .kloop_graph import OpKind
from .kloop_scheduler import ScheduledKLoop


class KLoopDriver(ABC):
    """Base class for K-loop pipeline drivers."""

    def __init__(self, pgr: int = 1, num_buffers: int = 2) -> None:
        if pgr > num_buffers:
            raise ValueError(
                f"PGR={pgr} > num_buffers={num_buffers}")
        self.pgr = pgr
        self.num_buffers = num_buffers

    @property
    def loads_before_reads(self) -> bool:
        return self.pgr < self.num_buffers

    @abstractmethod
    def emit_kloop(self, ctx: AsmContext, schedule: ScheduledKLoop,
                   loader: GlobalLoader, reader: LDSReader,
                   scale_loader: object = None) -> None:
        """Emit the complete K-loop (ramp-up + body + drain)."""

    # -- shared helpers --

    def _emit_ramp_up(self, ctx: AsmContext, loader: GlobalLoader,
                      scale_loader: object, schedule: ScheduledKLoop) -> None:
        """Emit PGR ramp-up stages (global loads to fill pipeline)."""
        for p in range(self.pgr):
            ctx.comment(f"Ramp-up stage {p}/{self.pgr}: G(tile={p})")
            loader.emit_loads()
            if p == 0:
                # First tile: wait + barrier (data needed immediately)
                if schedule and schedule.prologue_scale_ops:
                    for op in schedule.prologue_scale_ops:
                        if op.emit:
                            op.emit()
                    extra = len(schedule.prologue_scale_ops)
                    ctx.s_waitcnt(f"vmcnt({extra})",
                                  comment=f"wait DTL (leave {extra} scales)")
                else:
                    ctx.s_waitcnt("vmcnt(0)", comment="wait DTL loads")
                ctx.s_barrier(comment="sync tile 0")
            else:
                # Subsequent tiles: fire-and-forget prefetch
                loader.advance()
                loader.toggle_write()
                loader.emit_loads()
                if (schedule and schedule.scale_advance_op
                        and schedule.scale_advance_op.emit):
                    schedule.scale_advance_op.emit()
            ctx.raw("")

    def _emit_global_load_ops(self, ctx: AsmContext,
                              schedule: ScheduledKLoop,
                              loader: GlobalLoader,
                              scale_loader: object) -> None:
        """Emit G stage: advance + toggle + load."""
        for op in schedule.prefetch_ops:
            if op.emit:
                op.emit()
        if schedule.scale_advance_op and schedule.scale_advance_op.emit:
            schedule.scale_advance_op.emit()
        if (scale_loader and not scale_loader.has_cross_iter_prefetch
                and schedule.prologue_scale_ops):
            for op in schedule.prologue_scale_ops:
                if op.emit:
                    op.emit()

    def _emit_read_compute(self, ctx: AsmContext,
                           schedule: ScheduledKLoop,
                           reader: LDSReader,
                           loader: GlobalLoader,
                           scale_loader: object) -> None:
        """Emit R+M stages (from KLoopScheduler output)."""
        tile = ctx._metadata["tile"]
        nr = tile.mfma_n_repeat
        mr = tile.mfma_m_repeat
        ki_count = tile.k_iterations
        mfmas_per_mi = nr * ki_count
        partition_m = 4

        # Preamble reads
        ctx.comment("Preamble: A[m0] + B ki=1")
        for op in schedule.preamble_ops:
            if op.emit:
                op.emit()

        preamble_inflight = nr + 1
        if ki_count > 1:
            preamble_inflight += nr + 1
        first_batch = nr + 1
        remaining = preamble_inflight - first_batch
        wait_cnt = min(remaining, 15)
        ctx.s_waitcnt(f"lgkmcnt({wait_cnt})",
                      comment="wait B[ki=0] + A[m0,k0]")

        if scale_loader and hasattr(scale_loader, 'emit_scale_wait'):
            scale_loader.emit_scale_wait(loader)
        ctx.raw("")

        reader.emit_recompute_ki_bases()
        ctx.raw("")

        # MFMA body with interleaved side ops
        inflight_lgkm = preamble_inflight
        mfma_count = 0

        for i, mfma_op in enumerate(schedule.mfma_order):
            if i in schedule.waits:
                ctx.s_waitcnt(schedule.waits[i],
                              comment=f"auto-wait before MFMA[{i}]")
                inflight_lgkm = 0

            if mfma_count == nr and inflight_lgkm > 0:
                ctx.s_waitcnt("lgkmcnt(0)",
                              comment="wait B[ki=1] + A[m0,k1]")
                inflight_lgkm = 0

            if (mfma_count > 0 and mfma_count % mfmas_per_mi == 0
                    and inflight_lgkm > 0):
                ctx.s_waitcnt("lgkmcnt(0)",
                              comment=f"wait A[m{mfma_count // mfmas_per_mi}]")
                inflight_lgkm = 0

            if scale_loader:
                mps = partition_m * mfmas_per_mi
                n_st = mr // partition_m
                if mfma_count > 0 and mfma_count % mps == 0:
                    st_idx = mfma_count // mps
                    if st_idx < n_st:
                        num_dtl = (loader.num_inflight
                                   if hasattr(loader, "num_inflight") else 0)
                        ctx.s_waitcnt(
                            f"vmcnt({num_dtl})",
                            comment=f"wait scale_a subtile {st_idx}")

            if mfma_count % (partition_m * mfmas_per_mi) == 0:
                ctx.comment(
                    f"--- Partition "
                    f"{mfma_count // (partition_m * mfmas_per_mi)} ---")

            for op in schedule.side_ops[i]:
                if op.emit:
                    op.emit()
                if op.kind == OpKind.DS_READ:
                    inflight_lgkm += 1

            if mfma_op.emit:
                mfma_op.emit()
            mfma_count += 1

        for op in schedule.epilogue_ops:
            if op.emit:
                op.emit()


# ======================================================================
# Approach 1: Branch-based
# ======================================================================

class BranchDriver(KLoopDriver):
    """Conditional branch skips G stage during drain iterations.

    The loop body is identical for all iterations. A branch gates
    the G stage based on k_tiles > PGR-1.
    """

    def emit_kloop(self, ctx, schedule, loader, reader,
                   scale_loader=None):
        pgr = self.pgr
        ctx.comment(f"=== Branch-based K-loop (PGR={pgr}) ===")

        self._emit_ramp_up(ctx, loader, scale_loader, schedule)

        ctx.label("k_loop")
        ctx.raw("")

        if self.loads_before_reads:
            # G before R: load into free buffer
            ctx.comment(f"Early B reads (PGR={pgr})")
            for op in schedule.pre_body_ops:
                if op.emit:
                    op.emit()

            ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
                      comment="k_tiles--")
            ctx.inst("s_cmp_gt_u32", ctx.sreg("s_k_tiles"),
                     str(pgr - 1),
                     comment=f"k_tiles > {pgr - 1}?")
            ctx.inst("s_cbranch_scc0", "skip_G",
                     comment="skip G (drain)")
            self._emit_global_load_ops(ctx, schedule, loader, scale_loader)
            ctx.raw("")
            ctx.label("skip_G")
            loader.emit_sync()
            ctx.raw("")
        else:
            # G after R: read-before-write
            loader.emit_sync()
            ctx.raw("")
            ctx.comment(f"Early B reads (PGR={pgr})")
            for op in schedule.pre_body_ops:
                if op.emit:
                    op.emit()

        # R + M (identical for all PGR values)
        self._emit_read_compute(ctx, schedule, reader, loader, scale_loader)

        if not self.loads_before_reads:
            ctx.s_waitcnt("lgkmcnt(0)",
                          comment="wait ds_reads before overwriting LDS")
            ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
                      comment="k_tiles--")
            ctx.inst("s_cmp_gt_u32", ctx.sreg("s_k_tiles"),
                     str(pgr - 1),
                     comment=f"k_tiles > {pgr - 1}?")
            ctx.inst("s_cbranch_scc0", "skip_G",
                     comment="skip G (drain)")
            self._emit_global_load_ops(ctx, schedule, loader, scale_loader)
            ctx.raw("")
            ctx.label("skip_G")

        # Suffix + loop branch
        ctx.s_barrier(comment="sync")
        ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
                 comment="more?")
        ctx.inst("s_cbranch_scc1", "k_loop", comment="loop")
        ctx.raw("")


# ======================================================================
# Approach 2: Predicate-based (MLIR-style)
# ======================================================================

class PredicateDriver(KLoopDriver):
    """Every iteration runs the same body; stages are predicated.

    The loop runs T + PGR - 1 iterations total. Stage S is active
    when its tile index (iter - stage_num[S]) is in [0, T).

    No separate ramp-up code -- it's all in the loop.
    """

    def emit_kloop(self, ctx, schedule, loader, reader,
                   scale_loader=None):
        pgr = self.pgr
        ctx.comment(f"=== Predicate-based K-loop (PGR={pgr}) ===")

        # Iteration counter: 0 to T + PGR - 1
        ctx.alloc_sgpr_permanent(1, "s_iter")
        ctx.s_mov(ctx.sreg("s_iter"), "0", comment="iteration counter")
        # Total iterations = T + PGR (T stored in s_k_tiles)
        ctx.inst("s_add_u32", ctx.sreg("s_k_tiles"),
                 ctx.sreg("s_k_tiles"), str(pgr),
                 comment=f"total_iters = k_tiles + {pgr}")
        ctx.raw("")

        ctx.label("k_loop")
        ctx.raw("")

        # --- G stage: active when s_iter < T ---
        # (T = s_k_tiles - pgr, but we can check s_iter < s_k_tiles - pgr)
        ctx.inst("s_add_u32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_iter"), str(pgr),
                 comment=f"s_iter + {pgr}")
        ctx.inst("s_cmp_lt_u32", ctx.sreg("s_tmp0"),
                 ctx.sreg("s_k_tiles"),
                 comment="G active?")
        ctx.inst("s_cbranch_scc0", "pred_skip_G",
                 comment="G predicated off")
        if pgr > 0:
            # First iteration: G(0) needs wait + barrier
            ctx.inst("s_cmp_eq_u32", ctx.sreg("s_iter"), "0",
                     comment="first iter?")
            ctx.inst("s_cbranch_scc0", "pred_not_first_G",
                     comment="not first G")
            loader.emit_loads()
            ctx.s_waitcnt("vmcnt(0)", comment="wait first G")
            ctx.s_barrier(comment="sync first tile")
            ctx.inst("s_branch", "pred_skip_G", comment="done with first G")
            ctx.label("pred_not_first_G")
        self._emit_global_load_ops(ctx, schedule, loader, scale_loader)
        ctx.raw("")
        ctx.label("pred_skip_G")
        ctx.raw("")

        # --- R+M stage: active when s_iter >= PGR ---
        ctx.inst("s_cmp_ge_u32", ctx.sreg("s_iter"), str(pgr),
                 comment=f"R+M active? (iter >= {pgr})")
        ctx.inst("s_cbranch_scc0", "pred_skip_RM",
                 comment="R+M predicated off")
        loader.emit_sync()
        ctx.comment("Early B reads")
        for op in schedule.pre_body_ops:
            if op.emit:
                op.emit()
        self._emit_read_compute(ctx, schedule, reader, loader, scale_loader)
        ctx.label("pred_skip_RM")
        ctx.raw("")

        # Suffix + loop
        ctx.s_barrier(comment="sync")
        ctx.inst("s_add_u32", ctx.sreg("s_iter"),
                 ctx.sreg("s_iter"), "1", comment="iter++")
        ctx.inst("s_cmp_lt_u32", ctx.sreg("s_iter"),
                 ctx.sreg("s_k_tiles"), comment="more?")
        ctx.inst("s_cbranch_scc1", "k_loop", comment="loop")
        ctx.raw("")
