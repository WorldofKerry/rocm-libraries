"""Pipeline emitter: assembly code generation from a ScheduledPipeline.

Takes a ``ScheduledPipeline`` (from ``pipeline_scheduler.py``) and emits
the complete K-loop assembly: ramp-up prologue, steady-state body loop,
and drain epilogue.

Key design properties:
  - No ``isinstance`` checks -- works purely from the scheduled op list.
  - Waitcnts come from the scheduler, not manually computed here.
  - Produce-first vs consume-first structure is implicit in the body
    op ordering (determined by the scheduler).
  - Ops with ``emit=None`` are skipped with a comment (placeholder for
    Phase 4b wiring of actual codegen callbacks).
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from .kloop_graph import KLoopOp, OpKind
from .pipeline_scheduler import ScheduledPipeline
from .interleave import (
    classify_body_ops, build_dtl_sequence,
    emit_mfmas_with_dtl_interleaved, emit_dtl_load,
    emit_mfmas_with_reads_interleaved,
    emit_mfmas_with_reads_and_dtl,
)

if TYPE_CHECKING:
    from ..emit.context import AsmContext
    from ..memory.lds_stream import LDSBufferManager

__all__ = ["PipelineEmitter"]


class PipelineEmitter:
    """Emits assembly for a K-loop from a ``ScheduledPipeline``.

    The emitter is a thin code-generation layer.  All scheduling
    decisions (op ordering, waitcnt values, produce-first vs
    consume-first) are determined by the ``ScheduledPipeline``.

    Args:
        pipeline: Scheduled pipeline from ``PipelineScheduler``.
        buffer_mgr: LDS buffer manager (for barrier / negate helpers).
        ctx: Assembly emission context.
        double_copy: If True, emit two copies of the body per loop
            iteration (2x MFMAs, processing 2 DepthU chunks).
    """

    def __init__(
        self,
        pipeline: ScheduledPipeline,
        buffer_mgr: 'LDSBufferManager',
        ctx: 'AsmContext',
        *,
        double_copy: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.buffer_mgr = buffer_mgr
        self.ctx = ctx
        self.double_copy = double_copy

    # -- public ----------------------------------------------------

    def emit(self) -> None:
        """Emit the complete K-loop: ramp-up, body, drain."""
        self._emit_ramp_up()
        if self.pipeline.ki_phased:
            self._emit_pre_loop_ki0_reads()
        self._emit_body()
        # Drain iterations are handled by the body's skip-check
        # (producers are skipped when k_tiles is exhausted).
        # Explicit drain stages are only needed for PGR >= 2 where
        # the body loop exits early and extra consumer-only
        # iterations remain.
        if self.pipeline.drain:
            self._emit_drain()

    # -- ramp-up ---------------------------------------------------

    def _emit_ramp_up(self) -> None:
        """Emit prologue stages that prefetch tiles before the loop."""
        ctx = self.ctx
        pgr = self.pipeline.pgr
        stages = self.pipeline.ramp_up

        for stage_idx, stage_ops in enumerate(stages):
            is_first = stage_idx == 0
            is_last = stage_idx == pgr - 1
            ctx.comment(f"Pipeline ramp-up stage {stage_idx}/{pgr}")

            if is_first:
                self._emit_ramp_up_first(stage_ops)
            else:
                self._emit_ramp_up_subsequent(stage_ops, stage_idx, is_last)

            ctx.raw("")

    def _emit_ramp_up_first(self, ops: List[KLoopOp]) -> None:
        """Stage 0: load first tile, wait, barrier."""
        ctx = self.ctx
        has_writes = False

        needs_vmcnt = False  # track if vmcnt needed before ds_write
        for op in ops:
            if op.kind == OpKind.BARRIER:
                # Drain loads before barrier.
                if needs_vmcnt:
                    ctx.s_waitcnt("vmcnt(0)", comment="wait all loads")
                    needs_vmcnt = False
                if has_writes:
                    ctx.s_waitcnt("lgkmcnt(0)",
                                 comment="wait LDS writes")
                ctx.s_barrier(comment="sync tile 0")
                continue
            if op.kind == OpKind.DS_WRITE:
                # Must drain global loads before writing to LDS
                if needs_vmcnt:
                    ctx.s_waitcnt("vmcnt(0)",
                                 comment="wait global loads before ds_write")
                    needs_vmcnt = False
                has_writes = True
            if op.kind == OpKind.GLOBAL_LOAD:
                needs_vmcnt = True
            self._emit_op(op)

    def _emit_ramp_up_subsequent(
        self,
        ops: List[KLoopOp],
        stage_idx: int,
        is_last: bool,
    ) -> None:
        """Stage s > 0: guarded prefetch of next tile."""
        ctx = self.ctx
        skip_label = f"pgr_skip_{stage_idx}"

        # Guard: skip if not enough K-tiles.
        ctx.inst("s_cmp_le_u32", ctx.sreg("s_k_tiles"),
                 str(stage_idx),
                 comment=f"skip if k_tiles <= {stage_idx}")
        ctx.inst("s_cbranch_scc1", skip_label,
                 comment=f"not enough tiles for stage {stage_idx}")

        has_writes = False
        needs_vmcnt = False
        for op in ops:
            if op.kind == OpKind.BARRIER:
                if needs_vmcnt:
                    ctx.s_waitcnt("vmcnt(0)", comment="wait loads")
                    needs_vmcnt = False
                if has_writes:
                    ctx.s_waitcnt("lgkmcnt(0)",
                                 comment="wait LDS writes")
                ctx.s_barrier(comment=f"sync tile {stage_idx}")
                continue
            if op.kind == OpKind.DS_WRITE:
                # Must drain global loads before writing to LDS
                if needs_vmcnt:
                    ctx.s_waitcnt("vmcnt(0)",
                                 comment="wait global loads before ds_write")
                    needs_vmcnt = False
                has_writes = True
            if op.kind == OpKind.GLOBAL_LOAD:
                needs_vmcnt = True
            self._emit_op(op)

        ctx.label(skip_label)

    # -- body ------------------------------------------------------


    def _emit_pre_loop_ki0_reads(self) -> None:
        """Emit initial ki=0 reads before the loop body.

        For ki-phased scheduling, the loop body expects ki=0 reads
        to be in-flight (issued by the previous copy's Phase B-2).
        For the first iteration, this method issues those reads
        after the ramp-up.  They are drained by lgkmcnt(0) at the
        top of C0's Phase A.
        """
        ctx = self.ctx
        body = self.pipeline.body
        producer_start = len(self.pipeline.body)
        for i, op in enumerate(body):
            if op.iteration > 0:
                producer_start = i
                break

        ctx.comment("Pre-loop ki=0 reads for first C0")
        for i in range(producer_start):
            op = body[i]
            if op.kind in (OpKind.DS_READ, OpKind.SCALE_LOAD) and '_k1' not in op.name:
                self._emit_op(op)
        ctx.raw("")

    def _emit_body(self) -> None:
        """Emit the steady-state K-loop body."""
        ctx = self.ctx
        pipeline = self.pipeline
        body = pipeline.body
        pgr = pipeline.pgr

        ctx.label("k_loop")
        ctx.raw("")

        if pipeline.is_consume_first:
            self._emit_body_consume_first(body, pgr)
        else:
            self._emit_body_produce_first(body, pgr)

    def _emit_body_produce_first(
        self, body: List[KLoopOp], pgr: int
    ) -> None:
        """Produce-first body: pre-body -> skip-check -> producers -> barrier -> consumers.

        Pre-body reads (B ki=0) go before the skip check to overlap
        with the barrier stall.  Producer ops (iteration > 0) are
        wrapped in a skip-check so they're elided during drain
        iterations.
        """
        ctx = self.ctx
        waitcnts = self.pipeline.waitcnts
        pre_body_count = self.pipeline.pre_body_count
        barrier_pos = self.pipeline.body_barrier_pos

        # 1. Pre-body reads (overlap with arriving DTL data)
        if pre_body_count > 0:
            ctx.comment("Pre-body: early B reads (overlap with loads)")
        for i in range(pre_body_count):
            if i in waitcnts:
                ctx.s_waitcnt(waitcnts[i],
                              comment=f"auto-wait at pos {i}")
            self._emit_op(body[i])

        # 2. k_tiles-- and skip check
        ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
                  comment="k_tiles--")
        ctx.inst("s_cmp_gt_u32", ctx.sreg("s_k_tiles"),
                 str(pgr - 1),
                 comment=f"k_tiles > {pgr - 1}?")
        ctx.inst("s_cbranch_scc0", "load_skip_all",
                 comment="skip producers (drain)")

        # 3. Producers (skip-guarded)
        for i in range(pre_body_count, barrier_pos):
            if i in waitcnts:
                ctx.s_waitcnt(waitcnts[i],
                              comment=f"auto-wait at pos {i}")
            self._emit_op(body[i])

        # 4. load_skip_all: barrier + consumers
        ctx.raw("")
        ctx.label("load_skip_all")

        for i in range(barrier_pos, len(body)):
            if i in waitcnts:
                ctx.s_waitcnt(waitcnts[i],
                              comment=f"auto-wait at pos {i}")
            # Before barrier: drain any LDS writes from producers
            if body[i].kind == OpKind.BARRIER:
                # DTL loads must complete before barrier so LDS
                # data is visible to all waves after sync.
                has_global_loads = any(
                    op.kind == OpKind.GLOBAL_LOAD
                    for op in body[pre_body_count:barrier_pos])
                if has_global_loads:
                    ctx.s_waitcnt("vmcnt(0)",
                                  comment="wait DTL loads")
                has_ds_writes = any(
                    op.kind == OpKind.DS_WRITE
                    for op in body[pre_body_count:barrier_pos])
                if has_ds_writes:
                    ctx.s_waitcnt("lgkmcnt(0)",
                                  comment="wait LDS writes")
            self._emit_op(body[i])

        # Loop tail: negate DB step + branch.
        self._emit_loop_tail()

    def _emit_body_consume_first(
        self, body: List[KLoopOp], pgr: int
    ) -> None:
        """Consume-first body: barrier -> consumers -> skip-check -> producers.

        Barrier is at the top of the loop, syncing loads from the
        previous iteration.  Producers are at the bottom, wrapped
        in a skip-check.

        When ``double_copy`` is enabled, the consumers and producers
        are emitted twice (C0 then C1).  Each copy has its own
        skip-check that skips producers only (not the end-barrier,
        suffix toggle, or ki=0 reads).  The negate is emitted
        before the skip-check so it always runs; extra negates
        around the suffix ensure toggle_read uses the opposite
        step direction from toggle_write.
        """
        ctx = self.ctx
        waitcnts = self.pipeline.waitcnts

        # Find where producers start (first op with iteration > 0)
        producer_start = len(body)
        for i, op in enumerate(body):
            if op.iteration > 0:
                producer_start = i
                break

        if self.double_copy:
            if self.pipeline.ki_phased:
                # -- Ki-phased double-copy --------------------------
                ctx.comment("=== Copy C0 (ki-phased) ===")
                self._emit_copy_ki_phased(
                    body, producer_start, waitcnts,
                    skip_label="c0_prod_skip", pgr=pgr, copy_tag="C0",
                    is_first_copy=False)
                ctx.raw("")

                # Guard: skip C1 when C0 consumed the last tile.
                # k_tiles underflows (unsigned) if C1 decrements
                # past 0, so we must prevent C1 from running.
                ctx.inst("s_cmp_eq_u32", ctx.sreg("s_k_tiles"), "0",
                         comment="k_tiles == 0? (no data for C1)")
                ctx.inst("s_cbranch_scc1", "load_skip_all",
                         comment="skip C1 (all tiles consumed)")

                ctx.comment("=== Copy C1 (ki-phased) ===")
                self._emit_copy_ki_phased(
                    body, producer_start, waitcnts,
                    skip_label="c1_prod_skip", pgr=pgr, copy_tag="C1",
                    is_first_copy=False)
                ctx.raw("")

            else:
                # -- Standard interleaved double-copy ---------------
                ctx.comment("=== Copy C0 ===")
                self._emit_copy_interleaved(
                    body, producer_start, waitcnts,
                    skip_label="load_skip_all", pgr=pgr, copy_tag="C0")
                ctx.raw("")

                ctx.comment("=== Copy C1 ===")
                self._emit_copy_interleaved(
                    body, producer_start, waitcnts,
                    skip_label="load_skip_c1", pgr=pgr, copy_tag="C1")
                ctx.raw("")

            if not self.pipeline.ki_phased:
                # C1 skip target + loop tail (non-ki-phased path)
                ctx.label("load_skip_c1")
                self._emit_loop_tail()
                ctx.label("load_skip_all")
        else:
            # -- Single-copy (original) ----------------------------
            self._emit_copy_consumers(body, producer_start, waitcnts)

            ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
                      comment="k_tiles--")
            ctx.inst("s_cmp_gt_u32", ctx.sreg("s_k_tiles"),
                     str(pgr - 1),
                     comment=f"k_tiles > {pgr - 1}?")
            ctx.inst("s_cbranch_scc0", "load_skip_all",
                     comment="skip producers (drain)")

            self.buffer_mgr.emit_negate_step(ctx)
            self._emit_copy_producers(body, producer_start, waitcnts)

            ctx.raw("")
            ctx.label("load_skip_all")

            # Loop tail.
            self._emit_loop_tail()

        if self.double_copy and self.pipeline.ki_phased:
            # Loop tail for ki-phased double-copy (after C1).
            self._emit_loop_tail()
            ctx.label("load_skip_all")

    # -- drain -----------------------------------------------------

    def _emit_drain(self) -> None:
        """Emit drain iterations (consumer-only, no producers).

        Each drain stage d corresponds to ramp-up stage d+1.
        If that ramp-up stage was skipped (k_tiles_initial <= d+1),
        the drain stage must also be skipped to avoid reading
        from an uninitialized LDS buffer.

        Uses s_k_tiles_init (saved at loop entry) for the guard.
        """
        ctx = self.ctx
        pgr = self.pipeline.pgr

        # Save initial k_tiles before the loop decrements it.
        # This is emitted just before the drain (after the loop
        # exits), but the value was saved at loop entry.
        # Actually, k_tiles at this point = 0 or 1 (post-loop).
        # We need to compare against the ORIGINAL k_tiles.
        # Solution: use the pgr_skip labels -- if stage s was
        # skipped, k_tiles_init <= s. After the loop, k_tiles
        # has been decremented, but we can recover the original
        # from the fact that the loop ran (k_tiles_init - pgr)
        # iterations, consuming (k_tiles_init - pgr) tiles.
        # Simpler: just check if the ramp-up skip label was taken.
        # Even simpler: guard each drain stage with the same
        # condition as the corresponding ramp-up stage, but using
        # a saved register.

        # Guard: save initial k_tiles before loop decrements it
        # (emitted by the caller in pipeline.py)
        for d_idx, drain_ops in enumerate(self.pipeline.drain):
            skip_label = f"drain_skip_{d_idx}"
            ramp_stage = d_idx + 1  # drain[0] matches ramp-up[1]

            # Guard: skip drain if the corresponding ramp-up was skipped.
            # s_k_tiles_init is saved by the pipeline phase before
            # the loop starts. If it doesn't exist (unit tests),
            # drain runs unconditionally (assumes enough tiles).
            if self.double_copy:
                # Double-copy: body consumes 2 tiles/iter.  Drain
                # is needed only when k_tiles > 0 at loop exit
                # (odd total tile count).
                ctx.inst("s_cmp_eq_u32", ctx.sreg("s_k_tiles"), "0",
                         comment="k_tiles == 0? (all consumed)")
                ctx.inst("s_cbranch_scc1", skip_label,
                         comment=f"drain stage {d_idx} not needed")
            elif ctx.has("s_k_tiles_init"):
                ctx.inst("s_cmp_le_u32", ctx.sreg("s_k_tiles_init"),
                         str(ramp_stage),
                         comment=f"skip drain if k_tiles_init <= {ramp_stage}")
                ctx.inst("s_cbranch_scc1", skip_label,
                         comment=f"drain stage {d_idx} not needed")

            ctx.comment(f"Drain stage {d_idx}")
            for op in drain_ops:
                if op.kind == OpKind.BARRIER:
                    ctx.s_waitcnt("vmcnt(0)",
                                 comment="wait DTL loads")
                    ctx.s_waitcnt("lgkmcnt(0)",
                                 comment="wait LDS writes")
                self._emit_op(op)
            ctx.label(skip_label)
            ctx.raw("")

    # -- helpers ---------------------------------------------------

    def _emit_copy_consumers(
        self, body: List[KLoopOp], producer_start: int,
        waitcnts: dict,
    ) -> None:
        """Emit consumer ops for one copy (barrier + reads + MFMAs + toggles).

        Reusable for both C0 and C1 in double-copy mode.  The
        vmcnt(0) + lgkmcnt(0) before the barrier resets hw counter
        state, so the same *waitcnts* dict applies to every copy.
        """
        ctx = self.ctx
        for i in range(producer_start):
            op = body[i]
            if i in waitcnts:
                ctx.s_waitcnt(waitcnts[i],
                              comment=f"auto-wait at pos {i}")
            if op.kind == OpKind.BARRIER:
                ctx.s_waitcnt("vmcnt(0)",
                              comment="wait DTL loads from prev iter")
                ctx.s_waitcnt("lgkmcnt(0)",
                              comment="wait LDS writes from prev iter")
            self._emit_op(op)

    def _emit_copy_producers(
        self, body: List[KLoopOp], producer_start: int,
        waitcnts: dict,
    ) -> None:
        """Emit producer ops for one copy."""
        ctx = self.ctx
        for i in range(producer_start, len(body)):
            op = body[i]
            if i in waitcnts:
                ctx.s_waitcnt(waitcnts[i],
                              comment=f"auto-wait at pos {i}")
            self._emit_op(op)

    def _emit_copy_interleaved(
        self, body: List[KLoopOp], producer_start: int,
        waitcnts: dict, skip_label: str, pgr: int,
        copy_tag: str,
    ) -> None:
        """Emit one copy with producers interleaved among later MFMAs.

        Structure:
          barrier -> first-half consumers -> skip-check -> negate ->
          scalar producers -> second-half consumers interleaved with
          global-load producers

        This hides global load latency under MFMA execution instead
        of batching all loads after all MFMAs.
        """
        ctx = self.ctx
        ops = classify_body_ops(body, producer_start)
        scalar_prods = ops["scalar_prods"]
        load_prods = ops["load_prods"]

        # Find the MFMA midpoint in consumers (where to insert skip-check)
        consumer_mfma_indices = [
            i for i in range(producer_start) if body[i].kind == OpKind.MFMA
        ]
        total_mfmas = len(consumer_mfma_indices)

        # Split at ~25% of MFMAs: first part is pure consume,
        # second part interleaves global loads
        split_mfma = int(total_mfmas * 0.25)
        if split_mfma < 1:
            split_mfma = total_mfmas
        split_pos = consumer_mfma_indices[split_mfma - 1] + 1 if split_mfma > 0 else producer_start

        # Phase 1: barrier + first-half consumers
        for i in range(split_pos):
            op = body[i]
            if i in waitcnts:
                ctx.s_waitcnt(waitcnts[i],
                              comment=f"auto-wait at pos {i}")
            if op.kind == OpKind.BARRIER:
                ctx.s_waitcnt("vmcnt(0)",
                              comment="wait DTL loads from prev iter")
                ctx.s_waitcnt("lgkmcnt(0)",
                              comment="wait LDS writes from prev iter")
            self._emit_op(op)

        # Skip check + negate + scalar producers
        ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
                  comment=f"k_tiles-- ({copy_tag})")
        ctx.inst("s_cmp_gt_u32", ctx.sreg("s_k_tiles"),
                 str(pgr - 1),
                 comment=f"k_tiles > {pgr - 1}? ({copy_tag})")
        ctx.inst("s_cbranch_scc0", skip_label,
                 comment=f"skip {copy_tag} producers (drain)")

        self.buffer_mgr.emit_negate_step(ctx)

        for op in scalar_prods:
            self._emit_op(op)

        # Phase 2: remaining consumers with DTL interleaving
        remaining_mfma_ops = [
            body[i] for i in range(split_pos, producer_start)
            if body[i].kind == OpKind.MFMA
        ]
        remaining_non_mfma = [
            body[i] for i in range(split_pos, producer_start)
            if body[i].kind != OpKind.MFMA
        ]
        loader = ctx._state.get("_dtl_loader")

        # Emit non-MFMA ops first (reads with waitcnts)
        for i in range(split_pos, producer_start):
            op = body[i]
            if op.kind != OpKind.MFMA:
                if i in waitcnts:
                    ctx.s_waitcnt(waitcnts[i],
                                  comment=f"auto-wait at pos {i}")
                self._emit_op(op)

        if loader and len(remaining_mfma_ops) >= 4 and hasattr(loader, 'emit_dtl_load_a_single'):
            dtl_seq = build_dtl_sequence(load_prods, loader)
            emit_mfmas_with_dtl_interleaved(
                ctx, remaining_mfma_ops, dtl_seq, loader, self._emit_op,
                comment=f"remaining MFMAs + DTL ({copy_tag})")
        else:
            for mfma_op in remaining_mfma_ops:
                self._emit_op(mfma_op)
            for op in load_prods:
                self._emit_op(op)



    def _emit_copy_ki_phased(
        self, body: List[KLoopOp], producer_start: int,
        waitcnts: dict, skip_label: str, pgr: int,
        copy_tag: str, is_first_copy: bool,
    ) -> None:
        """Emit one copy with ki-phased structure and full ki=1 DTL spreading.

        Structure per copy (double-copy toggle fix):
          1. drain prev reads  (lgkmcnt/vmcnt)
          2. ki=0 MFMAs + ki=1 reads interleaved
          3. lgkmcnt(0) + s_barrier  (mid-copy: drain ki=1 reads)
          4. ALL ki=1 MFMAs  (consumers, always execute)
          5. k_tiles-- + negate (always)
          5b. skip-check → skip_label
          6. scalar_prods (toggle_write) + DTL loads
          7. [skip_label] vmcnt(0) + lgkmcnt(0) + s_barrier
          8. negate (flip step for toggle_read)
          9. suffix (toggle_read, uses flipped step)
          10. negate (restore step)
          11. ki=0 reads (from toggled buffer)

        ki=1 MFMAs are consumers (they use data from the CURRENT
        iteration loaded before the mid-barrier).  They must NOT
        be skipped during drain.  Only the producer ops (DTL loads,
        SRD advances) are skipped when there are no more tiles.

        Toggle invariant: toggle_write and toggle_read need opposite
        step directions.  The negate at step 5 sets the step for
        toggle_write; the extra negates at 8/10 flip it for
        toggle_read and restore it afterward.
        """
        ctx = self.ctx

        ops = classify_body_ops(body, producer_start)
        barrier_op = ops["barrier"]
        ki0_reads = ops["ki0_reads"]
        ki1_reads = ops["ki1_reads"]
        ki0_mfmas = ops["ki0_mfmas"]
        ki1_mfmas = ops["ki1_mfmas"]
        suffix_ops = ops["suffix"]
        scalar_prods = ops["scalar_prods"]
        load_prods = ops["load_prods"]

        # ─── Step 1: drain prev reads ───
        if is_first_copy:
            ctx.s_waitcnt("vmcnt(0)", comment=f"wait DTL ({copy_tag})")
            ctx.s_waitcnt("lgkmcnt(0)", comment=f"wait LDS ({copy_tag})")
            if barrier_op:
                self._emit_op(barrier_op)
            ctx.comment(f"ki=0 reads ({len(ki0_reads)} ops)")
            for r in ki0_reads:
                self._emit_op(r)
            if ki0_reads:
                ctx.s_waitcnt("vmcnt(0) lgkmcnt(0)",
                              comment="drain ki=0 reads + scale VMEM loads")
        else:
            has_vmem_scale = any(
                op.kind == OpKind.SCALE_LOAD for op in self.pipeline.body)
            if has_vmem_scale:
                ctx.s_waitcnt("vmcnt(0) lgkmcnt(0)",
                              comment=f"drain reads + scale VMEM ({copy_tag})")
            else:
                ctx.s_waitcnt("lgkmcnt(0)",
                              comment=f"drain ki=0 reads from prev B-2 ({copy_tag})")

        # ─── Step 2: ki=0 MFMAs + ki=1 reads interleaved ───
        emit_mfmas_with_reads_interleaved(
            ki0_mfmas, ki1_reads, self._emit_op,
            comment=f"ki=0 MFMAs + ki=1 reads", ctx=ctx)

        # ─── Step 3: mid-copy barrier ───
        ctx.s_waitcnt("lgkmcnt(0)", comment="drain ki=1 reads")
        ctx.s_barrier(comment=f"mid-copy barrier ({copy_tag})")

        # ─── Step 4: ki=1 MFMAs (consumers, always execute) ───
        for mfma_op in ki1_mfmas:
            self._emit_op(mfma_op)

        # ─── Step 5: k_tiles-- + negate (always runs) ───
        ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
                  comment=f"k_tiles-- ({copy_tag})")
        self.buffer_mgr.emit_negate_step(ctx)

        # ─── Step 5b: skip-check ───
        ctx.inst("s_cmp_gt_u32", ctx.sreg("s_k_tiles"),
                 str(pgr - 1),
                 comment=f"k_tiles > {pgr - 1}? ({copy_tag})")
        ctx.inst("s_cbranch_scc0", skip_label,
                 comment=f"skip {copy_tag} producers (drain)")

        # ─── Step 6: scalar_prods (toggle_write) + DTL loads ───
        for op in scalar_prods:
            self._emit_op(op)

        loader = ctx._state.get("_dtl_loader")
        if loader and hasattr(loader, 'emit_dtl_load_a_single'):
            dtl_seq = build_dtl_sequence(load_prods, loader)
            m0_state: dict = {}
            for kind, payload in dtl_seq:
                emit_dtl_load(kind, payload, loader, self._emit_op, m0_state)
        else:
            for op in load_prods:
                self._emit_op(op)

        # ─── Step 7: end-barrier ───
        ctx.label(skip_label)
        # Must drain ALL DTL loads before reading the buffer they
        # wrote to.  In double-copy, the other copy's ki=0 reads
        # access this buffer after toggle, so loads must be done.
        ctx.s_waitcnt("vmcnt(0) lgkmcnt(0)", comment=f"wait DTL+scale writes ({copy_tag})")
        ctx.s_barrier(comment=f"end-barrier ({copy_tag})")

        # ─── Step 8: negate (flip step for toggle_read) ───
        self.buffer_mgr.emit_negate_step(ctx)

        # ─── Step 9: suffix (toggle_read, uses flipped step) ───
        for op in suffix_ops:
            self._emit_op(op)

        # ─── Step 10: negate (restore step) ───
        self.buffer_mgr.emit_negate_step(ctx)

        # ─── Step 11: ki=0 reads (from toggled buffer) ───
        for r in ki0_reads:
            self._emit_op(r)


    def _emit_op(self, op: KLoopOp) -> None:
        """Emit a single op. Handles None callbacks gracefully."""
        ctx = self.ctx
        if op.kind == OpKind.BARRIER:
            ctx.s_barrier(comment=op.comment or "barrier")
            return
        if op.emit is not None:
            op.emit()
        else:
            # Placeholder: op not yet wired to codegen.
            ctx.comment(f"[TODO] {op.name} ({op.kind.value})")

    def _emit_loop_tail(self) -> None:
        """Negate DB step (produce-first only) and branch back."""
        ctx = self.ctx
        # In consume-first mode, negate is emitted before producers
        # (inside the skip-check guard), so skip it here.
        if not self.pipeline.is_consume_first:
            self.buffer_mgr.emit_negate_step(ctx)
        if self.pipeline.drain:
            # When explicit drain stages exist, exit the body loop
            # early so the drain handles the final pgr-1 consumer
            # iterations.  The body runs k_tiles - (pgr-1) iters.
            pgr = self.pipeline.pgr
            ctx.inst("s_cmp_gt_u32", ctx.sreg("s_k_tiles"),
                     str(pgr - 1),
                     comment=f"k_tiles > {pgr - 1}? (exit to drain)")
            ctx.inst("s_cbranch_scc1", "k_loop", comment="loop")
        else:
            ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
                     comment="more?")
            ctx.inst("s_cbranch_scc1", "k_loop", comment="loop")
        ctx.raw("")
