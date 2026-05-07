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
        are emitted twice (C0 then C1).  Each copy decrements
        k_tiles by 1 and has its own skip-check.  If C0 skips,
        C1 is bypassed entirely (jump to ``load_skip_all``).
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
            # -- Double-copy with interleaved producers -------------
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

            # C1 skip target + loop tail
            ctx.label("load_skip_c1")
            self._emit_loop_tail()
            # C0 skip target (after loop exit, drain follows)
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
            if ctx.has("s_k_tiles_init"):
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
        producers = [body[i] for i in range(producer_start, len(body))]

        # Separate producers into scalar (advance/toggle) and loads
        scalar_prods = [op for op in producers
                        if op.kind in (OpKind.SCALAR,)]
        load_prods = [op for op in producers
                      if op.kind == OpKind.GLOBAL_LOAD]

        # Find the MFMA midpoint in consumers (where to insert skip-check)
        consumer_mfma_indices = []
        for i in range(producer_start):
            if body[i].kind == OpKind.MFMA:
                consumer_mfma_indices.append(i)
        total_mfmas = len(consumer_mfma_indices)

        # Split at ~60% of MFMAs: first part is pure consume,
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

        # Emit scalar producers (advance SRDs, toggle write ptrs)
        for op in scalar_prods:
            self._emit_op(op)

        # Phase 2: remaining consumers with fine-grained DTL interleaving
        # Use per-line DTL methods to spread individual loads among MFMAs
        remaining_mfmas = sum(1 for i in range(split_pos, producer_start)
                              if body[i].kind == OpKind.MFMA)

        # Get the loader for per-line DTL
        loader = ctx._metadata.get("_dtl_loader")

        if loader and remaining_mfmas >= 4 and hasattr(loader, 'emit_dtl_load_a_single'):
            # Build individual DTL load sequence: scale_a, scale_b,
            # then alternating A[0..n] and B[0..n]
            dtl_seq = []
            for op in load_prods:
                if 'scale' in op.name:
                    dtl_seq.append(('scale_op', op))
            n_a = loader.num_loads_a
            n_b = loader.num_loads_b
            for j in range(max(n_a, n_b)):
                if j < n_a:
                    dtl_seq.append(('dtl_a', j))
                if j < n_b:
                    dtl_seq.append(('dtl_b', j))

            # Spread across remaining MFMAs
            total_loads = len(dtl_seq)
            interval = max(1, remaining_mfmas // (total_loads + 1))
            load_idx = 0
            mfma_ct = 0
            m0_a_set = False
            m0_b_set = False

            for i in range(split_pos, producer_start):
                op = body[i]
                if i in waitcnts:
                    ctx.s_waitcnt(waitcnts[i],
                                  comment=f"auto-wait at pos {i}")
                self._emit_op(op)

                if op.kind == OpKind.MFMA:
                    mfma_ct += 1
                    if mfma_ct % interval == 0 and load_idx < total_loads:
                        kind, payload = dtl_seq[load_idx]
                        if kind == 'scale_op':
                            self._emit_op(payload)
                        elif kind == 'dtl_a':
                            if not m0_a_set:
                                loader.emit_dtl_m0_a()
                                m0_a_set = True
                            loader.emit_dtl_load_a_single(payload)
                        elif kind == 'dtl_b':
                            if not m0_b_set:
                                loader.emit_dtl_m0_b()
                                m0_b_set = True
                            loader.emit_dtl_load_b_single(payload)
                        load_idx += 1

            # Remaining loads
            while load_idx < total_loads:
                kind, payload = dtl_seq[load_idx]
                if kind == 'scale_op':
                    self._emit_op(payload)
                elif kind == 'dtl_a':
                    if not m0_a_set:
                        loader.emit_dtl_m0_a()
                        m0_a_set = True
                    loader.emit_dtl_load_a_single(payload)
                elif kind == 'dtl_b':
                    if not m0_b_set:
                        loader.emit_dtl_m0_b()
                        m0_b_set = True
                    loader.emit_dtl_load_b_single(payload)
                load_idx += 1
        else:
            # Fallback: emit remaining consumers then bulk loads
            for i in range(split_pos, producer_start):
                op = body[i]
                if i in waitcnts:
                    ctx.s_waitcnt(waitcnts[i],
                                  comment=f"auto-wait at pos {i}")
                self._emit_op(op)
            for op in load_prods:
                self._emit_op(op)


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
