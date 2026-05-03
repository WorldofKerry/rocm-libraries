"""Auto-pipelined K-loop: derive loop structure from declared stages.

Users declare pipeline stages with dependencies and resources.
The framework auto-derives the loop structure and calls stage
emitters at the right points.

The key separation:
  - SoftwarePipeline: generic loop structure (ramp-up/body/drain)
  - StageEmitter: stage-specific instruction emission (pluggable)

Adding a new stage = implement a StageEmitter + declare a PipelineStage.
The framework handles everything else.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

from ..emit.context import AsmContext
from ..memory.global_loader import GlobalLoader
from ..memory.lds_reader import LDSReader
from .kloop_graph import (
    KLoopGraph, MFMABlock, DSReadBlock, GlobalLoadBlock,
    ScaleBlock, SuffixBlock, OpKind,
)
from .kloop_scheduler import KLoopScheduler, ScheduledKLoop
from .pipeline import ComputePipeline

__all__ = [
    "PipelineStage", "StageDep", "ResourceConfig",
    "SoftwarePipeline", "StageEmitter",
    "GlobalLoadStageEmitter", "ReadComputeStageEmitter",
    "AutoPipelinedCompute",
]


# ===================================================================
# Declarative types
# ===================================================================

@dataclass(frozen=True)
class PipelineStage:
    """A coarse-grained pipeline stage."""
    name: str
    distance: int = 0           # tile offset: 0 = current, 1 = next
    resource: Optional[str] = None
    mode: str = "none"          # "read", "write", "none"
    wait_counter: str = "none"  # "vmcnt", "lgkmcnt", "none"


@dataclass(frozen=True)
class StageDep:
    """Inter-iteration dependency between pipeline stages."""
    producer: str
    consumer: str
    distance: int = 1  # minimum iterations of separation


@dataclass(frozen=True)
class ResourceConfig:
    """Shared resource configuration."""
    name: str
    num_buffers: int = 2
    buf_size: int = 0  # bytes per buffer slice


# ===================================================================
# StageEmitter: pluggable stage-specific emission
# ===================================================================

class StageEmitter(ABC):
    """Interface for stage-specific instruction emission.

    Each pipeline stage has an emitter that knows how to produce
    its instructions. The SoftwarePipeline calls these at the
    right structural points (ramp-up, body, drain).
    """

    @abstractmethod
    def emit_ramp_up(self, ctx: AsmContext, stage_index: int,
                     is_first: bool) -> None:
        """Emit instructions for one ramp-up iteration.

        Args:
            ctx: Assembly context.
            stage_index: Which ramp-up stage (0, 1, ..., pgr-1).
            is_first: True for stage 0 (needs wait + barrier).
        """

    @abstractmethod
    def emit_produce(self, ctx: AsmContext) -> None:
        """Emit producer instructions in the loop body (G stage)."""

    @abstractmethod
    def emit_consume(self, ctx: AsmContext) -> None:
        """Emit consumer instructions in the loop body (R+M stage)."""


# ===================================================================
# SoftwarePipeline: inter-iteration structure
# ===================================================================

class SoftwarePipeline:
    """Derives K-loop structure from declared stages and dependencies.

    This is the generic framework. It handles:
    - Pipeline depth derivation (min_pgr)
    - Buffer lifecycle (loads_before_reads)
    - Loop structure generation via stage emitter callbacks

    It does NOT know about GEMM, MFMA, ds_reads, etc. Those are
    behind StageEmitter implementations.
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        deps: List[StageDep],
        resources: List[ResourceConfig],
        pgr: Optional[int] = None,
    ) -> None:
        self.stages = {s.name: s for s in stages}
        self.stage_list = list(stages)
        self.deps = deps
        self.resources = {r.name: r for r in resources}

        # Derive stage numbers (ASAP scheduling on dep graph)
        self.stage_nums = self._compute_stage_nums()
        self.min_pgr = max(self.stage_nums.values()) if self.stage_nums else 0

        # Resolve PGR
        self.pgr = pgr if pgr is not None else self.min_pgr
        if self.pgr < self.min_pgr:
            raise ValueError(
                f"PGR={self.pgr} < min_pgr={self.min_pgr} "
                f"(pipeline would stall)")

        # Validate buffer constraints
        self._validate_buffers()

        # Derive buffer lifecycle
        self._loads_before_reads = self._compute_loads_before_reads()

        # Classify stages
        self.producer_stages = [
            s for s in stages if s.mode == "write"
        ]
        self.consumer_stages = [
            s for s in stages if s.mode in ("read", "none")
        ]

    @property
    def loads_before_reads(self) -> bool:
        return self._loads_before_reads

    @property
    def num_buffers(self) -> int:
        for r in self.resources.values():
            return r.num_buffers
        return 2

    def _compute_stage_nums(self) -> Dict[str, int]:
        nums: Dict[str, int] = {name: 0 for name in self.stages}
        changed = True
        while changed:
            changed = False
            for dep in self.deps:
                new_val = nums[dep.producer] + dep.distance
                if new_val > nums.get(dep.consumer, 0):
                    nums[dep.consumer] = new_val
                    changed = True
        return nums

    def _validate_buffers(self) -> None:
        for dep in self.deps:
            prod = self.stages.get(dep.producer)
            cons = self.stages.get(dep.consumer)
            if (prod and cons and prod.resource
                    and prod.resource == cons.resource
                    and prod.mode == "write" and cons.mode == "read"):
                bufs = self.resources.get(prod.resource)
                if bufs and self.pgr > bufs.num_buffers:
                    raise ValueError(
                        f"PGR={self.pgr} > {bufs.num_buffers} buffers "
                        f"for resource '{prod.resource}'")

    def _compute_loads_before_reads(self) -> bool:
        for dep in self.deps:
            prod = self.stages.get(dep.producer)
            cons = self.stages.get(dep.consumer)
            if (prod and cons and prod.resource
                    and prod.resource == cons.resource
                    and prod.mode == "write" and cons.mode == "read"):
                bufs = self.resources.get(prod.resource)
                if bufs and self.pgr >= bufs.num_buffers:
                    return False
        return True

    # -- Loop emission (generic, calls stage emitters) --

    def emit_kloop(self, ctx: AsmContext,
                   emitters: Dict[str, StageEmitter]) -> None:
        """Emit the complete K-loop: ramp-up + body + drain.

        This is the generic entry point. It only knows about:
        - Stage ordering (from loads_before_reads)
        - Skip conditions (from pgr)
        - Barriers and loop control

        All stage-specific instructions come from emitters.
        """
        pgr = self.pgr

        # --- Ramp-up ---
        for stage_idx in range(pgr):
            is_first = (stage_idx == 0)
            ctx.comment(f"Pipeline ramp-up stage "
                        f"{stage_idx}/{pgr}")
            for s in self.producer_stages:
                if s.name in emitters:
                    emitters[s.name].emit_ramp_up(
                        ctx, stage_idx, is_first)
            ctx.raw("")

        # --- Loop body ---
        ctx.label("k_loop")
        ctx.raw("")

        if self.loads_before_reads:
            self._emit_body_produce_first(ctx, emitters, pgr)
        else:
            self._emit_body_consume_first(ctx, emitters, pgr)

    def _emit_body_produce_first(self, ctx, emitters, pgr):
        """Body: pre-body -> skip check -> G -> barrier -> R+M.

        Pre-body ops (early B reads) go first to overlap with
        arriving loads from the previous iteration.
        """
        # Pre-body ops from consumer (early B reads, overlap with loads)
        for s in self.consumer_stages:
            if s.name in emitters:
                emitter = emitters[s.name]
                if hasattr(emitter, 'emit_pre_body'):
                    emitter.emit_pre_body(ctx)

        # Skip check + producer
        ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
                  comment="k_tiles--")
        ctx.inst("s_cmp_gt_u32", ctx.sreg("s_k_tiles"),
                 str(pgr - 1),
                 comment=f"k_tiles > {pgr - 1}?")
        ctx.inst("s_cbranch_scc0", "load_skip_all",
                 comment="skip producers (drain phase)")

        for s in self.producer_stages:
            if s.name in emitters:
                emitters[s.name].emit_produce(ctx)

        ctx.raw("")
        ctx.label("load_skip_all")

        # Barrier between producer and consumer
        self._emit_sync(ctx, emitters)

        # Consumer (main body, excluding pre-body)
        for s in self.consumer_stages:
            if s.name in emitters:
                emitters[s.name].emit_consume(ctx)

        # Loop tail
        self._emit_loop_tail(ctx, emitters)

    def _emit_body_consume_first(self, ctx, emitters, pgr):
        """Body: barrier -> pre-body -> R+M -> lgkmcnt(0) -> G."""
        # Barrier first
        self._emit_sync(ctx, emitters)

        # Pre-body ops from consumer (early B reads)
        for s in self.consumer_stages:
            if s.name in emitters:
                emitter = emitters[s.name]
                if hasattr(emitter, 'emit_pre_body'):
                    emitter.emit_pre_body(ctx)

        # Consumer (main body)
        for s in self.consumer_stages:
            if s.name in emitters:
                emitters[s.name].emit_consume(ctx)

        # Wait for all consumer reads before overwriting
        ctx.s_waitcnt("lgkmcnt(0)",
                      comment="wait all reads before overwriting")

        # Skip check + producer
        ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
                  comment="k_tiles--")
        ctx.inst("s_cmp_gt_u32", ctx.sreg("s_k_tiles"),
                 str(pgr - 1),
                 comment=f"k_tiles > {pgr - 1}?")
        ctx.inst("s_cbranch_scc0", "load_skip_all",
                 comment="skip producers (drain phase)")

        for s in self.producer_stages:
            if s.name in emitters:
                emitters[s.name].emit_produce(ctx)

        ctx.raw("")
        ctx.label("load_skip_all")

        # Loop tail
        self._emit_loop_tail(ctx, emitters)

    def _emit_sync(self, ctx, emitters):
        """Emit barrier between producer and consumer stages."""
        # Check if any emitter has a custom sync; otherwise s_barrier
        for s in self.producer_stages:
            if s.name in emitters:
                emitter = emitters[s.name]
                if hasattr(emitter, 'emit_sync'):
                    emitter.emit_sync(ctx)
                    return
        ctx.s_barrier(comment="sync producer -> consumer")

    def _emit_loop_tail(self, ctx, emitters=None):
        """Emit loop branch back.

        If any producer emitter has cross_iter_prefetch, use an
        exit-branch pattern to skip prefetch on the last iteration.
        """
        has_cross_iter_pf = False
        if emitters:
            for s in self.producer_stages:
                if s.name in emitters:
                    e = emitters[s.name]
                    if (hasattr(e, 'scale_loader') and e.scale_loader
                            and e.scale_loader.has_cross_iter_prefetch):
                        has_cross_iter_pf = True
                        break

        ctx.s_barrier(comment="sync")
        ctx.inst("s_cmp_lg_u32", ctx.sreg("s_k_tiles"), "0",
                 comment="more?")
        if has_cross_iter_pf:
            ctx.inst("s_cbranch_scc0", "k_loop_end",
                     comment="exit if last")
            ctx.inst("s_branch", "k_loop", comment="loop back")
            ctx.label("k_loop_end")
        else:
            ctx.inst("s_cbranch_scc1", "k_loop", comment="loop")
        ctx.raw("")

    def describe(self, num_tiles: int = 8) -> List[str]:
        """Return textual description of the pipeline structure."""
        lines = [
            f"SoftwarePipeline: PGR={self.pgr}, min_pgr={self.min_pgr}, "
            f"loads_before_reads={self.loads_before_reads}",
            f"  Stages: {', '.join(f'{s.name}(d={s.distance})' for s in self.stage_list)}",
            f"  Deps: {', '.join(f'{d.producer}->{d.consumer}(d={d.distance})' for d in self.deps)}",
            "",
        ]
        for p in range(self.pgr):
            producers = [s.name for s in self.producer_stages]
            sync = " + WAIT+BARRIER" if p == 0 else ""
            lines.append(
                f"  ramp-up[{p}]: {'+'.join(producers)}(tile={p}){sync}")
        for i in range(num_tiles):
            parts = []
            if self.loads_before_reads:
                nt = i + self.pgr
                parts.append(f"G({nt})" if nt < num_tiles else "[skip G]")
                parts.append("BARRIER")
                parts.append(f"RM({i})")
            else:
                parts.append("BARRIER")
                parts.append(f"RM({i})")
                parts.append("LGKMCNT(0)")
                nt = i + self.pgr
                parts.append(f"G({nt})" if nt < num_tiles else "[skip G]")
            phase = "steady" if i + self.pgr < num_tiles else "drain "
            lines.append(f"  loop[{i:2d}] {phase}: {' '.join(parts)}")
        return lines


# ===================================================================
# Concrete StageEmitters for GEMM
# ===================================================================

class GlobalLoadStageEmitter(StageEmitter):
    """Emitter for the G stage (global load -> LDS)."""

    def __init__(self, loader: GlobalLoader,
                 schedule: ScheduledKLoop,
                 scale_loader: object = None) -> None:
        self.loader = loader
        self.schedule = schedule
        self.scale_loader = scale_loader

    def emit_ramp_up(self, ctx, stage_index, is_first):
        from ..memory.global_loader import BufferLoader
        loader = self.loader
        schedule = self.schedule
        scale_loader = self.scale_loader

        if is_first:
            loader.emit_loads()
            if schedule.prologue_scale_ops:
                for op in schedule.prologue_scale_ops:
                    if op.emit:
                        op.emit()
                extra = len(schedule.prologue_scale_ops)
                ctx.s_waitcnt(f"vmcnt({extra})",
                              comment=f"wait DTL (leave {extra} scales)")
            else:
                ctx.s_waitcnt("vmcnt(0)", comment="wait loads")
            # BufferLoader: data is in VGPRs, write to LDS before barrier
            if isinstance(loader, BufferLoader):
                loader._emit_ds_writes()
                ctx.s_waitcnt("lgkmcnt(0)", comment="wait LDS writes")
            ctx.s_barrier(comment="sync tile 0")
        else:
            ctx.inst("s_cmp_le_u32", ctx.sreg("s_k_tiles"),
                     str(stage_index),
                     comment=f"skip if k_tiles <= {stage_index}")
            ctx.inst("s_cbranch_scc1", f"pgr_skip_{stage_index}",
                     comment=f"not enough tiles for stage {stage_index}")
            loader.advance()
            loader.toggle_write()
            loader.emit_loads()
            if (schedule.scale_advance_op
                    and schedule.scale_advance_op.emit):
                schedule.scale_advance_op.emit()
            if (schedule.prologue_scale_ops and scale_loader
                    and not scale_loader.has_cross_iter_prefetch):
                for op in schedule.prologue_scale_ops:
                    if op.emit:
                        op.emit()
            ctx.label(f"pgr_skip_{stage_index}")

    def emit_produce(self, ctx):
        from ..memory.global_loader import BufferLoader
        schedule = self.schedule
        scale_loader = self.scale_loader
        for op in schedule.prefetch_ops:
            if op.emit:
                op.emit()
        if schedule.scale_advance_op and schedule.scale_advance_op.emit:
            schedule.scale_advance_op.emit()
        if scale_loader and not scale_loader.has_cross_iter_prefetch:
            for op in schedule.prologue_scale_ops:
                if op.emit:
                    op.emit()
        # BufferLoader: wait for global loads and write to LDS
        # This is inside the skip check, so it only runs when
        # new data was actually loaded.
        if isinstance(self.loader, BufferLoader):
            ctx.s_waitcnt("vmcnt(0)", comment="wait global loads")
            self.loader._emit_ds_writes()

    def emit_consume(self, ctx):
        pass  # G stage has no consumer phase

    def emit_sync(self, ctx):
        """Emit sync: for BufferLoader, ds_write + wait; then barrier."""
        self.loader.emit_sync()
        ctx.raw("")


class ReadComputeStageEmitter(StageEmitter):
    """Emitter for the R+M stage (LDS read + MFMA compute).

    Delegates to KLoopScheduler output for fine-grained
    ds_read/MFMA interleaving. This is where Level 2
    (intra-iteration) scheduling lives.
    """

    def __init__(self, schedule: ScheduledKLoop,
                 reader: LDSReader,
                 loader: GlobalLoader,
                 scale_loader: object = None) -> None:
        self.schedule = schedule
        self.reader = reader
        self.loader = loader
        self.scale_loader = scale_loader

    def emit_ramp_up(self, ctx, stage_index, is_first):
        pass  # R+M has no ramp-up phase (distance=0)

    def emit_produce(self, ctx):
        pass  # R+M has no producer phase

    def emit_pre_body(self, ctx):
        """Emit early B reads (overlap with arriving loads)."""
        for op in self.schedule.pre_body_ops:
            if op.emit:
                op.emit()

    def emit_consume(self, ctx):
        """Emit the R+M body: preamble + MFMA loop (pre-body is separate)."""
        schedule = self.schedule
        reader = self.reader
        loader = self.loader
        scale_loader = self.scale_loader
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

        if schedule.prologue_scale_ops:
            num_dtl = (loader.num_inflight
                       if hasattr(loader, "num_inflight") else 0)
            ctx.s_waitcnt(f"vmcnt({num_dtl})",
                          comment=f"wait scales (leave {num_dtl} DTL)")
        ctx.raw("")

        reader.emit_recompute_ki_bases()
        ctx.raw("")

        # MFMA body with interleaved side ops (Level 2 scheduling)
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
                ctx.s_waitcnt(
                    "lgkmcnt(0)",
                    comment=f"wait A[m{mfma_count // mfmas_per_mi}]")
                inflight_lgkm = 0

            if schedule.prologue_scale_ops:
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


# ===================================================================
# AutoPipelinedCompute: ties it all together
# ===================================================================

class AutoPipelinedCompute(ComputePipeline):
    """K-loop using SoftwarePipeline + pluggable StageEmitters.

    The pipeline framework handles loop structure.
    Stage emitters handle instruction emission.
    KLoopScheduler handles intra-iteration R+M interleaving.
    """

    def __init__(
        self,
        loader: GlobalLoader,
        reader: LDSReader,
        scale_loader: object = None,
        pgr: int = 1,
        num_lds_buffers: int = 2,
    ) -> None:
        self.loader = loader
        self.reader = reader
        self.scale_loader = scale_loader
        self._pgr = pgr
        self._num_lds_buffers = num_lds_buffers

        # Build pipeline from stage declarations
        stages = [
            PipelineStage("G", distance=1, resource="lds",
                          mode="write", wait_counter="vmcnt"),
            PipelineStage("RM", distance=0, resource="lds",
                          mode="read", wait_counter="lgkmcnt"),
        ]
        deps = [StageDep("G", "RM", distance=1)]
        resources = [ResourceConfig("lds", num_buffers=num_lds_buffers)]

        if scale_loader:
            stages.append(
                PipelineStage("S", distance=1, resource=None,
                              wait_counter="vmcnt"))
            deps.append(StageDep("S", "RM", distance=1))

        self.sw_pipeline = SoftwarePipeline(stages, deps, resources, pgr=pgr)

    @property
    def pgr(self) -> int:
        return self.sw_pipeline.pgr

    @property
    def loads_before_reads(self) -> bool:
        return self.sw_pipeline.loads_before_reads

    def emit(self, ctx: AsmContext) -> None:
        tile = ctx._metadata["tile"]
        problem = ctx._metadata["problem"]
        loader = self.loader
        reader = self.reader
        scale_loader = self.scale_loader
        elem = problem.element_bytes
        lds_data_half = int(
            (tile.wg_m + tile.wg_n) * tile.unroll_k * elem)

        ctx._metadata["_reader"] = reader

        # DB step register
        ctx.alloc_sgpr_permanent(1, "s_lds_db_step")
        ctx.s_mov(ctx.sreg("s_lds_db_step"), str(lds_data_half),
                  comment=f"DB step = {lds_data_half}")
        ctx.raw("")

        # Precompute offsets
        loader.precompute_soffsets()
        if scale_loader:
            scale_loader.precompute_soffsets()

        # Build dependency graph + schedule (Level 2)
        graph = KLoopGraph(tile, problem)
        GlobalLoadBlock(loader).register(graph)
        DSReadBlock(reader).register(graph)
        if scale_loader:
            ScaleBlock(scale_loader, tile).register(graph)
        MFMABlock(ctx, tile, scale_loader).register(graph)
        SuffixBlock(reader, scale_loader, loader).register(graph)
        graph.validate()

        schedule = KLoopScheduler(graph).schedule()

        # Build stage emitters
        emitters: Dict[str, StageEmitter] = {
            "G": GlobalLoadStageEmitter(loader, schedule, scale_loader),
            "RM": ReadComputeStageEmitter(
                schedule, reader, loader, scale_loader),
        }

        # Let the pipeline framework drive the loop structure
        self.sw_pipeline.emit_kloop(ctx, emitters)
