"""Auto-pipelined K-loop: derive loop structure from declared stages.

Users declare pipeline stages with dependencies and resources.
The framework auto-derives:
  - Minimum PGR (pipeline depth)
  - Buffer lifecycle (loads_before_reads)
  - Ramp-up, steady-state body, drain structure

Switchable against the manual ScheduledCompute for A/B comparison.

Usage:
    sw = SoftwarePipeline(stages, deps, resources, pgr=1)
    compute = AutoPipelinedCompute(sw, loader, reader, ...)
    compute.emit(ctx)
"""
from __future__ import annotations

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
    "SoftwarePipeline", "AutoPipelinedCompute",
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
# SoftwarePipeline: inter-iteration structure
# ===================================================================

class SoftwarePipeline:
    """Derives K-loop structure from declared stages and dependencies.

    This handles Level 1 (inter-iteration) scheduling:
    - Which stages are active per iteration
    - Buffer lifecycle (loads_before_reads)
    - Ramp-up / steady-state / drain phase counts
    - Stage ordering within the loop body

    Level 2 (intra-iteration instruction scheduling) is delegated
    to KLoopScheduler via the stage emitters.
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        deps: List[StageDep],
        resources: List[ResourceConfig],
        pgr: Optional[int] = None,
    ) -> None:
        self.stages = {s.name: s for s in stages}
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

        # Classify stages into producers (write) and consumers (read)
        self.producer_stages = [
            s for s in stages if s.mode == "write"
        ]
        self.consumer_stages = [
            s for s in stages if s.mode == "read" or s.mode == "none"
        ]

    @property
    def loads_before_reads(self) -> bool:
        return self._loads_before_reads

    @property
    def num_buffers(self) -> int:
        """Number of buffers for the primary shared resource."""
        for r in self.resources.values():
            return r.num_buffers
        return 2

    def _compute_stage_nums(self) -> Dict[str, int]:
        """ASAP stage assignment via longest-path on dependency graph."""
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
        """Check PGR doesn't exceed buffer count for any shared resource."""
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
        """True when PGR < num_buffers for all shared resources."""
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

    def describe(self, num_tiles: int = 8) -> List[str]:
        """Return textual description of the pipeline structure."""
        lines = [
            f"SoftwarePipeline: PGR={self.pgr}, min_pgr={self.min_pgr}, "
            f"loads_before_reads={self.loads_before_reads}",
            f"  Stages: {', '.join(f'{s.name}(d={s.distance})' for s in self.stages.values())}",
            f"  Deps: {', '.join(f'{d.producer}->{d.consumer}(d={d.distance})' for d in self.deps)}",
            "",
        ]

        # Ramp-up
        for p in range(self.pgr):
            producers = [s.name for s in self.producer_stages]
            sync = " + WAIT+BARRIER" if p == 0 else ""
            lines.append(f"  ramp-up[{p}]: {'+'.join(producers)}(tile={p}){sync}")

        # Loop iterations
        for i in range(num_tiles):
            parts = []
            if self.loads_before_reads:
                next_tile = i + self.pgr
                if next_tile < num_tiles:
                    parts.append(f"G({next_tile})")
                else:
                    parts.append("[skip G]")
                parts.append("BARRIER")
                parts.append(f"RM({i})")
            else:
                parts.append("BARRIER")
                parts.append(f"RM({i})")
                parts.append("LGKMCNT(0)")
                next_tile = i + self.pgr
                if next_tile < num_tiles:
                    parts.append(f"G({next_tile})")
                else:
                    parts.append("[skip G]")

            phase = "steady" if i + self.pgr < num_tiles else "drain "
            lines.append(f"  loop[{i:2d}] {phase}: {' '.join(parts)}")

        return lines


# ===================================================================
# AutoPipelinedCompute: generic ComputePipeline using SoftwarePipeline
# ===================================================================

class AutoPipelinedCompute(ComputePipeline):
    """K-loop using SoftwarePipeline for loop structure.

    Drop-in replacement for ScheduledCompute. Uses the same
    KLoopGraph + KLoopScheduler for intra-iteration scheduling,
    but derives the loop structure from the SoftwarePipeline
    instead of hardcoding it.
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
        lds_data_half = int((tile.wg_m + tile.wg_n) * tile.unroll_k * elem)

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

        # Build dependency graph (same as ScheduledCompute)
        graph = KLoopGraph(tile, problem)
        GlobalLoadBlock(loader).register(graph)
        DSReadBlock(reader).register(graph)
        if scale_loader:
            ScaleBlock(scale_loader, tile).register(graph)
        MFMABlock(ctx, tile, scale_loader).register(graph)
        SuffixBlock(reader, scale_loader, loader).register(graph)
        graph.validate()

        schedule = KLoopScheduler(graph).schedule()

        # Emit using auto-derived pipeline structure
        self._emit_ramp_up(ctx, loader, scale_loader, schedule)
        self._emit_loop(ctx, schedule, loader, reader, scale_loader)

    def _emit_ramp_up(self, ctx, loader, scale_loader, schedule):
        """Emit ramp-up stages. Identical logic to ScheduledCompute
        but driven by self.sw_pipeline.pgr."""
        pgr = self.sw_pipeline.pgr

        # Stage 0: load + wait + barrier
        ctx.comment(f"Auto-pipeline ramp-up stage 0/{pgr}: load tile 0")
        loader.emit_loads()
        if schedule and schedule.prologue_scale_ops:
            for op in schedule.prologue_scale_ops:
                if op.emit:
                    op.emit()
            extra = len(schedule.prologue_scale_ops)
            ctx.s_waitcnt(f"vmcnt({extra})",
                          comment=f"wait DTL (leave {extra} scale loads)")
        else:
            ctx.s_waitcnt("vmcnt(0)", comment="wait DTL loads")
        ctx.s_barrier(comment="sync tile 0")
        ctx.raw("")

        # Stages 1..PGR-1: prefetch
        for stage in range(1, pgr):
            ctx.comment(f"Auto-pipeline ramp-up stage {stage}/{pgr}: "
                        f"prefetch tile {stage}")
            ctx.inst("s_cmp_le_u32", ctx.sreg("s_k_tiles"),
                     str(stage),
                     comment=f"skip if k_tiles <= {stage}")
            ctx.inst("s_cbranch_scc1", f"pgr_skip_{stage}",
                     comment=f"not enough tiles for PGR stage {stage}")
            loader.advance()
            loader.toggle_write()
            loader.emit_loads()
            if (schedule and schedule.scale_advance_op
                    and schedule.scale_advance_op.emit):
                schedule.scale_advance_op.emit()
            if (schedule and schedule.prologue_scale_ops and scale_loader
                    and not scale_loader.has_cross_iter_prefetch):
                for op in schedule.prologue_scale_ops:
                    if op.emit:
                        op.emit()
            ctx.label(f"pgr_skip_{stage}")
            ctx.raw("")

    def _emit_global_load_ops(self, ctx, schedule, scale_loader):
        """Emit G stage: advance + toggle + load."""
        for op in schedule.prefetch_ops:
            if op.emit:
                op.emit()
        if schedule.scale_advance_op and schedule.scale_advance_op.emit:
            schedule.scale_advance_op.emit()
        if scale_loader and not scale_loader.has_cross_iter_prefetch:
            for op in schedule.prologue_scale_ops:
                if op.emit:
                    op.emit()

    def _emit_loop(self, ctx, schedule, loader, reader, scale_loader):
        """Emit K-loop body. Structure derived from SoftwarePipeline."""
        tile = ctx._metadata["tile"]
        pgr = self.sw_pipeline.pgr

        ctx.label("k_loop")
        ctx.raw("")

        # The pipeline determines stage ordering:
        # loads_before_reads=True:  G -> BARRIER -> R+M
        # loads_before_reads=False: BARRIER -> R+M -> lgkmcnt(0) -> G
        if self.sw_pipeline.loads_before_reads:
            # --- Producer before consumer ---
            ctx.comment(f"Auto-pipeline body (PGR={pgr}, "
                        f"loads_before_reads=True)")
            for op in schedule.pre_body_ops:
                if op.emit:
                    op.emit()

            ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
                      comment="k_tiles--")
            ctx.inst("s_cmp_gt_u32", ctx.sreg("s_k_tiles"),
                     str(pgr - 1),
                     comment=f"k_tiles > {pgr - 1}?")
            ctx.inst("s_cbranch_scc0", "load_skip_all",
                     comment="skip G (drain phase)")

            self._emit_global_load_ops(ctx, schedule, scale_loader)
            ctx.raw("")
            ctx.label("load_skip_all")
            loader.emit_sync()
            ctx.raw("")
        else:
            # --- Consumer before producer ---
            ctx.comment(f"Auto-pipeline body (PGR={pgr}, "
                        f"loads_before_reads=False)")
            loader.emit_sync()
            ctx.raw("")

            for op in schedule.pre_body_ops:
                if op.emit:
                    op.emit()

        # R+M body (from KLoopScheduler -- Level 2 scheduling)
        self._emit_read_compute(ctx, schedule, reader, loader, scale_loader)

        # Read-before-write: producer AFTER consumer
        if not self.sw_pipeline.loads_before_reads:
            ctx.s_waitcnt("lgkmcnt(0)",
                          comment="wait all ds_reads before overwriting LDS")
            ctx.s_sub(ctx.sreg("s_k_tiles"), ctx.sreg("s_k_tiles"), "1",
                      comment="k_tiles--")
            ctx.inst("s_cmp_gt_u32", ctx.sreg("s_k_tiles"),
                     str(pgr - 1),
                     comment=f"k_tiles > {pgr - 1}?")
            ctx.inst("s_cbranch_scc0", "load_skip_all",
                     comment="skip G (drain phase)")

            self._emit_global_load_ops(ctx, schedule, scale_loader)
            ctx.raw("")
            ctx.label("load_skip_all")

        # Loop branch
        ctx.s_barrier(comment="sync")
        has_pf = (scale_loader
                  and scale_loader.has_cross_iter_prefetch)
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

    def _emit_read_compute(self, ctx, schedule, reader, loader, scale_loader):
        """Emit R+M body (from KLoopScheduler). Identical to ScheduledCompute."""
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
