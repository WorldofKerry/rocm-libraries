"""Prototype: three approaches to software-pipelined K-loop.

Compare Option A (separate types), B (unified graph), C (hierarchical)
for PGR=0, 1, 2 with configurable buffer counts.

No assembly -- just derives the iteration structure as text.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ======================================================================
# Shared types
# ======================================================================

@dataclass(frozen=True)
class PipelineStage:
    """A coarse-grained pipeline stage."""
    name: str
    resource: Optional[str] = None   # shared resource (e.g. "lds_buf")
    mode: str = "none"               # "read", "write", or "none"

@dataclass(frozen=True)
class StageDep:
    """Inter-iteration dependency between stages."""
    producer: str
    consumer: str
    distance: int = 1    # minimum iterations of separation

@dataclass(frozen=True)
class ResourceConfig:
    """Buffer count for a shared resource."""
    name: str
    num_buffers: int = 2


# ======================================================================
# Option A: Separate types (two-level scheduling)
# ======================================================================

class OptionA_Pipeline:
    """Pipeline scheduling as a separate layer from instruction scheduling.

    Computes inter-iteration structure only. The intra-iteration
    schedule (how ops within R+M are ordered) is delegated to a
    separate scheduler (KLoopScheduler).
    """

    def __init__(self, stages: List[PipelineStage],
                 deps: List[StageDep],
                 resources: Dict[str, int],
                 pgr: Optional[int] = None):
        self.stages = {s.name: s for s in stages}
        self.deps = deps
        self.resources = resources

        # Step 1: compute stage numbers (ASAP)
        self.stage_num = self._compute_stage_nums()
        self.min_pgr = max(self.stage_num.values()) if self.stage_num else 0

        # Step 2: resolve PGR
        self.pgr = pgr if pgr is not None else self.min_pgr
        if self.pgr < self.min_pgr:
            raise ValueError(
                f"PGR={self.pgr} < min_pgr={self.min_pgr} "
                f"(pipeline would stall)")

        # Step 3: validate buffer constraints
        self._validate_buffers()

        # Step 4: determine ordering
        self.write_stages = {n for n, s in self.stages.items() if s.mode == "write"}
        self.read_stages = {n for n, s in self.stages.items() if s.mode == "read"}

    def _compute_stage_nums(self) -> Dict[str, int]:
        """ASAP stage assignment via longest-path."""
        nums: Dict[str, int] = {name: 0 for name in self.stages}
        changed = True
        while changed:
            changed = False
            for dep in self.deps:
                new_val = nums[dep.producer] + dep.distance
                if new_val > nums[dep.consumer]:
                    nums[dep.consumer] = new_val
                    changed = True
        return nums

    def _validate_buffers(self):
        """Check PGR doesn't exceed buffer count for any resource."""
        for dep in self.deps:
            prod = self.stages[dep.producer]
            cons = self.stages[dep.consumer]
            if prod.resource and prod.resource == cons.resource:
                if prod.mode == "write" and cons.mode == "read":
                    bufs = self.resources.get(prod.resource, 2)
                    if self.pgr > bufs:
                        raise ValueError(
                            f"PGR={self.pgr} > {bufs} buffers for "
                            f"resource '{prod.resource}'")

    @property
    def loads_before_reads(self) -> bool:
        for dep in self.deps:
            prod = self.stages[dep.producer]
            cons = self.stages[dep.consumer]
            if (prod.resource and prod.resource == cons.resource
                    and prod.mode == "write" and cons.mode == "read"):
                bufs = self.resources.get(prod.resource, 2)
                if self.pgr >= bufs:
                    return False
        return True

    def generate(self, num_tiles: int) -> List[str]:
        """Generate textual representation of each iteration."""
        lines = []
        pgr = self.pgr

        # Group stages by stage_num for ordering
        groups: Dict[int, List[str]] = {}
        for name, num in self.stage_num.items():
            groups.setdefault(num, []).append(name)

        # Ramp-up: emit write-stage-only iterations
        for p in range(pgr):
            active = []
            for name in sorted(groups.get(0, [])):
                if self.stages[name].mode == "write":
                    active.append(f"{name}(tile={p})")
            sync = " + WAIT+BARRIER" if p == 0 else ""
            lines.append(f"  ramp-up[{p}]: {' '.join(active)}{sync}")

        # Main loop: T iterations
        for i in range(num_tiles):
            active = []
            tile_for_stage = {}

            if self.loads_before_reads:
                # Write stages first (if not in drain)
                for sn in sorted(groups.get(0, [])):
                    t = i + pgr
                    if t < num_tiles:
                        active.append(f"{sn}({t})")
                        tile_for_stage[sn] = t
                if any(sn in tile_for_stage for sn in groups.get(0, [])):
                    active.append("SYNC")
                # Then read+compute stages
                for stage_n in sorted(groups.keys()):
                    if stage_n == 0:
                        continue
                    for sn in sorted(groups[stage_n]):
                        active.append(f"{sn}({i})")
            else:
                # Read+compute stages first
                active.append("SYNC")
                for stage_n in sorted(groups.keys()):
                    if stage_n == 0:
                        continue
                    for sn in sorted(groups[stage_n]):
                        active.append(f"{sn}({i})")
                # Then write stages (after reads done)
                active.append("LGKMCNT(0)")
                for sn in sorted(groups.get(0, [])):
                    t = i + pgr
                    if t < num_tiles:
                        active.append(f"{sn}({t})")

            phase = "steady" if i + pgr < num_tiles else "drain "
            lines.append(f"  loop[{i:2d}] {phase}: {' '.join(active)}")

        return lines


# ======================================================================
# Option B: Unified graph (single-level scheduling)
# ======================================================================

class OptionB_Unified:
    """All ops in a single graph with pipeline_stage field.

    Each op knows which pipeline stage it belongs to. The scheduler
    handles both inter- and intra-iteration ordering in one pass.
    """

    @dataclass
    class Op:
        name: str
        stage: str           # pipeline stage name
        pipeline_num: int    # derived stage number
        tile_offset: int     # relative to current tile (0=current, +N=future)

    def __init__(self, stages: List[PipelineStage],
                 deps: List[StageDep],
                 resources: Dict[str, int],
                 pgr: Optional[int] = None):
        self.stages = {s.name: s for s in stages}
        self.deps = deps
        self.resources = resources

        # Compute stage nums (same as Option A)
        self.stage_num = {}
        for s in stages:
            self.stage_num[s.name] = 0
        changed = True
        while changed:
            changed = False
            for dep in deps:
                nv = self.stage_num[dep.producer] + dep.distance
                if nv > self.stage_num[dep.consumer]:
                    self.stage_num[dep.consumer] = nv
                    changed = True
        self.min_pgr = max(self.stage_num.values()) if self.stage_num else 0
        self.pgr = pgr if pgr is not None else self.min_pgr
        if self.pgr < self.min_pgr:
            raise ValueError(f"PGR={self.pgr} < min_pgr={self.min_pgr}")

        # Validate buffers
        for dep in deps:
            p, c = self.stages[dep.producer], self.stages[dep.consumer]
            if (p.resource and p.resource == c.resource
                    and p.mode == "write" and c.mode == "read"):
                bufs = resources.get(p.resource, 2)
                if self.pgr > bufs:
                    raise ValueError(f"PGR={self.pgr} > {bufs} buffers")

    def generate(self, num_tiles: int) -> List[str]:
        """Generate iteration structure.

        In the unified model, each iteration contains ops from
        MULTIPLE pipeline stages (different tiles). The scheduler
        sees all ops together and orders them respecting both
        intra-iteration and inter-iteration deps.
        """
        lines = []
        pgr = self.pgr

        # Build per-iteration op lists
        # Each iteration i has:
        #   - compute ops for tile i (stages with stage_num > 0)
        #   - prefetch ops for tile i+pgr (stages with stage_num == 0)

        # Check buffer ordering
        loads_before = True
        for dep in self.deps:
            p, c = self.stages[dep.producer], self.stages[dep.consumer]
            if (p.resource and p.resource == c.resource
                    and p.mode == "write" and c.mode == "read"):
                if self.pgr >= self.resources.get(p.resource, 2):
                    loads_before = False

        # Ramp-up
        for p in range(pgr):
            ops = [f"{s}(tile={p})" for s in sorted(self.stage_num)
                   if self.stage_num[s] == 0
                   and self.stages[s].mode == "write"]
            sync = " + WAIT+BARRIER" if p == 0 else ""
            lines.append(f"  ramp-up[{p}]: {' '.join(ops)}{sync}")

        # Main loop -- unified: all ops visible to scheduler
        for i in range(num_tiles):
            all_ops = []
            prefetch_tile = i + pgr

            if loads_before:
                for s in sorted(self.stage_num):
                    if self.stage_num[s] == 0 and prefetch_tile < num_tiles:
                        all_ops.append(f"{s}({prefetch_tile})")
                if prefetch_tile < num_tiles:
                    all_ops.append("SYNC")
                for sn in sorted(set(self.stage_num.values())):
                    if sn == 0:
                        continue
                    for s in sorted(self.stage_num):
                        if self.stage_num[s] == sn:
                            all_ops.append(f"{s}({i})")
            else:
                all_ops.append("SYNC")
                for sn in sorted(set(self.stage_num.values())):
                    if sn == 0:
                        continue
                    for s in sorted(self.stage_num):
                        if self.stage_num[s] == sn:
                            all_ops.append(f"{s}({i})")
                all_ops.append("LGKMCNT(0)")
                for s in sorted(self.stage_num):
                    if self.stage_num[s] == 0 and prefetch_tile < num_tiles:
                        all_ops.append(f"{s}({prefetch_tile})")

            phase = "steady" if prefetch_tile < num_tiles else "drain "
            # Mark ops with [stage=N] for visibility
            lines.append(f"  loop[{i:2d}] {phase}: {' '.join(all_ops)}")

        return lines


# ======================================================================
# Option C: Hierarchical (stages contain sub-schedules)
# ======================================================================

class OptionC_Hierarchical:
    """PipelineStages are containers. Each has a sub-schedule of fine ops.

    The pipeline framework decides stage ordering.
    Within each stage, a sub-scheduler decides op ordering.
    """

    @dataclass
    class StageInstance:
        """A stage bound to a specific tile."""
        stage: PipelineStage
        tile: int
        sub_ops: List[str] = field(default_factory=list)

    def __init__(self, stages: List[PipelineStage],
                 deps: List[StageDep],
                 resources: Dict[str, int],
                 pgr: Optional[int] = None,
                 sub_ops: Optional[Dict[str, List[str]]] = None):
        self.stages = {s.name: s for s in stages}
        self.deps = deps
        self.resources = resources
        self.sub_ops = sub_ops or {}

        # Same ASAP computation
        self.stage_num = {}
        for s in stages:
            self.stage_num[s.name] = 0
        changed = True
        while changed:
            changed = False
            for dep in deps:
                nv = self.stage_num[dep.producer] + dep.distance
                if nv > self.stage_num[dep.consumer]:
                    self.stage_num[dep.consumer] = nv
                    changed = True
        self.min_pgr = max(self.stage_num.values()) if self.stage_num else 0
        self.pgr = pgr if pgr is not None else self.min_pgr
        if self.pgr < self.min_pgr:
            raise ValueError(f"PGR={self.pgr} < min_pgr={self.min_pgr}")

        for dep in deps:
            p, c = self.stages[dep.producer], self.stages[dep.consumer]
            if (p.resource and p.resource == c.resource
                    and p.mode == "write" and c.mode == "read"):
                bufs = resources.get(p.resource, 2)
                if self.pgr > bufs:
                    raise ValueError(f"PGR={self.pgr} > {bufs} buffers")

    def generate(self, num_tiles: int) -> List[str]:
        lines = []
        pgr = self.pgr

        loads_before = True
        for dep in self.deps:
            p, c = self.stages[dep.producer], self.stages[dep.consumer]
            if (p.resource and p.resource == c.resource
                    and p.mode == "write" and c.mode == "read"):
                if self.pgr >= self.resources.get(p.resource, 2):
                    loads_before = False

        # Ramp-up
        for p in range(pgr):
            parts = []
            for s in sorted(self.stage_num):
                if self.stage_num[s] == 0 and self.stages[s].mode == "write":
                    sub = self.sub_ops.get(s, [])
                    sub_str = f"[{','.join(sub)}]" if sub else ""
                    parts.append(f"{s}({p}){sub_str}")
            sync = " + WAIT+BARRIER" if p == 0 else ""
            lines.append(f"  ramp-up[{p}]: {' '.join(parts)}{sync}")

        # Main loop
        for i in range(num_tiles):
            parts = []
            pf = i + pgr

            if loads_before:
                for s in sorted(self.stage_num):
                    if self.stage_num[s] == 0 and pf < num_tiles:
                        sub = self.sub_ops.get(s, [])
                        sub_str = f"[{','.join(sub)}]" if sub else ""
                        parts.append(f"{s}({pf}){sub_str}")
                if pf < num_tiles:
                    parts.append("SYNC")
                for sn in sorted(set(self.stage_num.values())):
                    if sn == 0:
                        continue
                    for s in sorted(self.stage_num):
                        if self.stage_num[s] == sn:
                            sub = self.sub_ops.get(s, [])
                            sub_str = f"[{','.join(sub)}]" if sub else ""
                            parts.append(f"{s}({i}){sub_str}")
            else:
                parts.append("SYNC")
                for sn in sorted(set(self.stage_num.values())):
                    if sn == 0:
                        continue
                    for s in sorted(self.stage_num):
                        if self.stage_num[s] == sn:
                            sub = self.sub_ops.get(s, [])
                            sub_str = f"[{','.join(sub)}]" if sub else ""
                            parts.append(f"{s}({i}){sub_str}")
                parts.append("LGKMCNT(0)")
                for s in sorted(self.stage_num):
                    if self.stage_num[s] == 0 and pf < num_tiles:
                        sub = self.sub_ops.get(s, [])
                        sub_str = f"[{','.join(sub)}]" if sub else ""
                        parts.append(f"{s}({pf}){sub_str}")

            phase = "steady" if pf < num_tiles else "drain "
            lines.append(f"  loop[{i:2d}] {phase}: {' '.join(parts)}")

        return lines


# ======================================================================
# Compare all three
# ======================================================================

def compare(num_tiles: int = 8):
    stages = [
        PipelineStage("G", resource="lds", mode="write"),
        PipelineStage("R", resource="lds", mode="read"),
        PipelineStage("M"),
    ]
    deps = [
        StageDep("G", "R", distance=1),
        StageDep("R", "M", distance=0),
    ]
    sub_ops = {
        "G": ["dtl_load_a", "dtl_load_b"],
        "R": ["ds_read_b", "ds_read_a"],
        "M": ["mfma_x128"],
    }

    for pgr in [0, 1, 2]:
        for bufs in [2, 3]:
            print(f"\n{'='*60}")
            print(f"PGR={pgr}, buffers={bufs}, tiles={num_tiles}")
            print(f"{'='*60}")

            resources = {"lds": bufs}

            for label, Cls, extra in [
                ("A (separate)", OptionA_Pipeline, {}),
                ("B (unified)",  OptionB_Unified,  {}),
                ("C (hierarchical)", OptionC_Hierarchical, {"sub_ops": sub_ops}),
            ]:
                try:
                    p = Cls(stages, deps, resources, pgr=pgr, **extra)
                    info = (f"stage_nums={p.stage_num}, "
                            f"min_pgr={p.min_pgr}")
                    if hasattr(p, 'loads_before_reads'):
                        info += f", loads_before={p.loads_before_reads}"
                    else:
                        info += f", loads_before={'?'}"
                    print(f"\n--- {label} ({info}) ---")
                    for line in p.generate(num_tiles):
                        print(line)
                except ValueError as e:
                    print(f"\n--- {label} ---")
                    print(f"  REJECTED: {e}")


if __name__ == "__main__":
    compare(num_tiles=6)


# ======================================================================
# Option D: MLIR-style (stage annotation + peeling transform)
# ======================================================================

class OptionD_MLIR:
    """MLIR's createLoopPipeliningPass approach.

    Each op gets a stage number via a callback. The transform:
    1. Peels (num_stages - 1) ramp-up iterations from the top
    2. The steady-state body has ALL stages active
    3. Peels (num_stages - 1) drain iterations from the bottom
    4. Uses predicates (not conditionals) to mask inactive stages

    Key difference from A/B/C: the ORIGINAL loop body is untouched.
    The transform just replicates it with stage predicates. No
    separate "load path" vs "compute path" -- same body, different
    predicates per iteration.

    Reference: mlir/lib/Dialect/SCF/Transforms/LoopPipelining.cpp
    """

    def __init__(self, stages: list, deps: list, resources: dict,
                 pgr: int = None):
        self.stages = {s.name: s for s in stages}
        self.deps = deps
        self.resources = resources

        # MLIR uses getSchedule callback to assign stage nums
        self.stage_num = {}
        for s in stages:
            self.stage_num[s.name] = 0
        changed = True
        while changed:
            changed = False
            for dep in deps:
                nv = self.stage_num[dep.producer] + dep.distance
                if nv > self.stage_num[dep.consumer]:
                    self.stage_num[dep.consumer] = nv
                    changed = True

        self.num_stages = max(self.stage_num.values()) + 1 if self.stage_num else 1
        self.min_pgr = self.num_stages - 1
        self.pgr = pgr if pgr is not None else self.min_pgr
        if self.pgr < self.min_pgr:
            raise ValueError(f"PGR={self.pgr} < min_pgr={self.min_pgr}")

        # Buffer validation
        for dep in deps:
            p, c = self.stages[dep.producer], self.stages[dep.consumer]
            if (p.resource and p.resource == c.resource
                    and p.mode == "write" and c.mode == "read"):
                bufs = resources.get(p.resource, 2)
                if self.pgr > bufs:
                    raise ValueError(f"PGR={self.pgr} > {bufs} buffers")

    def generate(self, num_tiles: int) -> list:
        """MLIR-style: peel ramp-up, emit body with predicates, peel drain.

        In MLIR, predicates are SSA values. Here we show which stages
        are active per iteration. The KEY insight: the body code is
        the SAME in every iteration -- only the predicate differs.
        """
        lines = []
        pgr = self.pgr
        ns = self.num_stages

        # In MLIR, the original loop runs from 0 to T-1.
        # After pipelining:
        #   Ramp-up:  pgr peeled iterations (stages progressively added)
        #   Body:     T - pgr iterations (all stages active)
        #   Drain:    handled by predicating stages in the last pgr-1
        #             body iterations OR by peeling drain iterations
        #
        # MLIR peels ramp-up but keeps drain inside the loop with predicates.
        # For simplicity, we show both as explicit iterations.

        all_stages = sorted(self.stage_num.keys(),
                            key=lambda s: self.stage_num[s])

        # Every iteration executes the SAME body template.
        # The predicate for stage s in iteration i:
        #   stage s is active iff:
        #     tile_for_s = i - stage_num[s]  (which tile this stage works on)
        #     0 <= tile_for_s < num_tiles
        total_iters = num_tiles + pgr  # ramp-up + main

        for i in range(total_iters):
            active = []
            for s in all_stages:
                tile = i - self.stage_num[s]
                # Predicate: is this tile valid?
                if 0 <= tile < num_tiles:
                    active.append(f"{s}({tile})")
                else:
                    active.append(f"--")  # predicated off

            if i < pgr:
                phase = "ramp  "
            elif i >= num_tiles:
                phase = "drain "
            else:
                phase = "steady"

            pred_str = " ".join(active)
            lines.append(f"  iter[{i:2d}] {phase}: {pred_str}")

        return lines


# ======================================================================
# Option E: CUTLASS-style (producer/consumer with explicit barriers)
# ======================================================================

class OptionE_CUTLASS:
    """CUTLASS's PipelineAsync / PipelineTmaAsync pattern.

    Each stage is a producer or consumer of a shared resource (smem).
    Explicit acquire/release barriers control buffer lifecycle.

    Key concepts:
    - num_stages buffers form a circular queue
    - Producer: acquire(write, buf_idx) → produce → commit(buf_idx)
    - Consumer: wait(read, buf_idx) → consume → release(buf_idx)
    - buf_idx = iteration % num_stages

    The pipeline state machine tracks which buffers are:
    - empty (available for producer)
    - full (ready for consumer)
    - in-use (being consumed)

    Reference: cutlass/include/cutlass/pipeline/
    """

    def __init__(self, stages: list, deps: list, resources: dict,
                 pgr: int = None):
        self.stages = {s.name: s for s in stages}
        self.deps = deps
        self.resources = resources

        self.stage_num = {}
        for s in stages:
            self.stage_num[s.name] = 0
        changed = True
        while changed:
            changed = False
            for dep in deps:
                nv = self.stage_num[dep.producer] + dep.distance
                if nv > self.stage_num[dep.consumer]:
                    self.stage_num[dep.consumer] = nv
                    changed = True

        self.min_pgr = max(self.stage_num.values()) if self.stage_num else 0
        self.pgr = pgr if pgr is not None else self.min_pgr
        if self.pgr < self.min_pgr:
            raise ValueError(f"PGR={self.pgr} < min_pgr={self.min_pgr}")

        for dep in deps:
            p, c = self.stages[dep.producer], self.stages[dep.consumer]
            if (p.resource and p.resource == c.resource
                    and p.mode == "write" and c.mode == "read"):
                bufs = resources.get(p.resource, 2)
                if self.pgr > bufs:
                    raise ValueError(f"PGR={self.pgr} > {bufs} buffers")

    def generate(self, num_tiles: int) -> list:
        """CUTLASS-style: explicit buffer acquire/release protocol.

        Shows buffer index and acquire/release barriers.
        """
        lines = []
        pgr = self.pgr
        num_bufs = max(self.resources.values()) if self.resources else 2

        # Producer (G) runs pgr tiles ahead of consumer (R+M)
        # Producer and consumer each track their own buffer index

        # Ramp-up: producer fills pgr buffers
        for p in range(pgr):
            buf = p % num_bufs
            lines.append(
                f"  ramp-up[{p}]: "
                f"acquire_write(buf={buf}) → G({p}) → commit(buf={buf})"
                + (" → WAIT+BARRIER" if p == 0 else ""))

        # Main loop
        for i in range(num_tiles):
            parts = []
            read_buf = i % num_bufs
            write_tile = i + pgr
            write_buf = write_tile % num_bufs if write_tile < num_tiles else None

            # Consumer side
            parts.append(f"wait_read(buf={read_buf})")
            for s in sorted(self.stage_num):
                if self.stage_num[s] > 0:
                    parts.append(f"{s}({i})")
            parts.append(f"release(buf={read_buf})")

            # Producer side (if tiles remain)
            if write_buf is not None:
                parts.append(f"acquire_write(buf={write_buf})")
                for s in sorted(self.stage_num):
                    if self.stage_num[s] == 0 and self.stages[s].mode == "write":
                        parts.append(f"{s}({write_tile})")
                parts.append(f"commit(buf={write_buf})")

            phase = "steady" if write_buf is not None else "drain "
            lines.append(f"  loop[{i:2d}] {phase}: {' '.join(parts)}")

        return lines


# ======================================================================
# Option F: Rau's Modulo Scheduling (fine-grained, II-based)
# ======================================================================

class OptionF_ModuloSchedule:
    """Rau's modulo scheduling applied at the coarse stage level.

    Each stage has a schedule_time. II (initiation interval) = 1 iteration.
    stage_assignment = schedule_time // II.

    The modulo schedule kernel is the set of ops from all active stages
    in one II window. Ramp-up and drain are derived from stage assignments.

    This is the theoretical foundation -- at coarse granularity it
    produces the same result as Option A, but the formulation makes
    the II and stage concepts explicit.

    Reference: Rau, "Iterative Modulo Scheduling" (1994)
    """

    def __init__(self, stages: list, deps: list, resources: dict,
                 pgr: int = None):
        self.stages = {s.name: s for s in stages}
        self.deps = deps
        self.resources = resources

        # II = 1 (one tile per initiation)
        self.II = 1

        # Schedule time for each stage (ASAP, same as stage_num)
        self.sched_time = {}
        for s in stages:
            self.sched_time[s.name] = 0
        changed = True
        while changed:
            changed = False
            for dep in deps:
                nv = self.sched_time[dep.producer] + dep.distance
                if nv > self.sched_time[dep.consumer]:
                    self.sched_time[dep.consumer] = nv
                    changed = True

        # Stage assignment: stage = sched_time // II
        self.stage_of = {s: t // self.II for s, t in self.sched_time.items()}
        self.num_stages = max(self.stage_of.values()) + 1 if self.stage_of else 1

        # User can increase pipeline depth beyond minimum
        min_pgr = self.num_stages - 1
        self.pgr = pgr if pgr is not None else min_pgr
        if self.pgr < min_pgr:
            raise ValueError(f"PGR={self.pgr} < min required {min_pgr}")

        # If user requests more stages, push early stages earlier
        if self.pgr > min_pgr:
            extra = self.pgr - min_pgr
            for s in self.stage_of:
                if self.stage_of[s] == 0:
                    pass  # stays at 0
                else:
                    self.stage_of[s] += extra
            self.num_stages = max(self.stage_of.values()) + 1

        # Buffer validation
        for dep in deps:
            p, c = self.stages[dep.producer], self.stages[dep.consumer]
            if (p.resource and p.resource == c.resource
                    and p.mode == "write" and c.mode == "read"):
                bufs = resources.get(p.resource, 2)
                if self.pgr > bufs:
                    raise ValueError(f"PGR={self.pgr} > {bufs} buffers")

        # RecMII: recurrence minimum II
        # For us: write→read on same resource with distance d
        # RecMII = ceil(latency / distance) -- always 1 for our case
        self.RecMII = 1

        # ResMII: resource minimum II
        # Each resource can serve 1 write + 1 read per II → ResMII = 1
        self.ResMII = 1

        self.actual_II = max(self.RecMII, self.ResMII)

    def generate(self, num_tiles: int) -> list:
        """Modulo schedule: show the kernel with stage annotations.

        The 'kernel' is one II window containing ops from all stages.
        Ramp-up = num_stages - 1 iterations before kernel is fully active.
        Drain = num_stages - 1 iterations after last initiation.
        """
        lines = []
        ns = self.num_stages

        lines.append(f"  II={self.actual_II}, "
                     f"stages={self.stage_of}, "
                     f"num_stages={ns}")
        lines.append(f"  RecMII={self.RecMII}, ResMII={self.ResMII}")
        lines.append("")

        all_stages_sorted = sorted(self.stage_of.keys(),
                                   key=lambda s: self.stage_of[s])

        # Total iterations = T + (num_stages - 1)
        # Iteration j initiates tile j (if j < T)
        total = num_tiles + ns - 1

        for j in range(total):
            active = []
            for s in all_stages_sorted:
                st = self.stage_of[s]
                # In iteration j, stage s works on tile (j - st)
                tile = j - st
                if 0 <= tile < num_tiles:
                    active.append(f"{s}({tile})[s{st}]")
                else:
                    active.append(f"{'--':>8}")

            if j < ns - 1:
                phase = "ramp  "
            elif j >= num_tiles:
                phase = "drain "
            else:
                phase = "kernel"

            lines.append(f"  iter[{j:2d}] {phase}: {' '.join(active)}")

        return lines


# ======================================================================
# Updated comparison
# ======================================================================

def compare_all(num_tiles: int = 6):
    stages = [
        PipelineStage("G", resource="lds", mode="write"),
        PipelineStage("R", resource="lds", mode="read"),
        PipelineStage("M"),
    ]
    deps = [
        StageDep("G", "R", distance=1),
        StageDep("R", "M", distance=0),
    ]

    options = [
        ("A: Separate",     OptionA_Pipeline),
        ("B: Unified",      OptionB_Unified),
        ("C: Hierarchical", OptionC_Hierarchical),
        ("D: MLIR-style",   OptionD_MLIR),
        ("E: CUTLASS-style", OptionE_CUTLASS),
        ("F: Modulo Sched", OptionF_ModuloSchedule),
    ]

    for pgr in [0, 1, 2]:
        bufs = 2
        print(f"\n{'='*70}")
        print(f"PGR={pgr}, buffers={bufs}, tiles={num_tiles}")
        print(f"{'='*70}")

        resources = {"lds": bufs}

        for label, Cls in options:
            try:
                if Cls == OptionC_Hierarchical:
                    p = Cls(stages, deps, resources, pgr=pgr,
                            sub_ops={"G": ["dtl_a","dtl_b"],
                                     "R": ["ds_rd_b","ds_rd_a"],
                                     "M": ["mfma"]})
                else:
                    p = Cls(stages, deps, resources, pgr=pgr)
                print(f"\n--- {label} ---")
                for line in p.generate(num_tiles):
                    print(line)
            except ValueError as e:
                print(f"\n--- {label} ---")
                print(f"  REJECTED: {e}")


if __name__ == "__main__":
    compare_all(num_tiles=6)
