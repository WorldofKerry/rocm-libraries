"""Partition-based mainloop scheduler.

Builds the K-loop schedule by iterating over partitions. Each partition
contributes MFMA + LR + GR modules. Dependencies are wired across partition
boundaries. The final schedule is a flat list of ScheduleModules that the
SlotPlacer interleaves into MFMA slots.

Pipeline:
  1. build_modules()     -- Create MFMA/LR/GR modules per partition per subIterK
  2. wire_dependencies() -- Add dep edges (WAIT_GR, SYNC, LR before MFMA, etc.)
  3. emit()              -- Produce the interleaved instruction sequence

NGLL/NLL are derived by filtering modules from the mainloop schedule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from .partition import PartitionPlan, Partition, VGPRTileAllocator
from ..problem import TileConfig

__all__ = [
    "MainloopScheduler", "ScheduleModule", "ModuleKind", "DepEdge",
]


class ModuleKind(Enum):
    MFMA = auto()
    LR = auto()        # ds_read (local read)
    GR = auto()        # buffer_load (global read / DTL)
    WAIT_GR = auto()   # s_waitcnt vmcnt
    WAIT_LR = auto()   # s_waitcnt lgkmcnt
    SYNC = auto()       # s_barrier
    GR_INC = auto()     # pointer advance + LDS buffer swap for GR
    LR_INC = auto()     # LDS buffer swap for LR


@dataclass
class DepEdge:
    """Dependency edge between modules.

    Either kind (a sync/housekeeping op to emit) or module_ref
    (ordering constraint on another module).
    """
    kind: Optional[ModuleKind] = None
    emit_fn: Optional[Callable] = None
    module_ref: Optional[ScheduleModule] = None


@dataclass
class ScheduleModule:
    """One logical unit in the schedule.

    Contains emit_fn closures and dependency edges. Modules are the
    scheduling atoms -- internal instruction order is preserved,
    but modules can be reordered relative to each other (subject to deps).
    """
    kind: ModuleKind
    partition_id: int
    sub_iter_k: int = 0
    mt_iteration: str = "n"  # "n", "n+1", "n+2"
    emit_fns: List[Callable] = field(default_factory=list)
    before: List[DepEdge] = field(default_factory=list)  # must happen before this module
    after: List[DepEdge] = field(default_factory=list)    # must happen after this module
    # VGPR tile mappings (for MFMA and LR modules)
    vgpr_a: Dict[int, int] = field(default_factory=dict)  # subtile_idx -> vgpr_tile_id
    vgpr_b: Dict[int, int] = field(default_factory=dict)
    # Metadata
    comment: str = ""

    def __repr__(self):
        return (f"Module({self.kind.name}, P{self.partition_id}, "
                f"sik={self.sub_iter_k}, mt={self.mt_iteration}, "
                f"\"{self.comment}\")")


@dataclass
class PartitionSchedule:
    """Schedule for one partition -- contains subIterK steps."""
    partition_id: int
    sub_iter_k_steps: List[List[ScheduleModule]] = field(default_factory=list)


class MainloopScheduler:
    """Builds the mainloop schedule from a PartitionPlan.

    Usage:
        plan = PartitionPlan.from_tiling(tile, partition_m=2)
        sched = MainloopScheduler(plan)
        sched.build_modules(emit_mfma_fn, emit_lr_fn, emit_gr_fn)
        sched.wire_dependencies()
        modules = sched.mainloop_modules()  # flat list for SlotPlacer
    """

    def __init__(self, plan: PartitionPlan):
        self.plan = plan
        self.partition_schedules: List[PartitionSchedule] = []
        self._all_modules: List[ScheduleModule] = []

    def build_modules(
        self,
        make_mfma_emit: Callable,
        make_lr_emit: Callable,
        make_gr_emit: Callable,
    ) -> None:
        """Build MFMA + LR + GR modules for each partition and subIterK.

        Callback signatures:
            make_mfma_emit(partition, sub_iter_k, vgpr_a, vgpr_b) -> list[emit_fn]
            make_lr_emit(partition, sub_iter_k, lr_a_targets, lr_b_targets,
                         vgpr_a_alloc, vgpr_b_alloc) -> list[emit_fn]
            make_gr_emit(partition, gr_a_targets, gr_b_targets) -> list[emit_fn]
        """
        plan = self.plan
        ki = plan.ki_count
        allocator = plan.allocator

        for part in plan.partitions:
            ps = PartitionSchedule(partition_id=part.partition_id)

            for sik in range(ki):
                step_modules: List[ScheduleModule] = []

                # --- MFMA module ---
                vgpr_a = {mi: allocator.get("A", mi, sik)
                          for mi in part.tile_a_indices}
                vgpr_b = {ni: allocator.get("B", ni, sik)
                          for ni in part.tile_b_indices}

                mfma_emits = make_mfma_emit(part, sik, vgpr_a, vgpr_b)
                mfma_mod = ScheduleModule(
                    kind=ModuleKind.MFMA,
                    partition_id=part.partition_id,
                    sub_iter_k=sik,
                    mt_iteration="n",
                    emit_fns=mfma_emits,
                    vgpr_a=dict(vgpr_a),
                    vgpr_b=dict(vgpr_b),
                    comment=f"MFMA P{part.partition_id} sik={sik}",
                )
                step_modules.append(mfma_mod)

                # --- LR module (load next partition's data) ---
                # Determine what to load and into which VGPR tiles
                # LR for next partition's A subtiles
                lr_vgpr_a = {}
                for mi in part.lr_a_targets:
                    for lsik in range(ki):
                        if not allocator.is_allocated("A", mi, lsik):
                            tid = allocator.allocate("A", mi, lsik)
                        else:
                            tid = allocator.get("A", mi, lsik)
                        lr_vgpr_a[(mi, lsik)] = tid

                # LR for B (only when wrapping around)
                lr_vgpr_b = {}
                for ni in part.lr_b_targets:
                    for lsik in range(ki):
                        if not allocator.is_allocated("B", ni, lsik):
                            tid = allocator.allocate("B", ni, lsik)
                        else:
                            tid = allocator.get("B", ni, lsik)
                        lr_vgpr_b[(ni, lsik)] = tid

                if part.lr_a_targets or part.lr_b_targets:
                    lr_emits = make_lr_emit(
                        part, sik,
                        part.lr_a_targets, part.lr_b_targets,
                        lr_vgpr_a, lr_vgpr_b)
                    lr_mod = ScheduleModule(
                        kind=ModuleKind.LR,
                        partition_id=part.partition_id,
                        sub_iter_k=sik,
                        mt_iteration=part.lr_mt_iteration,
                        emit_fns=lr_emits,
                        vgpr_a={k[0]: v for k, v in lr_vgpr_a.items()},
                        vgpr_b={k[0]: v for k, v in lr_vgpr_b.items()},
                        comment=f"LR P{part.partition_id} sik={sik} "
                                f"A->{part.lr_a_targets} B->{part.lr_b_targets}",
                    )
                    step_modules.append(lr_mod)

                # --- GR module (only on sik=0 to avoid over-issuing) ---
                if sik == 0 and (part.gr_a_targets or part.gr_b_targets):
                    gr_emits = make_gr_emit(
                        part, part.gr_a_targets, part.gr_b_targets)
                    gr_mod = ScheduleModule(
                        kind=ModuleKind.GR,
                        partition_id=part.partition_id,
                        sub_iter_k=0,
                        mt_iteration=part.gr_mt_iteration,
                        emit_fns=gr_emits,
                        comment=f"GR P{part.partition_id} "
                                f"A->{part.gr_a_targets} B->{part.gr_b_targets}",
                    )
                    step_modules.append(gr_mod)

                ps.sub_iter_k_steps.append(step_modules)

            # Release VGPR tiles for this partition's A subtiles
            # (they can be reused by next partition's LR)
            for mi in part.tile_a_indices:
                allocator.release_all_for_subtile("A", mi)

            self.partition_schedules.append(ps)

    def wire_dependencies(self) -> None:
        """Wire dependency edges between modules across partitions.

        Rules:
          - LR(n+1) depends on WAIT_GR for the data it reads from LDS
          - MFMA depends on WAIT_LR for the data it consumes from VGPRs
          - GR_INC after last GR for an MT iteration
          - LR_INC after last LR for an MT iteration
          - SYNC between GR completion and LR start (barrier)
        """
        # Collect all modules in flat order
        all_mods = self.mainloop_modules()

        # Wire: each LR module gets a WAIT_LR after it
        for mod in all_mods:
            if mod.kind == ModuleKind.LR:
                mod.after.append(DepEdge(kind=ModuleKind.WAIT_LR))

        # Wire: each MFMA module depends on its LR being complete
        # (the LR that loaded this partition's data)
        lr_by_partition: Dict[int, ScheduleModule] = {}
        for mod in all_mods:
            if mod.kind == ModuleKind.LR:
                lr_by_partition[mod.partition_id] = mod

        # Wire: GR depends on prior barrier (can't DTL-write while reads in progress)
        # SYNC after all LRs for current buffer, before GR starts writing new data
        gr_mods = [m for m in all_mods if m.kind == ModuleKind.GR]
        for gr in gr_mods:
            gr.before.append(DepEdge(kind=ModuleKind.SYNC))

    def mainloop_modules(self) -> List[ScheduleModule]:
        """Flatten all partition schedules into a linear module list."""
        result = []
        for ps in self.partition_schedules:
            for step in ps.sub_iter_k_steps:
                result.extend(step)
        return result

    def mfma_modules(self) -> List[ScheduleModule]:
        """All MFMA modules in order."""
        return [m for m in self.mainloop_modules() if m.kind == ModuleKind.MFMA]

    def lr_modules(self) -> List[ScheduleModule]:
        """All LR modules in order."""
        return [m for m in self.mainloop_modules() if m.kind == ModuleKind.LR]

    def gr_modules(self) -> List[ScheduleModule]:
        """All GR modules in order."""
        return [m for m in self.mainloop_modules() if m.kind == ModuleKind.GR]

    def derive_ngll(self) -> List[ScheduleModule]:
        """NGLL: mainloop without GR(n+2) and GR_INC."""
        return [m for m in self.mainloop_modules()
                if not (m.kind == ModuleKind.GR and m.mt_iteration == "n+2")
                and m.kind != ModuleKind.GR_INC]

    def derive_nll(self) -> List[ScheduleModule]:
        """NLL: mainloop without any GR, LR(n+1), and associated ops."""
        return [m for m in self.mainloop_modules()
                if m.kind == ModuleKind.MFMA
                or (m.kind == ModuleKind.LR and m.mt_iteration == "n")]

    def summary(self) -> str:
        mods = self.mainloop_modules()
        by_kind = {}
        for m in mods:
            by_kind[m.kind.name] = by_kind.get(m.kind.name, 0) + 1
        lines = [f"MainloopScheduler: {len(mods)} modules, "
                 f"{len(self.partition_schedules)} partitions"]
        for k, v in sorted(by_kind.items()):
            lines.append(f"  {k}: {v}")
        total_mfma_fns = sum(len(m.emit_fns) for m in mods if m.kind == ModuleKind.MFMA)
        total_lr_fns = sum(len(m.emit_fns) for m in mods if m.kind == ModuleKind.LR)
        total_gr_fns = sum(len(m.emit_fns) for m in mods if m.kind == ModuleKind.GR)
        lines.append(f"  Total emit_fns: {total_mfma_fns} MFMA, {total_lr_fns} LR, {total_gr_fns} GR")
        lines.append(f"  VGPR tile peak: {self.plan.allocator.peak}")
        return "\n".join(lines)
