"""Partition plan: how the macrotile is divided into scheduling units.

A partition is a rectangle of subtiles (partition_m x nr) processed together.
Each partition's schedule contains:
  - MFMA ops consuming current partition's data ("n")
  - LR (ds_read) loading next partition's data ("n+1")
  - GR (buffer_load) issuing global reads for future iterations

VGPR tiles are reused across partitions via VGPRTileAllocator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..problem import TileConfig

__all__ = [
    "PartitionPlan", "Partition", "VGPRTileAllocator",
]


@dataclass
class Partition:
    """One partition: a rectangle of subtiles processed together."""
    partition_id: int
    # Subtile indices this partition's MFMAs consume
    tile_a_indices: List[int]  # mi values
    tile_b_indices: List[int]  # ni values (same for all partitions when pn=1)
    # What this partition loads for the NEXT partition
    lr_a_targets: List[int] = field(default_factory=list)
    lr_b_targets: List[int] = field(default_factory=list)
    lr_mt_iteration: str = "n"  # "n" = same K-tile, "n+1" = next K-tile (wrap-around)
    # What this partition's GR loads (global reads for future MT iterations)
    gr_a_targets: Set[int] = field(default_factory=set)
    gr_b_targets: Set[int] = field(default_factory=set)
    gr_mt_iteration: str = "n+1"


class _TilePool:
    """Single-matrix free-list tile allocator."""

    def __init__(self):
        self._next_id: int = 0
        self._free: List[int] = []
        self._map: Dict[Tuple[int, int], int] = {}
        self._peak: int = 0

    def allocate(self, subtile_idx: int, sub_iter_k: int) -> int:
        key = (subtile_idx, sub_iter_k)
        if self._free:
            tid = self._free.pop(0)
        else:
            tid = self._next_id
            self._next_id += 1
        self._map[key] = tid
        self._peak = max(self._peak, len(self._map))
        return tid

    def release(self, subtile_idx: int, sub_iter_k: int) -> None:
        key = (subtile_idx, sub_iter_k)
        tid = self._map.pop(key)
        self._free.append(tid)

    def release_all_for_subtile(self, subtile_idx: int) -> None:
        keys = [k for k in self._map if k[0] == subtile_idx]
        for k in keys:
            self._free.append(self._map.pop(k))

    def get(self, subtile_idx: int, sub_iter_k: int) -> int:
        return self._map[(subtile_idx, sub_iter_k)]

    def is_allocated(self, subtile_idx: int, sub_iter_k: int) -> bool:
        return (subtile_idx, sub_iter_k) in self._map


class VGPRTileAllocator:
    """Free-list VGPR tile allocator with separate A/B ID spaces.

    Partition-scoped: tiles allocated for partition N's compute are freed
    after N's MFMAs complete and reused for partition N+1's loads.
    """

    def __init__(self):
        self._a = _TilePool()
        self._b = _TilePool()

    def _pool(self, tc: str) -> _TilePool:
        return self._a if tc == "A" else self._b

    def allocate(self, tc: str, subtile_idx: int, sub_iter_k: int) -> int:
        return self._pool(tc).allocate(subtile_idx, sub_iter_k)

    def release(self, tc: str, subtile_idx: int, sub_iter_k: int) -> None:
        self._pool(tc).release(subtile_idx, sub_iter_k)

    def release_all_for_subtile(self, tc: str, subtile_idx: int) -> None:
        self._pool(tc).release_all_for_subtile(subtile_idx)

    def get(self, tc: str, subtile_idx: int, sub_iter_k: int) -> int:
        return self._pool(tc).get(subtile_idx, sub_iter_k)

    def is_allocated(self, tc: str, subtile_idx: int, sub_iter_k: int) -> bool:
        return self._pool(tc).is_allocated(subtile_idx, sub_iter_k)

    @property
    def peak(self) -> int:
        return self._a._peak + self._b._peak

    @property
    def peak_a(self) -> int:
        return self._a._peak

    @property
    def peak_b(self) -> int:
        return self._b._peak


@dataclass
class PartitionPlan:
    """Complete plan for how the macrotile is partitioned and scheduled.

    Derived from tiling parameters, not hand-specified.
    """
    tile: TileConfig
    partition_m: int  # how many mi indices per partition
    partitions: List[Partition] = field(default_factory=list)
    allocator: VGPRTileAllocator = field(default_factory=VGPRTileAllocator)

    @property
    def num_partitions(self) -> int:
        return len(self.partitions)

    @property
    def ki_count(self) -> int:
        return self.tile.k_iterations

    @property
    def mr(self) -> int:
        return self.tile.mfma_m_repeat

    @property
    def nr(self) -> int:
        return self.tile.mfma_n_repeat

    @staticmethod
    def from_tiling(tile: TileConfig, partition_m: int = 2) -> PartitionPlan:
        """Derive a partition plan from tile config.

        Args:
            tile: Tile configuration (from GemmTiling.to_tile_config()).
            partition_m: Number of mi indices per partition. Must divide mr.
                         E.g., partition_m=2 with mr=8 gives 4 partitions.
        """
        mr = tile.mfma_m_repeat
        nr = tile.mfma_n_repeat
        ki = tile.k_iterations

        if mr % partition_m != 0:
            raise ValueError(
                f"partition_m={partition_m} must divide mr={mr}")

        num_parts = mr // partition_m
        plan = PartitionPlan(tile=tile, partition_m=partition_m)

        # Build partitions
        for pi in range(num_parts):
            mi_start = pi * partition_m
            tile_a = list(range(mi_start, mi_start + partition_m))
            tile_b = list(range(nr))

            # LR targets: load next partition's A subtiles
            # (B is shared across all partitions, only loaded once)
            next_pi = (pi + 1) % num_parts
            next_mi_start = next_pi * partition_m
            lr_a = list(range(next_mi_start, next_mi_start + partition_m))

            # B is loaded in the prologue, not during mainloop partitions.
            # After buffer toggle, B is re-read at the top of the next
            # loop iteration. No partition needs to issue B LR.
            lr_b = []

            # Last partition does NOT do wrap-around LR for A.
            # A for the first partition of the next K-tile is loaded
            # after the postamble toggle, not during the current iteration.
            if next_pi == 0:
                lr_a = []
            lr_mt = "n+1" if next_pi == 0 else "n"

            # GR: DTL loads for next K-tile iteration
            # For simplicity, split evenly across partitions
            # Each partition issues GR for its own subtile range
            gr_a = set(tile_a)
            gr_b_per_part = nr // num_parts
            gr_b_start = pi * gr_b_per_part
            gr_b = set(range(gr_b_start, gr_b_start + gr_b_per_part))
            # Last partition picks up remainder
            if pi == num_parts - 1:
                gr_b = set(range(gr_b_start, nr))

            part = Partition(
                partition_id=pi,
                tile_a_indices=tile_a,
                tile_b_indices=tile_b,
                lr_a_targets=lr_a,
                lr_b_targets=lr_b,
                lr_mt_iteration=lr_mt,
                gr_a_targets=gr_a,
                gr_b_targets=gr_b,
                gr_mt_iteration="n+1",
            )
            plan.partitions.append(part)

        # Allocate initial VGPR tiles for first partition
        first = plan.partitions[0]
        for mi in first.tile_a_indices:
            for sik in range(ki):
                plan.allocator.allocate("A", mi, sik)
        for ni in first.tile_b_indices:
            for sik in range(ki):
                plan.allocator.allocate("B", ni, sik)

        return plan

    def summary(self) -> str:
        lines = [
            f"PartitionPlan: {self.num_partitions} partitions, "
            f"partition_m={self.partition_m}, "
            f"mr={self.mr} nr={self.nr} ki={self.ki_count}",
        ]
        for p in self.partitions:
            mfmas = len(p.tile_a_indices) * len(p.tile_b_indices) * self.ki_count
            lines.append(
                f"  P{p.partition_id}: A{p.tile_a_indices} B{p.tile_b_indices} "
                f"({mfmas} MFMAs) "
                f"LR_A->{p.lr_a_targets} LR_B->{p.lr_b_targets}({p.lr_mt_iteration}) "
                f"GR_A->{p.gr_a_targets} GR_B->{p.gr_b_targets}")
        lines.append(f"  VGPR tiles peak: {self.allocator.peak}")
        return "\n".join(lines)
