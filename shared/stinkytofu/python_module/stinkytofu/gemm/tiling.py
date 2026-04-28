# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""General recursive tiling via per-dimension TileDim chains.

Replaces the hardcoded ``wg_m``/``waves_m``/``mfma_m_repeat`` parameters
in ``TileConfig`` with a composable per-dimension hierarchy.  Each
dimension (M, N, K) has its own chain of splits with explicit
scheduling metadata (parallel / sequential / hardware).

The ``GemmTiling`` class composes per-dimension chains and derives
everything that ``TileConfig`` used to provide -- plus auto-generates
``TileLevel`` trees and ``TileDescriptor`` chains from the same source.

Usage::

    from stinkytofu.gemm.tiling import GemmTiling, TileDim, S

    # Standard GEMM tiling (equivalent to TileConfig defaults)
    tiling = GemmTiling.standard()

    # Custom tiling via per-dimension chains
    P, S, H = S.PARALLEL, S.SEQUENTIAL, S.HARDWARE
    tiling = GemmTiling(
        dim_m=TileDim("M", 256, P).split("M_wave", 64, P).split("M_mfma", 16, H),
        dim_n=TileDim("N", 128, P).split("N_wave", 64, P).split("N_mfma", 16, H),
        dim_k=TileDim("K", 64, S).split("K_mfma", 16, H),
        mfma=MfmaConfig.f16_16x16x16(),
    )

    # Use with the pipeline
    kernel = GemmKernel.build(problem, tiling.to_tile_config())

    # Or with subtiling (just add a level)
    dim_m = (TileDim("M", 128, P)
             .split("M_wave", 64, P)
             .split("M_subtile", 16, S)   # sequential = partition boundary
             .split("M_mfma", 16, H))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from .problem import MfmaConfig, TileConfig
from .tile import TileLevel
from .transforms import Dim, Tile, TileDescriptor

__all__ = [
    "ScheduleKind", "TileDim", "GemmTiling",
]


class ScheduleKind(Enum):
    """How a tiling level is realized in the generated kernel."""
    PARALLEL = auto()    # mapped to HW parallelism (grid, waves)
    SEQUENTIAL = auto()  # loop iteration (K-unroll, partition)
    HARDWARE = auto()    # fixed by instruction (MFMA tile)


# Shorthand alias
S = ScheduleKind


@dataclass(frozen=True)
class TileDim:
    """One level in a single dimension's tiling hierarchy.

    A linked list where each node describes one tiling split.
    ``count`` = how many inner tiles fit in this tile.

    Build chains with the fluent ``.split()`` API::

        dim = (TileDim("M", 128, PARALLEL)
               .split("M_wave", 64, PARALLEL)     # 2 waves
               .split("M_mfma", 16, HARDWARE))    # 4 repeats
    """
    name: str
    size: int
    schedule: ScheduleKind
    inner: Optional[TileDim] = None
    label: str = ""

    @property
    def is_leaf(self) -> bool:
        return self.inner is None

    @property
    def count(self) -> int:
        """How many inner tiles fit (this.size / inner.size)."""
        if self.inner is None:
            return 1
        return self.size // self.inner.size

    @property
    def depth(self) -> int:
        return 0 if self.inner is None else 1 + self.inner.depth

    @property
    def leaf_size(self) -> int:
        return self.size if self.inner is None else self.inner.leaf_size

    def levels(self) -> List[TileDim]:
        """All levels outermost-first."""
        result = [self]
        if self.inner:
            result.extend(self.inner.levels())
        return result

    def get_level(self, name: str) -> TileDim:
        """Find a level by name."""
        for lvl in self.levels():
            if lvl.name == name:
                return lvl
        raise KeyError(f"No level '{name}' in {self.summary()}")

    def split(self, name: str, inner_size: int,
              schedule: ScheduleKind,
              label: str = "") -> TileDim:
        """Append a split at the leaf, returning a new root.

        Immutable: returns a new chain, does not modify self.
        """
        if self.inner is None:
            if self.size % inner_size != 0:
                raise ValueError(
                    f"{self.name}: {self.size} not divisible by {inner_size}")
            return TileDim(
                self.name, self.size, self.schedule,
                inner=TileDim(name, inner_size, schedule, label=label),
                label=self.label,
            )
        return TileDim(
            self.name, self.size, self.schedule,
            inner=self.inner.split(name, inner_size, schedule, label=label),
            label=self.label,
        )

    def validate(self) -> None:
        if self.inner:
            if self.size % self.inner.size != 0:
                raise ValueError(
                    f"{self.name}({self.size}) not divisible by "
                    f"{self.inner.name}({self.inner.size})")
            self.inner.validate()

    def build_descriptor(self) -> TileDescriptor:
        """Auto-generate a TileDescriptor from this chain.

        Each non-leaf level becomes a ``Tile`` transform.
        """
        desc = TileDescriptor(self.name, [Dim(self.name, self.size)])
        for lvl in self.levels():
            if lvl.inner is not None:
                desc.add_transform(Tile(
                    Dim(lvl.name, lvl.size), lvl.inner.size,
                    outer_name=f"{lvl.inner.name}_id",
                    inner_name=lvl.inner.name,
                ))
        return desc

    def summary(self) -> str:
        parts = []
        for lvl in self.levels():
            tag = lvl.schedule.name[0]  # P/S/H
            parts.append(f"{lvl.name}({lvl.size})[{tag}]")
        return " -> ".join(parts)

    def __repr__(self) -> str:
        return self.summary()


@dataclass
class GemmTiling:
    """Complete GEMM tiling from per-dimension TileDim chains.

    The per-dimension chains are the source of truth.  All derived
    quantities (waves_m, mfma_m_repeat, etc.) come from walking
    the chains.  ``TileConfig``, ``TileLevel``, and ``TileDescriptor``
    are auto-generated.
    """
    dim_m: TileDim
    dim_n: TileDim
    dim_k: TileDim
    mfma: MfmaConfig
    wave_size: int = 64

    @staticmethod
    def standard(
        wg_m: int = 128, wg_n: int = 128, unroll_k: int = 32,
        waves_m: int = 2, waves_n: int = 2,
        mfma: Optional[MfmaConfig] = None,
        wave_size: int = 64,
    ) -> GemmTiling:
        """Build the standard 3-level GEMM tiling."""
        if mfma is None:
            mfma = MfmaConfig.f16_16x16x16()
        wave_m = wg_m // waves_m
        wave_n = wg_n // waves_n
        P, SEQ, H = S.PARALLEL, S.SEQUENTIAL, S.HARDWARE

        dim_m = (TileDim("M", wg_m, P)
                 .split("M_wave", wave_m, P)
                 .split("M_mfma", mfma.m, H))
        dim_n = (TileDim("N", wg_n, P)
                 .split("N_wave", wave_n, P)
                 .split("N_mfma", mfma.n, H))
        dim_k = (TileDim("K", unroll_k, SEQ)
                 .split("K_mfma", mfma.k, H))

        return GemmTiling(dim_m, dim_n, dim_k, mfma, wave_size)

    # -- Derived quantities (same interface as TileConfig) -----------

    @property
    def wg_m(self) -> int:
        return self.dim_m.size

    @property
    def wg_n(self) -> int:
        return self.dim_n.size

    @property
    def unroll_k(self) -> int:
        return self.dim_k.size

    @property
    def waves_m(self) -> int:
        return self.dim_m.count

    @property
    def waves_n(self) -> int:
        return self.dim_n.count

    @property
    def m_per_wave(self) -> int:
        return self.dim_m.inner.size

    @property
    def n_per_wave(self) -> int:
        return self.dim_n.inner.size

    @property
    def mfma_m_repeat(self) -> int:
        return self.dim_m.inner.count

    @property
    def mfma_n_repeat(self) -> int:
        return self.dim_n.inner.count

    @property
    def k_iterations(self) -> int:
        return self.dim_k.count

    @property
    def block_size(self) -> int:
        return self.waves_m * self.waves_n * self.wave_size

    # -- Auto-generate other representations -------------------------

    def to_tile_config(self) -> TileConfig:
        """Convert to legacy TileConfig for backward compatibility."""
        return TileConfig(
            wg_m=self.wg_m, wg_n=self.wg_n, unroll_k=self.unroll_k,
            waves_m=self.waves_m, waves_n=self.waves_n,
            mfma=self.mfma, wave_size=self.wave_size,
        )

    def build_tile_tree(self) -> TileLevel:
        """Auto-generate codegen TileLevel tree from dimension chains.

        Returns a workgroup->wave->mfma tree (no phases).  Use
        build_full_gemm_tree() from asm_emitter for a complete tree
        with workgroup/wave phases for assembly emission.
        """
        mfma_level = TileLevel(
            "mfma", m=self.mfma.m, n=self.mfma.n, k=self.mfma.k)
        wave_level = TileLevel(
            "wave", m=self.m_per_wave, n=self.n_per_wave,
            k=self.unroll_k, inner=mfma_level)
        workgroup_level = TileLevel(
            "workgroup", m=self.wg_m, n=self.wg_n, k=self.unroll_k,
            inner=wave_level, parallel=True)
        return workgroup_level

    def build_m_descriptor(self) -> TileDescriptor:
        return self.dim_m.build_descriptor()

    def build_n_descriptor(self) -> TileDescriptor:
        return self.dim_n.build_descriptor()

    def build_k_descriptor(self) -> TileDescriptor:
        return self.dim_k.build_descriptor()

    def validate(self) -> None:
        self.dim_m.validate()
        self.dim_n.validate()
        self.dim_k.validate()
        if self.dim_m.leaf_size != self.mfma.m:
            raise ValueError(
                f"M leaf size {self.dim_m.leaf_size} != mfma.m {self.mfma.m}")
        if self.dim_n.leaf_size != self.mfma.n:
            raise ValueError(
                f"N leaf size {self.dim_n.leaf_size} != mfma.n {self.mfma.n}")
        if self.dim_k.leaf_size != self.mfma.k:
            raise ValueError(
                f"K leaf size {self.dim_k.leaf_size} != mfma.k {self.mfma.k}")

    def summary(self) -> str:
        return (
            f"M: {self.dim_m}\n"
            f"N: {self.dim_n}\n"
            f"K: {self.dim_k}\n"
            f"Block: {self.block_size} threads  "
            f"MFMA: {self.mfma.m}x{self.mfma.n}x{self.mfma.k}  "
            f"Repeats: {self.mfma_m_repeat}x{self.mfma_n_repeat}"
            f"x{self.k_iterations}"
        )
