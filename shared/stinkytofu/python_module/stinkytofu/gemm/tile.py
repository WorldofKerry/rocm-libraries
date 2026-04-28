# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Recursive tile hierarchy with scoped register lifecycle.

A ``TileLevel`` is one level in the tile decomposition.  The full GEMM
hierarchy is just a tree of ``TileLevel`` nodes::

    grid
      workgroup  (m=128, n=128, k=32)
        wave       (m=64,  n=64)
          subtile    (m=16,  n=16,  partitioned=True)
            mfma       (m=16,  n=16,  k=16,  leaf=True)

Every level has the same interface:
  - dimensions (m, n, k -- whichever are relevant)
  - an ``inner`` child level (or None for leaves)
  - an optional custom ``emit`` callable
  - register allocation scope (auto-freed when the level completes)

The codegen walks the tree top-down.  At each level it:
  1. Computes how many inner tiles fit (``repeats_m``, ``repeats_n``, ...)
  2. Allocates operand registers (scoped -- auto-freed on exit)
  3. Loops over inner tiles, calling ``inner.execute()``
  4. Frees operand registers

A researcher overrides ``emit`` at one level to inject hand-tuned code.
Everything above and below stays auto-generated.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple


__all__ = [
    "TileLevel", "ScopedAllocator", "Lifetime",
    "build_gemm_tile_tree", "walk_tile_tree",
]

# Defer import to avoid circular dependency; context.py is standalone
def _import_context():
    from .context import TileContext
    return TileContext


# ===================================================================
# Scoped register allocator
# ===================================================================

class Lifetime(Enum):
    """Register lifetime policy."""
    SCOPED = auto()   # auto-freed when the owning tile level exits
    PERMANENT = auto() # lives for the entire kernel (accumulators, args)
    HELD = auto()      # manually managed (escape hatch for prefetch etc.)


@dataclass
class Allocation:
    """One named register allocation."""
    pool: str       # "v", "s", "acc"
    start: int
    count: int
    name: str
    lifetime: Lifetime
    level: str      # which TileLevel owns this


class ScopedAllocator:
    """Register allocator with scoped auto-free and manual escape hatches.

    Normal allocations tied to a tile level are freed when that level
    exits (``SCOPED``).  Kernel-wide registers (accumulators, args) use
    ``PERMANENT``.  For software pipelining, use ``HELD`` to keep
    registers live past their scope, then free manually.

    Usage::

        alloc = ScopedAllocator()

        # Permanent: lives forever
        alloc.alloc_vgpr(16, "acc_C", lifetime=Lifetime.PERMANENT)

        # Scoped: auto-freed when the scope exits
        with alloc.scope("subtile_0"):
            a = alloc.alloc_vgpr(2, "v_a", lifetime=Lifetime.SCOPED)
            # ... use v_a ...
        # v_a is freed here, VGPRs returned to pool

        # Held: manual control
        h = alloc.alloc_vgpr(4, "prefetch_buf", lifetime=Lifetime.HELD)
        # ... use across multiple scopes ...
        alloc.free("prefetch_buf")
    """

    def __init__(self) -> None:
        self._watermark = {"v": 0, "s": 0, "acc": 0}  # high-water mark
        self._free_ranges: Dict[str, List[Tuple[int, int]]] = {
            "v": [], "s": [], "acc": [],
        }
        self._allocations: Dict[str, Allocation] = {}
        self._scope_stack: List[str] = []
        self._peak = {"v": 0, "s": 0, "acc": 0}

    # -- allocation ---------------------------------------------------------

    def alloc_vgpr(self, count: int, name: str,
                   lifetime: Lifetime = Lifetime.SCOPED) -> int:
        return self._alloc("v", count, name, lifetime)

    def alloc_sgpr(self, count: int, name: str,
                   lifetime: Lifetime = Lifetime.SCOPED) -> int:
        return self._alloc("s", count, name, lifetime)

    def alloc_acc(self, count: int, name: str,
                  lifetime: Lifetime = Lifetime.SCOPED) -> int:
        return self._alloc("acc", count, name, lifetime)

    def _alloc(self, pool: str, count: int, name: str,
               lifetime: Lifetime) -> int:
        if name in self._allocations:
            raise ValueError(f"Duplicate allocation name: {name}")

        # Try to reuse a freed range
        start = self._try_reuse(pool, count)
        if start is None:
            start = self._watermark[pool]
            self._watermark[pool] += count

        level = self._scope_stack[-1] if self._scope_stack else "__global__"
        self._allocations[name] = Allocation(pool, start, count, name,
                                             lifetime, level)
        # Track peak usage
        in_use = self._count_in_use(pool)
        self._peak[pool] = max(self._peak[pool], in_use)
        return start

    def _try_reuse(self, pool: str, count: int) -> Optional[int]:
        """Find a free range that fits *count* registers."""
        for i, (start, size) in enumerate(self._free_ranges[pool]):
            if size >= count:
                # Use this range (or part of it)
                self._free_ranges[pool].pop(i)
                if size > count:
                    # Return leftover to free list
                    self._free_ranges[pool].append(
                        (start + count, size - count)
                    )
                return start
        return None

    def _count_in_use(self, pool: str) -> int:
        free = sum(size for _, size in self._free_ranges[pool])
        return self._watermark[pool] - free

    # -- free ---------------------------------------------------------------

    def free(self, name: str) -> None:
        """Manually free a named allocation (for HELD lifetime)."""
        if name not in self._allocations:
            raise ValueError(f"Unknown allocation: {name}")
        alloc = self._allocations.pop(name)
        self._free_ranges[alloc.pool].append((alloc.start, alloc.count))

    def _free_scoped(self, level: str) -> None:
        """Free all SCOPED allocations owned by *level*."""
        to_free = [
            name for name, a in self._allocations.items()
            if a.lifetime == Lifetime.SCOPED and a.level == level
        ]
        for name in to_free:
            self.free(name)

    # -- scope context manager ----------------------------------------------

    @contextmanager
    def scope(self, level_name: str):
        """Enter a tile-level scope.  SCOPED allocations are auto-freed on exit."""
        self._scope_stack.append(level_name)
        try:
            yield self
        finally:
            self._scope_stack.pop()
            self._free_scoped(level_name)

    @contextmanager
    def hold_registers(self):
        """Temporarily switch default lifetime to HELD.

        Use within a scope to keep registers live past the scope boundary
        (e.g., for software pipelining)::

            with alloc.scope("k_iter_0"):
                with alloc.hold_registers():
                    prefetch = alloc.alloc_vgpr(4, "prefetch")
                # prefetch is NOT freed when k_iter_0 exits
            # must free manually: alloc.free("prefetch")
        """
        # This is a marker -- the caller uses lifetime=Lifetime.HELD explicitly
        # This context manager is syntactic sugar that documents intent
        yield self

    # -- queries ------------------------------------------------------------

    def get(self, name: str) -> Allocation:
        return self._allocations[name]

    @property
    def vgpr_count(self) -> int:
        """Total VGPRs allocated (including freed gaps)."""
        return self._watermark["v"]

    @property
    def sgpr_count(self) -> int:
        return self._watermark["s"]

    @property
    def acc_count(self) -> int:
        return self._watermark["acc"]

    @property
    def vgpr_peak(self) -> int:
        """Peak simultaneous VGPR usage."""
        return self._peak["v"]

    @property
    def vgpr_in_use(self) -> int:
        """Currently live VGPRs."""
        return self._count_in_use("v")

    def summary(self) -> str:
        lines = [
            f"VGPRs: {self.vgpr_in_use} live, {self.vgpr_peak} peak, "
            f"{self.vgpr_count} watermark",
            f"SGPRs: {self.sgpr_count}",
            f"ACCs : {self.acc_count}",
        ]
        for name, a in sorted(self._allocations.items()):
            lines.append(
                f"  {name}: {a.pool}[{a.start}:{a.start + a.count}] "
                f"({a.lifetime.name}, {a.level})"
            )
        return "\n".join(lines)


# ===================================================================
# Recursive tile level
# ===================================================================

@dataclass
class TileLevel:
    """One level in the recursive tile hierarchy.

    Each level describes a tile size and how it decomposes into inner
    tiles.  The codegen walks the tree, allocating/freeing registers
    at each scope boundary.

    Attributes:
        name:   Human-readable level name ("workgroup", "wave", etc.)
        m, n:   Tile dimensions (output).  None if this dim isn't tiled
                at this level (e.g., K is only relevant at some levels).
        k:      K-dimension tile size (for levels that iterate over K).
        inner:  Child tile level, or None for leaf (MFMA instruction).
        emit:   Optional custom emitter.  Signature:
                ``(module, level, context) -> None``.
                When set, replaces the default recursive codegen for
                this level.  Everything above and below is unaffected.
        partitioned: If True, inner tiles are grouped into partitions
                and executed sequentially (enabling VGPR reuse).
        partition_m, partition_n: Inner tiles per partition along M/N.
        comment: Annotation for generated labels.

    Example -- building the hierarchy manually::

        mfma = TileLevel("mfma", m=16, n=16, k=16)
        subtile = TileLevel("subtile", m=16, n=16, inner=mfma,
                            partitioned=True, partition_m=2, partition_n=2)
        wave = TileLevel("wave", m=64, n=64, inner=subtile)
        wg = TileLevel("workgroup", m=128, n=128, k=32, inner=wave)

    Or use ``build_gemm_tile_tree()`` for the common case.
    """
    name: str
    m: Optional[int] = None
    n: Optional[int] = None
    k: Optional[int] = None
    inner: Optional[TileLevel] = None
    emit: Optional[Callable] = None  # custom emitter override

    # Partitioning (for VGPR reuse at this level)
    partitioned: bool = False
    partition_m: int = 1
    partition_n: int = 1

    comment: str = ""

    # Contract: what bindings this level reads / creates
    requires: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)

    # -- derived ------------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        return self.inner is None

    @property
    def repeats_m(self) -> int:
        """How many inner tiles along M."""
        if self.inner is None or self.inner.m is None or self.m is None:
            return 1
        return self.m // self.inner.m

    @property
    def repeats_n(self) -> int:
        if self.inner is None or self.inner.n is None or self.n is None:
            return 1
        return self.n // self.inner.n

    @property
    def repeats_k(self) -> int:
        if self.inner is None or self.inner.k is None or self.k is None:
            return 1
        return self.k // self.inner.k

    @property
    def total_inner_tiles(self) -> int:
        return self.repeats_m * self.repeats_n * self.repeats_k

    @property
    def num_partitions(self) -> int:
        if not self.partitioned:
            return 1
        pm = max(1, self.repeats_m // self.partition_m)
        pn = max(1, self.repeats_n // self.partition_n)
        return pm * pn

    @property
    def depth(self) -> int:
        """Number of levels from here to the leaf."""
        if self.inner is None:
            return 0
        return 1 + self.inner.depth

    def validate(self) -> None:
        """Check consistency of this level and all children."""
        if self.inner and self.m and self.inner.m:
            if self.m % self.inner.m != 0:
                raise ValueError(
                    f"{self.name}: m={self.m} not divisible by "
                    f"inner({self.inner.name}).m={self.inner.m}"
                )
        if self.inner and self.n and self.inner.n:
            if self.n % self.inner.n != 0:
                raise ValueError(
                    f"{self.name}: n={self.n} not divisible by "
                    f"inner({self.inner.name}).n={self.inner.n}"
                )
        if self.inner and self.k and self.inner.k:
            if self.k % self.inner.k != 0:
                raise ValueError(
                    f"{self.name}: k={self.k} not divisible by "
                    f"inner({self.inner.name}).k={self.inner.k}"
                )
        if self.partitioned:
            if self.repeats_m % self.partition_m != 0:
                raise ValueError(
                    f"{self.name}: repeats_m={self.repeats_m} not divisible "
                    f"by partition_m={self.partition_m}"
                )
            if self.repeats_n % self.partition_n != 0:
                raise ValueError(
                    f"{self.name}: repeats_n={self.repeats_n} not divisible "
                    f"by partition_n={self.partition_n}"
                )
        if self.inner:
            self.inner.validate()

    def walk(self) -> List[TileLevel]:
        """Return all levels from this node to the leaf, outermost first."""
        result = [self]
        if self.inner:
            result.extend(self.inner.walk())
        return result

    def find(self, name: str) -> Optional[TileLevel]:
        """Find a level by name in this subtree."""
        if self.name == name:
            return self
        if self.inner:
            return self.inner.find(name)
        return None

    def replace(self, name: str, **kwargs) -> TileLevel:
        """Return a copy of the tree with level *name*'s fields updated.

        This is the main API for researchers to modify one level::

            tree = build_gemm_tile_tree(...)
            tree = tree.replace("subtile", emit=my_custom_emitter)
        """
        if self.name == name:
            d = {
                "name": self.name, "m": self.m, "n": self.n, "k": self.k,
                "inner": self.inner, "emit": self.emit,
                "partitioned": self.partitioned,
                "partition_m": self.partition_m,
                "partition_n": self.partition_n,
                "comment": self.comment,
            }
            d.update(kwargs)
            return TileLevel(**d)
        if self.inner:
            return TileLevel(
                name=self.name, m=self.m, n=self.n, k=self.k,
                inner=self.inner.replace(name, **kwargs),
                emit=self.emit, partitioned=self.partitioned,
                partition_m=self.partition_m, partition_n=self.partition_n,
                comment=self.comment,
            )
        return self  # not found, return unchanged

    def summary(self, indent: int = 0) -> str:
        pad = "  " * indent
        parts = [f"{pad}{self.name}"]
        dims = []
        if self.m is not None:
            dims.append(f"m={self.m}")
        if self.n is not None:
            dims.append(f"n={self.n}")
        if self.k is not None:
            dims.append(f"k={self.k}")
        if dims:
            parts[0] += f"({', '.join(dims)})"
        if self.inner:
            reps = []
            if self.repeats_m > 1:
                reps.append(f"{self.repeats_m}m")
            if self.repeats_n > 1:
                reps.append(f"{self.repeats_n}n")
            if self.repeats_k > 1:
                reps.append(f"{self.repeats_k}k")
            if reps:
                parts[0] += f"  [{' x '.join(reps)} inner tiles]"
        if self.partitioned:
            parts[0] += (
                f"  partitioned({self.partition_m}x{self.partition_n}, "
                f"{self.num_partitions} parts)"
            )
        if self.emit:
            parts[0] += "  [custom emit]"
        if self.inner:
            parts.append(self.inner.summary(indent + 1))
        return "\n".join(parts)


# ===================================================================
# Tree builders
# ===================================================================

def build_gemm_tile_tree(
    wg_m: int = 128,
    wg_n: int = 128,
    unroll_k: int = 32,
    waves_m: int = 2,
    waves_n: int = 2,
    mfma_m: int = 16,
    mfma_n: int = 16,
    mfma_k: int = 16,
    subtile_m: Optional[int] = None,
    subtile_n: Optional[int] = None,
    partition_m: int = 2,
    partition_n: int = 2,
) -> TileLevel:
    """Build the standard GEMM tile tree.

    Without subtiling (subtile_m=None)::

        workgroup(m, n, k) -> wave(m/wm, n/wn) -> mfma(mm, mn, mk)

    With subtiling::

        workgroup -> wave -> subtile(sm, sn) [partitioned] -> mfma
    """
    mfma = TileLevel("mfma", m=mfma_m, n=mfma_n, k=mfma_k)

    wave_m = wg_m // waves_m
    wave_n = wg_n // waves_n

    if subtile_m is not None:
        st_m = subtile_m or mfma_m
        st_n = subtile_n or mfma_n
        subtile = TileLevel(
            "subtile", m=st_m, n=st_n, inner=mfma,
            partitioned=True,
            partition_m=partition_m, partition_n=partition_n,
        )
        wave = TileLevel("wave", m=wave_m, n=wave_n, inner=subtile)
    else:
        wave = TileLevel("wave", m=wave_m, n=wave_n, inner=mfma)

    workgroup = TileLevel(
        "workgroup", m=wg_m, n=wg_n, k=unroll_k, inner=wave,
    )
    return workgroup


def walk_tile_tree(
    root: TileLevel,
    ctx,
    visitor: Callable,
) -> None:
    """Walk the tile tree, calling *visitor* at each level with scoped alloc.

    The visitor receives ``(level, ctx)`` where *ctx* is a ``TileContext``.
    The visitor can:
    - Read bindings from outer levels via ``ctx.get(name)``
    - Allocate registers via ``ctx.alloc_vgpr(count, name)``
    - Emit instructions
    - Publish bindings for inner levels via ``ctx.bind(name, ...)``

    If ``level.emit`` is set, it is called instead of the default
    recursive walk for that level's subtree.
    """
    if root.emit is not None:
        # Validate contract before calling custom emitter
        if root.requires:
            ctx.validate_requires(root.requires, root.name)
        root.emit(root, ctx)
        if root.provides:
            ctx.validate_provides(root.provides, root.name)
        return

    # Validate requires
    if root.requires:
        ctx.validate_requires(root.requires, root.name)

    with ctx.scope(root.name):
        visitor(root, ctx)

        if root.inner is not None:
            if root.partitioned:
                _walk_partitioned(root, ctx, visitor)
            else:
                _walk_flat(root, ctx, visitor)

    # Validate provides
    if root.provides:
        ctx.validate_provides(root.provides, root.name)


def _walk_flat(root, ctx, visitor):
    """Default: iterate over all inner tiles sequentially."""
    for mi in range(root.repeats_m):
        for ni in range(root.repeats_n):
            for ki in range(root.repeats_k):
                ctx.set_index(root.name, "mi", mi)
                ctx.set_index(root.name, "ni", ni)
                ctx.set_index(root.name, "ki", ki)
                walk_tile_tree(root.inner, ctx, visitor)


def _walk_partitioned(root, ctx, visitor):
    """Partition-ordered: groups of inner tiles, VGPR reuse between groups."""
    pm, pn = root.partition_m, root.partition_n
    parts_m = root.repeats_m // pm
    parts_n = root.repeats_n // pn

    part_idx = 0
    for pmi in range(parts_m):
        for pni in range(parts_n):
            # Each partition gets its own scope for operand VGPR reuse
            with ctx.scope(f"{root.name}_part_{part_idx}"):
                for mi in range(pm):
                    for ni in range(pn):
                        for ki in range(root.repeats_k):
                            global_mi = pmi * pm + mi
                            global_ni = pni * pn + ni
                            ctx.set_index(root.name, "mi", global_mi)
                            ctx.set_index(root.name, "ni", global_ni)
                            ctx.set_index(root.name, "ki", ki)
                            ctx.set_index(root.name, "part", part_idx)
                            walk_tile_tree(root.inner, ctx, visitor)
            part_idx += 1
