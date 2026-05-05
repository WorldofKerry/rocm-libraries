# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Recursive tile hierarchy with scoped register lifecycle and phases.

A ``TileLevel`` is one level in the tile decomposition.  The full GEMM
hierarchy is just a tree of ``TileLevel`` nodes::

    grid
      workgroup  (m=128, n=128, k=32)
        wave       (m=64,  n=64)
          mfma       (m=16,  n=16,  k=16,  leaf=True)

Each level can have **prologue** and **epilogue phases** -- named,
replaceable steps that run before/after the inner tile iteration.
For example, the workgroup level's prologue loads kernargs and sets up
addresses; its epilogue stores the result matrix.

Every level has the same interface:
  - dimensions (m, n, k -- whichever are relevant)
  - an ``inner`` child level (or None for leaves)
  - an optional custom ``emit`` callable
  - register allocation scope (auto-freed when the level completes)

Phases make each sub-step independently replaceable.  A researcher
calls ``tree.replace_phase("global_load", my_func)`` to swap one step.

The codegen walks the tree top-down.  At each level it:
  1. Runs prologue phases (setup for this level)
  2. Iterates over inner tiles, calling ``inner.execute()``
  3. Runs epilogue phases (teardown for this level)

A researcher overrides ``emit`` at one level to inject hand-tuned code.
Everything above and below stays auto-generated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


__all__ = [
    "TilePhase", "TileLevel",  
     "walk_tile_tree",
]

# Defer import to avoid circular dependency; context.py is standalone
@dataclass
class TilePhase:
    """A named, replaceable step at a tile level.

    Phases run before (prologue) or after (epilogue) the inner tile
    iteration.  Signature: ``emit(level: TileLevel, ctx) -> None``.

    Replace a phase to customize one step without touching the rest::

        tree = tree.replace_phase("global_load", my_custom_load)
    """
    name: str
    emit: Callable  # (level: TileLevel, ctx) -> None


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
                ``(level, context) -> None``.
                When set, replaces the default recursive codegen for
                this level.  Everything above and below is unaffected.
        prologue_phases: Named steps that run before the body.
        epilogue_phases: Named steps that run after the body.
        parallel: If True, inner tiles are HW-mapped (waves); the walker
                walks inner once instead of iterating repeats.
        partitioned: If True, inner tiles are grouped into partitions
                and executed sequentially (enabling VGPR reuse).
        partition_m, partition_n: Inner tiles per partition along M/N.
        comment: Annotation for generated labels.

    Example -- building the hierarchy with phases::

        mfma = TileLevel("mfma", m=16, n=16, k=16)
        wave = TileLevel("wave", m=64, n=64, k=32, inner=mfma,
                         prologue_phases=[
                             TilePhase("global_load", my_load),
                             TilePhase("lds_write", my_write),
                         ],
                         epilogue_phases=[
                             TilePhase("k_advance", my_advance),
                         ])
        wg = TileLevel("workgroup", m=128, n=128, k=32,
                       inner=wave, parallel=True,
                       prologue_phases=[...], epilogue_phases=[...])

    Replace a single phase::

        tree = tree.replace_phase("global_load", my_custom_load)
    """
    name: str
    m: Optional[int] = None
    n: Optional[int] = None
    k: Optional[int] = None
    inner: Optional[TileLevel] = None
    emit: Optional[Callable] = None  # custom emitter override

    # Phases: named steps that run before/after the inner tile iteration.
    prologue_phases: List[TilePhase] = field(default_factory=list)
    epilogue_phases: List[TilePhase] = field(default_factory=list)

    # If True, inner tiles are mapped to HW parallelism (e.g. waves).
    # The walker walks inner once instead of iterating repeats_m * repeats_n.
    parallel: bool = False

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
                "prologue_phases": list(self.prologue_phases),
                "epilogue_phases": list(self.epilogue_phases),
                "parallel": self.parallel,
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
                prologue_phases=list(self.prologue_phases),
                epilogue_phases=list(self.epilogue_phases),
                parallel=self.parallel,
            )
        return self  # not found, return unchanged

    def replace_phase(self, phase_name: str,
                      new_emit: Callable) -> TileLevel:
        """Return a copy of the tree with the named phase's emit replaced.

        Searches this level's phases first, then recursively into inner.
        """
        def _replace_in_list(phases: list) -> tuple:
            found = False
            result = []
            for p in phases:
                if p.name == phase_name:
                    result.append(TilePhase(p.name, new_emit))
                    found = True
                else:
                    result.append(p)
            return result, found

        new_pro, found_pro = _replace_in_list(self.prologue_phases)
        new_epi, found_epi = _replace_in_list(self.epilogue_phases)

        if found_pro or found_epi:
            return TileLevel(
                name=self.name, m=self.m, n=self.n, k=self.k,
                inner=self.inner, emit=self.emit,
                prologue_phases=new_pro, epilogue_phases=new_epi,
                partitioned=self.partitioned,
                partition_m=self.partition_m, partition_n=self.partition_n,
                parallel=self.parallel, comment=self.comment,
            )
        # Not found at this level -- recurse into inner
        if self.inner:
            return TileLevel(
                name=self.name, m=self.m, n=self.n, k=self.k,
                inner=self.inner.replace_phase(phase_name, new_emit),
                emit=self.emit,
                prologue_phases=list(self.prologue_phases),
                epilogue_phases=list(self.epilogue_phases),
                partitioned=self.partitioned,
                partition_m=self.partition_m, partition_n=self.partition_n,
                parallel=self.parallel, comment=self.comment,
            )
        return self

    def get_phase(self, phase_name: str) -> Optional[TilePhase]:
        """Find a phase by name in this subtree."""
        for p in self.prologue_phases + self.epilogue_phases:
            if p.name == phase_name:
                return p
        if self.inner:
            return self.inner.get_phase(phase_name)
        return None

    def phase_names(self) -> List[str]:
        """All phase names in the entire subtree, depth-first."""
        names = [p.name for p in self.prologue_phases + self.epilogue_phases]
        if self.inner:
            names.extend(self.inner.phase_names())
        return names

    def __repr__(self) -> str:
        return self.summary()

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
        if self.parallel:
            parts[0] += "  [parallel/HW-mapped]"
        if self.prologue_phases:
            phase_str = ", ".join(p.name for p in self.prologue_phases)
            parts.append(f"{pad}  prologue: [{phase_str}]")
        if self.epilogue_phases:
            phase_str = ", ".join(p.name for p in self.epilogue_phases)
            parts.append(f"{pad}  epilogue: [{phase_str}]")
        if self.inner:
            parts.append(self.inner.summary(indent + 1))
        return "\n".join(parts)


# ===================================================================
# Tree builders
# ===================================================================

def walk_tile_tree(
    root: TileLevel,
    ctx: Any,
    visitor: Optional[Callable] = None,
) -> None:
    """Walk the tile tree, calling phases and *visitor* at each level.

    At each level:
      1. Run prologue phases
      2. Call visitor (if provided) -- backward-compat hook
      3. Iterate inner tiles (body)
      4. Run epilogue phases

    If ``level.emit`` is set, it is called instead of the default
    recursive walk for that level's subtree.

    The *visitor* receives ``(level, ctx)`` and is called at every
    level that doesn't have a custom ``emit``.  It can read bindings
    from outer levels via ``ctx.get(name)``.
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
        # Prologue phases
        for phase in root.prologue_phases:
            phase.emit(root, ctx)

        # Visitor callback (backward compat / general hook)
        if visitor is not None:
            visitor(root, ctx)

        # Body: iterate inner tiles
        if root.inner is not None:
            if root.parallel:
                # HW-mapped: walk inner once (HW handles parallelism)
                walk_tile_tree(root.inner, ctx, visitor)
            elif root.partitioned:
                _walk_partitioned(root, ctx, visitor)
            else:
                _walk_flat(root, ctx, visitor)

        # Epilogue phases
        for phase in root.epilogue_phases:
            phase.emit(root, ctx)

    # Validate provides
    if root.provides:
        ctx.validate_provides(root.provides, root.name)


def _walk_flat(root: TileLevel, ctx: Any, visitor: Optional[Callable]) -> None:
    """Default: iterate over all inner tiles sequentially."""
    for mi in range(root.repeats_m):
        for ni in range(root.repeats_n):
            for ki in range(root.repeats_k):
                ctx.set_index(root.name, "mi", mi)
                ctx.set_index(root.name, "ni", ni)
                ctx.set_index(root.name, "ki", ki)
                walk_tile_tree(root.inner, ctx, visitor)


def _walk_partitioned(root: TileLevel, ctx: Any, visitor: Optional[Callable]) -> None:
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
