# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TileContext: named register bindings flowing through the tile tree.

``TileContext`` is the interface between the auto-generated tile tree
walker and user-provided codegen functions.  It provides:

- **Named bindings** -- users reference ``ctx.get("current_a")``, never
  raw register indices.
- **Scoped lifetime** -- bindings created inside a scope are auto-freed
  when the scope exits.  ``held=True`` keeps a binding alive past the
  scope (for prefetch / cross-iteration state).
- **Stinkytofu resolution** -- ``ctx.vgpr("current_a")`` returns a
  stinkytofu register object that can be passed directly to stinkytofu
  instruction constructors.
- **Loop indices** -- ``ctx.indices`` tracks the current iteration at
  each tile level.
- **Contract validation** -- ``validate_requires`` / ``validate_provides``
  catch binding mismatches early.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

__all__ = ["Lifetime", "Binding", "TileContext"]


class Lifetime(Enum):
    """Register binding lifetime policy."""
    SCOPED = auto()    # auto-freed when owning scope exits
    PERMANENT = auto() # lives for entire kernel
    HELD = auto()      # survives scope exit, must be freed manually


@dataclass
class Binding:
    """A named register binding."""
    name: str
    pool: str          # "v", "s", "acc"
    start: int         # register start index
    count: int         # number of consecutive registers
    lifetime: Lifetime
    scope: str         # which scope owns this


class TileContext:
    """Context flowing through the tile tree.

    Provides named register bindings, scoped allocation, loop index
    tracking, and optional stinkytofu register resolution.

    Usage by a custom codegen function::

        def my_mfma(level, ctx):
            # Read bindings from outer levels
            a = ctx.vgpr("current_a")      # -> stinkytofu register object
            b = ctx.vgpr("current_b")
            acc = ctx.acc("acc_C")

            # Check loop state
            ki = ctx.indices.get("workgroup.ki", 0)

            # Allocate a temporary (auto-freed when this level exits)
            ctx.alloc_vgpr(1, "tmp")
            tmp = ctx.vgpr("tmp")

            # Emit stinkytofu instructions directly
            import stinkytofu as st
            ctx.module.add(st.MFMA("f16", "f32", 16, 16, 16, 1, False,
                                    acc, a, b, comment=f"mfma k={ki}"))
    """

    def __init__(self, module=None):
        self._bindings: Dict[str, Binding] = {}
        self._scope_stack: List[str] = []
        self._next: Dict[str, int] = {"v": 0, "s": 0, "acc": 0}
        self._free_ranges: Dict[str, List[Tuple[int, int]]] = {
            "v": [], "s": [], "acc": [],
        }
        self._peak: Dict[str, int] = {"v": 0, "s": 0, "acc": 0}
        self.indices: Dict[str, int] = {}
        self.module = module  # stinkytofu LogicalModule (can be None for dry run)

    # -- Scope management ---------------------------------------------------

    @contextmanager
    def scope(self, name: str):
        """Enter a tile-level scope.  SCOPED bindings auto-freed on exit."""
        self._scope_stack.append(name)
        try:
            yield self
        finally:
            self._scope_stack.pop()
            self._free_scoped(name)

    @property
    def current_scope(self) -> str:
        return self._scope_stack[-1] if self._scope_stack else "__global__"

    # -- Binding management -------------------------------------------------

    def bind(self, name: str, pool: str, start: int, count: int,
             held: bool = False) -> Binding:
        """Create or update a named binding.

        Args:
            name: Binding name (e.g. ``"current_a"``).
            pool: Register pool (``"v"``, ``"s"``, ``"acc"``).
            start: Register start index.
            count: Number of consecutive registers.
            held: If ``True``, binding survives scope exit (for prefetch
                  / cross-iteration state).
        """
        lt = Lifetime.HELD if held else Lifetime.SCOPED
        b = Binding(name, pool, start, count, lt, self.current_scope)
        self._bindings[name] = b
        return b

    def bind_permanent(self, name: str, pool: str, start: int,
                       count: int) -> Binding:
        """Create a PERMANENT binding (lives for entire kernel)."""
        b = Binding(name, pool, start, count, Lifetime.PERMANENT,
                    "__global__")
        self._bindings[name] = b
        return b

    # -- Allocation (alloc + bind in one step) ------------------------------

    def alloc_vgpr(self, count: int, name: str,
                   held: bool = False) -> int:
        """Allocate VGPRs and bind them under *name*.  Returns start index."""
        start = self._alloc("v", count)
        lt = Lifetime.HELD if held else Lifetime.SCOPED
        self._bindings[name] = Binding(name, "v", start, count, lt,
                                       self.current_scope)
        return start

    def alloc_sgpr(self, count: int, name: str,
                   held: bool = False) -> int:
        """Allocate SGPRs and bind them under *name*.  Returns start index."""
        start = self._alloc("s", count)
        lt = Lifetime.HELD if held else Lifetime.SCOPED
        self._bindings[name] = Binding(name, "s", start, count, lt,
                                       self.current_scope)
        return start

    def alloc_acc(self, count: int, name: str,
                  held: bool = False) -> int:
        """Allocate accumulators and bind under *name*.  Returns start index."""
        start = self._alloc("acc", count)
        lt = Lifetime.HELD if held else Lifetime.SCOPED
        self._bindings[name] = Binding(name, "acc", start, count, lt,
                                       self.current_scope)
        return start

    def alloc_vgpr_permanent(self, count: int, name: str) -> int:
        """Allocate VGPRs with PERMANENT lifetime.  Returns start index."""
        start = self._alloc("v", count)
        self._bindings[name] = Binding(name, "v", start, count,
                                       Lifetime.PERMANENT, "__global__")
        return start

    def alloc_sgpr_permanent(self, count: int, name: str) -> int:
        """Allocate SGPRs with PERMANENT lifetime.  Returns start index."""
        start = self._alloc("s", count)
        self._bindings[name] = Binding(name, "s", start, count,
                                       Lifetime.PERMANENT, "__global__")
        return start

    def alloc_acc_permanent(self, count: int, name: str) -> int:
        """Allocate accumulators with PERMANENT lifetime.  Returns start index."""
        start = self._alloc("acc", count)
        self._bindings[name] = Binding(name, "acc", start, count,
                                       Lifetime.PERMANENT, "__global__")
        return start

    def _alloc(self, pool: str, count: int) -> int:
        """Try to reuse a freed range, otherwise bump-allocate.
        
        Multi-register allocations are aligned: 2-regs to even,
        4+ regs to 4-aligned boundary.
        """
        # Align multi-register allocations for HW tuple requirements
        if count >= 4:
            self._next[pool] = (self._next[pool] + 3) & ~3
        elif count >= 2:
            self._next[pool] = (self._next[pool] + 1) & ~1
        for i, (start, size) in enumerate(self._free_ranges[pool]):
            if size >= count:
                self._free_ranges[pool].pop(i)
                if size > count:
                    self._free_ranges[pool].append(
                        (start + count, size - count)
                    )
                self._update_peak(pool)
                return start
        # Bump allocate
        start = self._next[pool]
        self._next[pool] += count
        self._update_peak(pool)
        return start

    def _update_peak(self, pool: str) -> None:
        in_use = self._next[pool] - sum(s for _, s in self._free_ranges[pool])
        self._peak[pool] = max(self._peak[pool], in_use)

    # -- Free ---------------------------------------------------------------

    def free(self, name: str) -> None:
        """Manually free a binding (for HELD lifetime)."""
        b = self._bindings.pop(name, None)
        if b is None:
            raise ValueError(f"Unknown binding: {name}")
        self._free_ranges[b.pool].append((b.start, b.count))

    def _free_scoped(self, scope_name: str) -> None:
        """Free all SCOPED bindings owned by *scope_name*."""
        to_free = [
            name for name, b in self._bindings.items()
            if b.lifetime == Lifetime.SCOPED and b.scope == scope_name
        ]
        for name in to_free:
            b = self._bindings.pop(name)
            self._free_ranges[b.pool].append((b.start, b.count))

    # -- Lookup / resolution ------------------------------------------------

    def get(self, name: str) -> Binding:
        """Get a binding by name.  Raises ``KeyError`` if not found."""
        if name not in self._bindings:
            available = sorted(self._bindings.keys())
            raise KeyError(
                f"Binding '{name}' not found. Available: {available}"
            )
        return self._bindings[name]

    def has(self, name: str) -> bool:
        """Return ``True`` if *name* is a live binding."""
        return name in self._bindings

    def vgpr(self, name: str, offset: int = 0,
             count: Optional[int] = None):
        """Resolve a binding to a stinkytofu VGPR register object.

        Imports stinkytofu lazily so dry-run mode works without it.
        """
        import stinkytofu as st
        b = self.get(name)
        c = count if count is not None else b.count
        return st.vgpr(b.start + offset, c)

    def sgpr(self, name: str, offset: int = 0,
             count: Optional[int] = None):
        """Resolve a binding to a stinkytofu SGPR register object."""
        import stinkytofu as st
        b = self.get(name)
        c = count if count is not None else b.count
        return st.sgpr(b.start + offset, c)

    def acc(self, name: str, offset: int = 0,
            count: Optional[int] = None):
        """Resolve a binding to a stinkytofu accumulator register object."""
        import stinkytofu as st
        b = self.get(name)
        c = count if count is not None else b.count
        return st.accvgpr(b.start + offset, c)

    # -- Contract validation ------------------------------------------------

    def validate_requires(self, requires: List[str],
                          level_name: str) -> None:
        """Check that all required bindings exist before entering a level."""
        missing = [r for r in requires if r not in self._bindings]
        if missing:
            raise ValueError(
                f"Level '{level_name}' requires bindings {missing} "
                f"but they are not in context. "
                f"Available: {sorted(self._bindings.keys())}"
            )

    def validate_provides(self, provides: List[str],
                          level_name: str) -> None:
        """Check that a level created all the bindings it promised."""
        missing = [p for p in provides if p not in self._bindings]
        if missing:
            raise ValueError(
                f"Level '{level_name}' was expected to provide bindings "
                f"{missing} but did not create them."
            )

    # -- Index management ---------------------------------------------------

    def set_index(self, level: str, dim: str, value: int) -> None:
        """Set a loop index for a tile level."""
        self.indices[f"{level}.{dim}"] = value

    def get_index(self, level: str, dim: str) -> int:
        """Get the current loop index for a level."""
        key = f"{level}.{dim}"
        if key not in self.indices:
            raise KeyError(
                f"Index '{key}' not set. "
                f"Available: {sorted(self.indices.keys())}"
            )
        return self.indices[key]

    # -- Queries ------------------------------------------------------------

    @property
    def vgpr_peak(self) -> int:
        """Peak simultaneous VGPR usage observed so far."""
        return self._peak["v"]

    @property
    def vgpr_in_use(self) -> int:
        """Currently live VGPRs."""
        return self._next["v"] - sum(s for _, s in self._free_ranges["v"])

    @property
    def bindings(self) -> Dict[str, Binding]:
        """All current bindings (read-only snapshot)."""
        return dict(self._bindings)

    @property
    def binding_names(self) -> List[str]:
        """Sorted list of live binding names."""
        return sorted(self._bindings.keys())

    def summary(self) -> str:
        """Human-readable summary of context state."""
        lines = [
            f"VGPRs: {self.vgpr_in_use} live, {self.vgpr_peak} peak",
            f"SGPRs: {self._next['s']}",
            f"ACCs : {self._next['acc']}",
            f"Scopes: {' > '.join(self._scope_stack) or '(none)'}",
            f"Indices: {self.indices}",
            f"Bindings ({len(self._bindings)}):",
        ]
        for name, b in sorted(self._bindings.items()):
            lines.append(
                f"  {name}: {b.pool}[{b.start}:{b.start + b.count}] "
                f"{b.lifetime.name} @{b.scope}"
            )
        return "\n".join(lines)
