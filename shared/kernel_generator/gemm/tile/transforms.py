# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coordinate transform system for tiling GEMM kernels.

Transforms describe how indices in one coordinate space map to indices
in another.  They compose: applying transform A then transform B gives
a combined mapping from A's input space to B's output space.

The key insight (from CK / rocRoller) is that ALL tiling decisions --
workgroup mapping, wave mapping, LDS layout, register layout -- can be
expressed as compositions of a small set of index transforms.

Hierarchy used for GEMM:
    Problem dims  (M, N, K)
      -> Workgroup tiles  (M_wg, N_wg, K_unroll)
        -> Wave tiles  (M_wave, N_wave)
          -> MFMA tiles  (M_mfma, N_mfma, K_mfma)
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

__all__ = [
    "Dim", "Transform",  "Tile",  
    "Embed",  "TileDescriptor", 
]


# ---------------------------------------------------------------------------
# Dimension
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Dim:
    """A named dimension with a size.

    Dimensions are the nodes in the coordinate graph.  Every transform
    maps between sets of dimensions.

    Examples::

        Dim("M", 256)      # matrix M dimension, size 256
        Dim("K", 64)       # unroll depth
        Dim("tid", 256)    # thread ID within workgroup
    """
    name: str
    size: int

    def __repr__(self) -> str:
        return f"{self.name}:{self.size}"


# ---------------------------------------------------------------------------
# Transform base
# ---------------------------------------------------------------------------

class Transform(ABC):
    """Base class for coordinate transforms.

    A transform maps from *lower* dimensions (closer to memory / hardware)
    to *upper* dimensions (closer to the problem description).

    ``lower_dims  <--  Transform  <--  upper_dims``

    ``forward(upper_indices) -> lower_indices``  (used at codegen time)
    """

    @property
    @abstractmethod
    def upper_dims(self) -> List[Dim]:
        """Dimensions produced (output / upper space)."""
        ...

    @property
    @abstractmethod
    def lower_dims(self) -> List[Dim]:
        """Dimensions consumed (input / lower space)."""
        ...

    @abstractmethod
    def forward(self, upper_indices: Dict[str, int]) -> Dict[str, int]:
        """Map upper indices to lower indices (concrete ints)."""
        ...

    @abstractmethod
    def codegen_forward(self, upper_exprs: Dict[str, str]) -> Dict[str, str]:
        """Map upper index *expressions* to lower index expressions.

        Returns dict of ``lower_dim_name -> expression_string`` suitable
        for embedding in generated code.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete transforms
# ---------------------------------------------------------------------------

class Tile(Transform):
    """Split one dimension into an outer and inner pair (tiling).

    ``Tile(Dim("M", 256), tile_size=64)`` produces::

        upper:  [Dim("M_outer", 4),  Dim("M_inner", 64)]
        lower:  [Dim("M", 256)]
        forward:  M = M_outer * 64 + M_inner

    This is CK's ``UnMerge`` / rocRoller's ``Split``.
    """

    def __init__(
        self,
        dim: Dim,
        tile_size: int,
        outer_name: Optional[str] = None,
        inner_name: Optional[str] = None,
    ) -> None:
        if dim.size % tile_size != 0:
            raise ValueError(
                f"Dim {dim.name}={dim.size} not divisible by tile_size={tile_size}"
            )
        self._dim = dim
        self._tile_size = tile_size
        n_tiles = dim.size // tile_size
        self._outer = Dim(outer_name or f"{dim.name}_outer", n_tiles)
        self._inner = Dim(inner_name or f"{dim.name}_inner", tile_size)

    # -- public accessors ---------------------------------------------------
    @property
    def upper_dims(self) -> List[Dim]:
        return [self._outer, self._inner]

    @property
    def lower_dims(self) -> List[Dim]:
        return [self._dim]

    @property
    def outer(self) -> Dim:
        return self._outer

    @property
    def inner(self) -> Dim:
        return self._inner

    @property
    def tile_size(self) -> int:
        return self._tile_size

    # -- mapping ------------------------------------------------------------
    def forward(self, upper_indices: Dict[str, int]) -> Dict[str, int]:
        o = upper_indices[self._outer.name]
        i = upper_indices[self._inner.name]
        return {self._dim.name: o * self._tile_size + i}

    def codegen_forward(self, upper_exprs: Dict[str, str]) -> Dict[str, str]:
        o = upper_exprs[self._outer.name]
        i = upper_exprs[self._inner.name]
        if self._tile_size == 1:
            return {self._dim.name: o}
        return {self._dim.name: f"({o} * {self._tile_size} + {i})"}

    def __repr__(self) -> str:
        return f"Tile({self._dim} -> {self._outer}, {self._inner})"


class Embed(Transform):
    """Affine index map: ``lower = sum(upper_i * coefficient_i)``.

    Used for stride computation::

        Embed([Dim("row",M), Dim("col",N)], Dim("off",M*stride), [stride, 1])
        forward:  off = row * stride + col
    """

    def __init__(
        self, upper: List[Dim], lower: Dim, coefficients: List[int]
    ) -> None:
        if len(upper) != len(coefficients):
            raise ValueError("upper dims and coefficients must have same length")
        self._upper = list(upper)
        self._lower = lower
        self._coefficients = list(coefficients)

    @property
    def upper_dims(self) -> List[Dim]:
        return list(self._upper)

    @property
    def lower_dims(self) -> List[Dim]:
        return [self._lower]

    def forward(self, upper_indices: Dict[str, int]) -> Dict[str, int]:
        val = sum(upper_indices[d.name] * c
                  for d, c in zip(self._upper, self._coefficients))
        return {self._lower.name: val}

    def codegen_forward(self, upper_exprs: Dict[str, str]) -> Dict[str, str]:
        terms: list[str] = []
        for d, c in zip(self._upper, self._coefficients):
            if c == 0:
                continue
            expr = upper_exprs[d.name]
            terms.append(str(expr) if c == 1 else f"({expr} * {c})")
        return {self._lower.name: " + ".join(terms) if terms else "0"}

    def __repr__(self) -> str:
        pairs = [f"{d.name}*{c}" for d, c in zip(self._upper, self._coefficients)]
        return f"Embed({' + '.join(pairs)} -> {self._lower})"


class TileDescriptor:
    """A tensor view described by a chain of coordinate transforms.

    Conceptually similar to CK's ``TensorDescriptor``: a base layout plus
    a sequence of transforms that reshape / remap indices.

    Example -- describing a tiled matrix ``A[M, K]``::

        desc = TileDescriptor("A", [Dim("M", 256), Dim("K", 64)])
        desc.add_transform(Tile(Dim("M", 256), tile_size=128))
        desc.add_transform(Tile(Dim("K", 64), tile_size=32))
        # visible dims: [M_outer, M_inner, K_outer, K_inner]
    """

    def __init__(self, name: str, base_dims: List[Dim]) -> None:
        self.name = name
        self._base_dims = list(base_dims)
        self._transforms: List[Transform] = []
        self._visible_dims: List[Dim] = list(base_dims)

    # -- read-only properties -----------------------------------------------

    @property
    def base_dims(self) -> List[Dim]:
        return list(self._base_dims)

    @property
    def visible_dims(self) -> List[Dim]:
        return list(self._visible_dims)

    @property
    def transforms(self) -> List[Transform]:
        return list(self._transforms)

    # -- mutators -----------------------------------------------------------

    def add_transform(self, transform: Transform) -> TileDescriptor:
        """Apply *transform*, replacing its lower dims with upper dims."""
        lower_names = {d.name for d in transform.lower_dims}
        visible_names = {d.name for d in self._visible_dims}
        missing = lower_names - visible_names
        if missing:
            raise ValueError(
                f"Transform {transform} needs dims {missing} "
                f"not in visible set {visible_names}"
            )

        self._transforms.append(transform)

        # Replace the first occurrence of any consumed lower dim with the
        # transform's upper dims; drop remaining consumed dims.
        new_visible: list[Dim] = []
        inserted = False
        for d in self._visible_dims:
            if d.name in lower_names:
                if not inserted:
                    new_visible.extend(transform.upper_dims)
                    inserted = True
                # else: skip -- this lower dim is consumed
            else:
                new_visible.append(d)
        self._visible_dims = new_visible
        return self

    def get_dim(self, name: str) -> Optional[Dim]:
        """Look up a visible dimension by name."""
        for d in self._visible_dims:
            if d.name == name:
                return d
        return None

    def __repr__(self) -> str:
        dims_str = ", ".join(str(d) for d in self._visible_dims)
        return f"TileDescriptor({self.name}: [{dims_str}])"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

