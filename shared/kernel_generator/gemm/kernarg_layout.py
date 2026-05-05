# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Data-driven kernarg layout descriptors.

Each data type defines a ``KernargLayout`` that describes operand
pointer slots, stride offsets, element packing, and scale metadata.
All code that used to branch on ``is_mx`` queries the layout instead.

Usage::

    from .kernarg_layout import layout_for
    layout = layout_for(problem.dtype, tile.mfma)
    if layout.has_scales:
        ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

__all__ = [
    "OperandSlot", "KernargLayout",
    "FP16_LAYOUT", "BF16_LAYOUT", "MXFP4_LAYOUT",
    "layout_for",
]


@dataclass(frozen=True)
class OperandSlot:
    """One pointer-sized slot in the kernarg buffer."""
    name: str            # "A", "B", "D", "ScaleA", "ScaleB"
    offset: int          # byte offset within kernarg
    is_scale: bool = False
    scale_for: str = ""  # which data operand this scales ("A" or "B")


@dataclass(frozen=True)
class KernargLayout:
    """Data-driven kernarg descriptor for a GEMM data type.

    Replaces scattered ``is_mx`` branches with table lookups.
    """
    name: str                          # "fp16", "bf16", "mxfp4"
    kernarg_size: int                  # total bytes (104, 136)
    element_bytes_numer: int           # numerator for input element size
    element_bytes_denom: int = 1       # denominator (2 for mxfp4: 1/2 byte)
    output_element_bytes: int = 2      # output is always fp16 or bf16

    operands: tuple = ()               # ordered OperandSlot entries
    strides_offset: int = 0            # byte offset of strideD0

    scale_block: int = 0               # elements per scale group (32 for MX)
    scale_element_bytes: int = 1       # bytes per scale value

    # MFMA instruction uses scale operands
    mfma_has_scale_operands: bool = False

    # Output conversion for colmajor TensileLite path
    colmajor_output_bf16: bool = False

    @property
    def has_scales(self) -> bool:
        return any(op.is_scale for op in self.operands)

    @property
    def flags_ptr_offset(self) -> int:
        """StreamK flags reuse the strideD0/D1 slots."""
        return self.strides_offset

    @property
    def num_operand_ptrs(self) -> int:
        """Number of pointer slots (for void** arg count)."""
        return len(self.operands)

    def element_bytes(self) -> float:
        """Input element size as float (0.5 for FP4)."""
        return self.element_bytes_numer / self.element_bytes_denom

    def k_stride_bytes(self, k: int) -> int:
        """Bytes per row in K dimension (always int)."""
        return k * self.element_bytes_numer // self.element_bytes_denom

    def k_offset_bytes(self, iter_start: int, unroll_k: int) -> int:
        """Byte offset for a K-tile iteration (always int)."""
        return iter_start * unroll_k * self.element_bytes_numer // self.element_bytes_denom

    def b_ptr_offset(self) -> int:
        return next(op.offset for op in self.operands if op.name == "B")

    def scale_operands(self) -> List[OperandSlot]:
        return [op for op in self.operands if op.is_scale]

    def data_operands(self) -> List[OperandSlot]:
        return [op for op in self.operands if not op.is_scale]


# --- Concrete layouts ---

FP16_LAYOUT = KernargLayout(
    name="fp16",
    kernarg_size=104,
    element_bytes_numer=2,
    strides_offset=64,
    operands=(
        OperandSlot("D", 32), OperandSlot("C", 40),
        OperandSlot("A", 48), OperandSlot("B", 56),
    ),
)

BF16_LAYOUT = KernargLayout(
    name="bf16",
    kernarg_size=104,
    element_bytes_numer=2,
    strides_offset=64,
    operands=(
        OperandSlot("D", 32), OperandSlot("C", 40),
        OperandSlot("A", 48), OperandSlot("B", 56),
    ),
)

MXFP4_LAYOUT = KernargLayout(
    name="mxfp4",
    kernarg_size=136,
    element_bytes_numer=1,
    element_bytes_denom=2,
    strides_offset=80,
    scale_block=32,
    mfma_has_scale_operands=True,
    colmajor_output_bf16=True,
    operands=(
        OperandSlot("D", 32), OperandSlot("C", 40),
        OperandSlot("A", 48),
        OperandSlot("ScaleA", 56, is_scale=True, scale_for="A"),
        OperandSlot("B", 64),
        OperandSlot("ScaleB", 72, is_scale=True, scale_for="B"),
    ),
)

_LAYOUT_MAP = {
    "f16": FP16_LAYOUT,
    "bf16": BF16_LAYOUT,
    "mxfp4": MXFP4_LAYOUT,
}


def layout_for(dtype, mfma=None) -> KernargLayout:
    """Look up the KernargLayout for a DataType enum or string."""
    key = dtype.value if hasattr(dtype, 'value') else str(dtype)
    if key not in _LAYOUT_MAP:
        raise ValueError(f"No KernargLayout for data type '{key}'")
    return _LAYOUT_MAP[key]
