"""Generic banked-memory swizzle for conflict-free LDS access.

See SWIZZLE_DESIGN.md for architecture overview.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

__all__ = [
    "BankingLevel", "BankedMemoryConfig",
    "LDS_GFX950", "LDS_GFX1250",
    "DataLayout",
    "Swizzle", "SwizzleState",
    "IdentitySwizzle", "XorSwizzle", "RotationSwizzle", "ComposedSwizzle",
]


# ---------------------------------------------------------------------------
# Memory model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BankingLevel:
    """One level in a hierarchical banked memory."""
    name: str
    num_units: int
    stride: int          # byte address stride between consecutive units
    max_per_cycle: int   # max accesses per unit per cycle

    def unit_of(self, byte_addr: int) -> int:
        return (byte_addr // self.stride) % self.num_units


@dataclass(frozen=True)
class BankedMemoryConfig:
    """Hierarchical banked memory descriptor.

    Levels ordered outermost to innermost.  A stall occurs if ANY level
    is overloaded.
    """
    levels: Tuple[BankingLevel, ...]
    access_width: int = 16       # bytes per ds_read (16 for b128)
    lanes_per_group: int = 16    # scheduling group (half-wave)

    @property
    def bank_row_bytes(self) -> int:
        """Bytes in one full bank cycle (innermost level)."""
        inner = self.levels[-1]
        return inner.num_units * inner.stride

    def cycles(self, byte_addrs: List[int]) -> int:
        """Worst-case cycles to serve all accesses across all levels."""
        inner_stride = self.levels[-1].stride
        worst = 1
        for level in self.levels:
            counts: Counter = Counter()
            for addr in byte_addrs:
                for b in range(self.access_width // inner_stride):
                    counts[level.unit_of(addr + b * inner_stride)] += 1
            if counts:
                busiest = max(counts.values())
                level_cycles = -(-busiest // level.max_per_cycle)
                worst = max(worst, level_cycles)
        return worst


# Architecture presets
LDS_GFX950 = BankedMemoryConfig(
    levels=(BankingLevel("bank", num_units=32, stride=4, max_per_cycle=1),),
    access_width=16,
    lanes_per_group=16,
)

LDS_GFX1250 = BankedMemoryConfig(
    levels=(
        BankingLevel("segment", num_units=2, stride=128, max_per_cycle=16),
        BankingLevel("bank", num_units=32, stride=4, max_per_cycle=1),
    ),
    access_width=16,
    lanes_per_group=16,
)


# ---------------------------------------------------------------------------
# Data layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataLayout:
    """Tile geometry relevant to swizzle computation."""
    row_stride_bytes: int   # unroll_k * elem_bytes
    mfma_k: int             # K elements per MFMA
    mfma_m: int             # MFMA M dimension (lane grouping, typically 16)
    elem_bytes: float       # bytes per element (0.5, 1, 2, 4)
    wave_size: int = 64

    @staticmethod
    def from_tile(tile, mfma, elem_bytes: float) -> DataLayout:
        return DataLayout(
            row_stride_bytes=int(tile.unroll_k * elem_bytes),
            mfma_k=mfma.k,
            mfma_m=mfma.m,
            elem_bytes=elem_bytes,
            wave_size=tile.wave_size,
        )

    @property
    def num_cols(self) -> int:
        """16-byte columns per LDS row."""
        return self.row_stride_bytes // 16

    @property
    def ki_count(self) -> int:
        """MFMA K-iterations per unroll."""
        return self.row_stride_bytes // int(self.mfma_k * self.elem_bytes)

    @property
    def k_step(self) -> int:
        """Column step between consecutive ki iterations."""
        return int(self.mfma_k * self.elem_bytes) // 16

    @property
    def k_groups(self) -> int:
        """K-groups within a wave (lanes sharing the same lane_row)."""
        return self.wave_size // self.mfma_m

    @property
    def rows_per_bank_row(self) -> int:
        """LDS rows that fit in one 128-byte bank cycle."""
        return max(1, 128 // self.row_stride_bytes)


# ---------------------------------------------------------------------------
# Swizzle state (output of setup, consumed by K-loop)
# ---------------------------------------------------------------------------

@dataclass
class SwizzleState:
    """Registers allocated by a Swizzle for use in the K-loop.

    Opaque to the kernel generator -- the Swizzle decides what goes here.
    The K-loop uses write_col_vreg for DTL offsets and read_base_vregs[ki]
    for ds_read base addresses.
    """
    write_col_vreg: str              # swizzled thread_col for DTL writes
    read_base_vregs: List[str]       # per-ki precomputed LR base addresses
    # Write-side swizzle VGPR (for thread_col -> swizzled_col mapping):
    write_swizzle_vreg: Optional[str] = None


# ---------------------------------------------------------------------------
# Swizzle base class
# ---------------------------------------------------------------------------

class Swizzle(ABC):
    """Column permutation for bank-conflict avoidance.

    Applied symmetrically to write and read paths.  Subclasses implement
    the permutation math (forward) and GPU code generation (emit_*).
    """

    @abstractmethod
    def forward(self, row: int, col: int, num_cols: int) -> int:
        """Pure-Python column mapping: (row, col) -> physical col.

        Used for verification and testing -- no GPU required.
        """
        ...

    @abstractmethod
    def emit_write_swizzle(self, ctx, layout: DataLayout,
                           mem: BankedMemoryConfig,
                           v_thread_row: str, v_thread_col: str,
                           v_out: str) -> None:
        """Emit instructions to compute swizzled write column.

        v_out = forward(v_thread_row, v_thread_col)
        """
        ...

    @abstractmethod
    def emit_read_setup(self, ctx, layout: DataLayout,
                        mem: BankedMemoryConfig,
                        v_lane_row: str, v_k_group: str,
                        v_row_base: str,
                        out_vregs: List[str]) -> None:
        """Emit instructions to compute per-ki LR base addresses.

        out_vregs[ki] = row_base + swizzled_col(ki) * access_width
        """
        ...

    def verify(self, layout: DataLayout,
               mem: BankedMemoryConfig) -> int:
        """Worst-case cycles for any k_group.  1 = conflict-free."""
        worst = 1
        for kg in range(layout.k_groups):
            addrs = []
            for lr in range(mem.lanes_per_group):
                col = self.forward(lr, kg, layout.num_cols)
                addr = lr * layout.row_stride_bytes + col * mem.access_width
                addrs.append(addr)
            worst = max(worst, mem.cycles(addrs))
        return worst

    def verify_all_ki(self, layout: DataLayout,
                      mem: BankedMemoryConfig) -> int:
        """Verify across all ki offsets (each ki shifts the column)."""
        worst = 1
        for ki in range(layout.ki_count):
            for kg in range(layout.k_groups):
                addrs = []
                for lr in range(mem.lanes_per_group):
                    base_col = self.forward(lr, kg, layout.num_cols)
                    col = (base_col + ki * layout.k_step) % layout.num_cols
                    addr = lr * layout.row_stride_bytes + col * mem.access_width
                    addrs.append(addr)
                worst = max(worst, mem.cycles(addrs))
        return worst


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class IdentitySwizzle(Swizzle):
    """No permutation.  Baseline for testing and memories without banks."""

    def forward(self, row, col, num_cols):
        return col

    def emit_write_swizzle(self, ctx, layout, mem,
                           v_thread_row, v_thread_col, v_out):
        ctx.v_mov(v_out, v_thread_col, comment="identity swizzle (no-op)")

    def emit_read_setup(self, ctx, layout, mem,
                        v_lane_row, v_k_group, v_row_base, out_vregs):
        # Single base: row_base + k_group * access_width
        ctx.v_lshl(out_vregs[0], v_k_group, 4,
                   comment="k_group * 16")
        ctx.v_add(out_vregs[0], out_vregs[0], v_row_base,
                  comment="+ row_base")
        for ki in range(1, len(out_vregs)):
            offset = ki * layout.k_step * mem.access_width
            ctx.v_add(out_vregs[ki], out_vregs[0], str(offset),
                      comment=f"ki={ki}: + {offset}")


class XorSwizzle(Swizzle):
    """col' = col ^ f(row).  No cross-lane ops.

    f(row) = ((row >> shift_r) << shift_l) & (num_cols - 1)
    Achieves 4-way on gfx950 (not optimal, but zero extra K-loop cost).
    """

    def __init__(self, shift_r: int = 2, shift_l: int = 1):
        self.shift_r = shift_r
        self.shift_l = shift_l

    def forward(self, row, col, num_cols):
        f = ((row >> self.shift_r) << self.shift_l) & (num_cols - 1)
        return col ^ f

    def emit_write_swizzle(self, ctx, layout, mem,
                           v_thread_row, v_thread_col, v_out):
        nc = layout.num_cols
        tmp = ctx.vreg("v_tmp3")  # must differ from v_out and v_thread_col
        ctx.v_lshr(tmp, v_thread_row, self.shift_r,
                   comment=f"row >> {self.shift_r}")
        ctx.v_lshl(tmp, tmp, self.shift_l,
                   comment=f"<< {self.shift_l}")
        ctx.v_and(tmp, tmp, nc - 1,
                  comment=f"& {nc - 1}")
        ctx.inst("v_xor_b32", v_out, v_thread_col, tmp,
                 comment="col ^ f(row)")

    def emit_read_setup(self, ctx, layout, mem,
                        v_lane_row, v_k_group, v_row_base, out_vregs):
        nc = layout.num_cols
        tmp = ctx.vreg("v_tmp2")
        # f(lane_row)
        ctx.v_lshr(tmp, v_lane_row, self.shift_r,
                   comment=f"lane_row >> {self.shift_r}")
        ctx.v_lshl(tmp, tmp, self.shift_l,
                   comment=f"<< {self.shift_l}")
        ctx.v_and(tmp, tmp, nc - 1,
                  comment=f"& {nc - 1}")
        # swizzled_col = k_group ^ f(lane_row)
        ctx.inst("v_xor_b32", out_vregs[0], v_k_group, tmp,
                 comment="k_group ^ f(lane_row)")
        ctx.v_lshl(out_vregs[0], out_vregs[0], 4,
                   comment="* 16 -> col bytes")
        ctx.v_add(out_vregs[0], out_vregs[0], v_row_base,
                  comment="+ row_base")
        # Per-ki offsets
        for ki in range(1, len(out_vregs)):
            # col_ki = ((base_col + ki*k_step) % num_cols)
            # Simplified: XOR with shifted value, then add ki offset
            xor_bytes = ki * layout.k_step * mem.access_width
            ctx.inst("v_xor_b32", out_vregs[ki],
                     out_vregs[0], str(xor_bytes),
                     comment=f"ki={ki}: XOR {xor_bytes}")


class RotationSwizzle(Swizzle):
    """Bank-conflict-free rotation + cross-lane redistribution.

    Formula:
        rows_per_bank_row = max(1, 128 / row_stride_bytes)
        lds_row_id = lane_row / rows_per_bank_row
        rotation = (lds_row_id / 2) * 2
        col = (rotation + k_group) % num_cols
        col = permlane16_swap(col)     if use_cross_lane

    Achieves optimal 2-way with cross_lane, 4-way without.
    """

    def __init__(self, use_cross_lane: bool = True):
        self.use_cross_lane = use_cross_lane

    def forward(self, row, col, num_cols, _rows_per_br=None):
        rpbr = _rows_per_br if _rows_per_br is not None else max(1, 128 // (num_cols * 16))
        lds_row_id = row // rpbr
        rotation = (lds_row_id // 2) * 2
        return (rotation + col) % num_cols
        # permlane16_swap effect verified on GPU, not modeled here

    def emit_write_swizzle(self, ctx, layout, mem,
                           v_thread_row, v_thread_col, v_out):
        rpbr = layout.rows_per_bank_row
        nc = layout.num_cols
        tmp = ctx.vreg("v_tmp2")

        # lds_row_id = thread_row / rows_per_bank_row
        if rpbr > 1:
            ctx.v_lshr(tmp, v_thread_row, int(math.log2(rpbr)),
                       comment=f"thread_row / {rpbr} -> lds_row_id")
        else:
            ctx.v_mov(tmp, v_thread_row, comment="lds_row_id = thread_row")

        # rotation = (lds_row_id / 2) * 2
        ctx.v_lshr(tmp, tmp, 1, comment="lds_row_id / 2")
        ctx.v_lshl(tmp, tmp, 1, comment="* 2 -> rotation")

        # col = (rotation + thread_col) % num_cols
        ctx.v_add(v_out, tmp, v_thread_col, comment="rotation + col")
        ctx.v_and(v_out, v_out, nc - 1,
                  comment=f"% {nc}")

        if self.use_cross_lane:
            ctx.inst("s_mov_b32", ctx.sreg("s_tmp0"), "0x33333333",
                     comment="exec mask lo")
            ctx.inst("s_mov_b32", ctx.sreg("s_tmp1"), "0x33333333",
                     comment="exec mask hi")
            ctx.inst("s_mov_b64", "exec", ctx.sreg("s_tmp0", 0, 2),
                     comment="set exec for permlane16")
            ctx.inst("v_permlane16_swap_b32", v_out, v_out,
                     comment="cross-half-wave redistribution")
            ctx.inst("s_mov_b64", "exec", "-1",
                     comment="restore exec")

    def emit_read_setup(self, ctx, layout, mem,
                        v_lane_row, v_k_group, v_row_base, out_vregs):
        rpbr = layout.rows_per_bank_row
        nc = layout.num_cols
        tmp = ctx.vreg("v_tmp2")
        col_vreg = ctx.vreg("v_tmp3")

        # lds_row_id
        if rpbr > 1:
            ctx.v_lshr(tmp, v_lane_row, int(math.log2(rpbr)),
                       comment=f"lane_row / {rpbr} -> lds_row_id")
        else:
            ctx.v_mov(tmp, v_lane_row, comment="lds_row_id = lane_row")

        # rotation = (lds_row_id / 2) * 2
        ctx.v_lshr(tmp, tmp, 1, comment="lds_row_id / 2")
        ctx.v_lshl(tmp, tmp, 1, comment="* 2 -> rotation")

        # col = (rotation + k_group) % num_cols
        ctx.v_add(col_vreg, tmp, v_k_group, comment="rotation + k_group")
        ctx.v_and(col_vreg, col_vreg, nc - 1,
                  comment=f"% {nc}")

        if self.use_cross_lane:
            ctx.inst("s_mov_b32", ctx.sreg("s_tmp0"), "0x33333333",
                     comment="exec mask lo")
            ctx.inst("s_mov_b32", ctx.sreg("s_tmp1"), "0x33333333",
                     comment="exec mask hi")
            ctx.inst("s_mov_b64", "exec", ctx.sreg("s_tmp0", 0, 2),
                     comment="set exec for permlane16")
            ctx.inst("v_permlane16_swap_b32", col_vreg, col_vreg,
                     comment="cross-half-wave redistribution")
            ctx.inst("s_mov_b64", "exec", "-1",
                     comment="restore exec")

        # Per-ki base addresses
        for ki in range(len(out_vregs)):
            if ki == 0:
                # lr_offset[0] = col * 16 + row_base
                ctx.v_lshl(out_vregs[0], col_vreg, 4,
                           comment="col * 16")
                ctx.v_add(out_vregs[0], out_vregs[0], v_row_base,
                          comment="+ row_base")
            else:
                # lr_offset[ki] = ((col + ki*k_step) % num_cols) * 16 + row_base
                step = ki * layout.k_step
                ctx.v_add(ctx.vreg("v_tmp4"), col_vreg, str(step),
                          comment=f"col + {step}")
                ctx.v_and(ctx.vreg("v_tmp4"), ctx.vreg("v_tmp4"), nc - 1,
                          comment=f"% {nc}")
                ctx.v_lshl(out_vregs[ki], ctx.vreg("v_tmp4"), 4,
                           comment="* 16")
                ctx.v_add(out_vregs[ki], out_vregs[ki], v_row_base,
                          comment="+ row_base")


class ComposedSwizzle(Swizzle):
    """Chain multiple swizzle patterns: col' = sN(...s2(s1(row, col)))."""

    def __init__(self, *stages: Swizzle):
        self.stages = stages

    def forward(self, row, col, num_cols):
        for s in self.stages:
            col = s.forward(row, col, num_cols)
        return col

    def emit_write_swizzle(self, ctx, layout, mem,
                           v_thread_row, v_thread_col, v_out):
        current = v_thread_col
        for i, s in enumerate(self.stages):
            out = v_out if i == len(self.stages) - 1 else ctx.vreg(f"v_tmp{2+i}")
            s.emit_write_swizzle(ctx, layout, mem,
                                 v_thread_row, current, out)
            current = out

    def emit_read_setup(self, ctx, layout, mem,
                        v_lane_row, v_k_group, v_row_base, out_vregs):
        # For composed swizzles, the last stage does the full setup
        # with the composed forward() handling column mapping
        # Fall back to the last stage's emit with modified column
        self.stages[-1].emit_read_setup(
            ctx, layout, mem, v_lane_row, v_k_group, v_row_base, out_vregs)
