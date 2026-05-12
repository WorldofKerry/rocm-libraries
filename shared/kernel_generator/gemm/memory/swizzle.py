"""Generic banked-memory swizzle for conflict-free LDS access.

See SWIZZLE_DESIGN.md for architecture overview.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..emit.context import AsmContext
    from ..problem import TileConfig, MfmaConfig

__all__ = [
    "BankingLevel", "BankedMemoryConfig",
    "LDS_GFX950", "LDS_GFX1250",
    "DataLayout",
    "Swizzle",
    "IdentitySwizzle", "XorSwizzle", "RotationSwizzle", "RowRotationSwizzle",
    "ComposedSwizzle", "PairedRowRotationSwizzle", "PairedRowLayout",
    "DTLRotationSwizzle",
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
    levels=(BankingLevel("bank", num_units=64, stride=4, max_per_cycle=1),),
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

    def from_tile(tile: TileConfig, mfma: MfmaConfig, elem_bytes: float) -> DataLayout:
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
    def emit_write_swizzle(self, ctx: AsmContext, layout: DataLayout,
                           mem: BankedMemoryConfig,
                           v_thread_row: str, v_thread_col: str,
                           v_out: str) -> None:
        """Emit instructions to compute swizzled write column.

        v_out = forward(v_thread_row, v_thread_col)
        """
        ...

    @abstractmethod
    def emit_read_setup(self, ctx: AsmContext, layout: DataLayout,
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

    def forward(self, row: int, col: int, num_cols: int) -> int:
        return col

    def emit_write_swizzle(self, ctx: AsmContext, layout: DataLayout, mem: BankedMemoryConfig,
                           v_thread_row: str, v_thread_col: str, v_out: str) -> None:
        ctx.v_mov(v_out, v_thread_col, comment="identity swizzle (no-op)")

    def emit_read_setup(self, ctx: AsmContext, layout: DataLayout, mem: BankedMemoryConfig,
                        v_lane_row: str, v_k_group: str, v_row_base: str, out_vregs: List[str]) -> None:
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

    def __init__(self, shift_r: int = 2, shift_l: int = 1) -> None:
        self.shift_r = shift_r
        self.shift_l = shift_l

    def forward(self, row: int, col: int, num_cols: int) -> int:
        f = ((row >> self.shift_r) << self.shift_l) & (num_cols - 1)
        return col ^ f

    def emit_write_swizzle(self, ctx: AsmContext, layout: DataLayout, mem: BankedMemoryConfig,
                           v_thread_row: str, v_thread_col: str, v_out: str) -> None:
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

    def emit_read_setup(self, ctx: AsmContext, layout: DataLayout, mem: BankedMemoryConfig,
                        v_lane_row: str, v_k_group: str, v_row_base: str, out_vregs: List[str]) -> None:
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
                     str(xor_bytes), out_vregs[0],
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

    def __init__(self, use_cross_lane: bool = True) -> None:
        self.use_cross_lane = use_cross_lane

    def forward(self, row: int, col: int, num_cols: int, _rows_per_br: Optional[int] = None) -> int:
        rpbr = _rows_per_br if _rows_per_br is not None else max(1, 128 // (num_cols * 16))
        lds_row_id = row // rpbr
        rotation = (lds_row_id // 2) * 2
        return (rotation + col) % num_cols
        # permlane16_swap effect verified on GPU, not modeled here

    def emit_write_swizzle(self, ctx: AsmContext, layout: DataLayout, mem: BankedMemoryConfig,
                           v_thread_row: str, v_thread_col: str, v_out: str) -> None:
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

    def emit_read_setup(self, ctx: AsmContext, layout: DataLayout, mem: BankedMemoryConfig,
                        v_lane_row: str, v_k_group: str, v_row_base: str, out_vregs: List[str]) -> None:
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



class RowRotationSwizzle(Swizzle):
    """col' = (col + row % num_cols) % num_cols.

    Simple additive rotation: each row starts at a different column.
    Achieves theoretical optimal (2 cycles) on gfx950/gfx1250.

    Key advantages over XOR:
    - ki stepping is just addition: (base + ki*k_step) % num_cols
    - Write/read paths are symmetric (both use addition)
    - No recomputation needed per ki iteration

    The rotation amount ``row % num_cols`` ensures all num_cols
    consecutive rows map to distinct starting columns, spreading
    bank accesses evenly.
    """

    def forward(self, row: int, col: int, num_cols: int) -> int:
        return (col + row) % num_cols

    def emit_write_swizzle(self, ctx: AsmContext, layout: DataLayout, mem: BankedMemoryConfig,
                           v_thread_row: str, v_thread_col: str, v_out: str) -> None:
        nc = layout.num_cols
        # col' = (thread_col + thread_row) % num_cols
        ctx.v_add(v_out, v_thread_col, v_thread_row,
                  comment="col + row (rotation)")
        ctx.v_and(v_out, v_out, nc - 1,
                  comment=f"% {nc}")

    def emit_read_setup(self, ctx: AsmContext, layout: DataLayout, mem: BankedMemoryConfig,
                        v_lane_row: str, v_k_group: str, v_row_base: str,
                        out_vregs: List[str]) -> None:
        nc = layout.num_cols
        col_vreg = ctx.vreg("v_tmp3")

        # base_col = (k_group + lane_row) % num_cols
        ctx.v_add(col_vreg, v_k_group, v_lane_row,
                  comment="k_group + lane_row (rotation)")
        ctx.v_and(col_vreg, col_vreg, nc - 1,
                  comment=f"% {nc}")

        # Per-ki: col_ki = (base_col + ki * k_step) % num_cols
        for ki in range(len(out_vregs)):
            if ki == 0:
                ctx.v_lshl(out_vregs[0], col_vreg, 4,
                           comment="col * 16")
                ctx.v_add(out_vregs[0], out_vregs[0], v_row_base,
                          comment="+ row_base")
            else:
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

    def __init__(self, *stages: Swizzle) -> None:
        self.stages = stages

    def forward(self, row: int, col: int, num_cols: int) -> int:
        for s in self.stages:
            col = s.forward(row, col, num_cols)
        return col

    def emit_write_swizzle(self, ctx: AsmContext, layout: DataLayout, mem: BankedMemoryConfig,
                           v_thread_row: str, v_thread_col: str, v_out: str) -> None:
        current = v_thread_col
        for i, s in enumerate(self.stages):
            out = v_out if i == len(self.stages) - 1 else ctx.vreg(f"v_tmp{2+i}")
            s.emit_write_swizzle(ctx, layout, mem,
                                 v_thread_row, current, out)
            current = out

    def emit_read_setup(self, ctx: AsmContext, layout: DataLayout, mem: BankedMemoryConfig,
                        v_lane_row: str, v_k_group: str, v_row_base: str, out_vregs: List[str]) -> None:
        # For composed swizzles, the last stage does the full setup
        # with the composed forward() handling column mapping
        # Fall back to the last stage's emit with modified column
        self.stages[-1].emit_read_setup(
            ctx, layout, mem, v_lane_row, v_k_group, v_row_base, out_vregs)


# ---------------------------------------------------------------------------
# Auto-derivation
# ---------------------------------------------------------------------------

def auto_derive_xor(layout: DataLayout,
                    mem: BankedMemoryConfig) -> Tuple[XorSwizzle, int]:
    """Find optimal XOR params via exhaustive search.

    Returns (best_swizzle, best_cycles). For typical configs (num_cols
    in {4, 8}), the search space is tiny (~25 combinations).
    """
    best_sw = XorSwizzle(0, 0)
    best_cycles = best_sw.verify_all_ki(layout, mem)
    max_shift = max(3, int(math.log2(max(layout.num_cols, 2))) + 2)

    for sr in range(max_shift):
        for sl in range(max_shift):
            sw = XorSwizzle(sr, sl)
            c = sw.verify_all_ki(layout, mem)
            if c < best_cycles:
                best_cycles = c
                best_sw = sw

    return best_sw, best_cycles


def auto_swizzle(layout: DataLayout,
                 mem: BankedMemoryConfig = LDS_GFX950) -> Swizzle:
    """Return the best swizzle for the given layout + memory config.

    Prefers XOR swizzle when it achieves zero bank conflicts, since
    XOR requires ~1 v_xor per read vs ~4 v_add for rotation-based
    swizzles.  Falls back to paired-row rotation when XOR cannot
    achieve zero conflicts.

    Data-type-independent: operates on DataLayout geometry only.
    """
    # XOR first: cheapest in instruction count (~1 v_xor per read)
    best_xor, xor_cycles = auto_derive_xor(layout, mem)
    if xor_cycles <= 1:
        return best_xor

    # Fallback: paired-row rotation (more instructions but always works)
    paired = PairedRowRotationSwizzle.from_layout(layout, mem)
    paired_cycles = paired.verify_paired(layout, mem)
    if paired_cycles <= 1:
        return paired

    # Last resort: pick whichever has fewer conflicts
    if xor_cycles <= paired_cycles:
        return best_xor
    return paired


@dataclass(frozen=True)
class PairedRowLayout:
    """LDS layout that packs multiple M-rows per LDS row for zero conflicts.

    With 64 banks, we need >= 16 columns (256B LDS rows) so that row
    rotation maps all 16 lanes to unique bank groups. When the natural
    row_stride is < 256B, we pack ``pair_factor`` consecutive M-rows
    side-by-side into one wider LDS row.

    This is data-type-independent: the pair_factor is derived purely
    from row_stride vs bank_row_bytes.
    """
    original_layout: DataLayout
    pair_factor: int           # M-rows packed per LDS row
    effective_layout: DataLayout  # layout with widened row_stride

    @staticmethod
    def from_layout(layout: DataLayout,
                    mem: BankedMemoryConfig = LDS_GFX950) -> 'PairedRowLayout':
        """Derive the paired-row layout for zero bank conflicts."""
        bank_row = mem.bank_row_bytes  # 256B for 64 banks
        pair_factor = max(1, -(-bank_row // layout.row_stride_bytes))
        eff_stride = layout.row_stride_bytes * pair_factor
        eff_layout = DataLayout(
            row_stride_bytes=eff_stride,
            mfma_k=layout.mfma_k,
            mfma_m=layout.mfma_m,
            elem_bytes=layout.elem_bytes,
            wave_size=layout.wave_size,
        )
        return PairedRowLayout(
            original_layout=layout,
            pair_factor=pair_factor,
            effective_layout=eff_layout,
        )

    @property
    def effective_cols(self) -> int:
        return self.effective_layout.num_cols

    @property
    def needs_pairing(self) -> bool:
        return self.pair_factor > 1

    def write_col(self, m_row: int, k_col: int) -> int:
        """Map (m_row, k_col) to column in the paired LDS row."""
        half = m_row % self.pair_factor
        orig_cols = self.original_layout.num_cols
        return half * orig_cols + k_col

    def read_col(self, m_row: int, k_group: int) -> int:
        """Map (m_row, k_group) to base column in the paired LDS row."""
        half = m_row % self.pair_factor
        orig_cols = self.original_layout.num_cols
        return half * orig_cols + k_group

    def lds_row(self, m_row: int) -> int:
        """Which LDS row an M-row belongs to."""
        return m_row // self.pair_factor

    def verify(self, mem: BankedMemoryConfig = LDS_GFX950) -> int:
        """Verify conflict cycles with row rotation on the paired layout."""
        sw = RowRotationSwizzle()
        return sw.verify_all_ki(self.effective_layout, mem)


class PairedRowRotationSwizzle(Swizzle):
    """Zero-conflict swizzle using paired rows + row rotation.

    Packs pair_factor consecutive M-rows into one wider LDS row,
    then applies additive row rotation on the wider column space.

    The rotation key is the actual M-row index (not just lane_row),
    ensuring write/read consistency across all mi iterations and waves.

    Formula:
        lds_row = m_row // pair_factor
        half_offset = (m_row % pair_factor) * orig_cols
        col' = (half_offset + col + lds_row) % effective_cols

    The rotation uses lds_row (not m_row) to avoid collisions
    within a paired row. M-rows sharing the same lds_row get
    different halves but the same rotation, ensuring the mapping
    is a bijection within each paired row.

    For pair_factor=2, orig_cols=8, effective_cols=16:
        M0 (lds_row=0): col' = (0 + col + 0) % 16 = col
        M1 (lds_row=0): col' = (8 + col + 0) % 16 = col + 8
        M2 (lds_row=1): col' = (0 + col + 1) % 16
        M3 (lds_row=1): col' = (8 + col + 1) % 16
    """

    def __init__(self, pair_factor: int = 2, orig_cols: int = 8) -> None:
        self.pair_factor = pair_factor
        self.orig_cols = orig_cols
        self.effective_cols = pair_factor * orig_cols

    @staticmethod
    def from_layout(layout: DataLayout,
                    mem: BankedMemoryConfig = LDS_GFX950) -> 'PairedRowRotationSwizzle':
        """Auto-derive from tile geometry."""
        bank_row = mem.bank_row_bytes
        pair_factor = max(1, -(-bank_row // layout.row_stride_bytes))
        return PairedRowRotationSwizzle(
            pair_factor=pair_factor,
            orig_cols=layout.num_cols,
        )

    def forward(self, row: int, col: int, num_cols: int) -> int:
        """Pure-Python column mapping for verification.

        Here row = M-row index, col = k_column within original row.
        num_cols is the ORIGINAL column count (before pairing).
        Rotation is by lds_row (row // pair_factor) to avoid
        collisions within a paired row.
        """
        lds_row = row // self.pair_factor
        half = row % self.pair_factor
        base = half * self.orig_cols + col
        return (base + lds_row) % self.effective_cols

    def lds_row_of(self, m_row: int) -> int:
        """Which LDS row an M-row maps to."""
        return m_row // self.pair_factor

    def emit_write_swizzle(self, ctx: AsmContext, layout: DataLayout,
                           mem: BankedMemoryConfig,
                           v_thread_row: str, v_thread_col: str,
                           v_out: str) -> None:
        """Compute swizzled write column.

        v_thread_row: the M-row index (within workgroup tile)
        v_thread_col: the k-column index (0..orig_cols-1)
        v_out: receives the swizzled column in the effective (wide) space

        Formula: col' = ((m_row % pair_factor) * orig_cols + thread_col + m_row) % eff_cols
        """
        pf = self.pair_factor
        oc = self.orig_cols
        ec = self.effective_cols
        tmp = ctx.vreg("v_tmp3")

        # half_offset = (thread_row % pair_factor) * orig_cols
        if pf == 2:
            ctx.v_and(tmp, v_thread_row, 1,
                      comment="m_row % 2")
            ctx.v_lshl(tmp, tmp, int(math.log2(oc)),
                       comment=f"* {oc} -> half_offset")
        elif pf == 4:
            ctx.v_and(tmp, v_thread_row, 3,
                      comment="m_row % 4")
            ctx.v_lshl(tmp, tmp, int(math.log2(oc)),
                       comment=f"* {oc} -> half_offset")
        else:
            ctx.v_and(tmp, v_thread_row, pf - 1,
                      comment=f"m_row % {pf}")
            ctx.v_mul(tmp, str(oc), tmp,
                      comment=f"* {oc} -> half_offset")

        # lds_row = thread_row / pair_factor (rotation key)
        ctx.v_lshr(ctx.vreg("v_tmp2"), v_thread_row, int(math.log2(pf)),
                   comment=f"lds_row = m_row / {pf}")

        # col' = (half_offset + thread_col + lds_row) % eff_cols
        ctx.v_add(v_out, tmp, v_thread_col,
                  comment="half_offset + col")
        ctx.v_add(v_out, v_out, ctx.vreg("v_tmp2"),
                  comment="+ lds_row (rotation)")
        ctx.v_and(v_out, v_out, ec - 1,
                  comment=f"% {ec}")

    def emit_read_setup(self, ctx: AsmContext, layout: DataLayout,
                        mem: BankedMemoryConfig,
                        v_lane_row: str, v_k_group: str,
                        v_row_base: str, out_vregs: List[str]) -> None:
        """Compute per-ki read base addresses.

        v_lane_row: actual M-row index (not just lane_id % mfma_m)
        v_k_group: lane_id / mfma_m (sub-column index within 16B chunks)
        v_row_base: byte offset to this lane's paired LDS row start

        The swizzle operates on 16B columns (k_col), but k_group is a
        finer granularity (multiple k_groups per column). We convert
        k_group to k_col and handle the byte-within-column offset.

        Internal scratch registers: v_tmp6 (col/swizzled_col), v_tmp7
        (byte_within), v_tmp8 (half_offset), v_tmp9 (lds_row/ki scratch).
        Chosen to avoid aliasing with any caller's parameters (callers
        pass v_tmp1-v_tmp5, v_k_group_a, v_lds_rd_a/b as arguments).
        """
        pf = self.pair_factor
        oc = self.orig_cols
        ec = self.effective_cols
        col_vreg = ctx.vreg("v_tmp6")
        bw_vreg = ctx.vreg("v_tmp7")       # byte_within
        half_vreg = ctx.vreg("v_tmp8")      # half_offset
        lds_row_vreg = ctx.vreg("v_tmp9")   # lds_row / ki scratch

        k_per_group = layout.mfma_k // (layout.wave_size // layout.mfma_m)
        bytes_per_kgroup = int(k_per_group * layout.elem_bytes)
        kgroups_per_col = 16 // bytes_per_kgroup  # typically 2

        # Phase 1: extract ALL values from parameters before any writes
        # that could alias them.

        # 1a. k_col = k_group / kgroups_per_col
        if kgroups_per_col > 1:
            log2_kpc = int(math.log2(kgroups_per_col))
            ctx.v_lshr(col_vreg, v_k_group, log2_kpc,
                       comment=f"k_col = k_group / {kgroups_per_col}")
        else:
            ctx.v_mov(col_vreg, v_k_group, comment="k_col = k_group")

        # 1b. byte_within = (k_group % kgroups_per_col) * bytes_per_kgroup
        if kgroups_per_col > 1:
            ctx.v_and(bw_vreg, v_k_group, kgroups_per_col - 1,
                      comment=f"k_group % {kgroups_per_col}")
            ctx.v_lshl(bw_vreg, bw_vreg, int(math.log2(bytes_per_kgroup)),
                       comment=f"* {bytes_per_kgroup} -> byte_within")
        # v_k_group is no longer needed

        # 1c. half_offset = (m_row % pair_factor) * orig_cols
        if pf == 2:
            ctx.v_and(half_vreg, v_lane_row, 1, comment="m_row % 2")
            ctx.v_lshl(half_vreg, half_vreg, int(math.log2(oc)),
                       comment=f"* {oc} -> half_offset")
        else:
            ctx.v_and(half_vreg, v_lane_row, pf - 1,
                      comment=f"m_row % {pf}")
            if oc > 1 and (oc & (oc - 1)) == 0:
                ctx.v_lshl(half_vreg, half_vreg, int(math.log2(oc)),
                           comment=f"* {oc}")
            else:
                ctx.v_mul(half_vreg, str(oc), half_vreg,
                          comment=f"* {oc}")

        # 1d. lds_row = m_row / pair_factor (rotation key)
        ctx.v_lshr(lds_row_vreg, v_lane_row, int(math.log2(pf)),
                   comment=f"lds_row = m_row / {pf}")
        # v_lane_row is no longer needed

        # Phase 2: swizzled_col = (half_offset + k_col + lds_row) % eff_cols
        ctx.v_add(col_vreg, half_vreg, col_vreg,
                  comment="half_offset + k_col")
        ctx.v_add(col_vreg, col_vreg, lds_row_vreg,
                  comment="+ lds_row (rotation)")
        ctx.v_and(col_vreg, col_vreg, ec - 1,
                  comment=f"% {ec}")

        # Phase 3: per-ki base addresses
        k_col_step = int(layout.mfma_k * layout.elem_bytes) // 16

        for ki in range(len(out_vregs)):
            if ki == 0:
                ctx.v_lshl(out_vregs[0], col_vreg, 4,
                           comment="swizzled_col * 16")
                ctx.v_add(out_vregs[0], out_vregs[0], v_row_base,
                          comment="+ row_base")
                if kgroups_per_col > 1:
                    ctx.v_add(out_vregs[0], out_vregs[0], bw_vreg,
                              comment="+ byte_within_col")
            else:
                step = ki * k_col_step
                # Reuse lds_row_vreg as scratch (no longer needed)
                ctx.v_add(lds_row_vreg, col_vreg, str(step),
                          comment=f"k_col + {step} (ki={ki})")
                ctx.v_and(lds_row_vreg, lds_row_vreg, ec - 1,
                          comment=f"% {ec}")
                ctx.v_lshl(out_vregs[ki], lds_row_vreg, 4,
                           comment="swizzled_col * 16")
                ctx.v_add(out_vregs[ki], out_vregs[ki], v_row_base,
                          comment="+ row_base")
                if kgroups_per_col > 1:
                    ctx.v_add(out_vregs[ki], out_vregs[ki], bw_vreg,
                              comment="+ byte_within_col")

    def verify_paired(self, layout: DataLayout,
                      mem: BankedMemoryConfig) -> int:
        """Verify conflict cycles using the actual paired access pattern."""
        worst = 1
        for ki in range(layout.ki_count):
            for kg in range(layout.k_groups):
                addrs = []
                for lane in range(mem.lanes_per_group):
                    # lane's M-row within one MFMA tile
                    m_row = lane  # lane 0-15 maps to M-rows 0-15
                    lds_row = self.lds_row_of(m_row)
                    eff_stride = self.effective_cols * 16
                    col = self.forward(m_row, kg, self.orig_cols)
                    col_ki = (col + ki * layout.k_step) % self.effective_cols
                    addr = lds_row * eff_stride + col_ki * 16
                    addrs.append(addr)
                worst = max(worst, mem.cycles(addrs))
        return worst


class DTLRotationSwizzle(Swizzle):
    """Rotation swizzle compatible with DTL (Direct-To-LDS) writes.

    DTL writes are sequential (LDS addr = m0 + tid*16), so we cannot
    swizzle the LDS write address. Instead, we rotate the global read
    column (DTL voffset) so different M-rows fetch from rotated K positions.
    The read side applies the matching de-rotation.

    Uses pair_factor=2: two consecutive M-rows share one wider LDS row.
    The rotation key is lds_row = m_row // pair_factor.

    Write (DTL voffset): global_k_col = (thread_col - lds_row) % orig_cols
    Read (ds_read):      read_col = half * orig_cols + (k_col + lds_row) % orig_cols

    This achieves 1-cycle (zero bank conflict) reads on GFX950 LDS.
    """

    def __init__(self, pair_factor: int = 2, orig_cols: int = 8) -> None:
        self.pair_factor = pair_factor
        self.orig_cols = orig_cols
        self.effective_cols = pair_factor * orig_cols

    @staticmethod
    def from_layout(layout: 'DataLayout',
                    mem: 'BankedMemoryConfig' = None) -> 'DTLRotationSwizzle':
        """Auto-derive from tile geometry."""
        if mem is None:
            mem = LDS_GFX950
        bank_row = mem.bank_row_bytes
        pair_factor = max(1, -(-bank_row // layout.row_stride_bytes))
        return DTLRotationSwizzle(
            pair_factor=pair_factor,
            orig_cols=layout.num_cols,
        )

    def forward(self, row: int, col: int, num_cols: int) -> int:
        """Column mapping for verification. Returns read-side swizzled column.

        Here row = M-row index, col = k_column within original row.
        The rotation maps: read_col = half * orig_cols + (col + lds_row) % orig_cols.
        """
        lds_row = row // self.pair_factor
        half = row % self.pair_factor
        return half * self.orig_cols + (col + lds_row) % self.orig_cols

    def lds_row_of(self, m_row: int) -> int:
        """Which LDS row an M-row maps to."""
        return m_row // self.pair_factor

    def emit_write_swizzle(self, ctx: 'AsmContext', layout: 'DataLayout',
                           mem: 'BankedMemoryConfig',
                           v_thread_row: str, v_thread_col: str,
                           v_out: str) -> None:
        """DTL writes are hardware-determined; this is a no-op placeholder.

        The actual rotation is applied in emit_dtl_voffset_rotation() which
        modifies the global read column instead of the LDS write column.
        """
        # DTL write addresses are fixed by hardware; nothing to swizzle.
        ctx.v_mov(v_out, v_thread_col,
                  comment="DTL write: identity (rotation applied to global read)")

    def emit_dtl_voffset_rotation(self, ctx: 'AsmContext',
                                  tile_config: 'TileConfig',
                                  v_thread_row: str,
                                  v_thread_col: str,
                                  v_out: str) -> None:
        """Apply inverse rotation to DTL column for the global read voffset.

        v_thread_row: thread_row = tid / threads_per_row (VGPR)
        v_thread_col: thread_col = tid % threads_per_row (VGPR, 0..orig_cols-1)
        v_out: receives rotated column index (VGPR)

        Formula: rotated_col = (thread_col - lds_row) % orig_cols
        where lds_row = thread_row / pair_factor
        """
        pf = self.pair_factor
        oc = self.orig_cols

        # lds_row = thread_row / pair_factor
        ctx.v_lshr(ctx.vreg("v_tmp2"), v_thread_row, int(math.log2(pf)),
                   comment=f"lds_row = thread_row / {pf}")
        # rotated_col = (thread_col + orig_cols - lds_row) % orig_cols
        ctx.v_add(v_out, v_thread_col, str(oc),
                  comment=f"thread_col + {oc}")
        ctx.inst("v_sub_u32", v_out, v_out, ctx.vreg("v_tmp2"),
                 comment="- lds_row")
        ctx.v_and(v_out, v_out, oc - 1,
                  comment=f"% {oc}")

    def emit_read_setup(self, ctx: 'AsmContext', layout: 'DataLayout',
                        mem: 'BankedMemoryConfig',
                        v_lane_row: str, v_k_group: str,
                        v_row_base: str, out_vregs: List[str]) -> None:
        """Compute per-ki read addresses with per-half rotation.

        Formula: read_col = half * orig_cols + (k_col + lds_row) % orig_cols
        addr = lds_row * eff_stride + read_col * 16 + byte_within_col

        Internal scratch registers: v_tmp6 (col), v_tmp7 (byte_within),
        v_tmp8 (half_offset), v_tmp9 (lds_row / ki scratch),
        v_tmp5 (saved rotated_k_col_0 for ki>0).
        """
        pf = self.pair_factor
        oc = self.orig_cols
        ec = self.effective_cols

        col_vreg = ctx.vreg("v_tmp6")
        bw_vreg = ctx.vreg("v_tmp7")
        half_vreg = ctx.vreg("v_tmp8")
        lds_row_vreg = ctx.vreg("v_tmp9")

        k_per_group = layout.mfma_k // (layout.wave_size // layout.mfma_m)
        bytes_per_kgroup = int(k_per_group * layout.elem_bytes)
        kgroups_per_col = 16 // bytes_per_kgroup

        # Phase 1: extract all values from parameters before writes

        # 1a. k_col = k_group / kgroups_per_col
        if kgroups_per_col > 1:
            log2_kpc = int(math.log2(kgroups_per_col))
            ctx.v_lshr(col_vreg, v_k_group, log2_kpc,
                       comment=f"k_col = k_group / {kgroups_per_col}")
        else:
            ctx.v_mov(col_vreg, v_k_group, comment="k_col = k_group")

        # 1b. byte_within = (k_group % kgroups_per_col) * bytes_per_kgroup
        if kgroups_per_col > 1:
            ctx.v_and(bw_vreg, v_k_group, kgroups_per_col - 1,
                      comment=f"k_group % {kgroups_per_col}")
            ctx.v_lshl(bw_vreg, bw_vreg, int(math.log2(bytes_per_kgroup)),
                       comment=f"* {bytes_per_kgroup} -> byte_within")

        # 1c. half_offset = (m_row % pair_factor) * orig_cols
        if pf == 2:
            ctx.v_and(half_vreg, v_lane_row, 1, comment="m_row % 2")
            ctx.v_lshl(half_vreg, half_vreg, int(math.log2(oc)),
                       comment=f"* {oc} -> half_offset")
        else:
            ctx.v_and(half_vreg, v_lane_row, pf - 1,
                      comment=f"m_row % {pf}")
            ctx.v_lshl(half_vreg, half_vreg, int(math.log2(oc)),
                       comment=f"* {oc}")

        # 1d. lds_row = m_row / pair_factor
        ctx.v_lshr(lds_row_vreg, v_lane_row, int(math.log2(pf)),
                   comment=f"lds_row = m_row / {pf}")

        # Phase 2: per-half rotation
        # rotated_k_col = (k_col + lds_row) % orig_cols
        # read_col = half_offset + rotated_k_col
        ctx.v_add(col_vreg, col_vreg, lds_row_vreg,
                  comment="k_col + lds_row")
        ctx.v_and(col_vreg, col_vreg, oc - 1,
                  comment=f"% {oc} (per-half wrap)")
        # Save rotated_k_col_0 before adding half_offset (needed for ki>0).
        # v_tmp5 is safe: callers pass v_lds_rd_a/v_tmp1/v_tmp2 as parameters,
        # and we've finished reading from v_k_group (v_tmp1) by this point.
        rot_base_vreg = ctx.vreg("v_tmp5")
        ctx.v_mov(rot_base_vreg, col_vreg,
                  comment="save rotated_k_col_0 for ki>0")
        ctx.v_add(col_vreg, col_vreg, half_vreg,
                  comment="+ half_offset -> read_col")

        # Phase 3: per-ki base addresses
        k_col_step = int(layout.mfma_k * layout.elem_bytes) // 16

        for ki in range(len(out_vregs)):
            if ki == 0:
                ctx.v_lshl(out_vregs[0], col_vreg, 4,
                           comment="read_col * 16")
                ctx.v_add(out_vregs[0], out_vregs[0], v_row_base,
                          comment="+ row_base")
                if kgroups_per_col > 1:
                    ctx.v_add(out_vregs[0], out_vregs[0], bw_vreg,
                              comment="+ byte_within_col")
            else:
                step = ki * k_col_step
                # rotated_k_col_ki = (rotated_k_col_0 + step) % oc
                # read_col_ki = half_offset + rotated_k_col_ki
                ctx.v_add(lds_row_vreg, rot_base_vreg, str(step),
                          comment=f"rotated_k_col_0 + {step} (ki={ki})")
                ctx.v_and(lds_row_vreg, lds_row_vreg, oc - 1,
                          comment=f"% {oc}")
                ctx.v_add(lds_row_vreg, lds_row_vreg, half_vreg,
                          comment="+ half_offset")
                ctx.v_lshl(out_vregs[ki], lds_row_vreg, 4,
                           comment="read_col * 16")
                ctx.v_add(out_vregs[ki], out_vregs[ki], v_row_base,
                          comment="+ row_base")
                if kgroups_per_col > 1:
                    ctx.v_add(out_vregs[ki], out_vregs[ki], bw_vreg,
                              comment="+ byte_within_col")

    def verify_paired(self, layout: DataLayout,
                      mem: BankedMemoryConfig) -> int:
        """Verify conflict cycles using the actual paired access pattern."""
        worst = 1
        for ki in range(layout.ki_count):
            for kg in range(layout.k_groups):
                addrs = []
                for lane in range(mem.lanes_per_group):
                    m_row = lane
                    lds_row = self.lds_row_of(m_row)
                    eff_stride = self.effective_cols * 16
                    col = self.forward(m_row, kg, self.orig_cols)
                    col_ki = (col + ki * layout.k_step) % self.effective_cols
                    addr = lds_row * eff_stride + col_ki * 16
                    addrs.append(addr)
                worst = max(worst, mem.cycles(addrs))
        return worst


def auto_swizzle_dtl(layout: DataLayout,
                     mem: BankedMemoryConfig = LDS_GFX950) -> Swizzle:
    """Select best swizzle for DTL (Direct-To-LDS) writes.

    DTL writes are sequential (hardware-determined LDS addresses), so XOR
    swizzle doesn't work. Uses DTLRotationSwizzle which rotates the global
    read column instead of the LDS write address.
    """
    return DTLRotationSwizzle.from_layout(layout, mem)
