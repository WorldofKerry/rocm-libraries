"""Scale loading strategies for MX (MXFP4) GEMM kernels.

MX data types require per-tile E8M0 scale factors loaded from global
memory alongside the data tiles.  This module extracts scale loading
logic into composable
classes so that different K-loop implementations (partitioned, simple,
software-pipelined) can reuse the same scale plumbing.

Three concrete strategies:

- ``NullScaleLoader``:  no-op for non-MX data types.
- ``VMEMScaleLoader``:  loads scales directly from global memory into
  VGPRs via ``buffer_load_dword``, bypassing LDS entirely.  Supports
  both the linear layout (standalone kernels) and the AITER
  pre-swizzled layout (``wave_abi`` / ``1d_grid`` modes).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from ..emit.context import AsmContext
from ..problem import TileConfig, MfmaConfig

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .global_loader import GlobalLoader

__all__ = [
    "ScaleLoader",
    "NullScaleLoader",
    "VMEMScaleLoader",
]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ScaleLoader(ABC):
    """Base class for MX scale loading strategies."""

    @abstractmethod
    def alloc_registers(self) -> None:
        """Allocate VGPRs/SGPRs for scale storage."""

    @abstractmethod
    def emit_setup(self) -> None:
        """Set up scale SRDs, voffsets, soffsets."""

    @abstractmethod
    def emit_initial_loads(self, partition_m: int) -> None:
        """Load scales for the first subtile + all B scales (prologue)."""

    @abstractmethod
    def emit_loop_loads(self) -> None:
        """Issue scale loads within the K-loop (after SRD advance)."""

    @abstractmethod
    def advance(self) -> None:
        """Advance scale SRDs by one K-step."""

    @property
    @abstractmethod
    def scale_names_a(self) -> dict:
        """Map ``(mi, ki)`` -> VGPR name for scale A."""

    @property
    @abstractmethod
    def scale_names_b(self) -> dict:
        """Map ``(ni, ki)`` -> VGPR name for scale B."""


# ---------------------------------------------------------------------------
# Null (no-op) implementation
# ---------------------------------------------------------------------------

class NullScaleLoader(ScaleLoader):
    """No-op loader for non-MX data types."""

    def alloc_registers(self) -> None:
        pass

    def emit_setup(self) -> None:
        pass

    def emit_initial_loads(self, partition_m: int) -> None:
        pass

    def emit_loop_loads(self) -> None:
        pass

    def advance(self) -> None:
        pass

    @property
    def scale_names_a(self) -> dict:
        return {}

    @property
    def scale_names_b(self) -> dict:
        return {}

    def emit_load_a(self, mi: int, ki: int) -> None:
        """No-op: emit a single scale-A load for (mi, ki)."""
        pass

    def emit_load_b(self, ni: int, ki: int) -> None:
        """No-op: emit a single scale-B load for (ni, ki)."""
        pass



# ---------------------------------------------------------------------------
    @property
    def has_scales(self) -> bool:
        return False

    @property
    def has_cross_iter_prefetch(self) -> bool:
        return False

    def precompute_soffsets(self) -> None:
        pass

    def num_initial_inflight(self, partition_m: int) -> int:
        return 0

    def cross_iter_inflight(self, partition_m: int, nr: int) -> int:
        return 0

    def emit_scale_wait(self, loader: GlobalLoader) -> None:
        pass

    def emit_subtile_wait(self, loader: GlobalLoader, st_idx: int) -> None:
        pass

    def emit_mfma(self, ctx: AsmContext, mfma: MfmaConfig, acc: str, a_reg: str, b_reg: str, mi: int, ni: int, ki: int) -> None:
        pass

# VMEM (buffer_load_dword) implementation
# ---------------------------------------------------------------------------

class VMEMScaleLoader(ScaleLoader):
    """Loads MX scales directly from global memory into VGPRs.

    Scale data is small enough (1 dword per MFMA tile per K-block) to
    bypass LDS entirely.  Each ``buffer_load_dword`` fetches 4 E8M0
    scale bytes covering one K-block of 32 elements.

    Two addressing modes:

    *Linear* (``swizzled=False``, standalone kernels):
        One VGPR per ``(mi)`` or ``(ni)`` index.  Per-mi/ni SGPRs
        (``s_soff_sa_{mi}``, ``s_soff_sb_{ni}``) provide the row
        offset as ``soffset`` to ``buffer_load_dword``.

    *Swizzled* (``swizzled=True``, AITER pre-shuffled layout):
        Two VGPRs per dimension (group0 = mi 0,1; group1 = mi 2,3).
        ``op_sel`` / ``op_sel_hi`` on the MFMA instruction select the
        correct byte within the dword for each ``(mi, ki)`` pair.

    Args:
        ctx: ``AsmContext`` used for register allocation and emission.
        tile: ``TileConfig`` describing the macro-tile geometry.
        swizzled: If ``True``, use the AITER pre-swizzled scale layout.
    """

    def __init__(self, ctx: AsmContext, tile: TileConfig, swizzled: bool = False) -> None:
        self._ctx = ctx
        self._tile = tile
        self._mfma = tile.mfma
        self._swizzled = swizzled

        self._mr = tile.mfma_m_repeat
        self._nr = tile.mfma_n_repeat
        self._ki_count = tile.k_iterations
        self._mx_block = self._mfma.mx_block  # 32

        # K-stride for SRD advance per K-loop iteration
        if swizzled:
            self._scale_k_stride = 256  # d3 stride in swizzled layout
        else:
            self._scale_k_stride = tile.unroll_k // self._mx_block

        # Populated by alloc_registers()
        self._scale_a_names: Dict[tuple, str] = {}
        self._scale_b_names: Dict[tuple, str] = {}

    # -- properties ---------------------------------------------------------

    @property
    def scale_k_stride(self) -> int:
        """Bytes added to each scale SRD per K-loop iteration."""
        return self._scale_k_stride

    @property
    def scale_names_a(self) -> dict:
        return dict(self._scale_a_names)

    @property
    def scale_names_b(self) -> dict:
        return dict(self._scale_b_names)

    # -- register allocation ------------------------------------------------

    def alloc_registers(self) -> None:
        """Allocate VGPRs for scale A and scale B storage."""
        ctx = self._ctx
        mr, nr, ki_count = self._mr, self._nr, self._ki_count

        if self._swizzled:
            # 2 VGPRs per dimension (group0, group1)
            for g in range(2):
                name = f"v_scale_a_g{g}"
                if not ctx.has(name):
                    ctx.alloc_vgpr_permanent(1, name)
                for mi_in_g in range(2):
                    mi_ = g * 2 + mi_in_g
                    if mi_ < mr:
                        for ki in range(ki_count):
                            self._scale_a_names[(mi_, ki)] = name
            for g in range(2):
                name = f"v_scale_b_g{g}"
                if not ctx.has(name):
                    ctx.alloc_vgpr_permanent(1, name)
                for ni_in_g in range(2):
                    ni_ = g * 2 + ni_in_g
                    if ni_ < nr:
                        for ki in range(ki_count):
                            self._scale_b_names[(ni_, ki)] = name
        else:
            # Linear: one VGPR per (mi, ki) for A and (ni, ki) for B.
            # Each ki covers a different K-range and needs different scale bytes.
            for mi in range(mr):
                for ki in range(ki_count):
                    name = f"v_scale_a_m{mi}k{ki}"
                    if not ctx.has(name):
                        ctx.alloc_vgpr_permanent(1, name)
                    self._scale_a_names[(mi, ki)] = name
            for ni in range(nr):
                for ki in range(ki_count):
                    name = f"v_scale_b_n{ni}k{ki}"
                    if not ctx.has(name):
                        ctx.alloc_vgpr_permanent(1, name)
                    self._scale_b_names[(ni, ki)] = name

    # -- SRD / offset setup -------------------------------------------------

    def emit_setup(self) -> None:
        """Precompute scale soffset SGPRs (linear mode only).

        For the swizzled mode the soffsets are computed in
        ``phase_mx_scale_setup`` (they depend on wave layout), so
        this method only handles the linear per-mi / per-ni offsets.
        """
        if self._swizzled:
            return

        ctx = self._ctx
        mfma = self._mfma
        mr, nr = self._mr, self._nr

        ctx.comment("Precompute scale soffsets")
        for mi_ in range(1, mr):
            soff_name = f"s_soff_sa_{mi_}"
            ctx.alloc_sgpr_permanent(1, soff_name)
            ctx.s_mul(ctx.sreg(soff_name), ctx.sreg("s_stride_scale_a"),
                      str(mi_ * mfma.m),
                      comment=f"soff_a[{mi_}] = stride * {mi_ * mfma.m}")
        for ni_ in range(1, nr):
            soff_name = f"s_soff_sb_{ni_}"
            ctx.alloc_sgpr_permanent(1, soff_name)
            ctx.s_mul(ctx.sreg(soff_name), ctx.sreg("s_stride_scale_b"),
                      str(ni_ * mfma.n),
                      comment=f"soff_b[{ni_}] = stride * {ni_ * mfma.n}")
        ctx.raw("")

    # -- initial (prologue) loads -------------------------------------------

    def emit_initial_loads(self, partition_m: int) -> None:
        """Issue ``buffer_load_dword`` for the first subtile of A + all B."""
        ctx = self._ctx
        nr = self._nr

        if self._swizzled:
            ctx.comment("Load swizzled scales A (2 groups)")
            ctx.inst("buffer_load_dword", ctx.vreg("v_scale_a_g0"),
                     ctx.vreg("v_dtl_off_scale_a"),
                     ctx.sreg("s_srd_scale_a", 0, 4),
                     ctx.sreg("s_scale_soff_a0"), "offen",
                     comment="scaleA group0 (mi=0,1)")
            ctx.inst("buffer_load_dword", ctx.vreg("v_scale_a_g1"),
                     ctx.vreg("v_dtl_off_scale_a"),
                     ctx.sreg("s_srd_scale_a", 0, 4),
                     ctx.sreg("s_scale_soff_a1"), "offen",
                     comment="scaleA group1 (mi=2,3)")
            ctx.comment("Load swizzled scales B (2 groups)")
            ctx.inst("buffer_load_dword", ctx.vreg("v_scale_b_g0"),
                     ctx.vreg("v_dtl_off_scale_b"),
                     ctx.sreg("s_srd_scale_b", 0, 4),
                     ctx.sreg("s_scale_soff_b0"), "offen",
                     comment="scaleB group0 (ni=0,1)")
            ctx.inst("buffer_load_dword", ctx.vreg("v_scale_b_g1"),
                     ctx.vreg("v_dtl_off_scale_b"),
                     ctx.sreg("s_srd_scale_b", 0, 4),
                     ctx.sreg("s_scale_soff_b1"), "offen",
                     comment="scaleB group1 (ni=2,3)")
        else:
            ki_count = self._ki_count
            ki_bytes = self._mfma.k // self._mx_block  # bytes per ki step
            ctx.comment("Load scales A subtile 0 (direct VGPR)")
            for mi_ in range(partition_m):
                for ki_ in range(ki_count):
                    soff = "0" if mi_ == 0 else ctx.sreg(f"s_soff_sa_{mi_}")
                    off = ki_ * ki_bytes
                    off_str = f" offset:{off}" if off > 0 else ""
                    ctx.inst("buffer_load_dword",
                             ctx.vreg(f"v_scale_a_m{mi_}k{ki_}"),
                             ctx.vreg("v_dtl_off_scale_a"),
                             ctx.sreg("s_srd_scale_a", 0, 4),
                             soff, f"offen{off_str}",
                             comment=f"scale A mi={mi_} ki={ki_}")
            ctx.comment("Load scales B (direct VGPR)")
            for ni_ in range(nr):
                for ki_ in range(ki_count):
                    soff = "0" if ni_ == 0 else ctx.sreg(f"s_soff_sb_{ni_}")
                    off = ki_ * ki_bytes
                    off_str = f" offset:{off}" if off > 0 else ""
                    ctx.inst("buffer_load_dword",
                             ctx.vreg(f"v_scale_b_n{ni_}k{ki_}"),
                             ctx.vreg("v_dtl_off_scale_b"),
                             ctx.sreg("s_srd_scale_b", 0, 4),
                             soff, f"offen{off_str}",
                             comment=f"scale B ni={ni_} ki={ki_}")

    # -- in-loop loads (after SRD advance) ----------------------------------

    def emit_loop_loads(self) -> None:
        """Issue scale loads inside the K-loop body.

        Swizzled mode reloads all groups every iteration.  Linear mode
        relies on cross-iteration prefetch (see ``make_prefetch_ops``),
        so this is a no-op for linear.
        """
        if not self._swizzled:
            return

        ctx = self._ctx
        ctx.comment("Load swizzled scales A (2 groups)")
        ctx.inst("buffer_load_dword", ctx.vreg("v_scale_a_g0"),
                 ctx.vreg("v_dtl_off_scale_a"),
                 ctx.sreg("s_srd_scale_a", 0, 4),
                 ctx.sreg("s_scale_soff_a0"), "offen",
                 comment="scaleA group0 (mi=0,1)")
        ctx.inst("buffer_load_dword", ctx.vreg("v_scale_a_g1"),
                 ctx.vreg("v_dtl_off_scale_a"),
                 ctx.sreg("s_srd_scale_a", 0, 4),
                 ctx.sreg("s_scale_soff_a1"), "offen",
                 comment="scaleA group1 (mi=2,3)")
        ctx.comment("Load swizzled scales B (2 groups)")
        ctx.inst("buffer_load_dword", ctx.vreg("v_scale_b_g0"),
                 ctx.vreg("v_dtl_off_scale_b"),
                 ctx.sreg("s_srd_scale_b", 0, 4),
                 ctx.sreg("s_scale_soff_b0"), "offen",
                 comment="scaleB group0 (ni=0,1)")
        ctx.inst("buffer_load_dword", ctx.vreg("v_scale_b_g1"),
                 ctx.vreg("v_dtl_off_scale_b"),
                 ctx.sreg("s_srd_scale_b", 0, 4),
                 ctx.sreg("s_scale_soff_b1"), "offen",
                 comment="scaleB group1 (ni=2,3)")

    # -- SRD advance --------------------------------------------------------

    def advance(self) -> None:
        """Advance both scale SRDs by ``scale_k_stride`` bytes."""
        ctx = self._ctx
        stride = self._scale_k_stride
        for srd_name in ["s_srd_scale_a", "s_srd_scale_b"]:
            ctx.inst("s_add_u32", ctx.sreg(srd_name, 0, 1),
                     ctx.sreg(srd_name, 0, 1), str(stride),
                     comment=f"{srd_name} += {stride}")
            ctx.inst("s_addc_u32", ctx.sreg(srd_name, 1, 1),
                     ctx.sreg(srd_name, 1, 1), "0", comment="carry")

    # -- per-index emission (for ScaleBlock integration) --------------------

    def emit_load_a(self, mi: int, ki: int) -> None:
        """Emit a single ``buffer_load_dword`` for scale A at (mi, ki).

        Swizzled mode emits per-group loads; linear mode emits
        per-(mi,ki) loads.
        """
        ctx = self._ctx
        if self._swizzled:
            group = mi // 2
            gname = f"v_scale_a_g{group}"
            soff_name = f"s_scale_soff_a{group}"
            ctx.inst("buffer_load_dword", ctx.vreg(gname),
                     ctx.vreg("v_dtl_off_scale_a"),
                     ctx.sreg("s_srd_scale_a", 0, 4),
                     ctx.sreg(soff_name), "offen",
                     comment=f"scaleA group{group} (mi={mi} ki={ki})")
        else:
            ki_bytes = self._mfma.k // self._mx_block
            soff = "0" if mi == 0 else ctx.sreg(f"s_soff_sa_{mi}")
            off = ki * ki_bytes
            off_str = f" offset:{off}" if off > 0 else ""
            ctx.inst("buffer_load_dword",
                     ctx.vreg(f"v_scale_a_m{mi}k{ki}"),
                     ctx.vreg("v_dtl_off_scale_a"),
                     ctx.sreg("s_srd_scale_a", 0, 4),
                     soff, f"offen{off_str}",
                     comment=f"scale A mi={mi} ki={ki}")

    def emit_load_b(self, ni: int, ki: int) -> None:
        """Emit a single ``buffer_load_dword`` for scale B at (ni, ki).

        Swizzled mode emits per-group loads; linear mode emits
        per-(ni,ki) loads.
        """
        ctx = self._ctx
        if self._swizzled:
            group = ni // 2
            gname = f"v_scale_b_g{group}"
            soff_name = f"s_scale_soff_b{group}"
            ctx.inst("buffer_load_dword", ctx.vreg(gname),
                     ctx.vreg("v_dtl_off_scale_b"),
                     ctx.sreg("s_srd_scale_b", 0, 4),
                     ctx.sreg(soff_name), "offen",
                     comment=f"scaleB group{group} (ni={ni} ki={ki})")
        else:
            ki_bytes = self._mfma.k // self._mx_block
            soff = "0" if ni == 0 else ctx.sreg(f"s_soff_sb_{ni}")
            off = ki * ki_bytes
            off_str = f" offset:{off}" if off > 0 else ""
            ctx.inst("buffer_load_dword",
                     ctx.vreg(f"v_scale_b_n{ni}k{ki}"),
                     ctx.vreg("v_dtl_off_scale_b"),
                     ctx.sreg("s_srd_scale_b", 0, 4),
                     soff, f"offen{off_str}",
                     comment=f"scale B ni={ni} ki={ki}")


    # -- composable K-loop integration -------------------------------------

    @property
    def has_scales(self) -> bool:
        """True if this loader provides scale operands for MFMAs."""
        return True

    @property
    def has_cross_iter_prefetch(self) -> bool:
        """Cross-iteration scale prefetch is not yet implemented.

        When True, the suffix uses vmcnt(N>0) assuming N prefetch
        loads are in-flight, and emit_produce skips scale re-emission.
        Since no code actually emits those prefetch loads, this causes:
        - DTL: vmcnt too high -> read toggle before DTL completes
        - Both: scale VGPRs never reloaded after iteration 0
        Disabled until the prefetch emission is implemented.
        """
        return False

    def precompute_soffsets(self) -> None:
        """Alias for emit_setup() -- called by ComposableKLoop."""
        self.alloc_registers()
        self.emit_setup()

    def num_initial_inflight(self, partition_m: int) -> int:
        """Number of vmcnt-tracked scale loads after emit_initial_loads."""
        if self._swizzled:
            return 4  # 2 groups A + 2 groups B
        return (partition_m + self._nr) * self._ki_count

    def cross_iter_inflight(self, partition_m: int, nr: int) -> int:
        """Number of prefetch loads in-flight at end of iteration."""
        if self._swizzled:
            return 0
        return (partition_m + nr) * self._ki_count

    def emit_scale_wait(self, loader: GlobalLoader) -> None:
        """Emit vmcnt wait for scale loads after preamble reads.

        For linear mode with DTL, leaves DTL loads in-flight.
        For swizzled mode, waits for all vmcnt.
        """
        ctx = self._ctx
        if not self._swizzled:
            num_dtl = loader.num_inflight
            ctx.s_waitcnt(f"vmcnt({num_dtl})",
                          comment=f"wait scales (leave {num_dtl} DTL in-flight)")
        else:
            ctx.s_waitcnt("vmcnt(0)", comment="wait scale VGPR loads")

    def emit_subtile_wait(self, loader: GlobalLoader, st_idx: int) -> None:
        """Emit vmcnt wait at subtile boundary for prefetched scale_a."""
        if self._swizzled:
            return
        ctx = self._ctx
        num_dtl = loader.num_inflight
        ctx.s_waitcnt(f"vmcnt({num_dtl})",
                      comment=f"wait scale_a subtile {st_idx} (leave DTL)")

    def emit_mfma(self, ctx: AsmContext, mfma: MfmaConfig, acc: str, a_reg: str, b_reg: str, mi: int, ni: int, ki: int) -> None:
        """Emit MFMA instruction with proper scale operands."""
        sa_name = self._scale_a_names.get((mi, ki))
        sb_name = self._scale_b_names.get((ni, ki))
        if sa_name is None or sb_name is None:
            # Fallback: constant scale
            ctx.inst(mfma.instruction_name, acc, a_reg, b_reg, acc,
                     ctx.vreg("v_mxscale"), ctx.vreg("v_mxscale"),
                     f"cbsz:{mfma.cbsz} blgp:{mfma.blgp}",
                     comment=f"MFMA m{mi}_n{ni}_k{ki}")
            return

        if self._swizzled:
            a_sel = mi % 2
            b_sel = ni % 2
            hi_a = ki
            hi_b = ki
            ctx.inst(mfma.instruction_name, acc, a_reg, b_reg, acc,
                     ctx.vreg(sa_name), ctx.vreg(sb_name),
                     f"op_sel:[{a_sel},{b_sel}] op_sel_hi:[{hi_a},{hi_b}]"
                     f" cbsz:{mfma.cbsz} blgp:{mfma.blgp}",
                     comment=f"MFMA m{mi}_n{ni}_k{ki}")
        else:
            ctx.inst(mfma.instruction_name, acc, a_reg, b_reg, acc,
                     ctx.vreg(sa_name), ctx.vreg(sb_name),
                     f"cbsz:{mfma.cbsz} blgp:{mfma.blgp}",
                     comment=f"MFMA m{mi}_n{ni}_k{ki}")


# ---------------------------------------------------------------------------
# LDS-based (DTL) implementation
# ---------------------------------------------------------------------------

class LDSScaleLoader(ScaleLoader):
    """Loads pre-swizzled MX scales via DTL into LDS, reads via ds_read_b32.

    Uses the AITER pre-swizzled scale layout (``--mx-scale-format 1``).
    Scale data is packed into 256-byte groups:

        group[g] = 256 bytes = 64 lanes × 4 bytes/lane
        byte layout per lane: [mi_even_ki0, mi_odd_ki0, mi_even_ki1, mi_odd_ki1]

    Per matrix: 4 groups (g0..g3 covering mi=0..7 or ni=0..7).
    Per wave partition: 1024 bytes (4 groups × 256).

    DTL loads 4096 bytes per matrix (1 load, 256 threads × 16 bytes).
    ds_read_b32 reads 4 bytes per group (4 reads per matrix = 8 total).
    MFMA uses op_sel/op_sel_hi to select the correct byte.

    Args:
        ctx: AsmContext for register allocation and emission.
        tile: TileConfig describing the macro-tile geometry.
        lds_scale_offset: Byte offset of scale A region within one
            LDS buffer (typically = lds_data_half).
    """

    def __init__(self, ctx: AsmContext, tile: TileConfig,
                 lds_scale_offset: int = 0) -> None:
        self._ctx = ctx
        self._tile = tile
        self._mfma = tile.mfma
        self._mr = tile.mfma_m_repeat
        self._nr = tile.mfma_n_repeat
        self._ki_count = tile.k_iterations
        self._mx_block = self._mfma.mx_block

        # Scale region sizes (4096 bytes each, matching DTL coverage)
        self._scale_a_lds_size = 4096
        self._scale_b_lds_size = 4096
        self._scale_a_lds_off = lds_scale_offset
        self._scale_b_lds_off = lds_scale_offset + self._scale_a_lds_size

        # K-stride for SRD advance: pre-swizzled layout uses 256 bytes
        # per d3 stride (see AITER e8m0_shuffle format)
        self._scale_k_stride = 256

        # Scale VGPR names: 2 groups per matrix (g0, g1 for A; g0, g1 for B)
        # Each group covers 2 mi (or ni) values × 2 ki values = 4 bytes
        self._scale_a_names: Dict[tuple, str] = {}
        self._scale_b_names: Dict[tuple, str] = {}
        self._num_groups = (self._mr + 1) // 2  # groups for A

    @property
    def scale_lds_size(self) -> int:
        """Total LDS bytes for scales (A + B), one buffer."""
        return self._scale_a_lds_size + self._scale_b_lds_size

    @property
    def scale_k_stride(self) -> int:
        return self._scale_k_stride

    @property
    def scale_names_a(self) -> dict:
        return dict(self._scale_a_names)

    @property
    def scale_names_b(self) -> dict:
        return dict(self._scale_b_names)

    def alloc_registers(self) -> None:
        """Allocate VGPRs: 1 per group (2 mi/ni per group)."""
        ctx = self._ctx
        mr, nr, ki_count = self._mr, self._nr, self._ki_count
        num_groups_a = (mr + 1) // 2
        num_groups_b = (nr + 1) // 2

        for g in range(num_groups_a):
            name = f"v_scale_a_g{g}"
            if not ctx.has(name):
                ctx.alloc_vgpr_permanent(1, name)
            for mi_in_g in range(2):
                mi = g * 2 + mi_in_g
                if mi < mr:
                    for ki in range(ki_count):
                        self._scale_a_names[(mi, ki)] = name

        for g in range(num_groups_b):
            name = f"v_scale_b_g{g}"
            if not ctx.has(name):
                ctx.alloc_vgpr_permanent(1, name)
            for ni_in_g in range(2):
                ni = g * 2 + ni_in_g
                if ni < nr:
                    for ki in range(ki_count):
                        self._scale_b_names[(ni, ki)] = name

    def emit_setup(self) -> None:
        """Set up scale DTL voffsets and ds_read base VGPRs.

        DTL voffset: tid * 16 (contiguous read from pre-swizzled buffer).
        ds_read base: wave_partition_offset + laneId * 4 + lds_scale_base.
        """
        ctx = self._ctx
        tile = self._tile

        # DTL voffset: tid * 16
        ctx.comment("Scale DTL voffset (pre-swizzled): tid * 16")
        if not ctx.has("v_dtl_off_scale_a_lds"):
            ctx.alloc_vgpr_permanent(1, "v_dtl_off_scale_a_lds")
        ctx.v_lshl(ctx.vreg("v_dtl_off_scale_a_lds"),
                   ctx.vreg("v_tid"), 4,
                   comment="scale DTL voff = tid * 16")
        if not ctx.has("v_dtl_off_scale_b_lds"):
            ctx.alloc_vgpr_permanent(1, "v_dtl_off_scale_b_lds")
        ctx.inst("v_mov_b32", ctx.vreg("v_dtl_off_scale_b_lds"),
                 ctx.vreg("v_dtl_off_scale_a_lds"),
                 comment="scale B DTL voff = same")
        ctx.raw("")

        # ds_read base: wave_partition * 1024 + laneId * 4 + lds_base
        ctx.comment("Scale ds_read base (pre-swizzled LDS)")
        if not ctx.has("v_scale_rd_a"):
            ctx.alloc_vgpr_permanent(1, "v_scale_rd_a")
        if not ctx.has("v_scale_rd_b"):
            ctx.alloc_vgpr_permanent(1, "v_scale_rd_b")

        # laneId = serial % 64
        ctx.inst("v_and_b32", ctx.vreg("v_tmp0"),
                 ctx.vreg("v_lane_id"), "63",
                 comment="laneId = lane_id & 63")
        ctx.v_lshl(ctx.vreg("v_tmp0"), ctx.vreg("v_tmp0"), 2,
                    comment="laneId * 4")

        # wave_m partition: waveId_m * 1024
        ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_wave_m"), 10,
                    comment="wave_m * 1024 (partition offset)")
        ctx.v_add(ctx.vreg("v_scale_rd_a"), ctx.vreg("v_tmp0"),
                  ctx.vreg("v_tmp1"),
                  comment="laneId*4 + partition_m")
        ctx.s_mov(ctx.sreg("s_tmp0"), str(self._scale_a_lds_off),
                  comment=f"scale_a_lds_off = {self._scale_a_lds_off}")
        ctx.v_add(ctx.vreg("v_scale_rd_a"), ctx.vreg("v_scale_rd_a"),
                  ctx.sreg("s_tmp0"),
                  comment="+ lds_base_a")

        # wave_n partition: waveId_n * 1024
        ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_wave_n"), 10,
                    comment="wave_n * 1024 (partition offset)")
        ctx.v_add(ctx.vreg("v_scale_rd_b"), ctx.vreg("v_tmp0"),
                  ctx.vreg("v_tmp1"),
                  comment="laneId*4 + partition_n")
        ctx.s_mov(ctx.sreg("s_tmp0"), str(self._scale_b_lds_off),
                  comment=f"scale_b_lds_off = {self._scale_b_lds_off}")
        ctx.v_add(ctx.vreg("v_scale_rd_b"), ctx.vreg("v_scale_rd_b"),
                  ctx.sreg("s_tmp0"),
                  comment="+ lds_base_b")
        ctx.raw("")

    # -- DTL emission -------------------------------------------------------

    def emit_dtl_loads(self) -> None:
        """Emit DTL loads for pre-swizzled scale A and B into LDS."""
        ctx = self._ctx
        ctx.inst("s_mov_b32", "m0",
                 ctx.sreg("s_lds_wr_scale_a"),
                 comment="m0 = LDS scale A write base")
        ctx.inst("buffer_load_dwordx4",
                 ctx.vreg("v_dtl_off_scale_a_lds"),
                 ctx.sreg("s_srd_scale_a", 0, 4),
                 "0", "offen offset:0, lds",
                 comment="DTL scale A (pre-swizzled)")
        ctx.inst("s_mov_b32", "m0",
                 ctx.sreg("s_lds_wr_scale_b"),
                 comment="m0 = LDS scale B write base")
        ctx.inst("buffer_load_dwordx4",
                 ctx.vreg("v_dtl_off_scale_b_lds"),
                 ctx.sreg("s_srd_scale_b", 0, 4),
                 "0", "offen offset:0, lds",
                 comment="DTL scale B (pre-swizzled)")

    def toggle_write(self) -> None:
        """Toggle scale LDS write bases for double-buffering."""
        ctx = self._ctx
        ctx.inst("s_xor_b32", ctx.sreg("s_lds_wr_scale_a"),
                 ctx.sreg("s_lds_wr_scale_a"),
                 ctx.sreg("s_scale_db_swap"),
                 comment="wr_scale_a ^= swap")
        ctx.inst("s_xor_b32", ctx.sreg("s_lds_wr_scale_b"),
                 ctx.sreg("s_lds_wr_scale_b"),
                 ctx.sreg("s_scale_db_swap"),
                 comment="wr_scale_b ^= swap")

    def toggle_read(self) -> None:
        """Toggle scale ds_read base VGPRs for double-buffering."""
        ctx = self._ctx
        ctx.inst("v_xor_b32", ctx.vreg("v_scale_rd_a"),
                 ctx.sreg("s_scale_db_swap"),
                 ctx.vreg("v_scale_rd_a"),
                 comment="scale_rd_a ^= swap")
        ctx.inst("v_xor_b32", ctx.vreg("v_scale_rd_b"),
                 ctx.sreg("s_scale_db_swap"),
                 ctx.vreg("v_scale_rd_b"),
                 comment="scale_rd_b ^= swap")

    # -- ds_read emission ---------------------------------------------------

    def emit_read_a(self, mi: int, ki: int) -> None:
        """Emit ds_read_b32 for scale A group containing (mi, ki)."""
        # Only emit for primary mi in each group (mi % 2 == 0)
        if mi % 2 != 0:
            return
        group = mi // 2
        ctx = self._ctx
        offset = group * 256
        name = f"v_scale_a_g{group}"
        ctx.ds_read(ctx.vreg(name), ctx.vreg("v_scale_rd_a"),
                    offset=offset, width=1,
                    comment=f"scale A group{group} (mi={mi},{mi+1})")

    def emit_read_b(self, ni: int, ki: int) -> None:
        """Emit ds_read_b32 for scale B group containing (ni, ki)."""
        if ni % 2 != 0:
            return
        group = ni // 2
        ctx = self._ctx
        offset = group * 256
        name = f"v_scale_b_g{group}"
        ctx.ds_read(ctx.vreg(name), ctx.vreg("v_scale_rd_b"),
                    offset=offset, width=1,
                    comment=f"scale B group{group} (ni={ni},{ni+1})")

    # -- SRD advance --------------------------------------------------------

    def advance(self) -> None:
        """Advance both scale SRDs by scale_k_stride bytes."""
        ctx = self._ctx
        stride = self._scale_k_stride
        for srd_name in ["s_srd_scale_a", "s_srd_scale_b"]:
            ctx.inst("s_add_u32", ctx.sreg(srd_name, 0, 1),
                     ctx.sreg(srd_name, 0, 1), str(stride),
                     comment=f"{srd_name} += {stride}")
            ctx.inst("s_addc_u32", ctx.sreg(srd_name, 1, 1),
                     ctx.sreg(srd_name, 1, 1), "0", comment="carry")

    # -- per-index stubs (for ScaleBlock compat) ----------------------------

    def emit_load_a(self, mi: int, ki: int) -> None:
        self.emit_read_a(mi, ki)

    def emit_load_b(self, ni: int, ki: int) -> None:
        self.emit_read_b(ni, ki)

    # -- composable K-loop integration -------------------------------------

    @property
    def has_scales(self) -> bool:
        return True

    @property
    def has_cross_iter_prefetch(self) -> bool:
        return False

    def precompute_soffsets(self) -> None:
        self.alloc_registers()
        self.emit_setup()

    def num_initial_inflight(self, partition_m: int) -> int:
        return 2  # 1 DTL scale A + 1 DTL scale B

    def cross_iter_inflight(self, partition_m: int, nr: int) -> int:
        return 0

    def emit_scale_wait(self, loader: 'GlobalLoader') -> None:
        pass  # Scales read via lgkmcnt, not vmcnt

    def emit_subtile_wait(self, loader: 'GlobalLoader', st_idx: int) -> None:
        pass  # Scales read via lgkmcnt

    def emit_mfma(self, ctx: AsmContext, mfma: MfmaConfig, acc: str,
                  a_reg: str, b_reg: str,
                  mi: int, ni: int, ki: int) -> None:
        """Emit MFMA with op_sel for pre-swizzled scale bytes."""
        sa_name = self._scale_a_names.get((mi, ki))
        sb_name = self._scale_b_names.get((ni, ki))
        if sa_name is None or sb_name is None:
            ctx.inst(mfma.instruction_name, acc, a_reg, b_reg, acc,
                     ctx.vreg("v_mxscale"), ctx.vreg("v_mxscale"),
                     f"cbsz:{mfma.cbsz} blgp:{mfma.blgp}",
                     comment=f"MFMA m{mi}_n{ni}_k{ki}")
            return

        a_sel = mi % 2
        b_sel = ni % 2
        hi_a = ki
        hi_b = ki
        ctx.inst(mfma.instruction_name, acc, a_reg, b_reg, acc,
                 ctx.vreg(sa_name), ctx.vreg(sb_name),
                 f"op_sel:[{a_sel},{b_sel}] op_sel_hi:[{hi_a},{hi_b}]"
                 f" cbsz:{mfma.cbsz} blgp:{mfma.blgp}",
                 comment=f"MFMA m{mi}_n{ni}_k{ki}")

    # -- Compatibility stubs -----------------------------------------------

    def emit_initial_loads(self, partition_m: int) -> None:
        pass  # DTL handled by DTLLoader

    def emit_loop_loads(self) -> None:
        pass  # DTL handled by DTLLoader
