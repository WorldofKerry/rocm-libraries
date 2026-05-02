"""Scale loading strategies for MX (MXFP4) GEMM kernels.

MX data types require per-tile E8M0 scale factors loaded from global
memory alongside the data tiles.  This module extracts scale loading
logic from the monolithic ``partitioned.py`` K-loop into composable
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
from typing import Dict, List, Optional

from ..schedule.slot_placer import PlacedOp, Path
from ..schedule.data_stream import (
    PrefetchOp,
    build_prefetch_path,
    compute_register_last_use,
    place_prefetch_path,
)

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
    def alloc_registers(self):
        """Allocate VGPRs/SGPRs for scale storage."""

    @abstractmethod
    def emit_setup(self):
        """Set up scale SRDs, voffsets, soffsets."""

    @abstractmethod
    def emit_initial_loads(self, partition_m: int):
        """Load scales for the first subtile + all B scales (prologue)."""

    @abstractmethod
    def emit_loop_loads(self):
        """Issue scale loads within the K-loop (after SRD advance)."""

    @abstractmethod
    def advance(self):
        """Advance scale SRDs by one K-step."""

    @abstractmethod
    def make_prefetch_ops(self, all_mfma_ops, partition_m, nr):
        """Return list of ``PlacedOp`` s for cross-iteration scale prefetch."""

    @abstractmethod
    def make_subtile_prefetch_paths(self, partition_m, num_subtiles):
        """Return list of ``Path`` s for subtile-level scale_a prefetch."""

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

    def alloc_registers(self):
        pass

    def emit_setup(self):
        pass

    def emit_initial_loads(self, partition_m: int):
        pass

    def emit_loop_loads(self):
        pass

    def advance(self):
        pass

    def make_prefetch_ops(self, all_mfma_ops, partition_m, nr):
        return []

    def make_subtile_prefetch_paths(self, partition_m, num_subtiles):
        return []

    @property
    def scale_names_a(self) -> dict:
        return {}

    @property
    def scale_names_b(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
    @property
    def has_scales(self) -> bool:
        return False

    @property
    def has_cross_iter_prefetch(self) -> bool:
        return False

    def precompute_soffsets(self):
        pass

    def num_initial_inflight(self, partition_m: int) -> int:
        return 0

    def cross_iter_inflight(self, partition_m: int, nr: int) -> int:
        return 0

    def emit_scale_wait(self, loader):
        pass

    def emit_subtile_wait(self, loader, st_idx: int):
        pass

    def emit_mfma(self, ctx, mfma, acc, a_reg, b_reg, mi, ni, ki):
        pass

    def place_cross_iter_prefetch(self, placer, all_mfma_ops,
                                  partition_m, nr):
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

    def __init__(self, ctx, tile, swizzled: bool = False):
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

    def alloc_registers(self):
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

    def emit_setup(self):
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

    def emit_initial_loads(self, partition_m: int):
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

    def emit_loop_loads(self):
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

    def advance(self):
        """Advance both scale SRDs by ``scale_k_stride`` bytes."""
        ctx = self._ctx
        stride = self._scale_k_stride
        for srd_name in ["s_srd_scale_a", "s_srd_scale_b"]:
            ctx.inst("s_add_u32", ctx.sreg(srd_name, 0, 1),
                     ctx.sreg(srd_name, 0, 1), str(stride),
                     comment=f"{srd_name} += {stride}")
            ctx.inst("s_addc_u32", ctx.sreg(srd_name, 1, 1),
                     ctx.sreg(srd_name, 1, 1), "0", comment="carry")

    # -- cross-iteration prefetch -------------------------------------------

    def make_prefetch_ops(self, all_mfma_ops, partition_m, nr):
        """Build ``PrefetchOp`` list and a ``Path`` for cross-iteration prefetch.

        Linear mode only.  After each scale register's last MFMA consumer
        in the current iteration, we issue a ``buffer_load_dword`` that
        loads the *next* iteration's scale into the same VGPR.  The SRD
        advance is bundled as the first op in the returned path.

        Returns:
            Tuple ``(prefetch_loads, path)`` where *prefetch_loads* is a
            list of ``PrefetchOp`` and *path* is a ``Path`` ready for
            ``place_prefetch_path``.  Returns ``([], None)`` for swizzled
            mode.
        """
        if self._swizzled:
            return ([], None)

        ctx = self._ctx

        ki_count = self._ki_count
        ki_bytes = self._mfma.k // self._mx_block

        # Determine last-use slot for each scale register
        scale_reg_names: List[str] = []
        for mi_ in range(partition_m):
            for ki_ in range(ki_count):
                scale_reg_names.append(f"v_scale_a_m{mi_}k{ki_}")
        for ni_ in range(nr):
            for ki_ in range(ki_count):
                scale_reg_names.append(f"v_scale_b_n{ni_}k{ki_}")

        last_use = compute_register_last_use(all_mfma_ops, scale_reg_names)

        prefetch_loads: List[PrefetchOp] = []

        for mi_ in range(partition_m):
            for ki_ in range(ki_count):
                reg = f"v_scale_a_m{mi_}k{ki_}"
                off = ki_ * ki_bytes
                off_str = f" offset:{off}" if off > 0 else ""

                def _mk_pf_a(m=mi_, k=ki_, o=off_str):
                    def emit():
                        soff = "0" if m == 0 else ctx.sreg(f"s_soff_sa_{m}")
                        ctx.inst("buffer_load_dword",
                                 ctx.vreg(f"v_scale_a_m{m}k{k}"),
                                 ctx.vreg("v_dtl_off_scale_a"),
                                 ctx.sreg("s_srd_scale_a", 0, 4),
                                 soff, f"offen{o}",
                                 comment=f"scale A mi={m} ki={k} (next K)")
                    return emit

                prefetch_loads.append(PrefetchOp(
                    reg_name=reg, emit_fn=_mk_pf_a(),
                    earliest_slot=last_use.get(reg, 0)))

        for ni_ in range(nr):
            for ki_ in range(ki_count):
                reg = f"v_scale_b_n{ni_}k{ki_}"
                off = ki_ * ki_bytes
                off_str = f" offset:{off}" if off > 0 else ""

                def _mk_pf_b(n=ni_, k=ki_, o=off_str):
                    def emit():
                        soff = "0" if n == 0 else ctx.sreg(f"s_soff_sb_{n}")
                        ctx.inst("buffer_load_dword",
                                 ctx.vreg(f"v_scale_b_n{n}k{k}"),
                                 ctx.vreg("v_dtl_off_scale_b"),
                                 ctx.sreg("s_srd_scale_b", 0, 4),
                                 soff, f"offen{o}",
                                 comment=f"scale B ni={n} ki={k} (next K)")
                    return emit

                prefetch_loads.append(PrefetchOp(
                    reg_name=reg, emit_fn=_mk_pf_b(),
                    earliest_slot=last_use.get(reg, 0)))

        pf_path = build_prefetch_path(
            all_mfma_ops, prefetch_loads,
            srd_advance_fn=self.advance)

        return (prefetch_loads, pf_path)

    # -- subtile prefetch paths ---------------------------------------------

    def make_subtile_prefetch_paths(self, partition_m, num_subtiles):
        """Build ``Path`` list for subtile-level scale_a prefetch.

        During subtile *N* 's MFMAs we issue ``buffer_load_dword`` for
        subtile *N+1* 's scale_a values.  This hides the global memory
        latency behind compute.

        Returns:
            List of ``Path`` objects, one per prefetch window (length =
            ``num_subtiles - 1``).  Empty list for swizzled mode.
        """
        if self._swizzled:
            return []

        ctx = self._ctx
        paths: List[Path] = []

        ki_count = self._ki_count
        ki_bytes = self._mfma.k // self._mx_block

        for st in range(num_subtiles - 1):
            path_ops: List[PlacedOp] = []
            for mi_in_st in range(partition_m):
                target_mi = (st + 1) * partition_m + mi_in_st
                for ki_ in range(ki_count):
                    off = ki_ * ki_bytes
                    off_str = f" offset:{off}" if off > 0 else ""

                    def _mk_scale_load(mi_=target_mi, k=ki_, o=off_str):
                        def emit():
                            soff = ("0" if mi_ == 0
                                    else ctx.sreg(f"s_soff_sa_{mi_}"))
                            ctx.inst("buffer_load_dword",
                                     ctx.vreg(f"v_scale_a_m{mi_}k{k}"),
                                     ctx.vreg("v_dtl_off_scale_a"),
                                     ctx.sreg("s_srd_scale_a", 0, 4),
                                     soff, f"offen{o}",
                                     comment=f"scale A mi={mi_} ki={k} (pf)")
                        return emit

                    path_ops.append(PlacedOp(
                        emit_fn=_mk_scale_load(), op_type="buffer_load",
                        comment=f"scale_a m{target_mi}k{ki_}"))
            paths.append(Path(ops=path_ops, reverse=False,
                              module_id=100 + st))

        return paths

    # -- composable K-loop integration -------------------------------------

    @property
    def has_scales(self) -> bool:
        """True if this loader provides scale operands for MFMAs."""
        return True

    @property
    def has_cross_iter_prefetch(self) -> bool:
        """True if linear mode uses cross-iteration prefetch."""
        return not self._swizzled

    def precompute_soffsets(self):
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

    def emit_scale_wait(self, loader):
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

    def emit_subtile_wait(self, loader, st_idx: int):
        """Emit vmcnt wait at subtile boundary for prefetched scale_a."""
        if self._swizzled:
            return
        ctx = self._ctx
        num_dtl = loader.num_inflight
        ctx.s_waitcnt(f"vmcnt({num_dtl})",
                      comment=f"wait scale_a subtile {st_idx} (leave DTL)")

    def emit_mfma(self, ctx, mfma, acc, a_reg, b_reg, mi, ni, ki):
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

    def place_cross_iter_prefetch(self, placer, all_mfma_ops,
                                  partition_m, nr):
        """Place cross-iteration prefetch ops into the SlotPlacer schedule."""
        if self._swizzled:
            return

        prefetch_loads, pf_path = self.make_prefetch_ops(
            all_mfma_ops, partition_m, nr)
        if pf_path is not None:
            place_prefetch_path(placer, pf_path, all_mfma_ops,
                                prefetch_loads)
