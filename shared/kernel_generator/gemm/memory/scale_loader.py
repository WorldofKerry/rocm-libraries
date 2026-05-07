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
from typing import TYPE_CHECKING, Dict

from ..emit.context import AsmContext
from ..problem import MfmaConfig, TileConfig

if TYPE_CHECKING:
    from .mfma_emitter import MFMAEmitter

__all__ = [
    "ScaleLoader",
    "NullScaleLoader",
    "VMEMScaleLoader",
    "LDSScaleLoader",
]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ScaleLoader(ABC):
    """Base class for MX scale loading strategies.

    Subclasses own all scale-related decisions: which LDS streams
    to create, which MFMAEmitter to use, and how much LDS to reserve.
    Callers never check flags -- they call methods on the loader.
    """

    @abstractmethod
    def alloc_registers(self) -> None:
        """Allocate VGPRs/SGPRs for scale storage."""

    @abstractmethod
    def emit_setup(self) -> None:
        """Set up scale SRDs, voffsets, soffsets."""

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

    def streams(self, tile: TileConfig) -> list:
        """LDS streams this loader needs (empty for non-LDS loaders)."""
        return []

    def mfma_emitter(self, mfma: MfmaConfig) -> MFMAEmitter:
        """Create the appropriate MFMAEmitter for this scale strategy."""
        from .mfma_emitter import MFMAEmitter
        return MFMAEmitter.for_non_mx(mfma)

    def lds_bytes_per_buffer(self) -> int:
        """LDS bytes per double-buffer slot for scale data."""
        return 0


# ---------------------------------------------------------------------------
# Null (no-op) implementation
# ---------------------------------------------------------------------------

class NullScaleLoader(ScaleLoader):
    """No-op loader for non-MX data types."""

    def alloc_registers(self) -> None:
        pass

    def emit_setup(self) -> None:
        pass


    def advance(self) -> None:
        pass

    @property
    def scale_names_a(self) -> dict:
        return {}

    @property
    def scale_names_b(self) -> dict:
        return {}


# ---------------------------------------------------------------------------


    def precompute_soffsets(self) -> None:
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

    def mfma_emitter(self, mfma: MfmaConfig) -> MFMAEmitter:
        """VMEM scales: MFMA reads scale from VGPRs loaded via buffer_load."""
        from .mfma_emitter import MFMAEmitter
        return MFMAEmitter.for_vmem_scales(mfma, self.scale_names_a, self.scale_names_b)

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
        """Precompute per-lane voffset and per-mi soffset for raw scale format.

        Raw scale layout: scale_A[m][k_block] stored contiguously with
        inner_stride = strideMXSA0 / 32 bytes per M-row.

        Per-lane voffset = (wave_m * mr * 16 + lane_id % 16) * inner_stride
        Per-mi soffset   = mi * 16 * inner_stride
        """
        if self._swizzled:
            return

        ctx = self._ctx
        mfma = self._mfma
        mr, nr = self._mr, self._nr

        # stride_scale_a/b = strideMXSA0 from kernarg = K/32 = per-M-row stride
        # Per-lane voffset = (wave_m * mr * 16 + lane_id & 15) * stride
        ctx.comment("Scale A per-lane voffset")
        if not ctx.has("v_scale_voff_a"):
            ctx.alloc_vgpr_permanent(1, "v_scale_voff_a")
        ctx.v_mul(ctx.vreg("v_tmp0"),
                  str(mr * mfma.m), ctx.vreg("v_wave_m"),
                  comment=f"wave_m * {mr * mfma.m}")
        ctx.inst("v_and_b32", ctx.vreg("v_tmp1"),
                 ctx.vreg("v_lane_id"), "15",
                 comment="lane_id & 15 (M-row within MFMA tile)")
        ctx.inst("v_add_u32", ctx.vreg("v_tmp0"),
                 ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
                 comment="M-row relative to wave's start")
        ctx.inst("v_mul_lo_u32", ctx.vreg("v_scale_voff_a"),
                 ctx.sreg("s_stride_scale_a"), ctx.vreg("v_tmp0"),
                 comment="* stride_scale_a -> byte offset")
        ctx.raw("")

        # Per-lane voffset for scale B:
        ctx.comment("Scale B per-lane voffset")
        if not ctx.has("v_scale_voff_b"):
            ctx.alloc_vgpr_permanent(1, "v_scale_voff_b")
        ctx.v_mul(ctx.vreg("v_tmp0"),
                  str(nr * mfma.n), ctx.vreg("v_wave_n"),
                  comment=f"wave_n * {nr * mfma.n}")
        ctx.inst("v_and_b32", ctx.vreg("v_tmp1"),
                 ctx.vreg("v_lane_id"), "15",
                 comment="lane_id & 15 (N-row within MFMA tile)")
        ctx.inst("v_add_u32", ctx.vreg("v_tmp0"),
                 ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
                 comment="N-row relative to wave's start")
        ctx.inst("v_mul_lo_u32", ctx.vreg("v_scale_voff_b"),
                 ctx.sreg("s_stride_scale_b"), ctx.vreg("v_tmp0"),
                 comment="* stride_scale_b -> byte offset")
        ctx.raw("")

        # Per-mi soffsets: mi * 16 * stride (16 M-rows per MFMA tile)
        ctx.comment("Precompute scale soffsets")
        for mi_ in range(1, mr):
            soff_name = f"s_soff_sa_{mi_}"
            ctx.alloc_sgpr_permanent(1, soff_name)
            ctx.inst("s_mul_i32", ctx.sreg(soff_name),
                     ctx.sreg("s_stride_scale_a"),
                     str(mi_ * mfma.m),
                     comment=f"soff_a[{mi_}] = stride * {mi_ * mfma.m}")
        for ni_ in range(1, nr):
            soff_name = f"s_soff_sb_{ni_}"
            ctx.alloc_sgpr_permanent(1, soff_name)
            ctx.inst("s_mul_i32", ctx.sreg(soff_name),
                     ctx.sreg("s_stride_scale_b"),
                     str(ni_ * mfma.n),
                     comment=f"soff_b[{ni_}] = stride * {ni_ * mfma.n}")
        ctx.raw("")


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


    def streams(self, tile: TileConfig) -> list:
        """Return dummy scale streams so the graph builder creates read ops.

        VMEMScaleLoader loads scales directly to VGPRs via buffer_load,
        not through LDS. But the pipeline graph needs scale read ops to
        schedule the loads at the right point. These streams have zero
        LDS region (scales bypass LDS entirely).
        """
        from .streams import ScaleStream
        # Use region_size=0 since VMEM doesn't use LDS for scales
        sa = ScaleStream("a", tile)
        sb = ScaleStream("b", tile)
        sa._region = 0  # Override: no LDS needed
        sb._region = 0
        return [sa, sb]

    def emit_read_a(self, mi: int, ki: int) -> None:
        """Emit buffer_load_dword for scale A at (mi, ki).

        Uses per-lane v_scale_voff_a for M-row addressing and per-mi
        soffset for tile offset. Each ki adds offset:(ki*4) to select
        the correct 4-byte group of k-block scales.
        """
        if mi % 2 != 0:
            return
        ctx = self._ctx
        voff = ctx.vreg("v_scale_voff_a")
        srd = ctx.sreg("s_srd_scale_a", 0, 4)
        for ki2 in range(self._ki_count):
            name = self._scale_a_names.get((mi, ki2))
            if name is None:
                continue
            soff = ctx.sreg(f"s_soff_sa_{mi}") if mi > 0 else "0"
            k_off = ki2 * 4  # 4 bytes per ki (4 k-blocks of 32)
            off_str = f"offen offset:{k_off}" if k_off > 0 else "offen"
            ctx.inst("buffer_load_dword",
                     ctx.vreg(name), voff, srd, soff, off_str,
                     comment=f"scale A m{mi} k{ki2}")
            # Also load mi+1 if it exists (different VGPR)
            name2 = self._scale_a_names.get((mi + 1, ki2))
            if name2 and name2 != name:
                soff2 = ctx.sreg(f"s_soff_sa_{mi+1}")
                ctx.inst("buffer_load_dword",
                         ctx.vreg(name2), voff, srd, soff2, off_str,
                         comment=f"scale A m{mi+1} k{ki2}")

    def emit_read_b(self, ni: int, ki: int) -> None:
        """Emit buffer_load_dword for scale B at (ni, ki)."""
        if ni % 2 != 0:
            return
        ctx = self._ctx
        voff = ctx.vreg("v_scale_voff_b")
        srd = ctx.sreg("s_srd_scale_b", 0, 4)
        for ki2 in range(self._ki_count):
            name = self._scale_b_names.get((ni, ki2))
            if name is None:
                continue
            soff = ctx.sreg(f"s_soff_sb_{ni}") if ni > 0 else "0"
            k_off = ki2 * 4
            off_str = f"offen offset:{k_off}" if k_off > 0 else "offen"
            ctx.inst("buffer_load_dword",
                     ctx.vreg(name), voff, srd, soff, off_str,
                     comment=f"scale B n{ni} k{ki2}")
            name2 = self._scale_b_names.get((ni + 1, ki2))
            if name2 and name2 != name:
                soff2 = ctx.sreg(f"s_soff_sb_{ni+1}")
                ctx.inst("buffer_load_dword",
                         ctx.vreg(name2), voff, srd, soff2, off_str,
                         comment=f"scale B n{ni+1} k{ki2}")

    def precompute_soffsets(self) -> None:
        """Alias for emit_setup() -- called by ComposableKLoop."""
        self.alloc_registers()
        self.emit_setup()


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
    Per wave partition: 512 bytes (4 groups × 256).

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

    def streams(self, tile: TileConfig) -> list:
        """LDS scale streams for DTL-based scale loading."""
        from .streams import ScaleStream
        return [ScaleStream("a", tile), ScaleStream("b", tile)]

    def mfma_emitter(self, mfma: MfmaConfig) -> MFMAEmitter:
        """LDS scales: MFMA reads scale from VGPRs loaded via ds_read."""
        from .mfma_emitter import MFMAEmitter
        return MFMAEmitter.for_lds_scales(mfma, self.scale_names_a, self.scale_names_b)

    def lds_bytes_per_buffer(self) -> int:
        """Total LDS bytes for scale A + scale B per buffer."""
        return self.scale_lds_size

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

        # DTL voffset: (tid % 16) * 16 + (tid / 16) * stride
        # Each group of 16 threads reads 256 bytes contiguously, groups
        # are spaced by the scale stride (matches pre-swizzled layout).
        ctx.comment("Scale DTL voffset (strided): (tid%16)*16 + (tid/16)*stride")
        if not ctx.has("v_dtl_off_scale_a_lds"):
            ctx.alloc_vgpr_permanent(1, "v_dtl_off_scale_a_lds")
        if not ctx.has("v_dtl_off_scale_b_lds"):
            ctx.alloc_vgpr_permanent(1, "v_dtl_off_scale_b_lds")
        # intra-group offset = (tid % 16) * 16
        ctx.v_and(ctx.vreg("v_tmp0"), ctx.vreg("v_tid"), "15",
                  comment="tid % 16")
        ctx.v_lshl(ctx.vreg("v_tmp0"), ctx.vreg("v_tmp0"), 4,
                   comment="* 16 -> intra-group byte offset")
        # inter-group offset = (tid / 16) * stride_scale_a
        ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_tid"), 4,
                   comment="tid / 16 (group index)")
        ctx.v_mul(ctx.vreg("v_tmp1"),
                  ctx.sreg("s_stride_scale_a"), ctx.vreg("v_tmp1"),
                  comment="* stride_scale_a")
        ctx.v_add(ctx.vreg("v_dtl_off_scale_a_lds"),
                  ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
                  comment="scale A voffset")
        # scale B: same intra-group, different stride
        ctx.v_lshr(ctx.vreg("v_tmp1"), ctx.vreg("v_tid"), 4,
                   comment="tid / 16")
        ctx.v_mul(ctx.vreg("v_tmp1"),
                  ctx.sreg("s_stride_scale_b"), ctx.vreg("v_tmp1"),
                  comment="* stride_scale_b")
        ctx.v_add(ctx.vreg("v_dtl_off_scale_b_lds"),
                  ctx.vreg("v_tmp0"), ctx.vreg("v_tmp1"),
                  comment="scale B voffset")
        ctx.raw("")

        # ds_read base: wave_partition * 512 + laneId * 4 + lds_base
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
        ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_wave_m"), 9,
                    comment="wave_m * 512 (partition offset)")
        ctx.v_add(ctx.vreg("v_scale_rd_a"), ctx.vreg("v_tmp0"),
                  ctx.vreg("v_tmp1"),
                  comment="laneId*4 + partition_m")
        ctx.s_mov(ctx.sreg("s_tmp0"), str(self._scale_a_lds_off),
                  comment=f"scale_a_lds_off = {self._scale_a_lds_off}")
        ctx.v_add(ctx.vreg("v_scale_rd_a"), ctx.vreg("v_scale_rd_a"),
                  ctx.sreg("s_tmp0"),
                  comment="+ lds_base_a")

        # wave_n partition: waveId_n * 1024
        ctx.v_lshl(ctx.vreg("v_tmp1"), ctx.vreg("v_wave_n"), 9,
                    comment="wave_n * 512 (partition offset)")
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
        """Issue global loads for scale data into VGPRs (async).

        Call emit_scale_ds_writes() after vmcnt(0) to write to LDS.
        """
        ctx = self._ctx
        for dw in range(4):
            ctx.inst("buffer_load_dword", ctx.vreg(f"v_tmp{dw}"),
                     ctx.vreg("v_dtl_off_scale_a_lds"),
                     ctx.sreg("s_srd_scale_a", 0, 4),
                     "0", f"offen offset:{dw * 4}",
                     comment=f"scale A dword {dw}")
        for dw in range(4):
            ctx.inst("buffer_load_dword", ctx.vreg(f"v_tmp{4 + dw}"),
                     ctx.vreg("v_dtl_off_scale_b_lds"),
                     ctx.sreg("s_srd_scale_b", 0, 4),
                     "0", f"offen offset:{dw * 4}",
                     comment=f"scale B dword {dw}")

    def emit_scale_ds_writes(self) -> None:
        """Write scale VGPRs to LDS using the strided DTL voffset.

        The LDS layout must match the ds_read pattern used later.
        The DTL voffset = (tid%16)*16 + (tid/16)*stride, which places
        each group of 16 threads' data at stride-spaced intervals.
        """
        ctx = self._ctx
        # Use the same strided voffset as the DTL global read so the
        # LDS layout matches the ds_read addressing pattern.
        ctx.v_add(ctx.vreg("v_tmp9"),
                  ctx.vreg("v_dtl_off_scale_a_lds"),
                  ctx.sreg("s_lds_wr_scale_a"),
                  comment="LDS addr A = wr_base_a + strided_voff")
        for dw in range(4):
            ctx.inst("ds_write_b32",
                     ctx.vreg("v_tmp9"), ctx.vreg(f"v_tmp{dw}"),
                     f"offset:{dw * 4}",
                     comment=f"scale A dw{dw} -> LDS")
        ctx.v_add(ctx.vreg("v_tmp9"),
                  ctx.vreg("v_dtl_off_scale_b_lds"),
                  ctx.sreg("s_lds_wr_scale_b"),
                  comment="LDS addr B = wr_base_b + strided_voff")
        for dw in range(4):
            ctx.inst("ds_write_b32",
                     ctx.vreg("v_tmp9"), ctx.vreg(f"v_tmp{4 + dw}"),
                     f"offset:{dw * 4}",
                     comment=f"scale B dw{dw} -> LDS")

    def toggle_write(self) -> None:
        """Toggle scale LDS write bases for double-buffering."""
        ctx = self._ctx
        ctx.inst("s_add_u32", ctx.sreg("s_lds_wr_scale_a"),
                 ctx.sreg("s_lds_wr_scale_a"),
                 ctx.sreg("s_lds_db_step"),
                 comment="wr_scale_a += db_step")
        ctx.inst("s_add_u32", ctx.sreg("s_lds_wr_scale_b"),
                 ctx.sreg("s_lds_wr_scale_b"),
                 ctx.sreg("s_lds_db_step"),
                 comment="wr_scale_b += db_step")

    def toggle_read(self) -> None:
        """Toggle scale ds_read base VGPRs for double-buffering."""
        ctx = self._ctx
        ctx.v_add(ctx.vreg("v_scale_rd_a"),
                  ctx.vreg("v_scale_rd_a"),
                  ctx.sreg("s_lds_db_step"),
                  comment="scale_rd_a += db_step")
        ctx.v_add(ctx.vreg("v_scale_rd_b"),
                  ctx.vreg("v_scale_rd_b"),
                  ctx.sreg("s_lds_db_step"),
                  comment="scale_rd_b += db_step")

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


    def precompute_soffsets(self) -> None:
        self.alloc_registers()
        self.emit_setup()
