"""Concrete LDSStream implementations.

Each stream handles one data channel (A data, B data, scale A,
scale B) in the unified LDSStream interface.  Each stream fully
owns its codegen -- advance, toggle, load, write -- so that
``emit_wiring.py`` can wire every stream identically with no
special-case branches.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .lds_stream import LDSStream

if TYPE_CHECKING:
    from ..emit.context import AsmContext
    from ..problem import TileConfig, GemmProblem

__all__ = [
    "DTLDataStream",
    "ScaleStream",
    ]


class DTLDataStream(LDSStream):
    """Matrix data (A or B) loaded via DTL (buffer_load_dwordx4 ... lds).

    Data goes directly from global memory to LDS via hardware DTL path.
    No VGPR intermediate. LDS writes are implicit (no emit_lds_writes).

    After construction, call :meth:`set_codegen` to wire a
    ``GlobalLoader`` and ``LDSReader`` before the first K-loop
    iteration.

    Args:
        matrix: "a" or "b".
        tile: Tile configuration.
        problem: GEMM problem (for element_bytes).
    """

    def __init__(self, matrix: str, tile: 'TileConfig',
                 problem: 'GemmProblem') -> None:
        assert matrix in ("a", "b")
        self._matrix = matrix
        # Codegen references -- set via set_codegen() before emission
        self._loader = None   # GlobalLoader (for emit_global_loads / advance)
        self._reader = None   # LDSReader   (for toggle_read on "a")

        self._tile = tile
        self._elem = problem.element_bytes
        self._lds_offset = 0

        tpr = int(tile.unroll_k * self._elem) // 16
        rpl = tile.block_size // tpr
        wg_dim = tile.wg_m if matrix == "a" else tile.wg_n
        self._num_loads = wg_dim // rpl
        self._k_stride = int(tile.unroll_k * self._elem)

        lds_row = int(rpl * tile.unroll_k * self._elem)
        self._region = lds_row * self._num_loads + tile.lds_pad * (self._num_loads - 1)

    # -- late-binding codegen references --------------------------------

    def set_codegen(self, loader: object, reader: object) -> None:
        """Wire the GlobalLoader and LDSReader used for emission.

        Called by ``wire_emit_callbacks`` before any ops fire, so the
        stream can delegate load/advance/toggle to the real codegen
        objects without emit_wiring needing special-case branches.

        Args:
            loader: ``GlobalLoader`` (DTLLoader or BufferLoader).
            reader: ``LDSReader`` for toggle_read (A buffer ping-pong).
        """
        self._loader = loader
        self._reader = reader

    @property
    def name(self) -> str:
        return f"data_{self._matrix}"

    @property
    def region_size(self) -> int:
        return self._region

    @property
    def num_global_loads(self) -> int:
        return self._num_loads

    @property
    def needs_lds_write(self) -> bool:
        return False  # DTL: hardware writes directly to LDS

    def setup(self, ctx: 'AsmContext', lds_offset: int) -> None:
        """Record LDS base offset (SRD setup is in kloop/setup.py)."""
        self._lds_offset = lds_offset

    def emit_global_loads(self, ctx: 'AsmContext') -> None:
        """Issue DTL loads via the wired GlobalLoader."""
        if self._matrix == "a":
            self._loader._emit_dtl_loads_a()
        else:
            self._loader._emit_dtl_loads_b()

    def emit_lds_writes(self, ctx: 'AsmContext') -> None:
        pass  # DTL: no VGPR intermediate

    def read_op_count(self) -> int:
        mr = self._tile.mfma_m_repeat if self._matrix == "a" else self._tile.mfma_n_repeat
        ki = self._tile.k_iterations
        return mr * ki

    def advance(self, ctx: 'AsmContext') -> None:
        """Advance the data SRD by one K-step."""
        srd = f"s_srd_{self._matrix}"
        stride = self._loader.k_stride
        ctx.inst("s_add_u32", ctx.sreg(srd, 0, 1),
                 ctx.sreg(srd, 0, 1), str(stride),
                 comment=f"{srd} += {stride}")
        ctx.inst("s_addc_u32", ctx.sreg(srd, 1, 1),
                 ctx.sreg(srd, 1, 1), "0", comment="carry")

    def toggle_write(self, ctx: 'AsmContext') -> None:
        """Toggle the LDS write base for double-buffering."""
        sg = f"s_lds_wr_{self._matrix}_sg"
        ctx.inst("s_add_u32", ctx.sreg(sg),
                 ctx.sreg(sg), ctx.sreg("s_lds_db_step"),
                 comment=f"{sg} += db")

    def toggle_read(self, ctx: 'AsmContext') -> None:
        """Toggle LDS read bases for double-buffering.

        For matrix A: delegates to LDSReader.toggle_read() which
        handles both A and B read pointers in one call.
        For matrix B: no-op (already handled by the A toggle).
        """
        if self._matrix == "a":
            self._reader.toggle_read()
        # matrix "b": no-op -- reader.toggle_read() covers both


class ScaleStream(LDSStream):
    """MX scale data loaded via true DTL (buffer_load_dwordx4 ... lds).

    Uses hardware DirectToLDS to load pre-swizzled scale data from
    global memory directly into LDS, bypassing VGPRs entirely.  Each
    of 256 threads loads 16 bytes -> 4096 bytes per matrix per K-step.

    DTL voffset = (tid%16)*16 + (tid/16)*strideMXS.
    m0 = LDS write base for this matrix's scale region.
    ds_read base = wave_partition*512 + laneId*4 + lds_scale_base.

    Args:
        matrix: "a" or "b".
        tile: Tile configuration.
        scale_k_stride: Bytes per K-step in the scale SRD (typically 256).
    """

    def __init__(self, matrix: str, tile: 'TileConfig',
                 scale_k_stride: int = 256) -> None:
        assert matrix in ("a", "b")
        self._matrix = matrix
        self._tile = tile
        self._scale_k_stride = scale_k_stride
        self._lds_offset = 0
        self._region = 4096  # 256 threads * 16 bytes

    @property
    def name(self) -> str:
        return f"scale_{self._matrix}"

    @property
    def region_size(self) -> int:
        return self._region

    @property
    def num_global_loads(self) -> int:
        if self._region == 0:
            return 0
        return 1  # single buffer_load_dwordx4 ... lds per matrix

    @property
    def needs_lds_write(self) -> bool:
        # True DTL: data goes directly from global to LDS, no ds_write
        return False

    def setup(self, ctx: 'AsmContext', lds_offset: int) -> None:
        if self._region == 0:
            return
        self._lds_offset = lds_offset

    def emit_global_loads(self, ctx: 'AsmContext') -> None:
        if self._region == 0:
            return
        # Set m0 to LDS write base for this matrix's scale region
        wr_base = f"s_lds_wr_scale_{self._matrix}"
        ctx.inst("s_mov_b32", "m0", ctx.sreg(wr_base),
                 comment=f"m0 = scale {self._matrix.upper()} LDS write base")
        # True DTL: buffer_load_dwordx4 vaddr, srd, soffset, offen lds
        # vaddr = per-lane offset, data goes to LDS at m0+vaddr
        srd = f"s_srd_scale_{self._matrix}"
        voff = f"v_dtl_off_scale_{self._matrix}_lds"
        ctx.inst("buffer_load_dwordx4",
                 ctx.vreg(voff), ctx.sreg(srd, 0, 4),
                 "0", "offen offset:0, lds",
                 comment=f"scale {self._matrix.upper()} DTL to LDS")

    def emit_lds_writes(self, ctx: 'AsmContext') -> None:
        # No-op: true DTL writes directly to LDS
        pass

    def read_op_count(self) -> int:
        # One read per 2-mi group (LDS scales group 2 mi values)
        mr = self._tile.mfma_m_repeat if self._matrix == "a" else self._tile.mfma_n_repeat
        return (mr + 1) // 2

    def advance(self, ctx: 'AsmContext') -> None:
        if self._region == 0:
            return  # VMEM path: advance handled by VMEMScaleLoader
        srd = f"s_srd_scale_{self._matrix}"
        stride = self._scale_k_stride
        ctx.inst("s_add_u32", ctx.sreg(srd, 0, 1),
                 ctx.sreg(srd, 0, 1), str(stride),
                 comment=f"{srd} += {stride}")
        ctx.inst("s_addc_u32", ctx.sreg(srd, 1, 1),
                 ctx.sreg(srd, 1, 1), "0", comment="carry")

    def toggle_write(self, ctx: 'AsmContext') -> None:
        if self._region == 0:
            return
        sg = f"s_lds_wr_scale_{self._matrix}"
        ctx.inst("s_add_u32", ctx.sreg(sg),
                 ctx.sreg(sg), ctx.sreg("s_lds_db_step"),
                 comment=f"wr_scale_{self._matrix} += db_step")

    def toggle_read(self, ctx: 'AsmContext') -> None:
        if self._region == 0:
            return
        rd = f"v_scale_rd_{self._matrix}"
        ctx.v_add(ctx.vreg(rd), ctx.vreg(rd),
                  ctx.sreg("s_lds_db_step"),
                  comment=f"scale_rd_{self._matrix} += db_step")
