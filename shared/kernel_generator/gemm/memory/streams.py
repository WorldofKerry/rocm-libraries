"""Concrete LDSStream implementations.

Wraps the existing loader/reader codegen into the unified LDSStream
interface. Each stream handles one data channel (A data, B data,
scale A, scale B).

Phase 1 of migration: these wrap existing code and are not yet
wired into the pipeline. They will replace GlobalLoader/ScaleLoader
in Phase 4.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .lds_stream import LDSStream

if TYPE_CHECKING:
    from ..emit.context import AsmContext
    from ..problem import TileConfig, GemmProblem, MfmaConfig

__all__ = [
    "DTLDataStream",
    "ScaleStream",
    "NullStream",
]


class DTLDataStream(LDSStream):
    """Matrix data (A or B) loaded via DTL (buffer_load_dwordx4 ... lds).

    Data goes directly from global memory to LDS via hardware DTL path.
    No VGPR intermediate. LDS writes are implicit (no emit_lds_writes).

    Args:
        matrix: "a" or "b".
        tile: Tile configuration.
        problem: GEMM problem (for element_bytes).
    """

    def __init__(self, matrix: str, tile: 'TileConfig',
                 problem: 'GemmProblem') -> None:
        assert matrix in ("a", "b")
        self._matrix = matrix
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
        self._lds_offset = lds_offset
        # SRD, voffset, soffset setup delegated to existing kloop/setup.py
        # (will be migrated here in Phase 4)

    def emit_global_loads(self, ctx: 'AsmContext') -> None:
        # Delegated to existing DTLLoader._emit_dtl_loads_a/b
        pass

    def emit_lds_writes(self, ctx: 'AsmContext') -> None:
        pass  # DTL: no VGPR intermediate

    def read_op_count(self) -> int:
        mr = self._tile.mfma_m_repeat if self._matrix == "a" else self._tile.mfma_n_repeat
        ki = self._tile.k_iterations
        return mr * ki

    def advance(self, ctx: 'AsmContext') -> None:
        srd = f"s_srd_{self._matrix}"
        ctx.inst("s_add_u32", ctx.sreg(srd, 0, 1),
                 ctx.sreg(srd, 0, 1), str(self._k_stride),
                 comment=f"{srd} += {self._k_stride}")
        ctx.inst("s_addc_u32", ctx.sreg(srd, 1, 1),
                 ctx.sreg(srd, 1, 1), "0", comment="carry")

    def toggle_write(self, ctx: 'AsmContext') -> None:
        sg = f"s_lds_wr_{self._matrix}_sg"
        ctx.inst("s_add_u32", ctx.sreg(sg),
                 ctx.sreg(sg), ctx.sreg("s_lds_db_step"),
                 comment=f"wr_{self._matrix} += db")

    def toggle_read(self, ctx: 'AsmContext') -> None:
        # Delegated to LDSReader.toggle_read() which handles
        # precomputed swizzle vs recompute vs plain toggle
        pass


class ScaleStream(LDSStream):
    """MX scale data loaded via buffer_load_dword + ds_write (2-step).

    Each thread loads 16 bytes from global into VGPRs, then writes to
    LDS via ds_write_b32. Uses strided voffset matching the pre-swizzled
    scale layout: (tid%16)*16 + (tid/16)*stride.

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
        return 4  # 4 x buffer_load_dword per matrix

    @property
    def needs_lds_write(self) -> bool:
        return True  # 2-step: buffer_load to VGPRs then ds_write

    def setup(self, ctx: 'AsmContext', lds_offset: int) -> None:
        self._lds_offset = lds_offset

    def emit_global_loads(self, ctx: 'AsmContext') -> None:
        # Issue 4 buffer_load_dword into tmp VGPRs
        srd = f"s_srd_scale_{self._matrix}"
        voff = f"v_dtl_off_scale_{self._matrix}_lds"
        base_tmp = 0 if self._matrix == "a" else 4
        for dw in range(4):
            ctx.inst("buffer_load_dword", ctx.vreg(f"v_tmp{base_tmp + dw}"),
                     ctx.vreg(voff), ctx.sreg(srd, 0, 4),
                     "0", f"offen offset:{dw * 4}",
                     comment=f"scale {self._matrix.upper()} dword {dw}")

    def emit_lds_writes(self, ctx: 'AsmContext') -> None:
        # Write 4 dwords to LDS at tid*16 + lds_write_base
        wr_base = f"s_lds_wr_scale_{self._matrix}"
        tmp_addr = "v_tmp8" if self._matrix == "a" else "v_tmp9"
        base_tmp = 0 if self._matrix == "a" else 4
        ctx.v_lshl(ctx.vreg("v_tmp8" if self._matrix == "a" else "v_tmp9"),
                    ctx.vreg("v_tid"), 4,
                    comment="tid * 16")
        ctx.v_add(ctx.vreg(tmp_addr), ctx.vreg(tmp_addr),
                  ctx.sreg(wr_base),
                  comment=f"LDS addr = wr_base_{self._matrix} + tid*16")
        for dw in range(4):
            ctx.inst("ds_write_b32",
                     ctx.vreg(tmp_addr), ctx.vreg(f"v_tmp{base_tmp + dw}"),
                     f"offset:{dw * 4}",
                     comment=f"scale {self._matrix.upper()} dw{dw} -> LDS")

    def read_op_count(self) -> int:
        # One read per 2-mi group (LDS scales group 2 mi values)
        mr = self._tile.mfma_m_repeat if self._matrix == "a" else self._tile.mfma_n_repeat
        return (mr + 1) // 2

    def advance(self, ctx: 'AsmContext') -> None:
        srd = f"s_srd_scale_{self._matrix}"
        stride = self._scale_k_stride
        ctx.inst("s_add_u32", ctx.sreg(srd, 0, 1),
                 ctx.sreg(srd, 0, 1), str(stride),
                 comment=f"{srd} += {stride}")
        ctx.inst("s_addc_u32", ctx.sreg(srd, 1, 1),
                 ctx.sreg(srd, 1, 1), "0", comment="carry")

    def toggle_write(self, ctx: 'AsmContext') -> None:
        sg = f"s_lds_wr_scale_{self._matrix}"
        ctx.inst("s_add_u32", ctx.sreg(sg),
                 ctx.sreg(sg), ctx.sreg("s_lds_db_step"),
                 comment=f"wr_scale_{self._matrix} += db_step")

    def toggle_read(self, ctx: 'AsmContext') -> None:
        rd = f"v_scale_rd_{self._matrix}"
        ctx.v_add(ctx.vreg(rd), ctx.vreg(rd),
                  ctx.sreg("s_lds_db_step"),
                  comment=f"scale_rd_{self._matrix} += db_step")


class NullStream(LDSStream):
    """No-op stream for non-MX kernels. Zero cost, zero LDS."""

    def __init__(self, matrix: str = "a") -> None:
        self._matrix = matrix

    @property
    def name(self) -> str:
        return f"null_{self._matrix}"

    @property
    def region_size(self) -> int:
        return 0

    @property
    def num_global_loads(self) -> int:
        return 0

    @property
    def needs_lds_write(self) -> bool:
        return False

    @property
    def has_reads(self) -> bool:
        return False

    def setup(self, ctx, lds_offset):
        pass

    def emit_global_loads(self, ctx):
        pass

    def emit_lds_writes(self, ctx):
        pass

    def read_op_count(self) -> int:
        return 0

    def advance(self, ctx):
        pass

    def toggle_write(self, ctx):
        pass

    def toggle_read(self, ctx):
        pass
