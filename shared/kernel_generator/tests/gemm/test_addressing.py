# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for address computation (addressing.py)."""
from __future__ import annotations

import pytest
from kernel_generator.gemm.addressing import AddressComputer
from kernel_generator.gemm.context import TileContext
from kernel_generator.gemm.problem import GemmProblem, TileConfig, MfmaConfig

try:
    import stinkytofu as _st
    HAS_ST = hasattr(_st, "LogicalModule")
except ImportError:
    HAS_ST = False

requires_st = pytest.mark.skipif(not HAS_ST, reason="stinkytofu C extension not available")


def _default_tile() -> TileConfig:
    return TileConfig(
        wg_m=128, wg_n=128, unroll_k=32,
        waves_m=2, waves_n=2,
        mfma=MfmaConfig.f16_16x16x16(),
        vector_width=8,
    )


def _default_problem() -> GemmProblem:
    return GemmProblem(m=4096, n=4096, k=4096)


# ===========================================================================
# Pure-Python offset calculations (no stinkytofu needed)
# ===========================================================================

class TestGlobalLoadCoords:
    def test_thread_0(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        row, col = ac.global_load_thread_coords_a(tid=0)
        assert row == 0
        assert col == 0

    def test_thread_1(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        row, col = ac.global_load_thread_coords_a(tid=1)
        assert row == 0
        assert col == 1  # col increments first

    def test_thread_8(self):
        """After vector_width=8 threads, row increments."""
        ac = AddressComputer(_default_problem(), _default_tile())
        row, col = ac.global_load_thread_coords_a(tid=8)
        assert row == 1
        assert col == 0

    def test_thread_255(self):
        """Last thread in the workgroup (block_size=256)."""
        ac = AddressComputer(_default_problem(), _default_tile())
        row, col = ac.global_load_thread_coords_a(tid=255)
        # 255 / 8 = 31, 255 % 8 = 7
        # cluster_m = min(256, 128) = 128, so 31 % 128 = 31
        assert row == 31
        assert col == 7

    def test_b_coords(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        row, col = ac.global_load_thread_coords_b(tid=10)
        # 10 / 8 = 1, 10 % 8 = 2
        assert row == 1
        assert col == 2


class TestLdsWriteOffset:
    def test_thread_0(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        off = ac.lds_write_offset_a(tid=0)
        assert off == 0  # (0 * 32 + 0) * 2 = 0

    def test_thread_1(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        off = ac.lds_write_offset_a(tid=1)
        # row=0, col=1 -> (0*32 + 1) * 2 = 2
        assert off == 2

    def test_thread_8(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        off = ac.lds_write_offset_a(tid=8)
        # row=1, col=0 -> (1*32 + 0) * 2 = 64
        assert off == 64

    def test_b_includes_offset(self):
        """B's LDS region starts after A's."""
        ac = AddressComputer(_default_problem(), _default_tile())
        off_b = ac.lds_write_offset_b(tid=0)
        # A occupies 128 * 32 * 2 = 8192 bytes
        assert off_b == 8192

    def test_b_thread_1(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        off_b = ac.lds_write_offset_b(tid=1)
        # 8192 + (0*32 + 1) * 2 = 8194
        assert off_b == 8194


class TestLdsReadOffset:
    def test_wave0_mi0_ki0_lane0(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        off = ac.lds_read_offset_a(wave_m=0, mfma_mi=0, ki=0, lane_id=0)
        # row = 0 + 0 + 0 = 0, col = 0 -> (0*32 + 0) * 2 = 0
        assert off == 0

    def test_wave1_mi0_ki0_lane0(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        off = ac.lds_read_offset_a(wave_m=1, mfma_mi=0, ki=0, lane_id=0)
        # row = 1*64 + 0 + 0 = 64, col = 0 -> (64*32) * 2 = 4096
        assert off == 4096

    def test_wave0_mi2_ki1_lane5(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        off = ac.lds_read_offset_a(wave_m=0, mfma_mi=2, ki=1, lane_id=5)
        mfma = _default_tile().mfma
        # row = 0 + 2*16 + (5 % 16) = 37
        # col = 1 * 16 = 16
        # offset = (37 * 32 + 16) * 2 = (1184 + 16) * 2 = 2400
        assert off == (37 * 32 + 16) * 2

    def test_b_includes_lds_offset(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        off_b = ac.lds_read_offset_b(wave_n=0, mfma_ni=0, ki=0, lane_id=0)
        # B starts at 8192 (wg_m * unroll_k * 2)
        assert off_b == 8192


class TestGlobalStoreOffset:
    def test_origin(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        off = ac.global_store_offset_d(
            wg_m=0, wg_n=0, wave_m=0, wave_n=0,
            mfma_mi=0, mfma_ni=0, lane_id=0, ldd=4096,
        )
        assert off == 0

    def test_lane_1(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        off = ac.global_store_offset_d(
            wg_m=0, wg_n=0, wave_m=0, wave_n=0,
            mfma_mi=0, mfma_ni=0, lane_id=1, ldd=4096,
        )
        # row = 1 % 16 = 1, col = 1 // 16 = 0
        # offset = (1 * 4096 + 0) * 2 = 8192
        assert off == 8192

    def test_wave1_mi1(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        off = ac.global_store_offset_d(
            wg_m=0, wg_n=0, wave_m=1, wave_n=0,
            mfma_mi=1, mfma_ni=0, lane_id=0, ldd=4096,
        )
        # row = 0 + 1*64 + 1*16 + 0 = 80
        # col = 0 + 0 + 0 + 0 = 0
        # offset = 80 * 4096 * 2 = 655360
        assert off == 80 * 4096 * 2


class TestEmbedTransforms:
    def test_lds_embed_a(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        embed = ac.lds_embed_a()
        # Verify: offset = row * 32 + col
        result = embed.forward({"row": 5, "col": 10})
        assert result["lds_a_offset"] == 5 * 32 + 10

    def test_lds_embed_b(self):
        ac = AddressComputer(_default_problem(), _default_tile())
        embed = ac.lds_embed_b()
        result = embed.forward({"row": 3, "col": 7})
        assert result["lds_b_offset"] == 3 * 32 + 7


# ===========================================================================
# Instruction emission tests (require stinkytofu)
# ===========================================================================

def _make_ctx_with_args(dry_run=False):
    """Create a TileContext with the standard kernel-arg bindings."""
    module = None
    if not dry_run:
        import stinkytofu as st
        module = st.LogicalModule("test_addr")
    ctx = TileContext(module=module)
    tile = _default_tile()
    problem = _default_problem()

    # Allocate the same bindings codegen_v2 would
    ctx.alloc_sgpr_permanent(2, "srd_A")
    ctx.alloc_sgpr_permanent(2, "srd_B")
    ctx.alloc_sgpr_permanent(2, "srd_D")
    ctx.alloc_sgpr_permanent(1, "s_M")
    ctx.alloc_sgpr_permanent(1, "s_N")
    ctx.alloc_sgpr_permanent(1, "s_K")
    ctx.alloc_sgpr_permanent(1, "s_lda")
    ctx.alloc_sgpr_permanent(1, "s_ldb")
    ctx.alloc_sgpr_permanent(1, "s_ldd")
    ctx.alloc_vgpr_permanent(1, "v_tid")
    ctx.alloc_vgpr_permanent(1, "v_wave_id")
    ctx.alloc_vgpr_permanent(1, "v_lane_id")
    ctx.alloc_vgpr_permanent(1, "v_wave_m")
    ctx.alloc_vgpr_permanent(1, "v_wave_n")
    ctx.alloc_vgpr_permanent(2, "v_addr_a")
    ctx.alloc_vgpr_permanent(2, "v_addr_b")
    ctx.alloc_vgpr_permanent(2, "v_addr_d")
    ctx.alloc_vgpr_permanent(1, "v_lds_write_a")
    ctx.alloc_vgpr_permanent(1, "v_lds_write_b")
    ctx.alloc_vgpr_permanent(1, "v_lds_read_a")
    ctx.alloc_vgpr_permanent(1, "v_lds_read_b")
    return ctx, tile, problem


@requires_st
class TestEmitGlobalLoadAddr:
    def test_emits_instructions(self):
        ctx, tile, problem = _make_ctx_with_args()
        ac = AddressComputer(problem, tile)
        with ctx.scope("test"):
            ac.emit_global_load_addr_a(ctx)
        dump = ctx.module.dump()
        assert "addr_A_lo" in dump
        assert "addr_A_hi" in dump
        assert "row * lda" in dump

    def test_b_emits(self):
        ctx, tile, problem = _make_ctx_with_args()
        ac = AddressComputer(problem, tile)
        with ctx.scope("test"):
            ac.emit_global_load_addr_b(ctx)
        dump = ctx.module.dump()
        assert "addr_B_lo" in dump


@requires_st
class TestEmitLdsWriteAddr:
    def test_emits_instructions(self):
        ctx, tile, problem = _make_ctx_with_args()
        ac = AddressComputer(problem, tile)
        with ctx.scope("test"):
            ac.emit_lds_write_addr(ctx)
        dump = ctx.module.dump()
        assert "lds_offset_b" in dump
        # Should include both A and B computations
        assert "row for A" in dump or "row *" in dump


@requires_st
class TestEmitLdsReadAddr:
    def test_emits_at_ki0(self):
        ctx, tile, problem = _make_ctx_with_args()
        ac = AddressComputer(problem, tile)
        ctx.set_index("wave", "mi", 0)
        ctx.set_index("wave", "ni", 0)
        ctx.set_index("wave", "ki", 0)
        with ctx.scope("test"):
            ac.emit_lds_read_addr(ctx)
        dump = ctx.module.dump()
        assert "lane_row" in dump
        assert "wave_m" in dump or "m_per_wave" in dump

    def test_emits_at_ki1(self):
        ctx, tile, problem = _make_ctx_with_args()
        ac = AddressComputer(problem, tile)
        ctx.set_index("wave", "mi", 2)
        ctx.set_index("wave", "ni", 1)
        ctx.set_index("wave", "ki", 1)
        with ctx.scope("test"):
            ac.emit_lds_read_addr(ctx)
        dump = ctx.module.dump()
        # Should have the ki*mfma_k offset
        assert "ki*mfma_k" in dump or "+ 16" in dump or "mfma_k (16)" in dump


@requires_st
class TestEmitAllPrologue:
    def test_emits_all(self):
        ctx, tile, problem = _make_ctx_with_args()
        ac = AddressComputer(problem, tile)
        ac.emit_all_prologue(ctx)
        dump = ctx.module.dump()
        # Global load addr A
        assert "addr_A_lo" in dump
        # Global load addr B
        assert "addr_B_lo" in dump
        # LDS write
        assert "lds_offset_b" in dump

    def test_temps_freed_after(self):
        """Temporary registers from prologue should be freed (scoped)."""
        ctx, tile, problem = _make_ctx_with_args()
        ac = AddressComputer(problem, tile)
        ac.emit_all_prologue(ctx)
        # Scoped temps like _tmp_row_a should be freed
        assert not ctx.has("_tmp_row_a")
        assert not ctx.has("_tmp_col_a")
