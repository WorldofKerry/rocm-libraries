# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for codegen layers (no stinkytofu binary required)."""
import pytest
from stinkytofu.gemm.codegen import (
    RegisterAllocator, ThreadMapping, GemmCodegen,
)
from stinkytofu.gemm.problem import (
    DataType, GemmProblem, MfmaConfig, TileConfig,
)


# ===========================================================================
# RegisterAllocator
# ===========================================================================

class TestRegisterAllocator:
    def test_vgpr(self):
        r = RegisterAllocator()
        s0 = r.alloc_vgpr(4, "foo")
        s1 = r.alloc_vgpr(2, "bar")
        assert s0 == 0
        assert s1 == 4
        assert r.vgpr_count == 6
        assert r.get("foo") == ("v", 0, 4)
        assert r.get("bar") == ("v", 4, 2)

    def test_sgpr(self):
        r = RegisterAllocator()
        r.alloc_sgpr(2, "ptr")
        assert r.sgpr_count == 2

    def test_acc(self):
        r = RegisterAllocator()
        r.alloc_acc(16, "accum")
        assert r.acc_count == 16
        assert r.get("accum") == ("acc", 0, 16)

    def test_unnamed(self):
        r = RegisterAllocator()
        s = r.alloc_vgpr(8)
        assert s == 0
        assert r.vgpr_count == 8

    def test_summary(self):
        r = RegisterAllocator()
        r.alloc_vgpr(4, "a")
        r.alloc_sgpr(2, "b")
        s = r.summary()
        assert "VGPRs: 4" in s
        assert "SGPRs: 2" in s


# ===========================================================================
# ThreadMapping
# ===========================================================================

class TestThreadMapping:
    def test_basic(self):
        t = TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
            vector_width=8,
        )
        m = ThreadMapping(t)
        assert m.a_loads_per_thread >= 1
        assert m.b_loads_per_thread >= 1
        assert m.lds_size_bytes > 0
        assert m.lds_offset_b == 128 * 32 * 2  # wg_m * unroll_k * sizeof(f16)

    def test_descriptors(self):
        t = TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
        )
        m = ThreadMapping(t)
        assert m.m_desc is not None
        assert m.n_desc is not None
        assert m.k_desc is not None


# ===========================================================================
# GemmCodegen (dry run -- no stinkytofu)
# ===========================================================================

class TestGemmCodegenDry:
    def _make(self, **kw):
        p = GemmProblem(m=4096, n=4096, k=4096, **kw)
        t = TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
            vector_width=8,
        )
        return GemmCodegen(p, t)

    def test_kernel_name(self):
        cg = self._make()
        assert "gemm_f16" in cg.kernel_name()
        assert "128x128x32" in cg.kernel_name()
        assert "mfma16x16x16" in cg.kernel_name()

    def test_dry_info(self):
        cg = self._make()
        info = cg.generate_dry()
        assert "name" in info
        assert "registers" in info
        assert "tile" in info
        assert "mapping" in info

    def test_register_allocation(self):
        cg = self._make()
        # Check that key register groups are allocated
        assert cg.regs.get("srd_A")[2] == 2      # 64-bit pointer
        assert cg.regs.get("acc_C")[2] > 0        # accumulators
        assert cg.regs.get("v_a")[2] > 0          # MFMA operand A

    def test_acc_count(self):
        """Accumulator count = mfma_m_repeat * mfma_n_repeat * acc_per_mfma."""
        cg = self._make()
        t = cg.tile
        expected = t.mfma_m_repeat * t.mfma_n_repeat * t.mfma.acc_vgprs
        _, _, actual = cg.regs.get("acc_C")
        assert actual == expected

    def test_flops_correctness(self):
        """Verify FLOP count from tile decomposition matches problem FLOPs."""
        p = GemmProblem(m=4096, n=4096, k=4096)
        t = TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
        )
        grid_m, grid_n = p.grid_dims(t)
        k_tiles = p.k // t.unroll_k
        waves_per_wg = t.waves_m * t.waves_n

        flops_from_tiles = (
            grid_m * grid_n       # workgroups
            * waves_per_wg        # waves per wg
            * t.total_mfma_per_wave  # MFMAs per wave per unroll
            * k_tiles             # unroll iterations
            * t.mfma.flops_per_instruction
        )
        assert flops_from_tiles == p.total_flops

    def test_different_tile_sizes(self):
        """32x32x8 MFMA variant."""
        p = GemmProblem(m=4096, n=4096, k=4096)
        t = TileConfig(
            wg_m=256, wg_n=128, unroll_k=32,
            waves_m=4, waves_n=2,
            mfma=MfmaConfig.f16_32x32x8(),
            vector_width=8,
        )
        cg = GemmCodegen(p, t)
        info = cg.generate_dry()
        assert "256x128x32" in info["name"]
