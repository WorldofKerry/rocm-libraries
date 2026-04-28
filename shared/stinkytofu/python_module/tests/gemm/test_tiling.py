# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for TileDim, ScheduleKind, GemmTiling."""
import pytest
from stinkytofu.gemm.tiling import TileDim, GemmTiling, ScheduleKind, S
from stinkytofu.gemm.problem import MfmaConfig, TileConfig

P, SEQ, H = S.PARALLEL, S.SEQUENTIAL, S.HARDWARE


class TestTileDim:
    def test_basic(self):
        d = TileDim("M", 128, P)
        assert d.size == 128
        assert d.is_leaf
        assert d.count == 1
        assert d.depth == 0

    def test_split(self):
        d = TileDim("M", 128, P).split("M_wave", 64, P)
        assert d.size == 128
        assert d.count == 2
        assert d.inner.size == 64
        assert d.inner.name == "M_wave"
        assert d.depth == 1

    def test_multi_split(self):
        d = (TileDim("M", 128, P)
             .split("M_wave", 64, P)
             .split("M_mfma", 16, H))
        assert d.count == 2       # 128 / 64
        assert d.inner.count == 4 # 64 / 16
        assert d.leaf_size == 16
        assert d.depth == 2

    def test_levels(self):
        d = (TileDim("M", 128, P)
             .split("wave", 64, P)
             .split("mfma", 16, H))
        levels = d.levels()
        assert len(levels) == 3
        assert [l.name for l in levels] == ["M", "wave", "mfma"]
        assert [l.size for l in levels] == [128, 64, 16]

    def test_get_level(self):
        d = (TileDim("M", 128, P)
             .split("wave", 64, P)
             .split("mfma", 16, H))
        assert d.get_level("wave").size == 64
        assert d.get_level("mfma").size == 16
        with pytest.raises(KeyError):
            d.get_level("nonexistent")

    def test_validate_ok(self):
        d = (TileDim("M", 128, P)
             .split("wave", 64, P)
             .split("mfma", 16, H))
        d.validate()  # should not raise

    def test_validate_bad(self):
        with pytest.raises(ValueError, match="not divisible"):
            TileDim("M", 128, P).split("bad", 30, P)

    def test_immutable(self):
        d1 = TileDim("M", 128, P)
        d2 = d1.split("wave", 64, P)
        assert d1.inner is None  # d1 unchanged
        assert d2.inner is not None

    def test_summary(self):
        d = (TileDim("M", 128, P)
             .split("wave", 64, P)
             .split("mfma", 16, H))
        s = d.summary()
        assert "M(128)[P]" in s
        assert "wave(64)[P]" in s
        assert "mfma(16)[H]" in s

    def test_build_descriptor(self):
        d = (TileDim("M", 128, P)
             .split("wave", 64, P)
             .split("mfma", 16, H))
        desc = d.build_descriptor()
        assert desc.name == "M"
        # Should have 2 Tile transforms applied
        assert len(desc.transforms) == 2

    def test_sequential(self):
        d = TileDim("K", 32, SEQ).split("K_mfma", 16, H)
        assert d.schedule == S.SEQUENTIAL
        assert d.inner.schedule == S.HARDWARE
        assert d.count == 2

    def test_subtile_level(self):
        """Adding a subtile level between wave and mfma."""
        d = (TileDim("M", 128, P)
             .split("M_wave", 64, P)
             .split("M_subtile", 16, SEQ)
             .split("M_mfma", 16, H))
        assert d.depth == 3
        levels = d.levels()
        assert levels[2].schedule == S.SEQUENTIAL
        assert levels[2].name == "M_subtile"


class TestGemmTiling:
    def test_standard(self):
        t = GemmTiling.standard()
        assert t.wg_m == 128
        assert t.wg_n == 128
        assert t.unroll_k == 32
        assert t.waves_m == 2
        assert t.waves_n == 2
        assert t.m_per_wave == 64
        assert t.mfma_m_repeat == 4
        assert t.k_iterations == 2
        assert t.block_size == 256

    def test_custom(self):
        t = GemmTiling.standard(wg_m=256, wg_n=128, unroll_k=64,
                                waves_m=4, waves_n=2)
        assert t.wg_m == 256
        assert t.waves_m == 4
        assert t.m_per_wave == 64
        assert t.unroll_k == 64
        assert t.k_iterations == 4

    def test_to_tile_config(self):
        t = GemmTiling.standard()
        tc = t.to_tile_config()
        assert isinstance(tc, TileConfig)
        assert tc.wg_m == 128
        assert tc.waves_m == 2
        assert tc.mfma.m == 16

    def test_matches_tile_config(self):
        """GemmTiling.standard() produces same values as TileConfig defaults."""
        t = GemmTiling.standard()
        tc = TileConfig()
        assert t.wg_m == tc.wg_m
        assert t.wg_n == tc.wg_n
        assert t.unroll_k == tc.unroll_k
        assert t.waves_m == tc.waves_m
        assert t.waves_n == tc.waves_n
        assert t.m_per_wave == tc.m_per_wave
        assert t.mfma_m_repeat == tc.mfma_m_repeat
        assert t.k_iterations == tc.k_iterations
        assert t.block_size == tc.block_size

    def test_build_tile_tree(self):
        t = GemmTiling.standard()
        tree = t.build_tile_tree()
        assert tree.name == "workgroup"
        assert tree.m == 128
        assert tree.parallel is True
        wave = tree.inner
        assert wave.name == "wave"
        assert wave.m == 64
        assert wave.inner.name == "mfma"
        assert wave.inner.m == 16
        tree.validate()

    def test_build_descriptors(self):
        t = GemmTiling.standard()
        md = t.build_m_descriptor()
        nd = t.build_n_descriptor()
        kd = t.build_k_descriptor()
        assert md.name == "M"
        assert nd.name == "N"
        assert kd.name == "K"

    def test_validate(self):
        t = GemmTiling.standard()
        t.validate()  # should not raise

    def test_validate_bad_leaf(self):
        """Leaf must match MFMA dim."""
        t = GemmTiling(
            dim_m=TileDim("M", 128, P).split("wave", 64, P).split("bad", 32, H),
            dim_n=TileDim("N", 128, P).split("wave", 64, P).split("mfma", 16, H),
            dim_k=TileDim("K", 32, SEQ).split("mfma_k", 16, H),
            mfma=MfmaConfig.f16_16x16x16(),
        )
        with pytest.raises(ValueError, match="M leaf size"):
            t.validate()

    def test_summary(self):
        t = GemmTiling.standard()
        s = t.summary()
        assert "M:" in s
        assert "N:" in s
        assert "K:" in s
        assert "Repeats:" in s

    def test_with_pipeline(self):
        """GemmTiling works with GemmKernel.build()."""
        from stinkytofu.gemm.kernel_pipeline import GemmKernel
        from stinkytofu.gemm.problem import GemmProblem

        tiling = GemmTiling.standard()
        kernel = GemmKernel.build(
            GemmProblem(4096, 4096, 4096), tiling=tiling)
        result = kernel.emit()
        assert len(result.asm_text) > 0
        assert "v_mfma_f32" in result.asm_text
