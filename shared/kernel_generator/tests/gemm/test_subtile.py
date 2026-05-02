# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the subtile layer: SubTileConfig, PartitionConfig, VGPRTileAllocator."""
import pytest
from kernel_generator.gemm.problem import (
    MfmaConfig, SubTileConfig, PartitionConfig, TileConfig,
)
# ===========================================================================
# SubTileConfig
# ===========================================================================

class TestSubTileConfig:
    def test_defaults(self):
        st = SubTileConfig()
        assert st.subtile_m == 16
        assert st.subtile_k_bytes == 128

    def test_k_elems_f16(self):
        st = SubTileConfig(subtile_k_bytes=128)
        # f16 = 2 bytes per element -> 128 / 2 = 64 elements
        assert st.subtile_k_elems(element_bytes=2) == 64

    def test_k_elems_f32(self):
        st = SubTileConfig(subtile_k_bytes=128)
        assert st.subtile_k_elems(element_bytes=4) == 32

    def test_num_subtiles_m(self):
        st = SubTileConfig(subtile_m=16)
        assert st.num_subtiles_m(wave_m=64) == 4
        assert st.num_subtiles_m(wave_m=32) == 2

    def test_num_subtiles_k(self):
        st = SubTileConfig(subtile_k_bytes=128)
        # f16, unroll_k=64: subtile_k_elems=64, so 64/64 = 1
        assert st.num_subtiles_k(unroll_k=64, element_bytes=2) == 1
        # f16, unroll_k=128: 128/64 = 2
        assert st.num_subtiles_k(unroll_k=128, element_bytes=2) == 2

    def test_subtile_k_mfmas(self):
        st = SubTileConfig(subtile_k_bytes=128)
        # f16: subtile_k_elems=64, mfma_k=16 -> 4 mfma K-iterations per subtile
        assert st.subtile_k_mfmas(mfma_k=16, element_bytes=2) == 4
        # mfma_k=32 -> 2
        assert st.subtile_k_mfmas(mfma_k=32, element_bytes=2) == 2


# ===========================================================================
# PartitionConfig
# ===========================================================================

class TestPartitionConfig:
    def test_defaults(self):
        pc = PartitionConfig()
        assert pc.subtiles_per_partition == 4  # 2x2

    def test_num_partitions(self):
        pc = PartitionConfig(partition_m=2, partition_n=2)
        # 4 subtiles_m x 4 subtiles_n, 2x2 per partition -> 4 partitions
        assert pc.num_partitions(4, 4) == 4
        # 8 x 4 -> 4*2 = 8
        assert pc.num_partitions(8, 4) == 8

    def test_single_partition(self):
        # Partition covers the whole wave tile
        pc = PartitionConfig(partition_m=4, partition_n=4)
        assert pc.num_partitions(4, 4) == 1


# ===========================================================================
# TileConfig with subtiling
# ===========================================================================

class TestTileConfigSubtile:
    def _tile(self, **kw):
        defaults = dict(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
            subtile=SubTileConfig(subtile_m=16, subtile_k_bytes=32),
            partition=PartitionConfig(partition_m=2, partition_n=2),
        )
        defaults.update(kw)
        return TileConfig(**defaults)

    def test_subtiling_enabled(self):
        t = self._tile()
        assert t.subtiling_enabled

    def test_not_enabled_by_default(self):
        t = TileConfig()
        assert not t.subtiling_enabled

    def test_num_subtiles(self):
        t = self._tile()
        # m_per_wave = 64, subtile_m = 16 -> 4 subtiles
        assert t.num_subtiles_m == 4
        assert t.num_subtiles_n == 4

    def test_num_partitions(self):
        t = self._tile()
        # 4x4 subtiles, 2x2 per partition -> 4 partitions
        assert t.num_partitions == 4

    def test_validate_ok(self):
        t = self._tile()
        t.validate()

    def test_validate_bad_subtile_m(self):
        t = self._tile(subtile=SubTileConfig(subtile_m=13))
        with pytest.raises(ValueError, match="divisible by subtile_m"):
            t.validate()

    def test_validate_partition_without_subtile(self):
        with pytest.raises(ValueError, match="partition requires subtile"):
            TileConfig(
                wg_m=128, wg_n=128, unroll_k=32,
                waves_m=2, waves_n=2,
                mfma=MfmaConfig.f16_16x16x16(),
                partition=PartitionConfig(),
            ).validate()

    def test_validate_bad_partition_size(self):
        t = self._tile(partition=PartitionConfig(partition_m=3))
        with pytest.raises(ValueError, match="divisible by partition_m"):
            t.validate()

    def test_live_vgprs_with_partition(self):
        t = self._tile()
        live = t.live_vgprs_per_partition
        # partition_m=2 * a_per_subtile + partition_n=2 * b_per_subtile
        assert live > 0
        # Should be less than the non-partitioned case
        t_nopart = self._tile(partition=None, subtile=None)
        assert live <= t_nopart.live_vgprs_per_partition

    def test_summary_includes_subtile(self):
        t = self._tile()
        s = t.summary()
        assert "Subtile" in s
        assert "Partitions" in s

    def test_flops_unchanged_by_subtiling(self):
        """Subtiling is a scheduling concern; FLOP count must not change."""
        t_flat = TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
        )
        t_sub = self._tile()
        assert t_flat.total_mfma_per_wave == t_sub.total_mfma_per_wave
        assert t_flat.flops_per_wave_per_unroll == t_sub.flops_per_wave_per_unroll


# ===========================================================================
# VGPRTileAllocator
# ===========================================================================
