"""Tests for the swizzle module -- pure Python, no GPU required."""
import pytest
from kernel_generator.gemm.memory.swizzle import (
    LDS_GFX950, LDS_GFX1250,
    DataLayout,
    IdentitySwizzle, XorSwizzle, RotationSwizzle, ComposedSwizzle,
)


# -- DataLayout fixtures for each data type --

def _layout(elem, mfma_k, unroll_k, mfma_m=16, ws=64):
    return DataLayout(
        row_stride_bytes=int(unroll_k * elem),
        mfma_k=mfma_k, mfma_m=mfma_m,
        elem_bytes=elem, wave_size=ws,
    )

MXFP4_UK256 = _layout(elem=0.5, mfma_k=128, unroll_k=256)
FP8_UK128   = _layout(elem=1,   mfma_k=64,  unroll_k=128)
FP16_UK64   = _layout(elem=2,   mfma_k=32,  unroll_k=64)
FP16_UK32   = _layout(elem=2,   mfma_k=32,  unroll_k=32)
BF16_UK32   = _layout(elem=2,   mfma_k=32,  unroll_k=32)

ALL_LAYOUTS = [MXFP4_UK256, FP8_UK128, FP16_UK64, FP16_UK32, BF16_UK32]


class TestDataLayout:
    def test_num_cols(self):
        assert MXFP4_UK256.num_cols == 8
        assert FP8_UK128.num_cols == 8
        assert FP16_UK64.num_cols == 8
        assert FP16_UK32.num_cols == 4

    def test_ki_count(self):
        assert MXFP4_UK256.ki_count == 2
        assert FP8_UK128.ki_count == 2
        assert FP16_UK64.ki_count == 2
        assert FP16_UK32.ki_count == 1

    def test_k_step(self):
        assert MXFP4_UK256.k_step == 4
        assert FP16_UK64.k_step == 4

    def test_k_groups(self):
        assert MXFP4_UK256.k_groups == 4

    def test_rows_per_bank_row(self):
        assert MXFP4_UK256.rows_per_bank_row == 1
        assert FP16_UK32.rows_per_bank_row == 2


class TestBankedMemoryConfig:
    def test_bank_row_bytes(self):
        assert LDS_GFX950.bank_row_bytes == 256  # 64 banks * 4B

    def test_no_conflict(self):
        # 16 accesses to 16 different banks -> 1 cycle
        addrs = [i * 16 for i in range(16)]
        assert LDS_GFX950.cycles(addrs) == 1  # 64 banks, 16*4=64 unique banks

    def test_worst_conflict(self):
        # All 16 lanes read from the same address -> 16-way on 4 banks
        addrs = [0] * 16
        assert LDS_GFX950.cycles(addrs) == 16

    def test_gfx1250_levels(self):
        assert len(LDS_GFX1250.levels) == 2
        assert LDS_GFX1250.levels[0].name == "segment"
        assert LDS_GFX1250.levels[1].name == "bank"


class TestIdentitySwizzle:
    def test_passthrough(self):
        s = IdentitySwizzle()
        assert s.forward(0, 3, 8) == 3
        assert s.forward(5, 7, 8) == 7

    @pytest.mark.parametrize("layout", ALL_LAYOUTS)
    def test_worst_case_conflict(self, layout):
        s = IdentitySwizzle()
        # Identity should have bad conflicts (16-way for 128-byte rows)
        cycles = s.verify(layout, LDS_GFX950)
        assert cycles >= 4  # at least 4-way conflict


class TestXorSwizzle:
    def test_basic(self):
        s = XorSwizzle(shift_r=2, shift_l=1)
        # row=0: f=0, col unchanged
        assert s.forward(0, 3, 8) == 3
        # row=4: f=((4>>2)<<1)&7 = 2, col=3^2=1
        assert s.forward(4, 3, 8) == 1

    def test_bijective(self):
        s = XorSwizzle()
        for nc in [4, 8]:
            for row in range(16):
                cols = [s.forward(row, c, nc) for c in range(nc)]
                assert sorted(cols) == list(range(nc)), \
                    f"not bijective at row={row}, nc={nc}"

    @pytest.mark.parametrize("layout", ALL_LAYOUTS)
    def test_reduces_conflicts(self, layout):
        identity = IdentitySwizzle()
        xor = XorSwizzle()
        id_cycles = identity.verify(layout, LDS_GFX950)
        xor_cycles = xor.verify(layout, LDS_GFX950)
        assert xor_cycles <= id_cycles

    @pytest.mark.parametrize("layout", ALL_LAYOUTS)
    def test_at_most_4way(self, layout):
        s = XorSwizzle()
        assert s.verify(layout, LDS_GFX950) <= 4


class TestRotationSwizzle:
    def test_basic(self):
        s = RotationSwizzle(use_cross_lane=False)
        # row=0, rows_per_bank_row=1: lds_row_id=0, rotation=0
        assert s.forward(0, 0, 8) == 0
        assert s.forward(0, 3, 8) == 3
        # row=2: lds_row_id=2, rotation=2
        assert s.forward(2, 0, 8) == 2
        assert s.forward(2, 3, 8) == 5

    def test_bijective(self):
        s = RotationSwizzle(use_cross_lane=False)
        for nc in [4, 8]:
            for row in range(16):
                cols = [s.forward(row, c, nc) for c in range(nc)]
                assert sorted(cols) == list(range(nc)), \
                    f"not bijective at row={row}, nc={nc}"

    @pytest.mark.parametrize("layout", ALL_LAYOUTS)
    def test_without_crosslane_at_most_4way(self, layout):
        s = RotationSwizzle(use_cross_lane=False)
        assert s.verify(layout, LDS_GFX950) <= 4

    @pytest.mark.parametrize("layout", ALL_LAYOUTS)
    def test_reduces_vs_identity(self, layout):
        identity = IdentitySwizzle()
        rot = RotationSwizzle(use_cross_lane=False)
        assert rot.verify(layout, LDS_GFX950) <= identity.verify(layout, LDS_GFX950)

    @pytest.mark.parametrize("layout", ALL_LAYOUTS)
    def test_all_ki_conflict_bounded(self, layout):
        s = RotationSwizzle(use_cross_lane=False)
        assert s.verify_all_ki(layout, LDS_GFX950) <= 4


class TestComposedSwizzle:
    def test_composition(self):
        s1 = XorSwizzle(shift_r=2, shift_l=1)
        s2 = XorSwizzle(shift_r=1, shift_l=0)
        composed = ComposedSwizzle(s1, s2)
        for row in range(16):
            for col in range(8):
                expected = s2.forward(row, s1.forward(row, col, 8), 8)
                assert composed.forward(row, col, 8) == expected

    def test_identity_composition(self):
        base = XorSwizzle()
        composed = ComposedSwizzle(IdentitySwizzle(), base)
        for row in range(16):
            for col in range(8):
                assert composed.forward(row, col, 8) == base.forward(row, col, 8)


class TestCrossDataType:
    """Verify swizzle works consistently across all supported data types."""

    @pytest.mark.parametrize("layout", ALL_LAYOUTS)
    def test_xor_all_dtypes(self, layout):
        s = XorSwizzle()
        cycles = s.verify(layout, LDS_GFX950)
        assert 1 <= cycles <= 4

    @pytest.mark.parametrize("layout", ALL_LAYOUTS)
    def test_rotation_all_dtypes(self, layout):
        s = RotationSwizzle(use_cross_lane=False)
        cycles = s.verify(layout, LDS_GFX950)
        assert 1 <= cycles <= 4
