# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the coordinate transform system."""
import pytest
from stinkytofu.gemm.transforms import (
    Dim, PassThrough, Tile, Flatten, Pad, Embed, Xor,
    TileDescriptor, tile_hierarchy,
)


# ===========================================================================
# Dim
# ===========================================================================

class TestDim:
    def test_creation(self):
        d = Dim("M", 256)
        assert d.name == "M"
        assert d.size == 256

    def test_frozen(self):
        d = Dim("M", 256)
        with pytest.raises(AttributeError):
            d.name = "N"

    def test_repr(self):
        assert repr(Dim("K", 64)) == "K:64"

    def test_equality(self):
        assert Dim("M", 128) == Dim("M", 128)
        assert Dim("M", 128) != Dim("M", 256)
        assert Dim("M", 128) != Dim("N", 128)

    def test_hashable(self):
        s = {Dim("M", 128), Dim("M", 128), Dim("N", 64)}
        assert len(s) == 2


# ===========================================================================
# PassThrough
# ===========================================================================

class TestPassThrough:
    def test_identity(self):
        d = Dim("M", 128)
        pt = PassThrough(d)
        assert pt.upper_dims == [d]
        assert pt.lower_dims == [d]
        assert pt.forward({"M": 42}) == {"M": 42}

    def test_codegen(self):
        pt = PassThrough(Dim("x", 10))
        assert pt.codegen_forward({"x": "v0"}) == {"x": "v0"}


# ===========================================================================
# Tile
# ===========================================================================

class TestTile:
    def test_basic_split(self):
        t = Tile(Dim("M", 256), 64)
        assert t.outer == Dim("M_outer", 4)
        assert t.inner == Dim("M_inner", 64)
        assert t.tile_size == 64

    def test_custom_names(self):
        t = Tile(Dim("M", 256), 64, outer_name="wg_id", inner_name="wg_off")
        assert t.outer.name == "wg_id"
        assert t.inner.name == "wg_off"

    def test_forward(self):
        t = Tile(Dim("M", 256), 64)
        # outer=2, inner=10 -> M = 2*64 + 10 = 138
        assert t.forward({"M_outer": 2, "M_inner": 10}) == {"M": 138}

    def test_forward_tile1(self):
        t = Tile(Dim("M", 8), 1)
        assert t.forward({"M_outer": 5, "M_inner": 0}) == {"M": 5}

    def test_codegen(self):
        t = Tile(Dim("M", 256), 64)
        result = t.codegen_forward({"M_outer": "wg_m", "M_inner": "lane"})
        assert result == {"M": "(wg_m * 64 + lane)"}

    def test_not_divisible(self):
        with pytest.raises(ValueError, match="not divisible"):
            Tile(Dim("M", 255), 64)

    def test_dims(self):
        t = Tile(Dim("M", 256), 64)
        assert t.upper_dims == [t.outer, t.inner]
        assert t.lower_dims == [Dim("M", 256)]


# ===========================================================================
# Flatten
# ===========================================================================

class TestFlatten:
    def test_two_dims(self):
        f = Flatten([Dim("A", 4), Dim("B", 8)])
        assert f.merged == Dim("A_B", 32)
        assert f.forward({"A": 2, "B": 5}) == {"A_B": 2 * 8 + 5}

    def test_three_dims(self):
        f = Flatten([Dim("X", 2), Dim("Y", 3), Dim("Z", 4)])
        assert f.merged.size == 24
        # X=1, Y=2, Z=3 -> 1*12 + 2*4 + 3 = 23
        assert f.forward({"X": 1, "Y": 2, "Z": 3}) == {"X_Y_Z": 23}

    def test_custom_name(self):
        f = Flatten([Dim("A", 4), Dim("B", 8)], merged_name="flat")
        assert f.merged.name == "flat"

    def test_codegen(self):
        f = Flatten([Dim("A", 4), Dim("B", 8)])
        result = f.codegen_forward({"A": "x", "B": "y"})
        assert "A_B" in result

    def test_too_few_dims(self):
        with pytest.raises(ValueError):
            Flatten([Dim("A", 4)])


# ===========================================================================
# Pad
# ===========================================================================

class TestPad:
    def test_basic(self):
        p = Pad(Dim("M", 250), 256)
        assert p.padded == Dim("M_padded", 256)
        assert p.forward({"M_padded": 100}) == {"M": 100}

    def test_no_pad_needed(self):
        p = Pad(Dim("M", 256), 256)
        assert p.padded.size == 256

    def test_too_small(self):
        with pytest.raises(ValueError):
            Pad(Dim("M", 256), 128)


# ===========================================================================
# Embed
# ===========================================================================

class TestEmbed:
    def test_row_major_offset(self):
        # offset = row * stride + col
        e = Embed(
            [Dim("row", 64), Dim("col", 128)],
            Dim("offset", 64 * 128),
            [128, 1],
        )
        assert e.forward({"row": 3, "col": 7}) == {"offset": 3 * 128 + 7}

    def test_codegen(self):
        e = Embed(
            [Dim("r", 4), Dim("c", 8)],
            Dim("off", 32),
            [8, 1],
        )
        result = e.codegen_forward({"r": "s_row", "c": "s_col"})
        assert "off" in result
        # Should contain the multiplication by stride
        assert "8" in result["off"]

    def test_zero_coefficient(self):
        e = Embed(
            [Dim("a", 4), Dim("b", 8)],
            Dim("x", 4),
            [1, 0],
        )
        assert e.forward({"a": 3, "b": 999}) == {"x": 3}


# ===========================================================================
# Xor
# ===========================================================================

class TestXor:
    def test_basic(self):
        x = Xor(Dim("row", 64), Dim("col", 8))
        result = x.forward({"row": 5, "col": 3})
        assert result["row_xor"] == 5 ^ 3
        assert result["col"] == 3

    def test_with_shift(self):
        x = Xor(Dim("row", 64), Dim("col", 8), shift=2)
        result = x.forward({"row": 5, "col": 12})
        assert result["row_xor"] == 5 ^ (12 >> 2)

    def test_codegen(self):
        x = Xor(Dim("r", 64), Dim("c", 8), shift=3)
        result = x.codegen_forward({"r": "v_r", "c": "v_c"})
        assert "^" in result["r_xor"]
        assert ">>" in result["r_xor"]


# ===========================================================================
# TileDescriptor
# ===========================================================================

class TestTileDescriptor:
    def test_basic(self):
        d = TileDescriptor("A", [Dim("M", 256), Dim("K", 64)])
        assert len(d.visible_dims) == 2
        assert d.get_dim("M") == Dim("M", 256)

    def test_add_transform(self):
        d = TileDescriptor("A", [Dim("M", 256), Dim("K", 64)])
        d.add_transform(Tile(Dim("M", 256), 64))
        # M is replaced by M_outer, M_inner; K stays
        names = [dim.name for dim in d.visible_dims]
        assert "M_outer" in names
        assert "M_inner" in names
        assert "K" in names
        assert "M" not in names

    def test_chained_transforms(self):
        d = TileDescriptor("A", [Dim("M", 256)])
        d.add_transform(Tile(Dim("M", 256), 64,
                             outer_name="wg_id", inner_name="wg"))
        d.add_transform(Tile(Dim("wg", 64), 16,
                             outer_name="wave_id", inner_name="wave"))
        names = [dim.name for dim in d.visible_dims]
        assert names == ["wg_id", "wave_id", "wave"]

    def test_missing_dim_error(self):
        d = TileDescriptor("A", [Dim("M", 256)])
        with pytest.raises(ValueError, match="not in visible"):
            d.add_transform(Tile(Dim("X", 256), 64))

    def test_chaining_returns_self(self):
        d = TileDescriptor("test", [Dim("X", 128)])
        result = d.add_transform(Tile(Dim("X", 128), 32))
        assert result is d


# ===========================================================================
# tile_hierarchy
# ===========================================================================

class TestTileHierarchy:
    def test_two_levels(self):
        tiles = tile_hierarchy(Dim("M", 256), [
            (128, "M_wg_id", "M_wg"),
            (32, "M_wave_id", "M_wave"),
        ])
        assert len(tiles) == 2
        assert tiles[0].outer.name == "M_wg_id"
        assert tiles[0].inner.name == "M_wg"
        assert tiles[1].outer.name == "M_wave_id"
        assert tiles[1].inner.name == "M_wave"
        # First tile: 256 / 128 = 2 outer tiles
        assert tiles[0].outer.size == 2
        # Second tile operates on the inner of the first: 128 / 32 = 4
        assert tiles[1].outer.size == 4
        assert tiles[1].inner.size == 32

    def test_three_levels(self):
        tiles = tile_hierarchy(Dim("M", 512), [
            (256, "L0_out", "L0_in"),
            (64,  "L1_out", "L1_in"),
            (16,  "L2_out", "L2_in"),
        ])
        assert len(tiles) == 3
        assert tiles[2].inner.size == 16
        assert tiles[2].outer.size == 4  # 64 / 16
