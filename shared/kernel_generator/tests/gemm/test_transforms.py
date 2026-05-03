# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the coordinate transform system."""
import pytest
from kernel_generator.gemm.tile.transforms import (
    Dim, Tile, Embed, TileDescriptor,
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
