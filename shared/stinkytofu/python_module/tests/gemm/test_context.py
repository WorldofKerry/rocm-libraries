# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for TileContext: bindings, scoped lifetime, reuse, contracts."""
from __future__ import annotations

import pytest
from stinkytofu.gemm.context import TileContext, Binding, Lifetime


# ===========================================================================
# Bindings
# ===========================================================================

class TestBindings:
    def test_bind_and_get(self):
        ctx = TileContext()
        ctx.bind("foo", "v", 0, 4)
        b = ctx.get("foo")
        assert b.pool == "v"
        assert b.start == 0
        assert b.count == 4

    def test_get_missing_raises(self):
        ctx = TileContext()
        with pytest.raises(KeyError, match="not found"):
            ctx.get("nonexistent")

    def test_has(self):
        ctx = TileContext()
        assert not ctx.has("x")
        ctx.bind("x", "v", 0, 1)
        assert ctx.has("x")

    def test_bind_overwrites(self):
        ctx = TileContext()
        ctx.bind("x", "v", 0, 4)
        ctx.bind("x", "v", 10, 2)
        assert ctx.get("x").start == 10
        assert ctx.get("x").count == 2


# ===========================================================================
# Allocation
# ===========================================================================

class TestAllocation:
    def test_alloc_vgpr(self):
        ctx = TileContext()
        s0 = ctx.alloc_vgpr(4, "a")
        s1 = ctx.alloc_vgpr(2, "b")
        assert s0 == 0
        assert s1 == 4
        assert ctx.get("a").start == 0
        assert ctx.get("b").start == 4

    def test_alloc_sgpr(self):
        ctx = TileContext()
        s = ctx.alloc_sgpr(2, "ptr")
        assert s == 0

    def test_alloc_acc(self):
        ctx = TileContext()
        s = ctx.alloc_acc(16, "acc_C")
        assert s == 0
        assert ctx.get("acc_C").count == 16


# ===========================================================================
# Scoped lifetime
# ===========================================================================

class TestScopedLifetime:
    def test_scoped_freed_on_exit(self):
        ctx = TileContext()
        ctx.alloc_vgpr(4, "permanent", held=False)  # global scope
        with ctx.scope("wave"):
            ctx.alloc_vgpr(2, "wave_tmp")
            assert ctx.has("wave_tmp")
        # wave_tmp freed after scope exit
        assert not ctx.has("wave_tmp")
        # global-scope binding still alive
        assert ctx.has("permanent")

    def test_nested_scopes(self):
        ctx = TileContext()
        with ctx.scope("wave"):
            ctx.alloc_vgpr(4, "wave_reg")
            with ctx.scope("subtile"):
                ctx.alloc_vgpr(2, "subtile_reg")
                assert ctx.has("wave_reg")
                assert ctx.has("subtile_reg")
            # subtile_reg freed
            assert not ctx.has("subtile_reg")
            # wave_reg still alive
            assert ctx.has("wave_reg")
        assert not ctx.has("wave_reg")

    def test_held_survives_scope(self):
        ctx = TileContext()
        with ctx.scope("k_iter_0"):
            ctx.alloc_vgpr(4, "prefetch", held=True)
        # prefetch survives scope exit
        assert ctx.has("prefetch")
        # must free manually
        ctx.free("prefetch")
        assert not ctx.has("prefetch")

    def test_permanent_survives_everything(self):
        ctx = TileContext()
        ctx.alloc_vgpr_permanent(16, "acc_C")
        with ctx.scope("wave"):
            with ctx.scope("subtile"):
                assert ctx.has("acc_C")
        assert ctx.has("acc_C")


# ===========================================================================
# Register reuse
# ===========================================================================

class TestRegisterReuse:
    def test_freed_regs_reused(self):
        ctx = TileContext()
        with ctx.scope("part_0"):
            s0 = ctx.alloc_vgpr(4, "operand_a")
        # operand_a freed, regs [0:4] available
        with ctx.scope("part_1"):
            s1 = ctx.alloc_vgpr(4, "operand_a")
        # Should reuse the same register range
        assert s1 == s0

    def test_peak_tracks_max(self):
        ctx = TileContext()
        ctx.alloc_vgpr_permanent(4, "perm")
        with ctx.scope("part_0"):
            ctx.alloc_vgpr(8, "tmp0")  # 4 + 8 = 12 live
        # tmp0 freed -> 4 live
        with ctx.scope("part_1"):
            ctx.alloc_vgpr(8, "tmp1")  # reused -> peak still 12
        assert ctx.vgpr_peak == 12


# ===========================================================================
# Contract validation
# ===========================================================================

class TestContractValidation:
    def test_requires_ok(self):
        ctx = TileContext()
        ctx.bind("a", "v", 0, 4)
        ctx.bind("b", "v", 4, 4)
        ctx.validate_requires(["a", "b"], "test_level")  # no error

    def test_requires_missing(self):
        ctx = TileContext()
        ctx.bind("a", "v", 0, 4)
        with pytest.raises(ValueError, match="requires bindings"):
            ctx.validate_requires(["a", "b", "c"], "test_level")

    def test_provides_ok(self):
        ctx = TileContext()
        ctx.bind("out", "v", 0, 4)
        ctx.validate_provides(["out"], "test_level")  # no error

    def test_provides_missing(self):
        ctx = TileContext()
        with pytest.raises(ValueError, match="expected to provide"):
            ctx.validate_provides(["missing_output"], "test_level")


# ===========================================================================
# Indices
# ===========================================================================

class TestIndices:
    def test_set_get(self):
        ctx = TileContext()
        ctx.set_index("wave", "mi", 3)
        assert ctx.get_index("wave", "mi") == 3

    def test_missing_index(self):
        ctx = TileContext()
        with pytest.raises(KeyError, match="not set"):
            ctx.get_index("wave", "mi")

    def test_multiple_levels(self):
        ctx = TileContext()
        ctx.set_index("workgroup", "ki", 5)
        ctx.set_index("wave", "mi", 2)
        ctx.set_index("wave", "ni", 1)
        assert ctx.get_index("workgroup", "ki") == 5
        assert ctx.get_index("wave", "mi") == 2
        assert ctx.indices == {
            "workgroup.ki": 5, "wave.mi": 2, "wave.ni": 1,
        }


# ===========================================================================
# Prefetch pattern (double-buffered)
# ===========================================================================

class TestPrefetchPattern:
    """Test the double-buffered prefetch pattern end to end."""

    def test_double_buffer_lifecycle(self):
        ctx = TileContext()
        ctx.alloc_acc_permanent(16, "acc_C")

        # Allocate two prefetch buffers with HELD lifetime
        buf_a = ctx.alloc_vgpr(4, "buf_0", held=True)
        buf_b = ctx.alloc_vgpr(4, "buf_1", held=True)

        # Simulate: pre-load into buf_0
        ctx.bind("current_a", "v", buf_a, 4, held=True)

        for ki in range(4):
            ctx.set_index("k_loop", "ki", ki)

            with ctx.scope(f"k_iter_{ki}"):
                # Inner levels see "current_a"
                assert ctx.has("current_a")
                b = ctx.get("current_a")
                assert b.count == 4

            # Swap buffers
            next_buf = buf_b if ki % 2 == 0 else buf_a
            ctx.bind("current_a", "v", next_buf, 4, held=True)

        # Cleanup
        ctx.free("current_a")
        ctx.free("buf_0")
        ctx.free("buf_1")
        assert not ctx.has("buf_0")
        assert not ctx.has("buf_1")


# ===========================================================================
# Summary
# ===========================================================================

class TestSummary:
    def test_summary_has_info(self):
        ctx = TileContext()
        ctx.alloc_vgpr(4, "foo")
        ctx.set_index("wave", "mi", 2)
        s = ctx.summary()
        assert "foo" in s
        assert "wave.mi" in s
