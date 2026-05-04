# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the GemmKernel pipeline with tree-driven phases."""
import pytest
from kernel_generator.gemm.kernel import (
    GemmKernel, default_mfma_visitor,
)
from kernel_generator.gemm.problem import GemmProblem, TileConfig


class TestGemmKernelBuild:
    def test_build_default(self):
        kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
        assert kernel.mfma_visitor is default_mfma_visitor
        assert kernel.tile_tree is not None
        assert kernel.tile_tree.name == "workgroup"

    def test_build_has_phases(self):
        kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
        tree = kernel.tile_tree
        # Workgroup level has prologue and epilogue phases
        pro_names = [p.name for p in tree.prologue_phases]
        assert "dtl_setup" in pro_names
        assert "scheduled_k_loop" in pro_names
        epi_names = [p.name for p in tree.epilogue_phases]
        assert "store_d" in epi_names

    def test_build_wave_phases(self):
        kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
        wave = kernel.tile_tree.inner
        assert wave.name == "wave"
        # Wave uses noop emit; no prologue/epilogue phases
        assert len(wave.prologue_phases) == 0
        assert len(wave.epilogue_phases) == 0

    def test_build_custom_tile(self):
        tile = TileConfig(wg_m=256, wg_n=128, unroll_k=64,
                          waves_m=4, waves_n=2)
        kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096), tile)
        assert kernel.tile.wg_m == 256
        assert kernel.tile.waves_m == 4

    def test_layouts_built(self):
        kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
        assert kernel.layouts is not None
        assert kernel.layouts.lds_b_offset > 0
        assert kernel.layouts.elem_bytes == 2

    def test_tile_tree_valid(self):
        kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
        kernel.tile_tree.validate()  # should not raise

    def test_all_phase_names(self):
        kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
        names = kernel.tile_tree.phase_names()
        expected = [
            "dtl_setup", "mx_scale_setup", "scheduled_k_loop",
            "store_d",
        ]
        for name in expected:
            assert name in names, f"Phase '{name}' not found in tree"


class TestGemmKernelEmit:
    def test_emit_produces_assembly(self):
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        result = kernel.emit()
        assert len(result.asm_text) > 0
        assert "gemm_kernel" in result.asm_text
        assert ".amdhsa_kernel" in result.asm_text
        assert ".amdgpu_metadata" in result.asm_text

    def test_emit_assembles(self):
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        result = kernel.emit()
        import os, tempfile
        co = result.assemble(output_path=os.path.join(
            tempfile.gettempdir(), "test_pipeline.co"))
        assert os.path.exists(co)
        assert os.path.getsize(co) > 0
        os.unlink(co)

    def test_emit_register_counts(self):
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        result = kernel.emit()
        assert result.vgpr_count > 0
        assert result.sgpr_count > 0
        assert result.acc_count > 0

    def test_emit_has_mfma(self):
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        result = kernel.emit()
        mfma_lines = [l for l in result.ctx.lines if 'v_mfma_f32' in l]
        assert len(mfma_lines) > 0

    def test_emit_has_k_loop(self):
        """Assembly should contain k_loop label and branch."""
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        result = kernel.emit()
        assert any("k_loop:" in l for l in result.ctx.lines)

    def test_emit_has_buffer_load(self):
        """Scheduled path uses buffer_load for DTL."""
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        result = kernel.emit()
        loads = [l for l in result.ctx.lines if 'buffer_load' in l]
        assert len(loads) > 0

    def test_emit_has_store(self):
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        result = kernel.emit()
        stores = [l for l in result.ctx.lines if 'global_store' in l or 'buffer_store' in l]
        assert len(stores) > 0


class TestPhaseReplacement:
    def test_replace_store_phase(self):
        """Replace the store_d phase -- no stores emitted."""
        def noop_store(level, ctx):
            ctx.comment("store replaced")

        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        kernel.tile_tree = kernel.tile_tree.replace_phase(
            "store_d", noop_store)
        result = kernel.emit()
        stores = [l for l in result.ctx.lines if 'global_store' in l or 'buffer_store' in l]
        assert len(stores) == 0
        assert any("store replaced" in l for l in result.ctx.lines)

    def test_scheduled_kernel_emits_mfma(self):
        """Scheduled kernel emits MFMA instructions."""
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        result = kernel.emit()
        mfma_lines = [l for l in result.ctx.lines if "v_mfma_f32" in l]
        assert len(mfma_lines) > 0

    def test_replace_scheduled_k_loop(self):
        """Replace the scheduled_k_loop phase with a marker."""
        from kernel_generator.gemm.schedule.kloop_scheduler import scheduled_kloop_phase

        def custom_kloop(level, ctx):
            ctx.comment("CUSTOM_MARKER")
            scheduled_kloop_phase(level, ctx)

        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        kernel.tile_tree = kernel.tile_tree.replace_phase(
            "scheduled_k_loop", custom_kloop)
        result = kernel.emit()
        assert any("CUSTOM_MARKER" in l for l in result.ctx.lines)

    def test_tile_tree_structure(self):
        """Tile tree has expected structure for scheduled path."""
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        mfma = kernel.tile_tree.find("mfma")
        assert mfma is not None
        assert mfma.m == kernel.tile.mfma.m
        wave = kernel.tile_tree.find("wave")
        assert wave is not None

    def test_get_phase(self):
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        phase = kernel.tile_tree.get_phase("scheduled_k_loop")
        assert phase is not None
        assert phase.name == "scheduled_k_loop"

    def test_get_phase_not_found(self):
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        phase = kernel.tile_tree.get_phase("nonexistent")
        assert phase is None


class TestMemoryViews:
    def test_views_registered(self):
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        result = kernel.emit()
        ctx = result.ctx
        a_view = ctx.get_view("A")
        b_view = ctx.get_view("B")
        assert a_view.source == "lds"
        assert b_view.source == "lds"
        assert b_view.base_offset > 0  # B has LDS offset

    def test_view_in_custom_phase(self):
        """Access MemoryView from a custom prologue phase."""
        views_found = []
        def check_views(level, ctx):
            for name in ["A", "B"]:
                view = ctx.get_view(name)
                if view:
                    views_found.append(name)

        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        from kernel_generator.gemm.tile.tree import TilePhase
        kernel.tile_tree.prologue_phases.insert(0,
            TilePhase("check_views", check_views))
        kernel.emit()
        assert "A" in views_found
        assert "B" in views_found

    def test_view_layout_matches(self):
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        result = kernel.emit()
        a_view = result.ctx.get_view("A")
        # LDS A layout: row * unroll_k + col
        assert a_view.layout._coefficients[0] == kernel.tile.unroll_k
        assert a_view.layout._coefficients[1] == 1
        assert a_view.elem_bytes == 2

    def test_missing_view_raises(self):
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        result = kernel.emit()
        with pytest.raises(KeyError, match="No MemoryView 'C'"):
            result.ctx.get_view("C")


class TestTreeStructure:
    """Test the tree structure produced by GemmKernel.build()."""

    def test_workgroup_parallel(self):
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        assert kernel.tile_tree.parallel is True

    def test_wave_not_parallel(self):
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        assert kernel.tile_tree.inner.parallel is False

    def test_tree_depth(self):
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        assert kernel.tile_tree.depth == 2  # workgroup -> wave -> mfma

    def test_mfma_is_leaf(self):
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        mfma = kernel.tile_tree.find("mfma")
        assert mfma is not None
        assert mfma.is_leaf is True

    def test_summary_shows_phases(self):
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        s = kernel.tile_tree.summary()
        assert "prologue:" in s
        assert "epilogue:" in s
        assert "scheduled_k_loop" in s
        assert "store_d" in s


class TestAddToggle:
    """Verify ADD-based double-buffer toggling and PGR=2 support."""

    def test_add_toggle_with_negate(self):
        """Assembly should use v_add/s_add for toggle with negate."""
        kernel = GemmKernel.build(GemmProblem(256, 256, 64))
        result = kernel.emit()
        asm = result.asm_text
        # ADD toggle should be present
        assert "v_add" in asm, "Missing v_add for read toggle"
        assert "s_add_u32" in asm, "Missing s_add_u32 for write toggle"
        # Negate step should be present (s_sub_u32 for db_step)
        assert any("negate db_step" in line for line in asm.split(chr(10))), \
            "Missing negate instruction for ADD-based toggle"

    def test_pgr2_disabled_safely(self):
        """PGR=2 flag accepted but disabled (falls back to PGR=1)."""
        kernel = GemmKernel.build(GemmProblem(256, 256, 256), pgr2=True)
        result = kernel.emit()
        co = result.assemble()
        assert co is not None, "PGR=2 (disabled) kernel should assemble"

    def test_pgr2_assembles(self):
        """PGR=2 kernel should assemble without errors."""
        kernel = GemmKernel.build(GemmProblem(256, 256, 256), pgr2=True)
        result = kernel.emit()
        co = result.assemble()
        assert co is not None, "PGR=2 kernel failed to assemble"

    def test_pgr2_mxfp4_assembles(self):
        """PGR=2 with MXFP4 should assemble."""
        from kernel_generator.gemm.problem import DataType, MfmaConfig
        from kernel_generator.gemm.tiling import GemmTiling
        mx = MfmaConfig.mxfp4_16x16x128()
        t = GemmTiling.high_perf(wg_m=256, wg_n=256, unroll_k=256,
                                  mfma=mx, lds_swizzle=True)
        p = GemmProblem(256, 256, 256, dtype=DataType.MXFP4)
        kernel = GemmKernel.build(p, tiling=t, pgr2=True)
        result = kernel.emit()
        co = result.assemble()
        assert co is not None, "PGR=2 MXFP4 kernel failed to assemble"

    def test_pgr1_still_works(self):
        """Default PGR=1 should still emit and assemble."""
        kernel = GemmKernel.build(GemmProblem(256, 256, 256))
        result = kernel.emit()
        co = result.assemble()
        assert co is not None, "PGR=1 kernel failed to assemble"
