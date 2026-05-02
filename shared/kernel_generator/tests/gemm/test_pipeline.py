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
        assert "load_kernargs" in pro_names
        assert "thread_indexing" in pro_names
        assert "k_loop_label" in pro_names
        epi_names = [p.name for p in tree.epilogue_phases]
        assert "store_d" in epi_names

    def test_build_wave_phases(self):
        kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
        wave = kernel.tile_tree.inner
        assert wave.name == "wave"
        pro_names = [p.name for p in wave.prologue_phases]
        assert "global_load" in pro_names
        assert "lds_write" in pro_names
        epi_names = [p.name for p in wave.epilogue_phases]
        assert "k_advance" in epi_names
        assert "k_loop_control" in epi_names

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
            "load_kernargs", "thread_indexing", "load_cluster_setup",
            "lds_addrs", "init_acc",
            "global_addrs", "k_loop_init", "k_loop_label",
            "store_d",
            "global_load", "lds_write",
            "k_advance", "k_loop_control",
        ]
        for name in expected:
            assert name in names, f"Phase '{name}' not found in tree"


class TestGemmKernelEmit:
    def test_emit_produces_assembly(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        result = kernel.emit()
        assert len(result.asm_text) > 0
        assert "gemm_kernel" in result.asm_text
        assert ".amdhsa_kernel" in result.asm_text
        assert ".amdgpu_metadata" in result.asm_text

    def test_emit_assembles(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        result = kernel.emit()
        import os, tempfile
        co = result.assemble(output_path=os.path.join(
            tempfile.gettempdir(), "test_pipeline.co"))
        assert os.path.exists(co)
        assert os.path.getsize(co) > 0
        os.unlink(co)

    def test_emit_register_counts(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        result = kernel.emit()
        assert result.vgpr_count > 0
        assert result.sgpr_count > 0
        assert result.acc_count == 64  # 4x4 MFMA tiles * 4 acc each

    def test_emit_has_mfma(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        result = kernel.emit()
        mfma_lines = [l for l in result.ctx.lines if 'v_mfma_f32' in l]
        expected = (kernel.tile.mfma_m_repeat *
                    kernel.tile.mfma_n_repeat *
                    kernel.tile.k_iterations)
        assert len(mfma_lines) == expected

    def test_emit_has_k_loop(self):
        """Assembly should contain k_loop label and branch."""
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        result = kernel.emit()
        assert any("k_loop:" in l for l in result.ctx.lines)
        assert any("s_cbranch_scc1" in l and "k_loop" in l
                    for l in result.ctx.lines)

    def test_emit_has_global_load(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        result = kernel.emit()
        loads = [l for l in result.ctx.lines if 'global_load_' in l]
        assert len(loads) > 0

    def test_emit_has_store(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        result = kernel.emit()
        stores = [l for l in result.ctx.lines if 'global_store' in l or 'buffer_store' in l]
        assert len(stores) > 0


class TestPhaseReplacement:
    def test_replace_store_phase(self):
        """Replace the store_d phase -- no stores emitted."""
        def noop_store(level, ctx):
            ctx.comment("store replaced")

        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        kernel.tile_tree = kernel.tile_tree.replace_phase(
            "store_d", noop_store)
        result = kernel.emit()
        stores = [l for l in result.ctx.lines if 'global_store' in l or 'buffer_store' in l]
        assert len(stores) == 0
        assert any("store replaced" in l for l in result.ctx.lines)

    def test_replace_mfma_visitor(self):
        """Count MFMA invocations via custom visitor."""
        count = [0]
        def counting_visitor(level, ctx):
            if level.name == "mfma":
                count[0] += 1
            default_mfma_visitor(level, ctx)

        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        kernel.mfma_visitor = counting_visitor
        kernel.emit()
        expected = (kernel.tile.mfma_m_repeat *
                    kernel.tile.mfma_n_repeat *
                    kernel.tile.k_iterations)
        assert count[0] == expected

    def test_replace_k_advance(self):
        """Custom K-advance adds a marker comment."""
        from kernel_generator.gemm.emit.phases import phase_k_advance

        def custom_advance(level, ctx):
            ctx.comment("CUSTOM_MARKER")
            phase_k_advance(level, ctx)

        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        kernel.tile_tree = kernel.tile_tree.replace_phase(
            "k_advance", custom_advance)
        result = kernel.emit()
        assert any("CUSTOM_MARKER" in l for l in result.ctx.lines)

    def test_replace_tile_tree_mfma(self):
        """Replace the MFMA leaf via tile tree."""
        emit_count = [0]
        def my_mfma(level, ctx):
            emit_count[0] += 1
            default_mfma_visitor(level, ctx)

        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        kernel.tile_tree = kernel.tile_tree.replace("mfma", emit=my_mfma)
        kernel.emit()
        assert emit_count[0] > 0

    def test_replace_global_load(self):
        """Replace global_load phase with a marker."""
        def custom_load(level, ctx):
            ctx.comment("CUSTOM_LOAD")
            # Still need to do the actual load for correctness
            from kernel_generator.gemm.emit.phases import phase_global_load
            phase_global_load(level, ctx)

        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        kernel.tile_tree = kernel.tile_tree.replace_phase(
            "global_load", custom_load)
        result = kernel.emit()
        assert any("CUSTOM_LOAD" in l for l in result.ctx.lines)

    def test_get_phase(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        phase = kernel.tile_tree.get_phase("global_load")
        assert phase is not None
        assert phase.name == "global_load"

    def test_get_phase_not_found(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        phase = kernel.tile_tree.get_phase("nonexistent")
        assert phase is None


class TestMemoryViews:
    def test_views_registered(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        result = kernel.emit()
        ctx = result.ctx
        a_view = ctx.get_view("A")
        b_view = ctx.get_view("B")
        assert a_view.source == "lds"
        assert b_view.source == "lds"
        assert b_view.base_offset > 0  # B has LDS offset

    def test_view_in_custom_phase(self):
        """Access MemoryView from a custom compute phase."""
        views_found = []
        def custom_visitor(level, ctx):
            if level.name == "mfma":
                views_found.append(ctx.get_view("A").name)
                views_found.append(ctx.get_view("B").name)
            default_mfma_visitor(level, ctx)

        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        kernel.mfma_visitor = custom_visitor
        kernel.emit()
        assert "A" in views_found
        assert "B" in views_found

    def test_view_layout_matches(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        result = kernel.emit()
        a_view = result.ctx.get_view("A")
        # LDS A layout: row * unroll_k + col
        assert a_view.layout._coefficients[0] == kernel.tile.unroll_k
        assert a_view.layout._coefficients[1] == 1
        assert a_view.elem_bytes == 2

    def test_missing_view_raises(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        result = kernel.emit()
        with pytest.raises(KeyError, match="No MemoryView 'C'"):
            result.ctx.get_view("C")


class TestTreeStructure:
    """Test the tree structure produced by GemmKernel.build()."""

    def test_workgroup_parallel(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        assert kernel.tile_tree.parallel is True

    def test_wave_not_parallel(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        assert kernel.tile_tree.inner.parallel is False

    def test_tree_depth(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        assert kernel.tile_tree.depth == 2  # workgroup -> wave -> mfma

    def test_mfma_is_leaf(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        mfma = kernel.tile_tree.find("mfma")
        assert mfma is not None
        assert mfma.is_leaf is True

    def test_summary_shows_phases(self):
        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        s = kernel.tile_tree.summary()
        assert "prologue:" in s
        assert "epilogue:" in s
        assert "parallel/HW-mapped" in s
        assert "global_load" in s
        assert "store_d" in s
