# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the GemmKernel pipeline with replaceable phases."""
import pytest
from stinkytofu.gemm.kernel_pipeline import (
    GemmKernel, MemoryView, default_prologue, default_epilogue,
    default_global_load, default_lds_write, default_compute,
    default_k_advance, default_loop_control, default_mfma_visitor,
)
from stinkytofu.gemm.problem import GemmProblem, TileConfig, MfmaConfig


class TestGemmKernelBuild:
    def test_build_default(self):
        kernel = GemmKernel.build(GemmProblem(4096, 4096, 4096))
        assert kernel.prologue is default_prologue
        assert kernel.epilogue is default_epilogue
        assert kernel.k_loop.global_load is default_global_load
        assert kernel.k_loop.compute is default_compute
        assert kernel.mfma_visitor is default_mfma_visitor

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


class TestPhaseReplacement:
    def test_replace_epilogue(self):
        """Skip epilogue entirely."""
        def noop_epilogue(ctx, kernel):
            ctx.comment("epilogue replaced")

        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        kernel.epilogue = noop_epilogue
        result = kernel.emit()
        stores = [l for l in result.ctx.lines if 'global_store' in l]
        assert len(stores) == 0
        assert any("epilogue replaced" in l for l in result.ctx.lines)

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
        def custom_advance(ctx, kernel):
            ctx.comment("CUSTOM_MARKER")
            default_k_advance(ctx, kernel)

        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        kernel.k_loop.k_advance = custom_advance
        result = kernel.emit()
        assert any("CUSTOM_MARKER" in l for l in result.ctx.lines)

    def test_replace_tile_tree(self):
        """Replace the MFMA leaf via tile tree."""
        emit_count = [0]
        def my_mfma(level, ctx):
            emit_count[0] += 1
            default_mfma_visitor(level, ctx)

        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        kernel.tile_tree = kernel.tile_tree.replace("mfma", emit=my_mfma)
        kernel.emit()
        assert emit_count[0] > 0


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
        def custom_compute(ctx, kernel):
            views_found.append(ctx.get_view("A").name)
            views_found.append(ctx.get_view("B").name)
            default_compute(ctx, kernel)

        kernel = GemmKernel.build(GemmProblem(128, 128, 32))
        kernel.k_loop.compute = custom_compute
        kernel.emit()
        assert views_found == ["A", "B"]

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
