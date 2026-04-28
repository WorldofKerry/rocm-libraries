# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for codegen_v2: tree-walking codegen with TileContext."""
import pytest
from stinkytofu.gemm.codegen_v2 import (
    generate_from_tree, default_visitor, GenerateResult, EMIT_REGISTRY,
)
from stinkytofu.gemm.problem import GemmProblem, TileConfig, MfmaConfig
from stinkytofu.gemm.tile import TileLevel, build_gemm_tile_tree
from stinkytofu.gemm.context import TileContext

try:
    import stinkytofu as _st
    HAS_ST = hasattr(_st, "LogicalModule")
except ImportError:
    HAS_ST = False

requires_st = pytest.mark.skipif(not HAS_ST, reason="stinkytofu C extension not available")


# ===========================================================================
# Dry-run tests (pure Python)
# ===========================================================================

class TestDryRun:
    def test_basic(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        r = generate_from_tree(p, dry_run=True)
        assert isinstance(r, GenerateResult)
        assert r.module is None
        assert r.ctx is not None

    def test_summary(self):
        p = GemmProblem(m=2048, n=1024, k=512)
        r = generate_from_tree(p, dry_run=True)
        s = r.summary()
        assert "2048" in s
        assert "Tree:" in s

    def test_register_allocation(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        r = generate_from_tree(p, dry_run=True)
        assert r.ctx.has("srd_A")
        assert r.ctx.has("acc_C")
        assert r.ctx.has("v_a")
        assert r.ctx.has("v_gload_a")

    def test_acc_count(self):
        t = TileConfig(wg_m=128, wg_n=128, unroll_k=32, waves_m=2, waves_n=2,
                       mfma=MfmaConfig.f16_16x16x16())
        p = GemmProblem(m=4096, n=4096, k=4096)
        r = generate_from_tree(p, tile=t, dry_run=True)
        expected = t.mfma_m_repeat * t.mfma_n_repeat * t.mfma.acc_vgprs
        assert r.ctx.get("acc_C").count == expected

    def test_indices_populated(self):
        p = GemmProblem(m=128, n=128, k=32)
        t = TileConfig(wg_m=128, wg_n=128, unroll_k=32, waves_m=2, waves_n=2,
                       mfma=MfmaConfig.f16_16x16x16())
        r = generate_from_tree(p, tile=t, dry_run=True)
        # After a full walk, indices should have been set
        assert "wave.ki" in r.ctx.indices
        assert "wave.mi" in r.ctx.indices
        assert "wave.ni" in r.ctx.indices

    def test_custom_tile_tree(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        tree = build_gemm_tile_tree(wg_m=64, wg_n=64, unroll_k=16,
                                    waves_m=1, waves_n=1,
                                    mfma_m=16, mfma_n=16, mfma_k=16)
        t = TileConfig(wg_m=64, wg_n=64, unroll_k=16, waves_m=1, waves_n=1,
                       mfma=MfmaConfig.f16_16x16x16())
        r = generate_from_tree(p, tile=t, tile_tree=tree, dry_run=True)
        assert "64x64x16" in r.summary()


# ===========================================================================
# Custom emit override tests
# ===========================================================================

class TestOverrides:
    def test_replace_mfma(self):
        call_log = []

        def my_mfma(level, ctx):
            mi = ctx.indices.get("wave.mi", 0)
            ni = ctx.indices.get("wave.ni", 0)
            ki = ctx.indices.get("wave.ki", 0)
            call_log.append((mi, ni, ki))

        p = GemmProblem(m=128, n=128, k=32)
        t = TileConfig(wg_m=128, wg_n=128, unroll_k=32, waves_m=2, waves_n=2,
                       mfma=MfmaConfig.f16_16x16x16())
        tree = build_gemm_tile_tree(wg_m=128, wg_n=128, unroll_k=32,
                                    waves_m=2, waves_n=2,
                                    mfma_m=16, mfma_n=16, mfma_k=16)
        tree = tree.replace("mfma", emit=my_mfma)
        r = generate_from_tree(p, tile=t, tile_tree=tree, dry_run=True)

        # Should be called mfma_m_repeat * mfma_n_repeat * k_iterations times
        expected = t.mfma_m_repeat * t.mfma_n_repeat * t.k_iterations
        assert len(call_log) == expected

    def test_replace_wave(self):
        """Override the wave level and verify inner levels are skipped."""
        wave_calls = []

        def my_wave(level, ctx):
            wave_calls.append(ctx.indices.get("wave.ki", -1))
            # Don't recurse into inner -- this replaces the whole subtree

        p = GemmProblem(m=128, n=128, k=32)
        t = TileConfig(wg_m=128, wg_n=128, unroll_k=32, waves_m=2, waves_n=2,
                       mfma=MfmaConfig.f16_16x16x16())
        tree = build_gemm_tile_tree(wg_m=128, wg_n=128, unroll_k=32,
                                    waves_m=2, waves_n=2,
                                    mfma_m=16, mfma_n=16, mfma_k=16)
        tree = tree.replace("wave", emit=my_wave)
        r = generate_from_tree(p, tile=t, tile_tree=tree, dry_run=True)

        # Wave is root, called exactly once
        assert len(wave_calls) == 1


# ===========================================================================
# FLOP correctness
# ===========================================================================

class TestFlopCorrectness:
    @pytest.mark.parametrize("m,n,k", [
        (128, 128, 32),
        (1024, 1024, 1024),
        (4096, 4096, 4096),
    ])
    def test_mfma_count_matches_flops(self, m, n, k):
        """Count MFMA calls via custom emit and verify against 2*M*N*K."""
        mfma_count = [0]

        def counting_mfma(level, ctx):
            mfma_count[0] += 1

        p = GemmProblem(m=m, n=n, k=k)
        t = TileConfig(wg_m=128, wg_n=128, unroll_k=32, waves_m=2, waves_n=2,
                       mfma=MfmaConfig.f16_16x16x16())
        tree = build_gemm_tile_tree(wg_m=128, wg_n=128, unroll_k=32,
                                    waves_m=2, waves_n=2,
                                    mfma_m=16, mfma_n=16, mfma_k=16)
        tree = tree.replace("mfma", emit=counting_mfma)
        generate_from_tree(p, tile=t, tile_tree=tree, dry_run=True)

        # mfma_count is per-workgroup-per-wave. Multiply out.
        grid_m, grid_n = p.grid_dims(t)
        k_tiles = p.k // t.unroll_k
        waves = t.waves_m * t.waves_n
        # Our walk covers one WG's one unroll iteration.
        # mfma_count = mfma_m_repeat * mfma_n_repeat * k_iterations (per wave)
        mfmas_per_wg = mfma_count[0]
        # Total across all WGs and K-tiles
        total_flops = (grid_m * grid_n * waves * mfmas_per_wg
                       * k_tiles * t.mfma.flops_per_instruction)
        if m % t.wg_m == 0 and n % t.wg_n == 0 and k % t.unroll_k == 0:
            assert total_flops == p.total_flops


# ===========================================================================
# Full generation (requires stinkytofu)
# ===========================================================================

@requires_st
class TestFullGeneration:
    def test_generates_module(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        r = generate_from_tree(p)
        assert r.module is not None
        dump = r.module.dump()
        assert "MFMA" in dump

    def test_mfma_count(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        t = TileConfig(wg_m=128, wg_n=128, unroll_k=32, waves_m=2, waves_n=2,
                       mfma=MfmaConfig.f16_16x16x16())
        r = generate_from_tree(p, tile=t)
        dump = r.module.dump()
        expected = t.mfma_m_repeat * t.mfma_n_repeat * t.k_iterations
        assert dump.count("MFMA m") == expected

    def test_has_barrier(self):
        p = GemmProblem(m=128, n=128, k=32)
        r = generate_from_tree(p)
        dump = r.module.dump()
        assert "Barrier" in dump or "barrier" in dump.lower()

    def test_has_lds_ops(self):
        import stinkytofu as st
        p = GemmProblem(m=128, n=128, k=32)
        r = generate_from_tree(p)
        dump = r.module.dump()
        assert "DSLoad" in dump or "ds_load" in dump.lower()
        assert "DSStore" in dump or "ds_store" in dump.lower()

    def test_has_global_loads(self):
        p = GemmProblem(m=128, n=128, k=32)
        r = generate_from_tree(p)
        dump = r.module.dump()
        assert "BufferLoad" in dump or "buffer_load" in dump.lower()

    def test_has_epilogue(self):
        p = GemmProblem(m=128, n=128, k=32)
        r = generate_from_tree(p)
        dump = r.module.dump()
        assert "epilogue" in dump.lower()
        assert "kernel_end" in dump.lower()

    def test_custom_mfma_changes_output(self):
        import stinkytofu as st

        class _Counter:
            n = 0

        def double_mfma(level, ctx):
            """Emit each MFMA twice."""
            tile = ctx._metadata["tile"]
            mfma = tile.mfma
            mi = ctx.indices.get("wave.mi", 0)
            ni = ctx.indices.get("wave.ni", 0)
            ki = ctx.indices.get("wave.ki", 0)
            acc_per = mfma.acc_vgprs
            acc_off = (mi * tile.mfma_n_repeat + ni) * acc_per
            for _ in range(2):
                ctx.module.add(st.MFMA(
                    instType=mfma.input_type, accType=mfma.acc_type,
                    m=mfma.m, n=mfma.n, k=mfma.k,
                    blocks=mfma.blocks, mfma1k=False,
                    acc=ctx.acc("acc_C", acc_off, acc_per),
                    a=ctx.vgpr("v_a", 0, mfma.a_vgprs),
                    b=ctx.vgpr("v_b", 0, mfma.b_vgprs),
                    comment=f"MFMA m{mi}_n{ni} k{ki} (doubled)",
                ))

        p = GemmProblem(m=128, n=128, k=32)
        t = TileConfig(wg_m=128, wg_n=128, unroll_k=32, waves_m=2, waves_n=2,
                       mfma=MfmaConfig.f16_16x16x16())
        tree = build_gemm_tile_tree(wg_m=128, wg_n=128, unroll_k=32,
                                    waves_m=2, waves_n=2,
                                    mfma_m=16, mfma_n=16, mfma_k=16)

        r_normal = generate_from_tree(p, tile=t)
        r_double = generate_from_tree(p, tile=t,
                                      tile_tree=tree.replace("mfma", emit=double_mfma))

        n_normal = r_normal.module.dump().count("MFMA m")
        n_double = r_double.module.dump().count("MFMA m")
        assert n_double == 2 * n_normal
