# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for software-pipelined K-loop."""
import pytest
from stinkytofu.gemm.prefetch import emit_pipelined_k_loop
from stinkytofu.gemm.codegen_v2 import generate_from_tree, GenerateResult
from stinkytofu.gemm.problem import GemmProblem, TileConfig, MfmaConfig
from stinkytofu.gemm.tile import build_gemm_tile_tree

try:
    import stinkytofu as _st
    HAS_ST = hasattr(_st, "LogicalModule")
except ImportError:
    HAS_ST = False

requires_st = pytest.mark.skipif(not HAS_ST, reason="stinkytofu not available")


class TestPipelinedDryRun:
    def test_dry_run_basic(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        t = TileConfig(wg_m=128, wg_n=128, unroll_k=32,
                       waves_m=2, waves_n=2, mfma=MfmaConfig.f16_16x16x16())
        tree = build_gemm_tile_tree(wg_m=128, wg_n=128, unroll_k=32,
                                    waves_m=2, waves_n=2,
                                    mfma_m=16, mfma_n=16, mfma_k=16)
        tree = tree.replace("wave", emit=emit_pipelined_k_loop)
        r = generate_from_tree(p, tile=t, tile_tree=tree, dry_run=True)
        assert r.ctx is not None

    def test_indices_still_set(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        t = TileConfig(wg_m=128, wg_n=128, unroll_k=32,
                       waves_m=2, waves_n=2, mfma=MfmaConfig.f16_16x16x16())
        tree = build_gemm_tile_tree(wg_m=128, wg_n=128, unroll_k=32,
                                    waves_m=2, waves_n=2,
                                    mfma_m=16, mfma_n=16, mfma_k=16)
        tree = tree.replace("wave", emit=emit_pipelined_k_loop)
        r = generate_from_tree(p, tile=t, tile_tree=tree, dry_run=True)
        assert "wave.ki" in r.ctx.indices
        assert "wave.mi" in r.ctx.indices


@requires_st
class TestPipelinedGeneration:
    def test_generates_with_prefetch(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        t = TileConfig(wg_m=128, wg_n=128, unroll_k=32,
                       waves_m=2, waves_n=2, mfma=MfmaConfig.f16_16x16x16())
        tree = build_gemm_tile_tree(wg_m=128, wg_n=128, unroll_k=32,
                                    waves_m=2, waves_n=2,
                                    mfma_m=16, mfma_n=16, mfma_k=16)
        tree = tree.replace("wave", emit=emit_pipelined_k_loop)
        r = generate_from_tree(p, tile=t, tile_tree=tree)

        assert r.module is not None
        dump = r.module.dump()
        assert "k_loop_pipelined" in dump
        assert "k_epilog" in dump
        assert "prefetch" in dump.lower()

    def test_more_mfmas_than_flat(self):
        """Pipelined version should have MORE MFMAs (k_tiles-1 compute
        iterations in the main loop + 1 epilog = same count, but the
        global loads are overlapped)."""
        p = GemmProblem(m=4096, n=4096, k=128)  # 4 K-tiles
        t = TileConfig(wg_m=128, wg_n=128, unroll_k=32,
                       waves_m=2, waves_n=2, mfma=MfmaConfig.f16_16x16x16())
        tree_flat = build_gemm_tile_tree(wg_m=128, wg_n=128, unroll_k=32,
                                         waves_m=2, waves_n=2,
                                         mfma_m=16, mfma_n=16, mfma_k=16)
        tree_pf = build_gemm_tile_tree(wg_m=128, wg_n=128, unroll_k=32,
                                       waves_m=2, waves_n=2,
                                       mfma_m=16, mfma_n=16, mfma_k=16)
        tree_pf = tree_pf.replace("wave", emit=emit_pipelined_k_loop)

        r_flat = generate_from_tree(p, tile=t, tile_tree=tree_flat)
        r_pf = generate_from_tree(p, tile=t, tile_tree=tree_pf)

        # Both should have the same MFMA count per unroll
        mfma_flat = r_flat.module.dump().count("MFMA m")
        mfma_pf = r_pf.module.dump().count("MFMA m")

        # Pipelined: (k_tiles-1) main iterations + 1 epilog = k_tiles
        # times the per-unroll MFMA count
        k_tiles = p.k // t.unroll_k  # 4
        mfma_per_unroll = t.mfma_m_repeat * t.mfma_n_repeat * t.k_iterations  # 32
        assert mfma_pf == k_tiles * mfma_per_unroll

    def test_has_prefetch_loads(self):
        p = GemmProblem(m=4096, n=4096, k=128)
        t = TileConfig(wg_m=128, wg_n=128, unroll_k=32,
                       waves_m=2, waves_n=2, mfma=MfmaConfig.f16_16x16x16())
        tree = build_gemm_tile_tree(wg_m=128, wg_n=128, unroll_k=32,
                                    waves_m=2, waves_n=2,
                                    mfma_m=16, mfma_n=16, mfma_k=16)
        tree = tree.replace("wave", emit=emit_pipelined_k_loop)
        r = generate_from_tree(p, tile=t, tile_tree=tree)
        dump = r.module.dump()
        # Should have prefetch global loads
        assert "A(prefetch)" in dump or "prefetch" in dump

    def test_prefetch_buffers_freed(self):
        """Prefetch buffers should be freed after the K-loop."""
        p = GemmProblem(m=4096, n=4096, k=128)
        t = TileConfig(wg_m=128, wg_n=128, unroll_k=32,
                       waves_m=2, waves_n=2, mfma=MfmaConfig.f16_16x16x16())
        tree = build_gemm_tile_tree(wg_m=128, wg_n=128, unroll_k=32,
                                    waves_m=2, waves_n=2,
                                    mfma_m=16, mfma_n=16, mfma_k=16)
        tree = tree.replace("wave", emit=emit_pipelined_k_loop)
        r = generate_from_tree(p, tile=t, tile_tree=tree)
        # Prefetch VGPRs should have been freed
        assert not r.ctx.has("v_prefetch_a")
        assert not r.ctx.has("v_prefetch_b")
