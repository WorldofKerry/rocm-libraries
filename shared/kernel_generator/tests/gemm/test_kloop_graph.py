"""Tests for KLoopGraph: validate dependency-driven scheduling
matches the manual ComposableKLoop output."""
from __future__ import annotations

import pytest

from kernel_generator.gemm.problem import (
    GemmProblem, DataType, MfmaConfig, TileConfig,
)
from kernel_generator.gemm.tiling import GemmTiling
from kernel_generator.gemm.schedule.kloop_graph import (
    OpKind, DepKind, KLoopOp, Dep, KLoopGraph,
    MFMABlock, DSReadBlock, GlobalLoadBlock,
)


# ===================================================================
# Graph construction tests
# ===================================================================

class TestKLoopGraphBasic:
    """Test that building blocks register ops and deps correctly."""

    def _make_tile(self, wg_m=256, wg_n=256, unroll_k=64):
        tiling = GemmTiling.high_perf(wg_m=wg_m, wg_n=wg_n,
                                       unroll_k=unroll_k)
        return tiling.to_tile_config()

    def test_graph_empty(self):
        tile = self._make_tile()
        problem = GemmProblem(256, 256, 64)
        g = KLoopGraph(tile, problem)
        assert len(g.ops) == 0
        assert len(g.deps) == 0

    def test_global_load_block(self):
        """GlobalLoadBlock registers advance, toggle, load, barrier."""
        tile = self._make_tile()
        problem = GemmProblem(256, 256, 64)
        g = KLoopGraph(tile, problem)

        class FakeLoader:
            def advance(self): pass
            def toggle_write(self): pass
            def emit_loads(self): pass
            def emit_sync(self): pass

        GlobalLoadBlock(FakeLoader()).register(g)

        assert "advance" in g.ops
        assert "toggle" in g.ops
        assert "global_load_next" in g.ops
        assert "barrier" in g.ops

        # iteration=1 for prefetch ops
        assert g.ops["advance"].iteration == 1
        assert g.ops["global_load_next"].iteration == 1
        assert g.ops["barrier"].iteration == 0

        # Deps: advance -> load, toggle -> load, load -> barrier
        dep_pairs = [(d.producer, d.consumer) for d in g.deps]
        assert ("advance", "global_load_next") in dep_pairs
        assert ("toggle", "global_load_next") in dep_pairs
        assert ("global_load_next", "barrier") in dep_pairs

    def test_ds_read_block_counts(self):
        """DSReadBlock produces correct number of ops and deps."""
        tile = self._make_tile()
        problem = GemmProblem(256, 256, 64)
        g = KLoopGraph(tile, problem)

        # Need barrier op (normally from GlobalLoadBlock)
        g.add_op(KLoopOp("barrier", OpKind.BARRIER))

        mr = tile.mfma_m_repeat  # 8
        nr = tile.mfma_n_repeat  # 8
        ki = tile.k_iterations   # 2

        class FakeReader:
            def emit_read_a(self, mi, ki, buf): pass
            def emit_read_b(self, ni, ki): pass

        DSReadBlock(FakeReader()).register(g)

        # A reads: mr * ki = 8 * 2 = 16, B reads: nr * ki = 8 * 2 = 16
        a_reads = [n for n in g.ops if n.startswith("read_a")]
        b_reads = [n for n in g.ops if n.startswith("read_b")]
        assert len(a_reads) == mr * ki
        assert len(b_reads) == nr * ki

    def test_a_pingpong_war_deps(self):
        """A ping-pong creates WAR deps: mi=2 depends on mi=0's last MFMA."""
        tile = self._make_tile()
        problem = GemmProblem(256, 256, 64)
        g = KLoopGraph(tile, problem)
        g.add_op(KLoopOp("barrier", OpKind.BARRIER))

        nr = tile.mfma_n_repeat
        ki_count = tile.k_iterations

        class FakeReader:
            def emit_read_a(self, mi, ki, buf): pass
            def emit_read_b(self, ni, ki): pass

        DSReadBlock(FakeReader()).register(g)

        # mi=0,1 have no WAR deps (first use of buf0, buf1)
        war_deps = [d for d in g.deps if d.kind == DepKind.WAR]

        # mi=2 (buf0) has WAR dep on mfma_m0_n{nr-1}_k{ki-1}
        war_on_m2 = [d for d in war_deps
                     if d.consumer.startswith("read_a_m2")]
        assert len(war_on_m2) == ki_count
        for d in war_on_m2:
            assert d.producer == f"mfma_m0_n{nr-1}_k{ki_count-1}"

    def test_mfma_block_counts(self):
        """MFMABlock produces mr * nr * ki MFMAs."""
        tile = self._make_tile()
        problem = GemmProblem(256, 256, 64)
        g = KLoopGraph(tile, problem)

        # Need barrier + all ds_reads first
        g.add_op(KLoopOp("barrier", OpKind.BARRIER))

        class FakeReader:
            def emit_read_a(self, mi, ki, buf): pass
            def emit_read_b(self, ni, ki): pass

        DSReadBlock(FakeReader()).register(g)

        from kernel_generator.gemm.emit.context import AsmContext
        ctx = AsmContext()
        ctx._metadata = {"tile": tile, "problem": problem}
        MFMABlock(ctx, tile).register(g)

        mr = tile.mfma_m_repeat
        nr = tile.mfma_n_repeat
        ki_count = tile.k_iterations
        mfmas = g.mfma_ops()
        assert len(mfmas) == mr * nr * ki_count

    def test_full_graph_validates(self):
        """A complete graph with all blocks passes validation."""
        tile = self._make_tile()
        problem = GemmProblem(256, 256, 64)
        g = KLoopGraph(tile, problem)

        class FakeLoader:
            def advance(self): pass
            def toggle_write(self): pass
            def emit_loads(self): pass
            def emit_sync(self): pass

        class FakeReader:
            def emit_read_a(self, mi, ki, buf): pass
            def emit_read_b(self, ni, ki): pass

        GlobalLoadBlock(FakeLoader()).register(g)
        DSReadBlock(FakeReader()).register(g)

        from kernel_generator.gemm.emit.context import AsmContext
        ctx = AsmContext()
        ctx._metadata = {"tile": tile, "problem": problem}
        MFMABlock(ctx, tile).register(g)

        g.validate()  # should not raise

        # Verify totals
        mr, nr, ki = tile.mfma_m_repeat, tile.mfma_n_repeat, tile.k_iterations
        assert len(g.mfma_ops()) == mr * nr * ki
        assert len(g.ds_read_ops()) == mr * ki + nr * ki


class TestKLoopGraphMXFP4:
    """Test graph construction for MXFP4 (different tile/mfma config)."""

    def _make_tile(self):
        mx = MfmaConfig.mxfp4_16x16x128()
        tiling = GemmTiling.high_perf(
            wg_m=256, wg_n=256, unroll_k=256, mfma=mx)
        return tiling.to_tile_config()

    def test_mxfp4_mfma_count(self):
        tile = self._make_tile()
        problem = GemmProblem(256, 256, 256, dtype=DataType.MXFP4)
        g = KLoopGraph(tile, problem)
        g.add_op(KLoopOp("barrier", OpKind.BARRIER))

        class FakeReader:
            def emit_read_a(self, mi, ki, buf): pass
            def emit_read_b(self, ni, ki): pass

        DSReadBlock(FakeReader()).register(g)

        from kernel_generator.gemm.emit.context import AsmContext
        ctx = AsmContext()
        ctx._metadata = {"tile": tile, "problem": problem}
        MFMABlock(ctx, tile).register(g)

        mr = tile.mfma_m_repeat   # 16
        nr = tile.mfma_n_repeat   # 16
        ki_count = tile.k_iterations  # 2
        assert len(g.mfma_ops()) == mr * nr * ki_count

    def test_mxfp4_war_dep_structure(self):
        """MXFP4 has more mi values, WAR deps should still be correct."""
        tile = self._make_tile()
        problem = GemmProblem(256, 256, 256, dtype=DataType.MXFP4)
        g = KLoopGraph(tile, problem)
        g.add_op(KLoopOp("barrier", OpKind.BARRIER))

        class FakeReader:
            def emit_read_a(self, mi, ki, buf): pass
            def emit_read_b(self, ni, ki): pass

        DSReadBlock(FakeReader()).register(g)

        mr = tile.mfma_m_repeat
        nr = tile.mfma_n_repeat
        ki_count = tile.k_iterations

        war_deps = [d for d in g.deps if d.kind == DepKind.WAR]

        # mi=0,1: no WAR. mi=2..mr-1: each has ki_count WAR deps
        expected_war = (mr - 2) * ki_count
        assert len(war_deps) == expected_war

        # Verify mi=4 (buf0) depends on mi=2's last MFMA (not mi=0)
        war_on_m4 = [d for d in war_deps
                     if d.consumer.startswith("read_a_m4")]
        for d in war_on_m4:
            assert d.producer == f"mfma_m2_n{nr-1}_k{ki_count-1}"


class TestGraphStructuralComparison:
    """Compare the graph-declared op structure against what
    the manual ComposableKLoop produces.

    This validates that the dependency graph captures the same
    operations the manual code would emit.
    """

    def _manual_op_names(self, tile):
        """Return the set of (kind, mi, ni, ki) tuples the manual code emits."""
        mr = tile.mfma_m_repeat
        nr = tile.mfma_n_repeat
        ki_count = tile.k_iterations

        mfmas = set()
        for mi in range(mr):
            for ki in range(ki_count):
                for ni in range(nr):
                    mfmas.add(("mfma", mi, ni, ki))

        a_reads = set()
        for mi in range(mr):
            for ki in range(ki_count):
                a_reads.add(("read_a", mi, ki, mi % 2))

        b_reads = set()
        for ni in range(nr):
            for ki in range(ki_count):
                b_reads.add(("read_b", ni, ki))

        return mfmas, a_reads, b_reads

    def _graph_op_names(self, graph):
        """Extract (kind, indices) from graph ops."""
        mfmas = set()
        a_reads = set()
        b_reads = set()

        for name, op in graph.ops.items():
            if op.kind == OpKind.MFMA:
                # Parse mfma_m{mi}_n{ni}_k{ki}
                parts = name.replace("mfma_m", "").replace("_n", " ").replace("_k", " ").split()
                mi, ni, ki = int(parts[0]), int(parts[1]), int(parts[2])
                mfmas.add(("mfma", mi, ni, ki))
            elif name.startswith("read_a_m"):
                # Parse read_a_m{mi}_k{ki}_buf{buf}
                parts = name.replace("read_a_m", "").replace("_k", " ").replace("_buf", " ").split()
                mi, ki, buf = int(parts[0]), int(parts[1]), int(parts[2])
                a_reads.add(("read_a", mi, ki, buf))
            elif name.startswith("read_b_n"):
                # Parse read_b_n{ni}_k{ki}
                parts = name.replace("read_b_n", "").replace("_k", " ").split()
                ni, ki = int(parts[0]), int(parts[1])
                b_reads.add(("read_b", ni, ki))

        return mfmas, a_reads, b_reads

    def test_fp16_256x256x64_ops_match(self):
        """Graph ops match manual ops for 256x256x64 fp16."""
        tiling = GemmTiling.high_perf(wg_m=256, wg_n=256, unroll_k=64)
        tile = tiling.to_tile_config()
        problem = GemmProblem(256, 256, 64)

        g = KLoopGraph(tile, problem)
        g.add_op(KLoopOp("barrier", OpKind.BARRIER))

        class FR:
            def emit_read_a(self, mi, ki, buf): pass
            def emit_read_b(self, ni, ki): pass

        DSReadBlock(FR()).register(g)

        from kernel_generator.gemm.emit.context import AsmContext
        ctx = AsmContext()
        ctx._metadata = {"tile": tile, "problem": problem}
        MFMABlock(ctx, tile).register(g)

        manual_m, manual_a, manual_b = self._manual_op_names(tile)
        graph_m, graph_a, graph_b = self._graph_op_names(g)

        assert graph_m == manual_m, f"MFMA mismatch: {graph_m ^ manual_m}"
        assert graph_a == manual_a, f"A-read mismatch: {graph_a ^ manual_a}"
        assert graph_b == manual_b, f"B-read mismatch: {graph_b ^ manual_b}"

    def test_mxfp4_256x256x256_ops_match(self):
        """Graph ops match manual ops for MXFP4 256x256x256."""
        mx = MfmaConfig.mxfp4_16x16x128()
        tiling = GemmTiling.high_perf(
            wg_m=256, wg_n=256, unroll_k=256, mfma=mx)
        tile = tiling.to_tile_config()
        problem = GemmProblem(256, 256, 256, dtype=DataType.MXFP4)

        g = KLoopGraph(tile, problem)
        g.add_op(KLoopOp("barrier", OpKind.BARRIER))

        class FR:
            def emit_read_a(self, mi, ki, buf): pass
            def emit_read_b(self, ni, ki): pass

        DSReadBlock(FR()).register(g)

        from kernel_generator.gemm.emit.context import AsmContext
        ctx = AsmContext()
        ctx._metadata = {"tile": tile, "problem": problem}
        MFMABlock(ctx, tile).register(g)

        manual_m, manual_a, manual_b = self._manual_op_names(tile)
        graph_m, graph_a, graph_b = self._graph_op_names(g)

        assert graph_m == manual_m
        assert graph_a == manual_a
        assert graph_b == manual_b
