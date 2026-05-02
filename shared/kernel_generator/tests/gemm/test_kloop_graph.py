"""Tests for KLoopGraph and KLoopScheduler.

Validates dependency-driven scheduling matches manual ComposableKLoop."""
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
from kernel_generator.gemm.schedule.kloop_scheduler import (
    KLoopScheduler, ScheduledKLoop,
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


# ===================================================================
# Scheduler tests
# ===================================================================

class TestKLoopSchedulerBasic:
    """Test that KLoopScheduler produces valid schedules."""

    def _build_graph(self, wg_m=256, wg_n=256, unroll_k=64):
        tiling = GemmTiling.high_perf(wg_m=wg_m, wg_n=wg_n,
                                       unroll_k=unroll_k)
        tile = tiling.to_tile_config()
        problem = GemmProblem(wg_m, wg_n, unroll_k)
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
        g.validate()
        return g, tile

    def test_mfma_order_matches_manual(self):
        """Scheduled MFMA order must match manual mi,ki,ni traversal."""
        g, tile = self._build_graph()
        scheduler = KLoopScheduler(g)
        result = scheduler.schedule()

        mr = tile.mfma_m_repeat
        nr = tile.mfma_n_repeat
        ki_count = tile.k_iterations

        expected = []
        for mi in range(mr):
            for ki in range(ki_count):
                for ni in range(nr):
                    expected.append(f"mfma_m{mi}_n{ni}_k{ki}")

        actual = [op.name for op in result.mfma_order]
        assert actual == expected

    def test_pre_body_has_b_reads_ki0(self):
        """B reads for ki=0 should be in pre-body (before barrier)."""
        g, tile = self._build_graph()
        result = KLoopScheduler(g).schedule()

        nr = tile.mfma_n_repeat
        pre_body_names = {op.name for op in result.pre_body_ops}
        for ni in range(nr):
            assert f"read_b_n{ni}_k0" in pre_body_names

    def test_preamble_has_a_m0_reads(self):
        """Preamble should contain A reads for m0 and B reads for ki=1."""
        g, tile = self._build_graph()
        result = KLoopScheduler(g).schedule()

        preamble_names = [op.name for op in result.preamble_ops]
        assert "read_a_m0_k0_buf0" in preamble_names
        if tile.k_iterations > 1:
            assert "read_a_m0_k1_buf0" in preamble_names

    def test_a_prefetch_respects_war_deps(self):
        """A reads for mi+2 must be placed after mi's last MFMA."""
        g, tile = self._build_graph()
        result = KLoopScheduler(g).schedule()

        mr = tile.mfma_m_repeat
        nr = tile.mfma_n_repeat
        ki_count = tile.k_iterations
        mfma_positions = {op.name: i for i, op in enumerate(result.mfma_order)}

        # Build read positions from side_ops
        read_mfma_pos = {}
        for i, ops in enumerate(result.side_ops):
            for op in ops:
                if op.kind == OpKind.DS_READ:
                    read_mfma_pos[op.name] = i

        # For mi >= 2, read_a should be placed after prior mi's last MFMA
        for mi in range(2, mr):
            prior_mi = mi - 2
            last_mfma_pos = mfma_positions[
                f"mfma_m{prior_mi}_n{nr-1}_k{ki_count-1}"]

            for ki in range(ki_count):
                buf = mi % 2
                read_name = f"read_a_m{mi}_k{ki}_buf{buf}"
                if read_name in read_mfma_pos:
                    assert read_mfma_pos[read_name] > last_mfma_pos, \
                        f"{read_name} at {read_mfma_pos[read_name]} " \
                        f"but must be after MFMA pos {last_mfma_pos}"

    def test_reads_before_consuming_mfma(self):
        """Every ds_read must be placed before its consuming MFMA."""
        g, tile = self._build_graph()
        result = KLoopScheduler(g).schedule()

        mfma_positions = {op.name: i for i, op in enumerate(result.mfma_order)}

        # Collect read positions
        read_positions = {}
        for op in result.pre_body_ops:
            read_positions[op.name] = -2
        for op in result.preamble_ops:
            read_positions[op.name] = -1
        for i, ops in enumerate(result.side_ops):
            for op in ops:
                if op.kind == OpKind.DS_READ:
                    read_positions[op.name] = i

        # Check: every RAW dep from ds_read to MFMA is satisfied
        for dep in g.deps:
            if dep.kind != DepKind.RAW:
                continue
            prod = g.ops.get(dep.producer)
            cons = g.ops.get(dep.consumer)
            if (prod and prod.kind == OpKind.DS_READ
                    and cons and cons.kind == OpKind.MFMA):
                read_pos = read_positions.get(dep.producer)
                mfma_pos = mfma_positions.get(dep.consumer)
                if read_pos is not None and mfma_pos is not None:
                    assert read_pos < mfma_pos, \
                        f"{dep.producer} at {read_pos} but " \
                        f"{dep.consumer} at {mfma_pos}"

    def test_no_unplaced_reads(self):
        """Every ds_read must be placed somewhere."""
        g, tile = self._build_graph()
        result = KLoopScheduler(g).schedule()

        all_reads = {op.name for op in g.ds_read_ops()}
        placed = set()
        for op in result.pre_body_ops:
            placed.add(op.name)
        for op in result.preamble_ops:
            placed.add(op.name)
        for ops in result.side_ops:
            for op in ops:
                if op.kind == OpKind.DS_READ:
                    placed.add(op.name)

        missing = all_reads - placed
        assert not missing, f"Unplaced reads: {missing}"

    def test_prefetch_ops_extracted(self):
        """Iteration=1 ops should be in prefetch_ops."""
        g, tile = self._build_graph()
        result = KLoopScheduler(g).schedule()

        prefetch_names = {op.name for op in result.prefetch_ops}
        assert "advance" in prefetch_names
        assert "toggle" in prefetch_names
        assert "global_load_next" in prefetch_names

    def test_mxfp4_schedule_valid(self):
        """MXFP4 config also produces valid schedule."""
        mx = MfmaConfig.mxfp4_16x16x128()
        tiling = GemmTiling.high_perf(
            wg_m=256, wg_n=256, unroll_k=256, mfma=mx)
        tile = tiling.to_tile_config()
        problem = GemmProblem(256, 256, 256, dtype=DataType.MXFP4)

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
        g.validate()

        result = KLoopScheduler(g).schedule()

        # All MFMAs present
        assert len(result.mfma_order) == tile.mfma_m_repeat * tile.mfma_n_repeat * tile.k_iterations

        # All reads placed
        all_reads = {op.name for op in g.ds_read_ops()}
        placed = set()
        for op in result.pre_body_ops:
            placed.add(op.name)
        for op in result.preamble_ops:
            placed.add(op.name)
        for ops in result.side_ops:
            for op in ops:
                if op.kind == OpKind.DS_READ:
                    placed.add(op.name)
        assert all_reads == placed


class TestScheduledVsManual:
    """Compare scheduled and manual (composable) kernel output."""

    def _emit_both(self, **kwargs):
        """Build and emit both composable and scheduled kernels."""
        from kernel_generator.gemm.kernel import GemmKernel

        problem = kwargs.pop("problem", GemmProblem(256, 256, 64))

        k_manual = GemmKernel.build(problem, composable=True, **kwargs)
        r_manual = k_manual.emit()

        k_sched = GemmKernel.build(problem, scheduled=True, **kwargs)
        r_sched = k_sched.emit()

        return r_manual, r_sched

    def _extract_mfma_lines(self, result):
        """Extract MFMA instruction lines (comment part)."""
        return [l.strip() for l in result.ctx.lines if 'v_mfma_f32' in l or 'v_mfma_scale' in l]

    def _extract_ds_read_lines(self, result):
        """Extract ds_read instruction lines."""
        return [l.strip() for l in result.ctx.lines if 'ds_read' in l]

    def _count(self, result, pattern):
        return sum(1 for l in result.ctx.lines if pattern in l)

    def test_fp16_mfma_count_matches(self):
        """Same number of MFMAs in both paths."""
        r_man, r_sch = self._emit_both()
        assert self._count(r_man, 'v_mfma_f32') == self._count(r_sch, 'v_mfma_f32')

    def test_fp16_ds_read_count_matches(self):
        """Same number of ds_reads in both paths."""
        r_man, r_sch = self._emit_both()
        assert self._count(r_man, 'ds_read') == self._count(r_sch, 'ds_read')

    def test_fp16_mfma_order_matches(self):
        """MFMA comments (mi,ni,ki) should be in same order."""
        r_man, r_sch = self._emit_both()
        man_mfmas = self._extract_mfma_lines(r_man)
        sch_mfmas = self._extract_mfma_lines(r_sch)
        # Extract just the MFMA comment (m{mi}_n{ni}_k{ki})
        import re
        def extract_comment(line):
            m = re.search(r'm\d+_n\d+_k\d+', line)
            return m.group(0) if m else line

        man_order = [extract_comment(l) for l in man_mfmas]
        sch_order = [extract_comment(l) for l in sch_mfmas]
        assert man_order == sch_order, (
            f"MFMA order differs at index "
            f"{next(i for i,(a,b) in enumerate(zip(man_order,sch_order)) if a!=b)}")

    def test_fp16_barrier_count_matches(self):
        """Same number of barriers."""
        r_man, r_sch = self._emit_both()
        assert self._count(r_man, 's_barrier') == self._count(r_sch, 's_barrier')

    def test_fp16_total_lines_within_10pct(self):
        """Total line count should be within 10%."""
        r_man, r_sch = self._emit_both()
        man_lines = len(r_man.ctx.lines)
        sch_lines = len(r_sch.ctx.lines)
        ratio = abs(man_lines - sch_lines) / max(man_lines, 1)
        assert ratio < 0.10, (
            f"Line count differs by {ratio:.0%}: "
            f"manual={man_lines}, scheduled={sch_lines}")

    def test_fp16_assembles(self):
        """Scheduled kernel assembles to a .co file."""
        import os, tempfile
        from kernel_generator.gemm.kernel import GemmKernel
        problem = GemmProblem(256, 256, 64)
        k = GemmKernel.build(problem, scheduled=True)
        result = k.emit()
        with tempfile.TemporaryDirectory() as d:
            co = result.assemble(output_path=os.path.join(d, "test.co"))
            assert os.path.exists(co)
            assert os.path.getsize(co) > 0


class TestScheduledMXFP4:
    """Scheduled path for MXFP4 kernels."""

    def test_mxfp4_emits(self):
        """MXFP4 scheduled kernel emits valid assembly."""
        from kernel_generator.gemm.kernel import GemmKernel
        mx = MfmaConfig.mxfp4_16x16x128()
        tiling = GemmTiling.high_perf(
            wg_m=256, wg_n=256, unroll_k=256, mfma=mx)
        problem = GemmProblem(256, 256, 256, dtype=DataType.MXFP4)
        k = GemmKernel.build(problem, tiling=tiling, scheduled=True)
        result = k.emit()
        # Should have MFMAs
        mfma_count = sum(1 for l in result.ctx.lines
                         if 'v_mfma_scale' in l or 'v_mfma_f32' in l)
        assert mfma_count > 0, "No MFMAs found"
        # Should have ds_reads
        ds_reads = sum(1 for l in result.ctx.lines if 'ds_read' in l)
        assert ds_reads > 0, "No ds_reads found"

    def test_mxfp4_mfma_count_matches_composable(self):
        """MXFP4 scheduled and composable produce same MFMA count."""
        from kernel_generator.gemm.kernel import GemmKernel
        mx = MfmaConfig.mxfp4_16x16x128()
        tiling = GemmTiling.high_perf(
            wg_m=256, wg_n=256, unroll_k=256, mfma=mx)
        problem = GemmProblem(256, 256, 256, dtype=DataType.MXFP4)

        k_man = GemmKernel.build(problem, tiling=tiling, composable=True)
        r_man = k_man.emit()

        k_sch = GemmKernel.build(problem, tiling=tiling, scheduled=True)
        r_sch = k_sch.emit()

        def count(r, pat):
            return sum(1 for l in r.ctx.lines if pat in l)

        assert count(r_man, 'v_mfma_scale') == count(r_sch, 'v_mfma_scale')

    def test_mxfp4_assembles(self):
        """MXFP4 scheduled kernel assembles to .co."""
        import os, tempfile
        from kernel_generator.gemm.kernel import GemmKernel
        mx = MfmaConfig.mxfp4_16x16x128()
        tiling = GemmTiling.high_perf(
            wg_m=256, wg_n=256, unroll_k=256, mfma=mx)
        problem = GemmProblem(256, 256, 256, dtype=DataType.MXFP4)
        k = GemmKernel.build(problem, tiling=tiling, scheduled=True)
        result = k.emit()
        with tempfile.TemporaryDirectory() as d:
            co = result.assemble(output_path=os.path.join(d, "mxfp4.co"))
            assert os.path.exists(co)
            assert os.path.getsize(co) > 0
