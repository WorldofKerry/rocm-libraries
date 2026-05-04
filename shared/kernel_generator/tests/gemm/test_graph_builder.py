"""Tests for build_kloop_graph.

Verifies graph structure (ops + deps) produced from LDSStream lists
for fp16 (data only) and MXFP4 (data + scale) configurations.
"""
from __future__ import annotations

import pytest

from kernel_generator.gemm.problem import (
    TileConfig, GemmProblem, MfmaConfig, DataType,
)
from kernel_generator.gemm.tiling import GemmTiling
from kernel_generator.gemm.memory.lds_stream import LDSStream
from kernel_generator.gemm.schedule.kloop_graph import (
    OpKind, DepKind, KLoopOp, KLoopGraph,
)
from kernel_generator.gemm.schedule.graph_builder import build_kloop_graph


# ===================================================================
# Minimal stub streams for graph-building tests.
# Real DTLDataStream / ScaleStream pull in tile geometry we don't
# need here; stubs let us control name, needs_lds_write, has_reads.
# ===================================================================

class StubDataStream(LDSStream):
    """Stub for a matrix data stream (DTL path, no LDS write)."""

    def __init__(self, matrix: str) -> None:
        assert matrix in ("a", "b")
        self._matrix = matrix

    @property
    def name(self):
        return f"data_{self._matrix}"

    @property
    def region_size(self):
        return 1024

    @property
    def num_global_loads(self):
        return 4

    @property
    def needs_lds_write(self):
        return False

    def setup(self, ctx, lds_offset):
        pass

    def emit_global_loads(self, ctx):
        pass

    def emit_lds_writes(self, ctx):
        pass

    def read_op_count(self):
        return 0

    def advance(self, ctx):
        pass

    def toggle_write(self, ctx):
        pass

    def toggle_read(self, ctx):
        pass


class StubScaleStream(LDSStream):
    """Stub for a scale stream (2-step: buffer_load + ds_write)."""

    def __init__(self, matrix: str) -> None:
        assert matrix in ("a", "b")
        self._matrix = matrix

    @property
    def name(self):
        return f"scale_{self._matrix}"

    @property
    def region_size(self):
        return 4096

    @property
    def num_global_loads(self):
        return 4

    @property
    def needs_lds_write(self):
        return True

    def setup(self, ctx, lds_offset):
        pass

    def emit_global_loads(self, ctx):
        pass

    def emit_lds_writes(self, ctx):
        pass

    def read_op_count(self):
        return 0

    def advance(self, ctx):
        pass

    def toggle_write(self, ctx):
        pass

    def toggle_read(self, ctx):
        pass


# ===================================================================
# Tile helpers
# ===================================================================

def _fp16_tile() -> TileConfig:
    """256x256x64, f16_16x16x32 -> mr=8, nr=8, ki=2."""
    return GemmTiling.high_perf(wg_m=256, wg_n=256,
                                unroll_k=64).to_tile_config()


def _mxfp4_tile() -> TileConfig:
    """256x256x256, mxfp4_16x16x128 -> mr=16, nr=16, ki=2."""
    mx = MfmaConfig.mxfp4_16x16x128()
    return GemmTiling.high_perf(wg_m=256, wg_n=256,
                                unroll_k=256, mfma=mx).to_tile_config()


# ===================================================================
# Tests
# ===================================================================

class TestFP16Config:
    """FP16: data_a + data_b, no scale streams."""

    def _build(self, pgr=1, num_buffers=2):
        tile = _fp16_tile()
        streams = [StubDataStream("a"), StubDataStream("b")]
        return build_kloop_graph(streams, tile, pgr=pgr,
                                 num_buffers=num_buffers), tile

    def test_correct_ops_created(self):
        """All expected op names are present."""
        g, tile = self._build()
        mr = tile.mfma_m_repeat
        nr = tile.mfma_n_repeat
        ki = tile.k_iterations

        # Producer ops
        for sn in ("data_a", "data_b"):
            assert f"advance_{sn}" in g.ops
            assert f"toggle_wr_{sn}" in g.ops
            assert f"load_{sn}" in g.ops
            # DTL streams: no write op
            assert f"write_{sn}" not in g.ops

        assert "barrier" in g.ops

        # Data reads
        for mi in range(mr):
            for ki_ in range(ki):
                assert f"read_data_a_m{mi}_k{ki_}" in g.ops
        for ni in range(nr):
            for ki_ in range(ki):
                assert f"read_data_b_n{ni}_k{ki_}" in g.ops

        # MFMAs
        for mi in range(mr):
            for ni in range(nr):
                for ki_ in range(ki):
                    assert f"mfma_m{mi}_n{ni}_k{ki_}" in g.ops

        # Suffix
        assert "toggle_rd_data_a" in g.ops
        assert "toggle_rd_data_b" in g.ops

        # No scale ops
        scale_ops = [n for n in g.ops if "scale" in n]
        assert len(scale_ops) == 0

    def test_producer_distance(self):
        """Producer ops have iteration=pgr."""
        g, _ = self._build(pgr=1)
        for sn in ("data_a", "data_b"):
            assert g.ops[f"advance_{sn}"].iteration == 1
            assert g.ops[f"toggle_wr_{sn}"].iteration == 1
            assert g.ops[f"load_{sn}"].iteration == 1

    def test_consumer_distance(self):
        """Consumer and MFMA ops have iteration=0."""
        g, tile = self._build()
        for name, op in g.ops.items():
            if name.startswith("read_") or name.startswith("mfma_"):
                assert op.iteration == 0, f"{name} has iteration={op.iteration}"

    def test_barrier_deps(self):
        """Terminal producer ops → barrier (SYNC)."""
        g, _ = self._build()
        barrier_preds = {d.producer for d in g.deps
                         if d.consumer == "barrier" and d.kind == DepKind.SYNC}
        # DTL streams: terminal is load (no write)
        assert "load_data_a" in barrier_preds
        assert "load_data_b" in barrier_preds

    def test_barrier_to_reads(self):
        """barrier → every read (SYNC)."""
        g, tile = self._build()
        mr, nr, ki = tile.mfma_m_repeat, tile.mfma_n_repeat, tile.k_iterations
        read_names = {n for n in g.ops if n.startswith("read_")}
        barrier_succs = {d.consumer for d in g.deps
                         if d.producer == "barrier" and d.kind == DepKind.SYNC}
        assert read_names == barrier_succs

    def test_reads_to_mfma(self):
        """RAW deps from data reads to MFMAs."""
        g, tile = self._build()
        mr, nr, ki = tile.mfma_m_repeat, tile.mfma_n_repeat, tile.k_iterations

        raw_deps = {(d.producer, d.consumer) for d in g.deps
                    if d.kind == DepKind.RAW}

        # Each MFMA depends on its A-read and B-read
        for mi in range(mr):
            for ni in range(nr):
                for ki_ in range(ki):
                    mname = f"mfma_m{mi}_n{ni}_k{ki_}"
                    assert (f"read_data_a_m{mi}_k{ki_}", mname) in raw_deps
                    assert (f"read_data_b_n{ni}_k{ki_}", mname) in raw_deps

    def test_mfma_to_toggle_rd(self):
        """Last MFMA → toggle_rd (RAW/ORDER)."""
        g, tile = self._build()
        mr, nr, ki = tile.mfma_m_repeat, tile.mfma_n_repeat, tile.k_iterations
        last_mfma = f"mfma_m{mr-1}_n{nr-1}_k{ki-1}"
        toggle_preds_a = {d.producer for d in g.deps
                          if d.consumer == "toggle_rd_data_a"}
        toggle_preds_b = {d.producer for d in g.deps
                          if d.consumer == "toggle_rd_data_b"}
        assert last_mfma in toggle_preds_a
        assert last_mfma in toggle_preds_b

    def test_pingpong_war_data_a(self):
        """A-matrix reads at mi >= num_buffers have WAR deps."""
        g, tile = self._build(num_buffers=2)
        mr, nr, ki = tile.mfma_m_repeat, tile.mfma_n_repeat, tile.k_iterations

        war_deps = [(d.producer, d.consumer) for d in g.deps
                    if d.kind == DepKind.WAR]

        # mi=0,1: no WAR
        for mi in range(2):
            for ki_ in range(ki):
                rname = f"read_data_a_m{mi}_k{ki_}"
                wars_on_r = [p for p, c in war_deps if c == rname]
                assert len(wars_on_r) == 0, f"Unexpected WAR on {rname}"

        # mi=2: WAR from last MFMA of mi=0
        for ki_ in range(ki):
            rname = f"read_data_a_m2_k{ki_}"
            wars_on_r = [p for p, c in war_deps if c == rname]
            assert len(wars_on_r) == 1
            assert wars_on_r[0] == f"mfma_m0_n{nr-1}_k{ki-1}"

    def test_no_war_data_b(self):
        """B-matrix reads have NO WAR deps (no ping-pong)."""
        g, tile = self._build()
        war_on_b = [d for d in g.deps
                    if d.kind == DepKind.WAR
                    and d.consumer.startswith("read_data_b")]
        assert len(war_on_b) == 0

    def test_graph_validates(self):
        """Full FP16 graph passes validation."""
        g, _ = self._build()
        g.validate()


class TestMXFP4Config:
    """MXFP4: data_a + data_b + scale_a + scale_b."""

    def _build(self, pgr=1):
        tile = _mxfp4_tile()
        streams = [
            StubDataStream("a"),
            StubDataStream("b"),
            StubScaleStream("a"),
            StubScaleStream("b"),
        ]
        return build_kloop_graph(streams, tile, pgr=pgr), tile

    def test_correct_ops_created(self):
        """All expected op names including scale ops are present."""
        g, tile = self._build()
        mr = tile.mfma_m_repeat
        nr = tile.mfma_n_repeat
        ki = tile.k_iterations

        # Scale producer ops (including DS_WRITE)
        for sn in ("scale_a", "scale_b"):
            assert f"advance_{sn}" in g.ops
            assert f"toggle_wr_{sn}" in g.ops
            assert f"load_{sn}" in g.ops
            assert f"write_{sn}" in g.ops  # 2-step stream
            assert g.ops[f"write_{sn}"].kind == OpKind.DS_WRITE

        # Scale reads: grouped by 2-mi
        num_a_groups = (mr + 1) // 2
        num_b_groups = (nr + 1) // 2
        for grp in range(num_a_groups):
            assert f"read_scale_a_g{grp}" in g.ops
        for grp in range(num_b_groups):
            assert f"read_scale_b_g{grp}" in g.ops

        # No extra scale read ops beyond groups
        all_scale_a_reads = [n for n in g.ops
                             if n.startswith("read_scale_a")]
        all_scale_b_reads = [n for n in g.ops
                             if n.startswith("read_scale_b")]
        assert len(all_scale_a_reads) == num_a_groups
        assert len(all_scale_b_reads) == num_b_groups

    def test_scale_read_grouped(self):
        """Scale reads are grouped: one per 2-mi group, not per mi*ki."""
        g, tile = self._build()
        mr = tile.mfma_m_repeat   # 16
        nr = tile.mfma_n_repeat   # 16

        # If not grouped, we'd have mr*ki or nr*ki reads.
        # With grouping: mr/2 and nr/2 reads.
        scale_a_reads = [n for n in g.ops if n.startswith("read_scale_a")]
        scale_b_reads = [n for n in g.ops if n.startswith("read_scale_b")]
        assert len(scale_a_reads) == mr // 2  # 8
        assert len(scale_b_reads) == nr // 2  # 8

    def test_scale_raw_to_mfma(self):
        """Scale reads have RAW deps to the correct MFMA group."""
        g, tile = self._build()
        mr, nr, ki = tile.mfma_m_repeat, tile.mfma_n_repeat, tile.k_iterations

        raw_deps = {(d.producer, d.consumer) for d in g.deps
                    if d.kind == DepKind.RAW}

        # read_scale_a_g0 covers mi=0,1 -> all ni, all ki
        for mi2 in (0, 1):
            for ki2 in range(ki):
                for ni in range(nr):
                    mname = f"mfma_m{mi2}_n{ni}_k{ki2}"
                    assert ("read_scale_a_g0", mname) in raw_deps

        # read_scale_b_g0 covers ni=0,1 -> all mi, all ki
        for ni2 in (0, 1):
            for ki2 in range(ki):
                for mi in range(mr):
                    mname = f"mfma_m{mi}_n{ni2}_k{ki2}"
                    assert ("read_scale_b_g0", mname) in raw_deps

    def test_scale_write_barrier(self):
        """Scale write ops feed into barrier (SYNC)."""
        g, _ = self._build()
        barrier_preds = {d.producer for d in g.deps
                         if d.consumer == "barrier" and d.kind == DepKind.SYNC}
        assert "write_scale_a" in barrier_preds
        assert "write_scale_b" in barrier_preds

    def test_suffix_includes_scale_toggles(self):
        """Suffix toggle_rd ops exist for all streams including scales."""
        g, _ = self._build()
        assert "toggle_rd_scale_a" in g.ops
        assert "toggle_rd_scale_b" in g.ops
        assert "toggle_rd_data_a" in g.ops
        assert "toggle_rd_data_b" in g.ops

    def test_graph_validates(self):
        """Full MXFP4 graph passes validation."""
        g, _ = self._build()
        g.validate()


class TestPGRValues:
    """Different pgr values produce correct iteration distances."""

    def test_pgr_0(self):
        """pgr=0: producer ops in current iteration."""
        tile = _fp16_tile()
        streams = [StubDataStream("a"), StubDataStream("b")]
        g = build_kloop_graph(streams, tile, pgr=0)

        for sn in ("data_a", "data_b"):
            assert g.ops[f"advance_{sn}"].iteration == 0
            assert g.ops[f"load_{sn}"].iteration == 0

    def test_pgr_1(self):
        """pgr=1: producer ops prefetched one iteration ahead."""
        tile = _fp16_tile()
        streams = [StubDataStream("a"), StubDataStream("b")]
        g = build_kloop_graph(streams, tile, pgr=1)

        for sn in ("data_a", "data_b"):
            assert g.ops[f"advance_{sn}"].iteration == 1
            assert g.ops[f"load_{sn}"].iteration == 1

    def test_pgr_2(self):
        """pgr=2: producer ops prefetched two iterations ahead."""
        tile = _fp16_tile()
        streams = [StubDataStream("a"), StubDataStream("b")]
        g = build_kloop_graph(streams, tile, pgr=2)

        for sn in ("data_a", "data_b"):
            assert g.ops[f"advance_{sn}"].iteration == 2
            assert g.ops[f"load_{sn}"].iteration == 2

        # Consumers still iteration=0
        assert g.ops["barrier"].iteration == 0
        for name, op in g.ops.items():
            if name.startswith("read_") or name.startswith("mfma_"):
                assert op.iteration == 0


class TestEdgeCases:
    """Edge cases and structural invariants."""

    def test_num_buffers_3(self):
        """num_buffers=3: WAR dep only for mi >= 3."""
        tile = _fp16_tile()
        streams = [StubDataStream("a"), StubDataStream("b")]
        g = build_kloop_graph(streams, tile, num_buffers=3)

        war_consumers = [d.consumer for d in g.deps
                         if d.kind == DepKind.WAR]

        # mi=0,1,2: no WAR
        ki = tile.k_iterations
        for mi in range(3):
            for ki_ in range(ki):
                assert f"read_data_a_m{mi}_k{ki_}" not in war_consumers

        # mi=3 has WAR from mi=0
        for ki_ in range(ki):
            assert f"read_data_a_m3_k{ki_}" in war_consumers

    def test_producer_chain_order(self):
        """advance → toggle_wr → load → [write] ordering."""
        tile = _mxfp4_tile()
        streams = [StubScaleStream("a")]
        g = build_kloop_graph(streams, tile)

        dep_pairs = [(d.producer, d.consumer) for d in g.deps]
        assert ("advance_scale_a", "toggle_wr_scale_a") in dep_pairs
        assert ("toggle_wr_scale_a", "load_scale_a") in dep_pairs
        assert ("load_scale_a", "write_scale_a") in dep_pairs

    def test_ds_write_hw_counter(self):
        """DS_WRITE ops get lgkmcnt hw_counter."""
        tile = _mxfp4_tile()
        streams = [StubScaleStream("a")]
        g = build_kloop_graph(streams, tile)
        assert g.ops["write_scale_a"].hw_counter == "lgkmcnt"

    def test_op_counts_fp16(self):
        """Verify total op counts for fp16 config."""
        tile = _fp16_tile()
        mr, nr, ki = tile.mfma_m_repeat, tile.mfma_n_repeat, tile.k_iterations
        streams = [StubDataStream("a"), StubDataStream("b")]
        g = build_kloop_graph(streams, tile)

        # Producers: 3 per data stream * 2 = 6
        # Barrier: 1
        # Data reads: mr*ki + nr*ki
        # MFMAs: mr*nr*ki
        # Suffix: 2
        expected = 6 + 1 + mr*ki + nr*ki + mr*nr*ki + 2
        assert len(g.ops) == expected

    def test_op_counts_mxfp4(self):
        """Verify total op counts for mxfp4 config."""
        tile = _mxfp4_tile()
        mr, nr, ki = tile.mfma_m_repeat, tile.mfma_n_repeat, tile.k_iterations
        streams = [
            StubDataStream("a"), StubDataStream("b"),
            StubScaleStream("a"), StubScaleStream("b"),
        ]
        g = build_kloop_graph(streams, tile)

        # Producers: 3 per data * 2 + 4 per scale * 2 = 14
        # Barrier: 1
        # Data reads: mr*ki + nr*ki
        # Scale reads: mr//2 + nr//2
        # MFMAs: mr*nr*ki
        # Suffix: 4
        expected = (6 + 8 + 1
                    + mr*ki + nr*ki
                    + mr//2 + nr//2
                    + mr*nr*ki
                    + 4)
        assert len(g.ops) == expected
