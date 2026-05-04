"""Tests for PipelineScheduler."""
import pytest

from kernel_generator.gemm.schedule.graph_builder import build_kloop_graph
from kernel_generator.gemm.schedule.pipeline_scheduler import (
    PipelineScheduler,
    ScheduledPipeline,
)
from kernel_generator.gemm.schedule.kloop_graph import OpKind
from kernel_generator.gemm.memory.streams import DTLDataStream, ScaleStream
from kernel_generator.gemm.problem import GemmProblem, DataType, MfmaConfig
from kernel_generator.gemm.tiling import GemmTiling


# ── helpers ────────────────────────────────────────────────────────

def _fp16_graph(pgr: int = 1):
    """Build a simple FP16 graph for testing."""
    p = GemmProblem(64, 64, 64)
    t = GemmTiling.standard(wg_m=64, wg_n=64, unroll_k=64)
    tc = t.to_tile_config()
    data_a = DTLDataStream("a", tc, p)
    data_b = DTLDataStream("b", tc, p)
    return build_kloop_graph([data_a, data_b], tc, pgr=pgr, problem=p)


def _mxfp4_graph(pgr: int = 1):
    """Build an MXFP4 graph with scales for testing."""
    p = GemmProblem(128, 128, 256, dtype=DataType.MXFP4)
    t = GemmTiling.mxfp4_standard()
    tc = t.to_tile_config()
    data_a = DTLDataStream("a", tc, p)
    data_b = DTLDataStream("b", tc, p)
    scale_a = ScaleStream("a", tc)
    scale_b = ScaleStream("b", tc)
    return build_kloop_graph(
        [data_a, data_b, scale_a, scale_b], tc, pgr=pgr, problem=p)


# ── PGR=1 produce-first ──────────────────────────────────────────

class TestPGR1ProduceFirst:
    """PGR=1 should produce-first: producers → barrier → consumers."""

    def test_not_consume_first(self):
        sp = PipelineScheduler(_fp16_graph(pgr=1)).schedule()
        assert not sp.is_consume_first

    def test_producers_before_barrier(self):
        sp = PipelineScheduler(_fp16_graph(pgr=1)).schedule()
        bp = sp.body_barrier_pos
        assert bp > 0, "barrier should not be first"
        for op in sp.body[:bp]:
            assert op.iteration > 0 or op.kind == OpKind.BARRIER, \
                f"non-producer op {op.name} before barrier"

    def test_consumers_after_barrier(self):
        sp = PipelineScheduler(_fp16_graph(pgr=1)).schedule()
        bp = sp.body_barrier_pos
        for op in sp.body[bp + 1:]:
            assert op.iteration == 0, \
                f"producer op {op.name} after barrier"

    def test_mfma_order(self):
        """MFMAs should be in canonical m,ki,ni order."""
        sp = PipelineScheduler(_fp16_graph(pgr=1)).schedule()
        mfmas = [op for op in sp.body if op.kind == OpKind.MFMA]
        names = [op.name for op in mfmas]
        # First MFMA should be m0_n0_k0
        assert names[0] == "mfma_m0_n0_k0"
        # Last should be m1_n1_k3
        assert names[-1] == "mfma_m1_n1_k3"


# ── PGR=2 consume-first ──────────────────────────────────────────

class TestPGR2ConsumeFirst:
    """PGR=2 should consume-first: barrier → consumers → producers."""

    def test_is_consume_first(self):
        sp = PipelineScheduler(_fp16_graph(pgr=2)).schedule()
        assert sp.is_consume_first

    def test_barrier_first(self):
        sp = PipelineScheduler(_fp16_graph(pgr=2)).schedule()
        assert sp.body_barrier_pos == 0

    def test_producers_after_consumers(self):
        sp = PipelineScheduler(_fp16_graph(pgr=2)).schedule()
        # Find first producer in body (after barrier).
        first_producer = None
        last_consumer = None
        for i, op in enumerate(sp.body):
            if op.kind == OpKind.BARRIER:
                continue
            if op.iteration > 0:
                if first_producer is None:
                    first_producer = i
            else:
                last_consumer = i
        assert first_producer is not None
        assert last_consumer is not None
        assert first_producer > last_consumer, \
            "producers should come after all consumers"

    def test_pgr_metadata(self):
        sp = PipelineScheduler(_fp16_graph(pgr=2)).schedule()
        assert sp.pgr == 2


# ── PGR=0 ────────────────────────────────────────────────────────

class TestPGR0:
    """PGR=0: no prefetch, no ramp-up, no drain."""

    def test_no_ramp_up(self):
        sp = PipelineScheduler(_fp16_graph(pgr=0)).schedule()
        assert len(sp.ramp_up) == 0

    def test_no_drain(self):
        sp = PipelineScheduler(_fp16_graph(pgr=0)).schedule()
        assert len(sp.drain) == 0

    def test_pgr_zero(self):
        sp = PipelineScheduler(_fp16_graph(pgr=0)).schedule()
        assert sp.pgr == 0

    def test_not_consume_first(self):
        sp = PipelineScheduler(_fp16_graph(pgr=0)).schedule()
        assert not sp.is_consume_first

    def test_all_ops_in_body(self):
        g = _fp16_graph(pgr=0)
        sp = PipelineScheduler(g).schedule()
        # Every graph op should appear in body.
        body_names = {op.name for op in sp.body}
        for name in g.ops:
            assert name in body_names, f"missing {name}"


# ── Ramp-up depth ────────────────────────────────────────────────

class TestRampUp:
    """Ramp-up depth should equal PGR."""

    @pytest.mark.parametrize("pgr", [0, 1, 2, 3])
    def test_ramp_up_depth(self, pgr):
        sp = PipelineScheduler(_fp16_graph(pgr=pgr)).schedule()
        assert len(sp.ramp_up) == pgr

    def test_pgr1_stage0_has_barrier(self):
        sp = PipelineScheduler(_fp16_graph(pgr=1)).schedule()
        has_barrier = any(
            op.kind == OpKind.BARRIER for op in sp.ramp_up[0])
        assert has_barrier

    def test_pgr2_stage0_has_barrier(self):
        sp = PipelineScheduler(_fp16_graph(pgr=2)).schedule()
        has_barrier = any(
            op.kind == OpKind.BARRIER for op in sp.ramp_up[0])
        assert has_barrier

    def test_pgr2_stage1_no_barrier(self):
        """Last ramp-up stage for PGR>=2 has no barrier."""
        sp = PipelineScheduler(_fp16_graph(pgr=2)).schedule()
        has_barrier = any(
            op.kind == OpKind.BARRIER for op in sp.ramp_up[1])
        assert not has_barrier


# ── Drain ─────────────────────────────────────────────────────────

class TestDrain:
    """Drain iterations for PGR>=2."""

    def test_pgr0_no_drain(self):
        sp = PipelineScheduler(_fp16_graph(pgr=0)).schedule()
        assert len(sp.drain) == 0

    def test_pgr1_no_drain(self):
        sp = PipelineScheduler(_fp16_graph(pgr=1)).schedule()
        assert len(sp.drain) == 0

    def test_pgr2_one_drain(self):
        sp = PipelineScheduler(_fp16_graph(pgr=2)).schedule()
        assert len(sp.drain) == 1

    def test_pgr3_two_drains(self):
        sp = PipelineScheduler(_fp16_graph(pgr=3)).schedule()
        assert len(sp.drain) == 2

    def test_drain_has_barrier_and_consumers(self):
        sp = PipelineScheduler(_fp16_graph(pgr=2)).schedule()
        drain_stage = sp.drain[0]
        has_barrier = any(op.kind == OpKind.BARRIER for op in drain_stage)
        has_mfma = any(op.kind == OpKind.MFMA for op in drain_stage)
        has_read = any(op.kind == OpKind.DS_READ for op in drain_stage)
        assert has_barrier
        assert has_mfma
        assert has_read

    def test_drain_no_producers(self):
        sp = PipelineScheduler(_fp16_graph(pgr=2)).schedule()
        for stage in sp.drain:
            for op in stage:
                assert op.kind not in (
                    OpKind.GLOBAL_LOAD, OpKind.DS_WRITE), \
                    f"producer op {op.name} in drain"


# ── Waitcnts ──────────────────────────────────────────────────────

class TestWaitcnts:
    """Auto-derived waitcnt values."""

    def test_first_mfma_has_lgkmcnt(self):
        sp = PipelineScheduler(_fp16_graph(pgr=1)).schedule()
        mfma_positions = [
            i for i, op in enumerate(sp.body)
            if op.kind == OpKind.MFMA
        ]
        first_mfma = mfma_positions[0]
        assert first_mfma in sp.waitcnts
        assert "lgkmcnt" in sp.waitcnts[first_mfma]

    def test_lgkmcnt_reaches_zero(self):
        """Some waitcnt before or at the last MFMA must reach lgkmcnt(0)."""
        sp = PipelineScheduler(_fp16_graph(pgr=1)).schedule()
        # With elision, lgkmcnt(0) may be on an earlier MFMA that
        # brings the inflight count to 0. All subsequent MFMAs
        # inherit that state and don't need another wait.
        has_lgkm_zero = any(
            "lgkmcnt(0)" in sp.waitcnts.get(i, "")
            for i, op in enumerate(sp.body)
            if op.kind == OpKind.MFMA
        )
        assert has_lgkm_zero, "No MFMA has lgkmcnt(0)"

    def test_lgkmcnt_decreases(self):
        """lgkmcnt values should monotonically decrease across MFMAs."""
        sp = PipelineScheduler(_fp16_graph(pgr=1)).schedule()
        lgkm_vals = []
        for i, op in enumerate(sp.body):
            if op.kind == OpKind.MFMA and i in sp.waitcnts:
                wc = sp.waitcnts[i]
                if "lgkmcnt" in wc:
                    val = int(wc.split("lgkmcnt(")[1].split(")")[0])
                    lgkm_vals.append(val)
        # Should be non-increasing.
        for j in range(1, len(lgkm_vals)):
            assert lgkm_vals[j] <= lgkm_vals[j - 1], \
                f"lgkmcnt increased: {lgkm_vals}"

    def test_no_negative_waits(self):
        sp = PipelineScheduler(_fp16_graph(pgr=1)).schedule()
        for pos, wc in sp.waitcnts.items():
            for part in wc.split():
                val = int(part.split("(")[1].rstrip(")"))
                assert val >= 0


# ── MXFP4 with scales ────────────────────────────────────────────

class TestMXFP4:
    """MXFP4 with scale streams."""

    def test_scale_writes_before_barrier(self):
        """ds_write for scales must be before barrier."""
        sp = PipelineScheduler(_mxfp4_graph(pgr=1)).schedule()
        bp = sp.body_barrier_pos
        writes = [
            i for i, op in enumerate(sp.body)
            if op.kind == OpKind.DS_WRITE
        ]
        for w in writes:
            assert w < bp, f"ds_write at {w} after barrier at {bp}"

    def test_scale_reads_after_barrier(self):
        """Scale ds_reads must be after barrier."""
        sp = PipelineScheduler(_mxfp4_graph(pgr=1)).schedule()
        bp = sp.body_barrier_pos
        scale_reads = [
            i for i, op in enumerate(sp.body)
            if op.kind == OpKind.DS_READ and "scale" in op.name
        ]
        for sr in scale_reads:
            assert sr > bp, f"scale read at {sr} before barrier at {bp}"

    def test_scale_writes_have_vmcnt(self):
        """Each ds_write should have a vmcnt wait (for buffer_load)."""
        sp = PipelineScheduler(_mxfp4_graph(pgr=1)).schedule()
        write_positions = [
            i for i, op in enumerate(sp.body)
            if op.kind == OpKind.DS_WRITE
        ]
        for wp in write_positions:
            assert wp in sp.waitcnts, f"no waitcnt for ds_write at {wp}"
            assert "vmcnt" in sp.waitcnts[wp]

    def test_scale_reads_before_mfmas(self):
        """Scale reads should appear before the MFMAs that use them."""
        sp = PipelineScheduler(_mxfp4_graph(pgr=1)).schedule()
        scale_read_positions = {
            op.name: i for i, op in enumerate(sp.body)
            if op.kind == OpKind.DS_READ and "scale" in op.name
        }
        mfma_positions = {
            op.name: i for i, op in enumerate(sp.body)
            if op.kind == OpKind.MFMA
        }
        for sr_name, sr_pos in scale_read_positions.items():
            # First MFMA should be after any scale read.
            first_mfma_pos = min(mfma_positions.values())
            assert sr_pos < first_mfma_pos

    def test_mxfp4_pgr2(self):
        """MXFP4 PGR=2: consume-first, scales properly ordered."""
        sp = PipelineScheduler(_mxfp4_graph(pgr=2)).schedule()
        assert sp.is_consume_first
        assert sp.body_barrier_pos == 0
        # Scale writes should be in the producer section (after consumers).
        write_positions = [
            i for i, op in enumerate(sp.body)
            if op.kind == OpKind.DS_WRITE
        ]
        mfma_positions = [
            i for i, op in enumerate(sp.body)
            if op.kind == OpKind.MFMA
        ]
        if write_positions and mfma_positions:
            # In consume-first, writes (producers) come after MFMAs.
            assert min(write_positions) > max(mfma_positions)
