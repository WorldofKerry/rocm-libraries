"""Tests for PipelineEmitter."""
import pytest

from kernel_generator.gemm.schedule.pipeline_emitter import PipelineEmitter
from kernel_generator.gemm.schedule.pipeline_scheduler import (
    PipelineScheduler, ScheduledPipeline,
)
from kernel_generator.gemm.schedule.graph_builder import build_kloop_graph
from kernel_generator.gemm.memory.lds_stream import LDSBufferManager
from kernel_generator.gemm.memory.streams import DTLDataStream, ScaleStream
from kernel_generator.gemm.problem import GemmProblem, DataType, MfmaConfig
from kernel_generator.gemm.tiling import GemmTiling
from kernel_generator.gemm.emit.context import AsmContext


# ── helpers ───────────────────────────────────────────────────────

def _make_ctx() -> AsmContext:
    """Create an AsmContext with the minimal registers the emitter needs."""
    ctx = AsmContext()
    ctx.alloc_sgpr_permanent(1, "s_k_tiles")
    ctx.alloc_sgpr_permanent(1, "s_lds_db_step")
    return ctx


def _fp16_tile():
    return GemmTiling.standard(
        wg_m=256, wg_n=256, unroll_k=64,
    ).to_tile_config()


def _mxfp4_tile():
    return GemmTiling.mxfp4_standard().to_tile_config()


def _fp16_streams(tile):
    p = GemmProblem(256, 256, 64)
    return [DTLDataStream("a", tile, p), DTLDataStream("b", tile, p)]


def _mxfp4_streams(tile):
    p = GemmProblem(128, 128, 256, dtype=DataType.MXFP4)
    return [
        DTLDataStream("a", tile, p),
        DTLDataStream("b", tile, p),
        ScaleStream("a", tile),
        ScaleStream("b", tile),
    ]


def _build_and_emit(streams, tile, pgr, num_buffers=2):
    """Build graph → schedule → emit, return assembly text."""
    p = GemmProblem(tile.wg_m, tile.wg_n, tile.unroll_k)
    graph = build_kloop_graph(streams, tile, pgr=pgr,
                              num_buffers=num_buffers)
    pipeline = PipelineScheduler(graph).schedule()
    mgr = LDSBufferManager(streams, num_buffers=num_buffers)
    mgr.compute_layout()
    ctx = _make_ctx()
    emitter = PipelineEmitter(pipeline, mgr, ctx)
    emitter.emit()
    return ctx.asm_text()


# ── PGR=0 ────────────────────────────────────────────────────────

class TestPGR0:
    """PGR=0: all ops in body, no ramp-up, no drain."""

    def test_no_crash_fp16(self):
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=0)
        assert "k_loop:" in asm

    def test_has_barrier(self):
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=0)
        assert "s_barrier" in asm

    def test_has_branch(self):
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=0)
        assert "s_cbranch_scc1 k_loop" in asm

    def test_no_ramp_up_comment(self):
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=0)
        assert "ramp-up" not in asm


# ── PGR=1 (produce-first) ────────────────────────────────────────

class TestPGR1:
    """PGR=1: produce-first, one ramp-up stage."""

    def test_no_crash_fp16(self):
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=1)
        assert "k_loop:" in asm

    def test_ramp_up_present(self):
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=1)
        assert "ramp-up stage 0/1" in asm

    def test_produce_first_order(self):
        """Producers (TODOs for loads) appear before barrier in body."""
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=1)
        lines = asm.split("\n")
        # Find k_loop label, then load_skip_all, then barrier
        k_loop_pos = next(i for i, l in enumerate(lines) if "k_loop:" in l)
        skip_pos = next(i for i, l in enumerate(lines)
                        if "load_skip_all:" in l and i > k_loop_pos)
        barrier_pos = next(i for i, l in enumerate(lines)
                           if "s_barrier" in l and i > k_loop_pos)
        # Skip label comes before or at barrier
        assert skip_pos <= barrier_pos

    def test_negate_db_step(self):
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=1)
        assert "negate db_step" in asm

    def test_has_k_tiles_decrement(self):
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=1)
        assert "k_tiles--" in asm

    def test_no_crash_mxfp4(self):
        tile = _mxfp4_tile()
        asm = _build_and_emit(_mxfp4_streams(tile), tile, pgr=1)
        assert "k_loop:" in asm


# ── PGR=2 (consume-first) ────────────────────────────────────────

class TestPGR2:
    """PGR=2: consume-first, two ramp-up stages."""

    def test_no_crash_fp16(self):
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=2)
        assert "k_loop:" in asm

    def test_two_ramp_up_stages(self):
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=2)
        assert "ramp-up stage 0/2" in asm
        assert "ramp-up stage 1/2" in asm

    def test_consume_first_order(self):
        """Barrier appears before producers in body."""
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=2)
        lines = asm.split("\n")
        k_loop_pos = next(i for i, l in enumerate(lines) if "k_loop:" in l)
        # In consume-first: barrier comes early, skip_all comes late
        barrier_pos = next(i for i, l in enumerate(lines)
                           if "s_barrier" in l and i > k_loop_pos)
        skip_pos = next(i for i, l in enumerate(lines)
                        if "load_skip_all:" in l and i > k_loop_pos)
        assert barrier_pos < skip_pos

    def test_pgr_skip_in_ramp_up(self):
        """Second ramp-up stage has a skip guard."""
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=2)
        assert "pgr_skip_1" in asm

    def test_no_crash_mxfp4(self):
        tile = _mxfp4_tile()
        asm = _build_and_emit(_mxfp4_streams(tile), tile, pgr=2)
        assert "k_loop:" in asm


# ── Structural checks ────────────────────────────────────────────

class TestStructural:
    """Cross-cutting structural properties."""

    def test_todo_comments_for_null_emit(self):
        """Ops with emit=None get [TODO] comments."""
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=1)
        assert "[TODO]" in asm

    def test_waitcnt_present(self):
        """At least one auto-wait should be in the body."""
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=1)
        # The ramp-up emits vmcnt(0); body may have lgkmcnt waits
        assert "s_waitcnt" in asm

    def test_mxfp4_has_scale_todos(self):
        """MXFP4 config should have scale-related TODO ops."""
        tile = _mxfp4_tile()
        asm = _build_and_emit(_mxfp4_streams(tile), tile, pgr=1)
        assert "scale" in asm.lower()

    def test_triple_buffer_no_crash(self):
        """num_buffers=3 doesn't crash."""
        tile = _fp16_tile()
        asm = _build_and_emit(
            _fp16_streams(tile), tile, pgr=1, num_buffers=3)
        assert "k_loop:" in asm

    def test_pgr3_no_crash(self):
        """PGR=3 emits three ramp-up stages."""
        tile = _fp16_tile()
        asm = _build_and_emit(_fp16_streams(tile), tile, pgr=3)
        assert "ramp-up stage 0/3" in asm
        assert "ramp-up stage 2/3" in asm
        assert "pgr_skip_2" in asm
