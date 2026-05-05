# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for LDSStream / LDSBufferManager and concrete stream classes."""

import pytest

from kernel_generator.gemm.memory.lds_stream import LDSStream, LDSBufferManager
from kernel_generator.gemm.memory.streams import DTLDataStream, ScaleStream
from kernel_generator.gemm.problem import GemmProblem, DataType, MfmaConfig
from kernel_generator.gemm.tiling import GemmTiling


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mxfp4_streams():
    """Build the 4-stream MXFP4 128x128x256 config used by most tests."""
    tiling = GemmTiling.mxfp4_standard(wg_m=128, wg_n=128, unroll_k=256)
    tile = tiling.to_tile_config()
    problem = GemmProblem(m=128, n=128, k=256, dtype=DataType.MXFP4)

    data_a = DTLDataStream("a", tile, problem)
    data_b = DTLDataStream("b", tile, problem)
    scale_a = ScaleStream("a", tile)
    scale_b = ScaleStream("b", tile)
    return [data_a, data_b, scale_a, scale_b], tiling, tile, problem


def _fp16_data_streams():
    """Build data-only streams for fp16 128x128x32."""
    tiling = GemmTiling.standard(wg_m=128, wg_n=128, unroll_k=32)
    tile = tiling.to_tile_config()
    problem = GemmProblem(m=128, n=128, k=256, dtype=DataType.F16)

    data_a = DTLDataStream("a", tile, problem)
    data_b = DTLDataStream("b", tile, problem)
    return [data_a, data_b], tiling, tile, problem


# ---------------------------------------------------------------------------
# 1. LDSBufferManager layout
# ---------------------------------------------------------------------------

class TestLDSBufferManagerLayout:
    def test_sequential_offsets(self):
        """compute_layout() assigns offsets equal to cumulative region sizes."""
        streams, *_ = _mxfp4_streams()
        mgr = LDSBufferManager(streams, num_buffers=2)
        mgr.compute_layout()

        expected_offset = 0
        for s in streams:
            assert mgr.stream_offset(s.name) == expected_offset
            expected_offset += s.region_size

    def test_buffer_size_is_sum(self):
        """buffer_size equals the sum of all region sizes."""
        streams, *_ = _mxfp4_streams()
        mgr = LDSBufferManager(streams, num_buffers=2)
        mgr.compute_layout()

        assert mgr.buffer_size == sum(s.region_size for s in streams)

    def test_total_lds_is_buffer_size_times_num_buffers(self):
        streams, *_ = _mxfp4_streams()
        mgr = LDSBufferManager(streams, num_buffers=2)
        mgr.compute_layout()

        assert mgr.total_lds_bytes == mgr.buffer_size * 2

    def test_db_step_equals_buffer_size(self):
        streams, *_ = _mxfp4_streams()
        mgr = LDSBufferManager(streams, num_buffers=2)
        mgr.compute_layout()

        assert mgr.db_step == mgr.buffer_size


# ---------------------------------------------------------------------------
# 2. DTLDataStream properties
# ---------------------------------------------------------------------------

class TestDTLDataStream:
    @pytest.fixture(params=["a", "b"])
    def stream_and_tile(self, request):
        matrix = request.param
        tiling = GemmTiling.mxfp4_standard(wg_m=128, wg_n=128, unroll_k=256)
        tile = tiling.to_tile_config()
        problem = GemmProblem(m=128, n=128, k=256, dtype=DataType.MXFP4)
        return DTLDataStream(matrix, tile, problem), tile, matrix

    def test_name(self, stream_and_tile):
        stream, _, matrix = stream_and_tile
        assert stream.name == f"data_{matrix}"

    def test_region_size_positive(self, stream_and_tile):
        stream, _, _ = stream_and_tile
        assert stream.region_size > 0

    def test_num_global_loads_positive(self, stream_and_tile):
        stream, _, _ = stream_and_tile
        assert stream.num_global_loads > 0

    def test_needs_lds_write_false(self, stream_and_tile):
        """DTL streams write directly to LDS via hardware."""
        stream, _, _ = stream_and_tile
        assert stream.needs_lds_write is False

    def test_has_reads_true(self, stream_and_tile):
        stream, _, _ = stream_and_tile
        assert stream.has_reads is True


# ---------------------------------------------------------------------------
# 3. ScaleStream properties
# ---------------------------------------------------------------------------

class TestScaleStream:
    def test_name(self):
        tiling = GemmTiling.mxfp4_standard(wg_m=128, wg_n=128, unroll_k=256)
        tile = tiling.to_tile_config()
        s = ScaleStream("a", tile)
        assert s.name == "scale_a"

    def test_region_size(self):
        tiling = GemmTiling.mxfp4_standard(wg_m=128, wg_n=128, unroll_k=256)
        tile = tiling.to_tile_config()
        s = ScaleStream("b", tile)
        assert s.region_size == 4096

    def test_num_global_loads(self):
        tiling = GemmTiling.mxfp4_standard(wg_m=128, wg_n=128, unroll_k=256)
        tile = tiling.to_tile_config()
        s = ScaleStream("a", tile)
        assert s.num_global_loads == 4

    def test_needs_lds_write_true(self):
        tiling = GemmTiling.mxfp4_standard(wg_m=128, wg_n=128, unroll_k=256)
        tile = tiling.to_tile_config()
        s = ScaleStream("a", tile)
        assert s.needs_lds_write is True

    def test_read_op_count(self):
        """read_op_count = ceil(mfma_{m,n}_repeat / 2) for scale streams."""
        tiling = GemmTiling.mxfp4_standard(wg_m=128, wg_n=128, unroll_k=256)
        tile = tiling.to_tile_config()
        for matrix in ("a", "b"):
            s = ScaleStream(matrix, tile)
            mr = tile.mfma_m_repeat if matrix == "a" else tile.mfma_n_repeat
            assert s.read_op_count() == (mr + 1) // 2


# ---------------------------------------------------------------------------

# 5. Layout matches current MXFP4 kernel (128x128x256)
# ---------------------------------------------------------------------------

class TestMXFP4Layout:
    def test_known_offsets(self):
        streams, *_ = _mxfp4_streams()
        mgr = LDSBufferManager(streams, num_buffers=2)
        mgr.compute_layout()

        assert mgr.stream_offset("data_a") == 0
        assert mgr.stream_offset("data_b") == 16384
        assert mgr.stream_offset("scale_a") == 32768
        assert mgr.stream_offset("scale_b") == 36864
        assert mgr.buffer_size == 40960


# ---------------------------------------------------------------------------
# 6. Layout for fp16 (no scales)
# ---------------------------------------------------------------------------

class TestFP16Layout:
    def test_buffer_size_data_only(self):
        streams, *_ = _fp16_data_streams()
        mgr = LDSBufferManager(streams, num_buffers=2)
        mgr.compute_layout()

        assert mgr.buffer_size == streams[0].region_size + streams[1].region_size
        # No scale streams, so buffer_size is just data_a + data_b
        assert mgr.total_lds_bytes == mgr.buffer_size * 2


# ---------------------------------------------------------------------------
# 7. Buffer count variations
# ---------------------------------------------------------------------------

class TestBufferCountVariations:
    def test_single_buffer(self):
        streams, *_ = _mxfp4_streams()
        mgr = LDSBufferManager(streams, num_buffers=1)
        mgr.compute_layout()

        assert mgr.total_lds_bytes == mgr.buffer_size * 1

    def test_triple_buffer(self):
        streams, *_ = _mxfp4_streams()
        mgr = LDSBufferManager(streams, num_buffers=3)
        mgr.compute_layout()

        assert mgr.total_lds_bytes == mgr.buffer_size * 3
