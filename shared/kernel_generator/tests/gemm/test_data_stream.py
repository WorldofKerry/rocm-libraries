"""Tests for DataStream abstraction."""
from kernel_generator.gemm.schedule.data_stream import (
    DataStream, StreamSource, StreamBuffering, StreamSchedule,
    make_gemm_streams,
)
from kernel_generator.gemm.problem import MfmaConfig


class TestDataStream:
    """Test DataStream properties."""

    def test_a_operand_varies_with_mi(self):
        s = DataStream(name="A", varies_with={"mi", "ki"}, source=StreamSource.LDS)
        assert s.varies_with_mi
        assert not s.varies_with_ni
        assert s.varies_with_ki
        assert s.reload_at_subtile
        assert not s.load_in_preamble

    def test_b_operand_varies_with_ni(self):
        s = DataStream(name="B", varies_with={"ni", "ki"}, source=StreamSource.LDS)
        assert not s.varies_with_mi
        assert s.varies_with_ni
        assert s.load_in_preamble
        assert not s.reload_at_subtile

    def test_scale_a_same_pattern_as_a(self):
        a = DataStream(name="A", varies_with={"mi", "ki"}, source=StreamSource.LDS)
        sa = DataStream(name="scale_a", varies_with={"mi"}, source=StreamSource.GLOBAL)
        assert a.reload_at_subtile == sa.reload_at_subtile
        assert a.load_in_preamble == sa.load_in_preamble

    def test_scale_b_same_pattern_as_b(self):
        b = DataStream(name="B", varies_with={"ni", "ki"}, source=StreamSource.LDS)
        sb = DataStream(name="scale_b", varies_with={"ni"}, source=StreamSource.GLOBAL)
        assert b.reload_at_subtile == sb.reload_at_subtile
        assert b.load_in_preamble == sb.load_in_preamble

    def test_waitcnt_type(self):
        lds = DataStream(name="A", varies_with={"mi"}, source=StreamSource.LDS)
        glb = DataStream(name="scale_a", varies_with={"mi"}, source=StreamSource.GLOBAL)
        sgpr = DataStream(name="bias", varies_with={"mi"}, source=StreamSource.SGPR)
        assert lds.waitcnt_type == "lgkmcnt"
        assert glb.waitcnt_type == "vmcnt"
        assert sgpr.waitcnt_type == "lgkmcnt"

    def test_bias_a_varies_mi_only(self):
        """Bias A varies with mi only, loaded at subtile boundary."""
        s = DataStream(name="bias_a", varies_with={"mi"}, source=StreamSource.GLOBAL)
        assert s.reload_at_subtile
        assert not s.load_in_preamble
        assert not s.varies_with_ki

    def test_bias_b_varies_ni_only(self):
        """Bias B varies with ni only, loaded once in preamble."""
        s = DataStream(name="bias_b", varies_with={"ni"}, source=StreamSource.GLOBAL)
        assert not s.reload_at_subtile
        assert s.load_in_preamble


class TestMakeGemmStreams:
    """Test factory function for GEMM streams."""

    def test_basic_gemm_no_scales(self):
        mx = MfmaConfig.mxfp4_16x16x128()
        streams = make_gemm_streams(mx, ki_count=1, use_scales=False)
        assert len(streams) == 2
        names = {s.name for s in streams}
        assert names == {"A", "B"}

    def test_mxfp4_with_scales(self):
        mx = MfmaConfig.mxfp4_16x16x128()
        streams = make_gemm_streams(mx, ki_count=1, use_scales=True)
        assert len(streams) == 4
        names = {s.name for s in streams}
        assert names == {"A", "B", "scale_a", "scale_b"}

    def test_a_ping_pong_b_persistent(self):
        mx = MfmaConfig.mxfp4_16x16x128()
        streams = make_gemm_streams(mx, ki_count=1)
        a = next(s for s in streams if s.name == "A")
        b = next(s for s in streams if s.name == "B")
        assert a.buffering == StreamBuffering.PING_PONG
        assert b.buffering == StreamBuffering.PERSISTENT


class TestStreamSchedule:
    """Test schedule derivation from streams."""

    def test_256x256_schedule(self):
        """256x256 tile (mr=8, nr=8, ki=1) with partition_m=2."""
        mx = MfmaConfig.mxfp4_16x16x128()
        streams = make_gemm_streams(mx, ki_count=1, use_scales=True)
        sched = StreamSchedule.from_streams(
            streams, mr=8, nr=8, ki_count=1, partition_m=2)

        # Preamble: B (8 loads) + scale_b (8 loads)
        preamble_names = {s.name for s, _ in sched.preamble_streams}
        assert "B" in preamble_names
        assert "scale_b" in preamble_names
        assert "A" not in preamble_names
        assert "scale_a" not in preamble_names

        # Subtile prefetch: 3 subtiles (0,1,2) prefetch for (1,2,3)
        assert len(sched.subtile_prefetch) == 3
        for st in [0, 1, 2]:
            assert st in sched.subtile_prefetch
            names = {s.name for s, _ in sched.subtile_prefetch[st]}
            assert "A" in names       # A prefetch (LDS reads)
            assert "scale_a" in names  # scale_a prefetch (global loads)

    def test_128x128_schedule(self):
        """128x128 tile (mr=4, nr=4, ki=2) with partition_m=2."""
        mx = MfmaConfig.mxfp4_16x16x128()
        streams = make_gemm_streams(mx, ki_count=2, use_scales=True)
        sched = StreamSchedule.from_streams(
            streams, mr=4, nr=4, ki_count=2, partition_m=2)

        # 2 subtiles, 1 prefetch (subtile 0 -> subtile 1)
        assert len(sched.subtile_prefetch) == 1
        assert 0 in sched.subtile_prefetch

        # A prefetch: partition_m * ki_count = 2 * 2 = 4 loads
        a_loads = next(n for s, n in sched.subtile_prefetch[0] if s.name == "A")
        assert a_loads == 4

        # scale_a prefetch: partition_m = 2 loads (ki packed)
        sa_loads = next(n for s, n in sched.subtile_prefetch[0] if s.name == "scale_a")
        assert sa_loads == 2

    def test_no_prefetch_single_subtile(self):
        """With partition_m == mr, only 1 subtile, no prefetch needed."""
        mx = MfmaConfig.mxfp4_16x16x128()
        streams = make_gemm_streams(mx, ki_count=1, use_scales=True)
        sched = StreamSchedule.from_streams(
            streams, mr=4, nr=4, ki_count=1, partition_m=4)

        # 1 subtile, no prefetch
        assert len(sched.subtile_prefetch) == 0
