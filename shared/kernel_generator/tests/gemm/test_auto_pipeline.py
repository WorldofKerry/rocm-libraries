"""Tests for auto-pipelined K-loop (SoftwarePipeline).

Verifies that AutoPipelinedCompute produces instruction-identical
assembly to the manual ScheduledCompute for all supported configs.
"""
import pytest
from kernel_generator.gemm.problem import GemmProblem, DataType
from kernel_generator.gemm.kernel import GemmKernel
from kernel_generator.gemm.schedule.auto_pipeline import (
    AutoPipelinedCompute,
    SoftwarePipeline, PipelineStage, StageDep, ResourceConfig,
)


def _strip_instructions(asm_text: str) -> list[str]:
    """Extract instruction lines, stripping comments and blanks."""
    result = []
    for line in asm_text.splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        if "//" in s:
            s = s[:s.index("//")].strip()
        result.append(s)
    return result


class TestSoftwarePipeline:
    """Unit tests for the SoftwarePipeline derivation logic."""

    def _gemm_stages(self):
        return (
            [
                PipelineStage("G", distance=1, resource="lds",
                              mode="write", wait_counter="vmcnt"),
                PipelineStage("RM", distance=0, resource="lds",
                              mode="read", wait_counter="lgkmcnt"),
            ],
            [StageDep("G", "RM", distance=1)],
        )

    def test_min_pgr(self):
        stages, deps = self._gemm_stages()
        sw = SoftwarePipeline(stages, deps, [ResourceConfig("lds", 2)])
        assert sw.min_pgr == 1

    def test_pgr1_loads_before_reads(self):
        stages, deps = self._gemm_stages()
        sw = SoftwarePipeline(stages, deps, [ResourceConfig("lds", 2)], pgr=1)
        assert sw.loads_before_reads is True

    def test_pgr2_two_buffers_read_before_write(self):
        stages, deps = self._gemm_stages()
        sw = SoftwarePipeline(stages, deps, [ResourceConfig("lds", 2)], pgr=2)
        assert sw.loads_before_reads is False

    def test_pgr2_three_buffers_loads_before_reads(self):
        stages, deps = self._gemm_stages()
        sw = SoftwarePipeline(stages, deps, [ResourceConfig("lds", 3)], pgr=2)
        assert sw.loads_before_reads is True

    def test_pgr_exceeds_buffers_raises(self):
        stages, deps = self._gemm_stages()
        with pytest.raises(ValueError, match="PGR=3"):
            SoftwarePipeline(stages, deps, [ResourceConfig("lds", 2)], pgr=3)

    def test_pgr_below_min_raises(self):
        stages, deps = self._gemm_stages()
        with pytest.raises(ValueError, match="min_pgr"):
            SoftwarePipeline(stages, deps, [ResourceConfig("lds", 2)], pgr=0)

    def test_scale_stage_min_pgr(self):
        stages = [
            PipelineStage("G", distance=1, resource="lds",
                          mode="write", wait_counter="vmcnt"),
            PipelineStage("S", distance=1, wait_counter="vmcnt"),
            PipelineStage("RM", distance=0, resource="lds",
                          mode="read", wait_counter="lgkmcnt"),
        ]
        deps = [
            StageDep("G", "RM", distance=1),
            StageDep("S", "RM", distance=1),
        ]
        sw = SoftwarePipeline(stages, deps, [ResourceConfig("lds", 2)])
        assert sw.min_pgr == 1  # both G and S have distance=1 to RM

    def test_describe_output(self):
        stages, deps = self._gemm_stages()
        sw = SoftwarePipeline(stages, deps, [ResourceConfig("lds", 2)], pgr=1)
        desc = sw.describe(num_tiles=4)
        assert any("ramp-up" in line for line in desc)
        assert any("steady" in line for line in desc)
        assert any("drain" in line for line in desc)


class TestAutoVsManualAssembly:
    """Verify auto-pipelined assembly matches manual assembly."""

    @pytest.mark.parametrize("dtype", [DataType.F16, DataType.BF16])
    def test_pgr1_identical(self, dtype):
        p = GemmProblem(4096, 4096, 4096, dtype=dtype)
        auto = GemmKernel.build(p, pipeline_strategy=AutoPipelinedCompute, pgr=1)
        manual = GemmKernel.build(p, pgr=1)

        auto_inst = _strip_instructions(auto.emit().asm_text)
        manual_inst = _strip_instructions(manual.emit().asm_text)
        assert auto_inst == manual_inst, (
            f"{dtype.value} PGR=1: instructions differ")

    @pytest.mark.parametrize("dtype", [DataType.F16, DataType.BF16])
    def test_pgr2_identical(self, dtype):
        p = GemmProblem(4096, 4096, 4096, dtype=dtype)
        auto = GemmKernel.build(p, pipeline_strategy=AutoPipelinedCompute, pgr=2)
        manual = GemmKernel.build(p, pgr=2)

        auto_inst = _strip_instructions(auto.emit().asm_text)
        manual_inst = _strip_instructions(manual.emit().asm_text)
        assert auto_inst == manual_inst, (
            f"{dtype.value} PGR=2: instructions differ")
