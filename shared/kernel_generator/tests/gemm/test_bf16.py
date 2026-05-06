"""Tests for BF16 GEMM kernel generation."""
import pytest

from kernel_generator.gemm.problem import GemmProblem, DataType, MfmaConfig
from kernel_generator.gemm.tiling import GemmTiling
from kernel_generator.gemm.kernel import GemmKernel
from kernel_generator.gemm.mainloop import mainloop_bf16


class TestBF16Emit:
    """BF16 kernel emission and assembly tests."""

    def test_bf16_emits(self) -> None:
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.BF16)
        k = GemmKernel.build(p)
        result = k.emit()
        assert result.vgpr_count > 0
        assert result.sgpr_count > 0
        assert "v_mfma_f32_16x16x32_bf16" in result.asm_text

    def test_bf16_assembles(self) -> None:
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.BF16)
        k = GemmKernel.build(p)
        co = k.emit().assemble()
        assert co.endswith(".co")

    def test_bf16_mfma_config(self) -> None:
        mfma = MfmaConfig.bf16_16x16x32()
        assert mfma.m == 16
        assert mfma.n == 16
        assert mfma.k == 32
        assert mfma.input_type == "bf16"
        assert mfma.acc_type == "f32"
        assert not mfma.is_mx

    def test_bf16_mainloop(self) -> None:
        ml = mainloop_bf16()
        assert ml.layout.name == "bf16"
        assert ml.pgr == 1
        assert not ml.is_streamk

    def test_bf16_pgr2(self) -> None:
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.BF16)
        k = GemmKernel.build(p, pgr=2)
        result = k.emit()
        co = result.assemble()
        assert co.endswith(".co")

    def test_bf16_streamk(self) -> None:
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.BF16)
        k = GemmKernel.build(p, streamk=True)
        result = k.emit()
        assert "StreamK" in result.asm_text or "sk_" in result.asm_text
        co = result.assemble()
        assert co.endswith(".co")

    def test_bf16_mainloop_api(self) -> None:
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.BF16)
        ml = mainloop_bf16(pgr=2, streamk=True)
        k = GemmKernel.build(p, mainloop=ml)
        result = k.emit()
        co = result.assemble()
        assert co.endswith(".co")

    def test_bf16_register_counts(self) -> None:
        """BF16 should use same register counts as FP16 (same tile shape)."""
        p_bf16 = GemmProblem(4096, 4096, 4096, dtype=DataType.BF16)
        p_fp16 = GemmProblem(4096, 4096, 4096, dtype=DataType.F16)
        r_bf16 = GemmKernel.build(p_bf16).emit()
        r_fp16 = GemmKernel.build(p_fp16).emit()
        assert r_bf16.vgpr_count == r_fp16.vgpr_count
        assert r_bf16.sgpr_count == r_fp16.sgpr_count

    def test_bf16_tile_sizes(self) -> None:
        """BF16 works with various problem sizes."""
        for m, n, k in [(512, 512, 512), (1024, 2048, 4096), (4096, 4096, 4096)]:
            p = GemmProblem(m, n, k, dtype=DataType.BF16)
            result = GemmKernel.build(p).emit()
            co = result.assemble()
            assert co.endswith(".co"), f"Failed for {m}x{n}x{k}"
