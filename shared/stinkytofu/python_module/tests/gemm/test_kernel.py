# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""End-to-end kernel generation tests.

Tests marked ``@requires_stinkytofu`` need the compiled stinkytofu
extension module.  All other tests work with pure Python (dry-run mode).
"""
import pytest
from stinkytofu.gemm.kernel import generate_gemm_kernel, KernelResult
from stinkytofu.gemm.problem import DataType, GemmProblem, MfmaConfig, TileConfig
from stinkytofu.gemm.codegen import Emitter, GemmSchedule

# Marker for tests requiring the stinkytofu C extension
try:
    import stinkytofu as _st
    HAS_STINKYTOFU = hasattr(_st, "LogicalModule")
except ImportError:
    HAS_STINKYTOFU = False

requires_stinkytofu = pytest.mark.skipif(
    not HAS_STINKYTOFU, reason="stinkytofu C extension not available"
)


# ===========================================================================
# Dry-run tests (pure Python, no stinkytofu binary)
# ===========================================================================

class TestDryRun:
    def test_basic(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        result = generate_gemm_kernel(p, dry_run=True)
        assert isinstance(result, KernelResult)
        assert result.module is None
        assert "gemm_f16" in result.name

    def test_summary(self):
        p = GemmProblem(m=2048, n=1024, k=512)
        result = generate_gemm_kernel(p, dry_run=True)
        s = result.summary()
        assert "2048" in s
        assert "Workgroup tile" in s
        assert "LDS" in s

    def test_default_tile(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        result = generate_gemm_kernel(p, dry_run=True)
        assert result.tile.wg_m == 128
        assert result.tile.mfma.input_type == "f16"

    def test_custom_tile(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        t = TileConfig(
            wg_m=256, wg_n=128, unroll_k=64,
            waves_m=4, waves_n=2,
            mfma=MfmaConfig.f16_32x32x8(),
        )
        result = generate_gemm_kernel(p, t, dry_run=True)
        assert result.tile.wg_m == 256

    def test_flops_sanity(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        result = generate_gemm_kernel(p, dry_run=True)
        info = result.dry_info()
        assert info["name"] == result.name

    def test_arithmetic_intensity(self):
        p = GemmProblem(m=4096, n=4096, k=4096)
        assert p.arithmetic_intensity > 100

    def test_small_problem(self):
        p = GemmProblem(m=128, n=128, k=32)
        result = generate_gemm_kernel(p, dry_run=True)
        grid_m, grid_n = p.grid_dims(result.tile)
        assert grid_m == 1
        assert grid_n == 1


# ===========================================================================
# FLOP correctness: tile decomposition must equal 2*M*N*K
# ===========================================================================

class TestFlopCorrectness:
    @pytest.mark.parametrize("m,n,k", [
        (1024, 1024, 1024),
        (4096, 4096, 4096),
        (2048, 1024, 512),
        (256, 256, 256),
        (128, 128, 32),
    ])
    def test_flops_match(self, m, n, k):
        p = GemmProblem(m=m, n=n, k=k)
        t = TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
        )
        grid_m, grid_n = p.grid_dims(t)
        k_tiles = p.k // t.unroll_k
        waves = t.waves_m * t.waves_n

        computed = (
            grid_m * grid_n * waves
            * t.total_mfma_per_wave * k_tiles
            * t.mfma.flops_per_instruction
        )
        # For tile-aligned problems, must match exactly
        if m % t.wg_m == 0 and n % t.wg_n == 0 and k % t.unroll_k == 0:
            assert computed == p.total_flops

    @pytest.mark.parametrize("mfma_fn,wg_m,wg_n,unroll_k,wm,wn", [
        (MfmaConfig.f16_16x16x16, 128, 128, 32, 2, 2),
        (MfmaConfig.f16_32x32x8,  256, 128, 32, 4, 2),
        (MfmaConfig.f16_16x16x16, 64,  64,  16, 1, 1),
    ])
    def test_flops_various_configs(self, mfma_fn, wg_m, wg_n, unroll_k, wm, wn):
        p = GemmProblem(m=4096, n=4096, k=4096)
        t = TileConfig(
            wg_m=wg_m, wg_n=wg_n, unroll_k=unroll_k,
            waves_m=wm, waves_n=wn,
            mfma=mfma_fn(),
        )
        grid_m, grid_n = p.grid_dims(t)
        k_tiles = p.k // t.unroll_k
        waves = t.waves_m * t.waves_n
        computed = (
            grid_m * grid_n * waves
            * t.total_mfma_per_wave * k_tiles
            * t.mfma.flops_per_instruction
        )
        assert computed == p.total_flops


# ===========================================================================
# Custom emitter / schedule override tests
# ===========================================================================

class TestOverrides:
    def test_custom_emitter_dry(self):
        """Verify a custom Emitter subclass is wired in."""
        call_log = []

        class LoggingEmitter(Emitter):
            def emit_mfma_block(self, module, k_iter):
                call_log.append(("mfma", k_iter))
                super().emit_mfma_block(module, k_iter)

        p = GemmProblem(m=128, n=128, k=32)
        t = TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
        )
        # Dry run still instantiates emitter/schedule for metadata
        cg = generate_gemm_kernel(p, t, emitter_cls=LoggingEmitter,
                                  dry_run=True)
        assert cg.codegen.emitter.__class__ is LoggingEmitter

    def test_custom_schedule_dry(self):
        class NoopSchedule(GemmSchedule):
            def emit_k_loop(self, module):
                pass  # skip K-loop entirely

        p = GemmProblem(m=128, n=128, k=32)
        result = generate_gemm_kernel(p, schedule_cls=NoopSchedule,
                                      dry_run=True)
        assert result.codegen.schedule.__class__ is NoopSchedule


# ===========================================================================
# Full generation (requires stinkytofu)
# ===========================================================================

@requires_stinkytofu
class TestFullGeneration:
    def test_generate_module(self):
        import stinkytofu as st
        p = GemmProblem(m=4096, n=4096, k=4096)
        result = generate_gemm_kernel(p)
        assert result.module is not None
        dump = result.module.dump()
        assert "MFMA" in dump or "mfma" in dump.lower()

    def test_instruction_counts(self):
        import stinkytofu as st
        p = GemmProblem(m=4096, n=4096, k=4096)
        t = TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
        )
        result = generate_gemm_kernel(p, t)
        # Expected: mfma_m_repeat * mfma_n_repeat * k_iterations
        expected = t.mfma_m_repeat * t.mfma_n_repeat * t.k_iterations
        # Count MFMA instructions in the IR dump
        dump = result.module.dump()
        # Each MFMA shows as "MFMA" in the logical IR dump
        n_mfma = dump.count("MFMA m")  # matches "MFMA m0_n0", "MFMA m1_n0", etc.
        assert n_mfma == expected, f"Expected {expected} MFMAs, got {n_mfma}"

    def test_has_barrier(self):
        import stinkytofu as st
        p = GemmProblem(m=128, n=128, k=32)
        result = generate_gemm_kernel(p)
        dump = result.module.dump()
        assert "Barrier" in dump or "barrier" in dump.lower()

    def test_has_lds_ops(self):
        import stinkytofu as st
        p = GemmProblem(m=128, n=128, k=32)
        result = generate_gemm_kernel(p)
        assert st.countLocalRead(result.module) > 0
        assert st.countLocalWrite(result.module) > 0

    def test_has_global_loads(self):
        import stinkytofu as st
        p = GemmProblem(m=128, n=128, k=32)
        result = generate_gemm_kernel(p)
        assert st.countGlobalRead(result.module) > 0

    def test_custom_mfma_emitter(self):
        """Override MFMA emission and verify it takes effect."""
        import stinkytofu as st

        class DoubledMfmaEmitter(Emitter):
            """Emit each MFMA twice (nonsensical, but tests override)."""
            def emit_mfma_block(self, module, k_iter):
                super().emit_mfma_block(module, k_iter)
                super().emit_mfma_block(module, k_iter)

        p = GemmProblem(m=128, n=128, k=32)
        t = TileConfig(
            wg_m=128, wg_n=128, unroll_k=32,
            waves_m=2, waves_n=2,
            mfma=MfmaConfig.f16_16x16x16(),
        )
        result_normal = generate_gemm_kernel(p, t)
        result_double = generate_gemm_kernel(p, t,
                                             emitter_cls=DoubledMfmaEmitter)
        n_normal = st.countMFMA(result_normal.module)
        n_double = st.countMFMA(result_double.module)
        assert n_double == 2 * n_normal
