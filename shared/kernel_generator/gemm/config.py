# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Typed kernel configuration -- replaces implicit ctx._metadata dict.

``KernelConfig`` is the explicit contract between kernel components.
Users can construct it directly from ``GemmProblem`` + ``GemmTiling``
without going through ``GemmKernel.build()``, enabling standalone
component use.

``DTypeConfig`` consolidates per-data-type constants (MFMA config,
TensileLite metadata codes, element sizing) into a single registry
so adding a new type is one table entry instead of editing 6 files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    from .emit.context import AsmContext
    from .emit.layouts import GemmLayouts
    from .kernarg_layout import KernargLayout
    from .problem import GemmProblem, MfmaConfig, TileConfig
    from .tiling import GemmTiling

__all__ = [
    "KernelConfig",
    "KLoopContract",
    "DTypeConfig",
    "DTYPE_REGISTRY",
    "dtype_config",
    "setup_kloop",
]


# ---------------------------------------------------------------------------
# Data type registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DTypeConfig:
    """Per-data-type constants consolidated in one place.

    Adding a new data type = one new entry in ``DTYPE_REGISTRY``.
    """
    # MFMA instruction config
    mfma_factory: Callable[[], 'MfmaConfig']

    # Max unroll-K the data type supports
    max_unroll_k: int

    # TensileLite metadata codes
    tensile_data_type: str      # "H", "B", "F4", ...
    tensile_dest_type: str      # "H", "B", ...
    tensile_mi: list            # MatrixInstruction YAML list

    # Element sizing
    element_bytes_numer: int    # numerator for input element size
    element_bytes_denom: int = 1

    # Whether this type uses MX block scales
    has_mx_scales: bool = False
    mx_block: int = 0

    # Kernel name fragment for TensileLite export
    kernel_name_fragment: str = ""

    # PGR default
    default_pgr: int = 1

    @property
    def element_bytes(self) -> float:
        return self.element_bytes_numer / self.element_bytes_denom


def _make_registry() -> Dict[str, DTypeConfig]:
    """Build the dtype registry. Deferred to avoid circular imports."""
    from .problem import MfmaConfig

    return {
        "fp16": DTypeConfig(
            mfma_factory=MfmaConfig.f16_16x16x16,
            max_unroll_k=64,
            tensile_data_type="H",
            tensile_dest_type="H",
            tensile_mi=[16, 16, 16, 1, 1, 8, 8, 2, 2],
            element_bytes_numer=2,
            kernel_name_fragment="HHS_BH",
        ),
        "bf16": DTypeConfig(
            mfma_factory=MfmaConfig.bf16_16x16x32,
            max_unroll_k=64,
            tensile_data_type="B",
            tensile_dest_type="B",
            tensile_mi=[16, 16, 32, 1, 1, 8, 8, 2, 2],
            element_bytes_numer=2,
            kernel_name_fragment="BBS_BB",
        ),
        "mxfp4": DTypeConfig(
            mfma_factory=lambda: MfmaConfig.mxfp4_16x16x128(),
            max_unroll_k=256,
            tensile_data_type="F4",
            tensile_dest_type="B",
            tensile_mi=[16, 16, 128, 1, 1, 8, 8, 2, 2],
            element_bytes_numer=1,
            element_bytes_denom=2,
            has_mx_scales=True,
            mx_block=32,
            kernel_name_fragment="F4BS_MXA32_MXB32",
            default_pgr=2,
        ),
    }


# Lazy-init to avoid import-time circular deps
_REGISTRY: Optional[Dict[str, DTypeConfig]] = None


def _get_registry() -> Dict[str, DTypeConfig]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _make_registry()
    return _REGISTRY


# Public access: DTYPE_REGISTRY["bf16"].mfma_factory()
class _RegistryProxy:
    """Dict-like proxy with lazy init."""
    def __getitem__(self, key: str) -> DTypeConfig:
        return _get_registry()[key]
    def __contains__(self, key: str) -> bool:
        return key in _get_registry()
    def keys(self):
        return _get_registry().keys()
    def values(self):
        return _get_registry().values()
    def items(self):
        return _get_registry().items()


DTYPE_REGISTRY = _RegistryProxy()


def dtype_config(dtype: str) -> DTypeConfig:
    """Look up DTypeConfig by name.  Raises KeyError if unknown."""
    return DTYPE_REGISTRY[dtype]


# ---------------------------------------------------------------------------
# Kernel config -- typed replacement for ctx._metadata
# ---------------------------------------------------------------------------

@dataclass
class KernelConfig:
    """Typed configuration passed to all kernel components.

    Replaces the untyped ``ctx._metadata`` dict. Users construct this
    directly to use components as a library without ``GemmKernel``.

    Example::

        from kernel_generator.gemm.config import KernelConfig
        from kernel_generator.gemm.problem import GemmProblem, DataType
        from kernel_generator.gemm.tiling import GemmTiling

        p = GemmProblem(4096, 4096, 4096, dtype=DataType.BF16)
        t = GemmTiling.high_perf(wg_m=256, wg_n=256, unroll_k=64)
        cfg = KernelConfig.from_problem(p, t)
        # Now use cfg.tile, cfg.problem, etc. in component constructors.
    """
    tile: 'TileConfig'
    problem: 'GemmProblem'
    layout: Optional['KernargLayout'] = None
    layouts: Optional['GemmLayouts'] = None
    mainloop: Optional[object] = None
    kernel: Optional[object] = None

    @classmethod
    def from_problem(cls, problem: 'GemmProblem',
                     tiling: Optional['GemmTiling'] = None,
                     mainloop: Optional[object] = None) -> 'KernelConfig':
        """Construct KernelConfig from problem + tiling without GemmKernel.

        This is the library entry point -- no GemmKernel needed.
        The layouts field (transform descriptors) is left None;
        it is only needed for the full GemmKernel path and requires
        tile-tree context to construct.
        """
        from .kernarg_layout import layout_for

        if tiling is None:
            from .tiling import GemmTiling
            tiling = GemmTiling.high_perf()

        tile = tiling.to_tile_config()
        layout = layout_for(problem.dtype)

        return cls(
            tile=tile,
            problem=problem,
            layout=layout,
            mainloop=mainloop,
        )


# ---------------------------------------------------------------------------
# K-loop register contract
# ---------------------------------------------------------------------------

@dataclass
class KLoopContract:
    """Declares the register interface between setup and K-loop.

    Setup (prologue) produces these bindings; the K-loop emitter
    consumes them.  This makes the dependency explicit so users can
    provide their own setup (e.g. data already in LDS) without
    going through the default prologue.

    Categories:
        thread_id:   Thread/wave identity (v_tid, v_lane_id, v_wave_m, ...)
        data_srd:    Buffer resource descriptors for global loads
        lds_addrs:   LDS read/write base addresses and double-buffer state
        dtl_offsets: Direct-to-LDS voffsets and soffsets
        accumulators: Accumulator registers (zeroed before K-loop)
        k_loop:      K-loop control (s_k_tiles, s_k_stride)
        scratch:     Temporary registers (v_tmp*, s_tmp*)
    """

    # -- Thread identity (read-only during K-loop) --
    thread_id: tuple = (
        "v_tid",        # threadIdx.x
        "v_lane_id",    # tid % 64
        "v_wave_id",    # tid / 64
        "v_wave_m",     # wave M partition index
        "v_wave_n",     # wave N partition index
    )

    # -- Data SRDs (advanced per K iteration) --
    data_srd: tuple = (
        "s_srd_a",      # 4 SGPRs: buffer SRD for matrix A
        "s_srd_b",      # 4 SGPRs: buffer SRD for matrix B
    )

    # -- LDS addressing --
    lds_addrs: tuple = (
        "v_lds_rd_a",   # LDS read base for A (toggled each iter)
        "v_lds_rd_b",   # LDS read base for B
        "s_lds_wr_a_sg", # LDS write base for A (toggled each iter)
        "s_lds_wr_b_sg", # LDS write base for B
        "s_lds_db_step", # Double-buffer toggle step size
        "s_rd_db",       # Read-side double-buffer step
    )

    # -- DTL offsets (only for DTL load path) --
    dtl_offsets: tuple = (
        "v_dtl_off_a",  # Per-thread DTL voffset for A
        "v_dtl_off_b",  # Per-thread DTL voffset for B
    )

    # -- K-loop control --
    k_loop: tuple = (
        "s_k_tiles",    # Remaining K iterations (loop counter)
        "s_k_stride",   # Bytes per K-step in data SRD
    )

    # -- Accumulators --
    accumulators: tuple = (
        "acc_C",        # mr * nr * acc_per_mfma accumulator VGPRs
    )

    @property
    def all_names(self) -> tuple:
        """All register names the K-loop emitter requires."""
        return (self.thread_id + self.data_srd + self.lds_addrs +
                self.dtl_offsets + self.k_loop + self.accumulators)

    def validate(self, ctx: 'AsmContext') -> None:
        """Check that all required bindings exist in the context.

        Raises KeyError with a clear message listing missing bindings.
        """
        missing = [n for n in self.all_names if not ctx.has(n)]
        if missing:
            raise KeyError(
                f"K-loop requires these bindings but they are missing "
                f"from ctx: {missing}. Run setup_kloop() or allocate "
                f"them manually before emitting the K-loop.")


def setup_kloop(ctx: 'AsmContext', config: 'KernelConfig') -> KLoopContract:
    """Standalone K-loop setup -- no tile tree or GemmKernel needed.

    Allocates all registers the K-loop emitter needs and returns
    a ``KLoopContract`` documenting them.  Does NOT emit setup
    instructions (SRD computation, kernarg loads, etc.) -- only
    register allocation.

    For a complete standalone kernel, call this then emit your own
    setup code (or use ``alloc_registers_dtl`` + the DTL setup phase).

    Example -- library-mode K-loop::

        cfg = KernelConfig.from_problem(problem, tiling)
        ctx = AsmContext(config=cfg)
        contract = setup_kloop(ctx, cfg)
        # ... emit your own setup instructions ...
        PipelineEmitter(pipeline, buf_mgr, ctx).emit()
        # Results in acc_C registers

    Example -- data already in LDS::

        contract = setup_kloop(ctx, cfg)
        # Skip DTL global loads entirely, just set up LDS read addrs
        # and run the K-loop consumer (ds_read + MFMA only)
    """
    from .emit.emitter import alloc_registers_dtl
    from .kernarg_layout import layout_for

    tile = config.tile
    problem = config.problem
    layout = config.layout or layout_for(problem.dtype)

    # Populate ctx.config and _metadata compat layer
    ctx.config = config
    ctx._metadata = {
        "tile": tile, "problem": problem,
        "layout": layout, "layouts": config.layouts,
        "mainloop": config.mainloop, "kernel": config.kernel,
    }

    # Allocate all registers (thread ID, SRDs, LDS addrs, accumulators, etc.)
    alloc_registers_dtl(ctx, problem, tile, layout)

    # Allocate K-loop-specific registers not covered by alloc_registers_dtl
    if not ctx.has("s_lds_db_step"):
        ctx.alloc_sgpr_permanent(1, "s_lds_db_step")
    if not ctx.has("s_rd_db"):
        ctx.alloc_sgpr_permanent(1, "s_rd_db")
    if not ctx.has("s_buf_wr_db"):
        ctx.alloc_sgpr_permanent(1, "s_buf_wr_db")

    contract = KLoopContract()
    return contract
