# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM problem description and tile configuration.

A ``GemmProblem`` captures the mathematical GEMM specification
(sizes, data types, transposes) while ``TileConfig`` captures the
tiling and mapping decisions that determine the generated kernel.

The tile hierarchy is::

    Problem (M x N x K)
      Workgroup tile (M_wg x N_wg x K_unroll)
        Wave tile (M_wave x N_wave)
          MFMA tile (M_mfma x N_mfma x K_mfma)  -- hardware instruction
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from .tile.transforms import Dim, Tile, TileDescriptor

__all__ = [
    "DataType", "GemmProblem", "MfmaConfig", "SubTileConfig",
    "PartitionConfig", "TileConfig",
]


class DataType(Enum):
    """Supported matrix element types."""
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    MXFP4 = "mxfp4"


# ---------------------------------------------------------------------------
# MFMA instruction descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MfmaConfig:
    """Describes one MFMA (Matrix Fused Multiply-Add) instruction variant.

    Fields mirror the stinkytofu ``MFMA()`` constructor parameters.
    """
    m: int            # output rows per instruction
    n: int            # output cols per instruction
    k: int            # reduction depth per instruction
    blocks: int       # number of output blocks
    input_type: str   # e.g. "f16", "bf16"
    acc_type: str     # e.g. "f32"
    # Number of VGPRs consumed by one MFMA's A / B / accumulator operands
    a_vgprs: int = 0
    b_vgprs: int = 0
    acc_vgprs: int = 0
    element_bits: int = 16   # bits per element (16 for fp16, 4 for mxfp4)
    cbsz: int = 0            # format selector for A operand (4 = FP4)
    blgp: int = 0            # format selector for B operand (4 = FP4)
    mx_block: int = 0         # MX block size (32 for MXFP4, 0 = no MX)
    is_mx: bool = False       # uses MX scale operands

    @property
    def flops_per_instruction(self) -> int:
        """FLOPs per single MFMA invocation (multiply + accumulate)."""
        return 2 * self.m * self.n * self.k

    @property
    def element_bytes(self) -> float:
        """Bytes per element. 0.5 for 4-bit types."""
        return self.element_bits / 8

    @property
    def instruction_name(self) -> str:
        """Full MFMA instruction name.

        For MX variants: ``v_mfma_scale_f32_16x16x128_f8f6f4``.
        """
        if self.is_mx:
            return f"v_mfma_scale_{self.acc_type}_{self.m}x{self.n}x{self.k}_{self.input_type}"
        return f"v_mfma_{self.acc_type}_{self.m}x{self.n}x{self.k}_{self.input_type}"

    @staticmethod
    def f16_16x16x16() -> MfmaConfig:
        """``v_mfma_f32_16x16x16_f16``: 16x16 output, K=16, 4 acc VGPRs."""
        return MfmaConfig(
            m=16, n=16, k=16, blocks=1,
            input_type="f16", acc_type="f32",
            a_vgprs=2, b_vgprs=2, acc_vgprs=4,
        )

    @staticmethod
    def f16_32x32x8() -> MfmaConfig:
        """``v_mfma_f32_32x32x8_f16``: 32x32 output, K=8, 16 acc VGPRs."""
        return MfmaConfig(
            m=32, n=32, k=8, blocks=1,
            input_type="f16", acc_type="f32",
            a_vgprs=2, b_vgprs=2, acc_vgprs=16,
        )

    @staticmethod
    def f16_16x16x32() -> MfmaConfig:
        """``v_mfma_f32_16x16x32_f16``: 16x16 output, K=32 (gfx950).

        2x the K-depth of f16_16x16x16 at the same cycle count (16 cycles),
        doubling FLOPs per instruction.  A/B operands are 4 VGPRs each
        (8 fp16 elements per thread), read via ds_read_b128.
        """
        return MfmaConfig(
            m=16, n=16, k=32, blocks=1,
            input_type="f16", acc_type="f32",
            a_vgprs=4, b_vgprs=4, acc_vgprs=4,
        )

    @staticmethod
    def mxfp4_16x16x128() -> MfmaConfig:
        """``v_mfma_scale_f32_16x16x128_f8f6f4``: MXFP4 on gfx950.

        MI_K=128, 4 VGPRs per A/B operand (128 * 0.5B / 64 lanes = 1B/lane,
        packed into 4 VGPRs due to instruction format). 4 acc VGPRs.
        """
        return MfmaConfig(
            m=16, n=16, k=128, blocks=1,
            input_type="f8f6f4", acc_type="f32",
            a_vgprs=4, b_vgprs=4, acc_vgprs=4,
            element_bits=4,
            cbsz=4, blgp=4,
            is_mx=True,
            mx_block=32,
        )


# ---------------------------------------------------------------------------
# Sub-tile configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubTileConfig:
    """Defines a sub-tile: the fundamental scheduling unit within a wave tile.

    A sub-tile groups a small number of MFMA instructions along M and K.
    The key property: VGPR operand registers are allocated per-subtile
    and can be **recycled** across partitions, reducing peak VGPR pressure.

    Dimensions (data-type-independent in bytes):
        ``subtile_m``      -- elements per sub-tile in M/N (typically = mfma.m)
        ``subtile_k_bytes`` -- bytes per sub-tile in K (typically 128 = 1 cache line)

    The shape ``[subtile_m_mfmas, subtile_k_mfmas]`` describes how many
    MFMA tiles fit in one sub-tile::

        subtile_m_mfmas = subtile_m / mfma.m   (usually 1)
        subtile_k_mfmas = subtile_k_elems / mfma.k  (usually 1-2)

    See ``shared/rocroller/docs/subtile_dimensions.md`` for diagrams.
    """
    subtile_m: int = 16        # elements per sub-tile in M/N
    subtile_k_bytes: int = 128 # bytes per sub-tile in K

    def subtile_k_elems(self, element_bytes: int) -> int:
        """K-dimension elements per sub-tile for a given data type."""
        return self.subtile_k_bytes // element_bytes

    def num_subtiles_m(self, wave_m: int) -> int:
        """Number of sub-tiles along M within one wave tile."""
        return wave_m // self.subtile_m

    def num_subtiles_k(self, unroll_k: int, element_bytes: int) -> int:
        """Number of sub-tiles along K within one unroll tile."""
        return unroll_k // self.subtile_k_elems(element_bytes)

    def subtile_k_mfmas(self, mfma_k: int, element_bytes: int) -> int:
        """MFMA K-iterations within one sub-tile's K span."""
        return self.subtile_k_elems(element_bytes) // mfma_k


@dataclass(frozen=True)
class PartitionConfig:
    """Groups of sub-tiles that are scheduled together.

    Within one partition, all LDS reads happen before all MFMAs.
    Across partitions, execution is strictly sequential::

        LDS_read_p0 -> MFMA_p0 -> LDS_read_p1 -> MFMA_p1 -> ...

    This enables VGPR reuse: operand registers from partition 0 are
    freed before partition 1 allocates its operands.

    ``partition_m x partition_n`` is the number of *sub-tiles* per
    partition along M and N.
    """
    partition_m: int = 2  # sub-tiles per partition along M
    partition_n: int = 2  # sub-tiles per partition along N

    @property
    def subtiles_per_partition(self) -> int:
        return self.partition_m * self.partition_n

    def num_partitions(self, num_subtiles_m: int, num_subtiles_n: int) -> int:
        """Total partitions covering the full wave tile."""
        pm = math.ceil(num_subtiles_m / self.partition_m)
        pn = math.ceil(num_subtiles_n / self.partition_n)
        return pm * pn


# ---------------------------------------------------------------------------
# Tile configuration
# ---------------------------------------------------------------------------

@dataclass
class TileConfig:
    """All tiling / mapping decisions for a GEMM kernel.

    The four-level hierarchy is fully determined by these parameters:

    1. **Workgroup tile** -- ``(wg_m, wg_n, unroll_k)``
    2. **Wave tile** -- derived from ``(waves_m, waves_n)`` split of the
       workgroup tile, combined with the MFMA tile.
    3. **MFMA tile** -- ``mfma.m x mfma.n x mfma.k`` (hardware).
    4. **Subtile / repeat** -- how many MFMAs each wave issues per
       K-iteration (``m_per_wave / mfma.m`` x ``n_per_wave / mfma.n``).
    """
    wg_m: int = 128          # workgroup tile M
    wg_n: int = 128          # workgroup tile N
    unroll_k: int = 32       # k-loop unroll depth
    waves_m: int = 2         # waves along M in the workgroup
    waves_n: int = 2         # waves along N in the workgroup
    wave_size: int = 64      # threads per wave
    mfma: MfmaConfig = field(default_factory=MfmaConfig.f16_16x16x16)
    lds_pad: int = 0         # LDS padding per row (bytes)
    lds_swizzle: bool = False # XOR-based LDS bank conflict avoidance (convenience)
    swizzle: object = None   # Swizzle instance (overrides lds_swizzle if set)
    prefetch_stages: int = 1 # number of software-pipeline stages
    vector_width: int = 8    # elements per global load (fp16: 8 = 16 B)
    subtile: Optional[SubTileConfig] = None     # None = subtiling disabled
    partition: Optional[PartitionConfig] = None  # None = no partitioning

    # -- swizzle resolution ------------------------------------------------

    def resolved_swizzle(self, elem_bytes: float = None):
        """Return the active Swizzle instance, or None.
        
        If self.swizzle is set, use it directly.
        If self.lds_swizzle is True, auto-create a RotationSwizzle (2-way optimal).
        """
        if self.swizzle is not None:
            return self.swizzle
        if self.lds_swizzle:
            from .memory.swizzle import RotationSwizzle
            return RotationSwizzle(use_cross_lane=True)
        return None

    # -- subtile-derived quantities -----------------------------------------

    @property
    def subtiling_enabled(self) -> bool:
        return self.subtile is not None

    @property
    def num_subtiles_m(self) -> int:
        """Sub-tiles along M within one wave tile."""
        if not self.subtile:
            return self.mfma_m_repeat
        return self.subtile.num_subtiles_m(self.m_per_wave)

    @property
    def num_subtiles_n(self) -> int:
        if not self.subtile:
            return self.mfma_n_repeat
        return self.subtile.num_subtiles_m(self.n_per_wave)  # same formula

    @property
    def num_partitions(self) -> int:
        """Total partitions per wave (1 if partitioning disabled)."""
        if not self.partition:
            return 1
        return self.partition.num_partitions(
            self.num_subtiles_m, self.num_subtiles_n,
        )

    @property
    def vgpr_a_per_subtile(self) -> int:
        """VGPR operand registers for A per sub-tile."""
        if not self.subtile:
            return self.mfma.a_vgprs
        k_mfmas = self.subtile.subtile_k_mfmas(self.mfma.k, 2)  # 2 = f16 bytes
        return self.mfma.a_vgprs * k_mfmas

    @property
    def vgpr_b_per_subtile(self) -> int:
        if not self.subtile:
            return self.mfma.b_vgprs
        k_mfmas = self.subtile.subtile_k_mfmas(self.mfma.k, 2)
        return self.mfma.b_vgprs * k_mfmas

    @property
    def live_vgprs_per_partition(self) -> int:
        """Peak VGPR operand pressure within one partition.

        This is what subtiling reduces: only one partition's operands
        are live at a time.
        """
        if not self.partition:
            # Without partitioning, all operands are live simultaneously
            return (self.mfma_m_repeat * self.mfma.a_vgprs
                    + self.mfma_n_repeat * self.mfma.b_vgprs)
        st = self.partition
        return (st.partition_m * self.vgpr_a_per_subtile
                + st.partition_n * self.vgpr_b_per_subtile)

    # -- derived quantities -------------------------------------------------

    @property
    def block_size(self) -> int:
        """Threads per workgroup."""
        return self.waves_m * self.waves_n * self.wave_size

    @property
    def m_per_wave(self) -> int:
        return self.wg_m // self.waves_m

    @property
    def n_per_wave(self) -> int:
        return self.wg_n // self.waves_n

    @property
    def mfma_m_repeat(self) -> int:
        """Number of MFMA tiles a single wave issues along M."""
        return self.m_per_wave // self.mfma.m

    @property
    def mfma_n_repeat(self) -> int:
        """Number of MFMA tiles a single wave issues along N."""
        return self.n_per_wave // self.mfma.n

    @property
    def k_iterations(self) -> int:
        """MFMA K-iterations within one unroll_k tile."""
        return self.unroll_k // self.mfma.k

    @property
    def total_mfma_per_wave(self) -> int:
        """Total MFMAs per wave per unroll iteration."""
        return self.mfma_m_repeat * self.mfma_n_repeat * self.k_iterations

    @property
    def flops_per_wave_per_unroll(self) -> int:
        """FLOPs one wave performs in one unroll-K iteration."""
        return self.total_mfma_per_wave * self.mfma.flops_per_instruction

    def validate(self) -> None:
        """Raise ``ValueError`` if the configuration is inconsistent."""
        if self.wg_m % self.waves_m != 0:
            raise ValueError("wg_m must be divisible by waves_m")
        if self.wg_n % self.waves_n != 0:
            raise ValueError("wg_n must be divisible by waves_n")
        if self.m_per_wave % self.mfma.m != 0:
            raise ValueError("m_per_wave must be divisible by mfma.m")
        if self.n_per_wave % self.mfma.n != 0:
            raise ValueError("n_per_wave must be divisible by mfma.n")
        if self.unroll_k % self.mfma.k != 0:
            raise ValueError("unroll_k must be divisible by mfma.k")
        if self.subtile:
            if self.m_per_wave % self.subtile.subtile_m != 0:
                raise ValueError("m_per_wave must be divisible by subtile_m")
            if self.n_per_wave % self.subtile.subtile_m != 0:
                raise ValueError("n_per_wave must be divisible by subtile_m")
        if self.partition and not self.subtile:
            raise ValueError("partition requires subtile to be set")
        if self.partition:
            if self.num_subtiles_m % self.partition.partition_m != 0:
                raise ValueError(
                    "num_subtiles_m must be divisible by partition_m"
                )
            if self.num_subtiles_n % self.partition.partition_n != 0:
                raise ValueError(
                    "num_subtiles_n must be divisible by partition_n"
                )

    # -- tile descriptors ---------------------------------------------------

    def build_m_descriptor(self) -> TileDescriptor:
        """Build the M-dimension tiling descriptor.

        ``M -> [M_wg_id, M_wave_id, M_mfma_id, M_mfma]``
        """
        d = TileDescriptor("M_tile", [Dim("M", self.wg_m)])
        # Level 1: split into wave tiles
        d.add_transform(Tile(
            Dim("M", self.wg_m), self.m_per_wave,
            outer_name="M_wave_id", inner_name="M_wave",
        ))
        # Level 2: split wave tile into MFMA repeats
        d.add_transform(Tile(
            Dim("M_wave", self.m_per_wave), self.mfma.m,
            outer_name="M_mfma_id", inner_name="M_mfma",
        ))
        return d

    def build_n_descriptor(self) -> TileDescriptor:
        """Build the N-dimension tiling descriptor."""
        d = TileDescriptor("N_tile", [Dim("N", self.wg_n)])
        d.add_transform(Tile(
            Dim("N", self.wg_n), self.n_per_wave,
            outer_name="N_wave_id", inner_name="N_wave",
        ))
        d.add_transform(Tile(
            Dim("N_wave", self.n_per_wave), self.mfma.n,
            outer_name="N_mfma_id", inner_name="N_mfma",
        ))
        return d

    def build_k_descriptor(self) -> TileDescriptor:
        """Build the K-dimension tiling descriptor."""
        d = TileDescriptor("K_tile", [Dim("K", self.unroll_k)])
        d.add_transform(Tile(
            Dim("K", self.unroll_k), self.mfma.k,
            outer_name="K_iter", inner_name="K_mfma",
        ))
        return d

    def summary(self) -> str:
        """Human-readable summary of the tile configuration."""
        lines = [
            f"Workgroup tile : {self.wg_m} x {self.wg_n} x {self.unroll_k}",
            f"Waves          : {self.waves_m} x {self.waves_n}  "
            f"({self.block_size} threads)",
            f"Wave tile      : {self.m_per_wave} x {self.n_per_wave}",
            f"MFMA           : {self.mfma.m}x{self.mfma.n}x{self.mfma.k} "
            f"({self.mfma.input_type} -> {self.mfma.acc_type})",
            f"MFMA repeats   : {self.mfma_m_repeat} x {self.mfma_n_repeat} "
            f"x {self.k_iterations}  "
            f"({self.total_mfma_per_wave} per wave per unroll)",
        ]
        if self.subtile:
            lines.append(
                f"Subtile        : {self.subtile.subtile_m} elems M, "
                f"{self.subtile.subtile_k_bytes} bytes K  "
                f"({self.num_subtiles_m}x{self.num_subtiles_n} per wave)"
            )
        if self.partition:
            lines.append(
                f"Partitions     : {self.partition.partition_m}x"
                f"{self.partition.partition_n} subtiles/part  "
                f"({self.num_partitions} total, "
                f"{self.live_vgprs_per_partition} operand VGPRs live)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GEMM problem
# ---------------------------------------------------------------------------

@dataclass
class GemmProblem:
    """Full specification of a GEMM problem.

    ``D[M, N] = alpha * A[M, K] @ B[K, N] + beta * C[M, N]``

    Attributes:
        m, n, k: matrix dimensions
        dtype: element data type (currently f16 supported)
        acc_type: accumulator type (f32)
        trans_a: whether A is transposed (column-major)
        trans_b: whether B is transposed (row-major)
        alpha, beta: scaling factors
    """
    m: int
    n: int
    k: int
    dtype: DataType = DataType.F16
    acc_type: DataType = DataType.F32
    trans_a: bool = False
    trans_b: bool = True
    alpha: float = 1.0
    beta: float = 0.0

    # -- derived ---

    @property
    def a_stride_row(self) -> int:
        """A's row stride (elements).  A is M x K."""
        return 1 if self.trans_a else self.k

    @property
    def a_stride_col(self) -> int:
        return self.m if self.trans_a else 1

    @property
    def b_stride_row(self) -> int:
        """B's row stride.  B is K x N."""
        return 1 if self.trans_b else self.n

    @property
    def b_stride_col(self) -> int:
        return self.k if self.trans_b else 1

    @property
    def element_bytes(self) -> float:
        return {DataType.F16: 2, DataType.BF16: 2, DataType.F32: 4,
                DataType.MXFP4: 0.5}[self.dtype]

    def grid_dims(self, tile: TileConfig) -> Tuple[int, int]:
        """Number of workgroups in (M, N) dimensions."""
        gm = math.ceil(self.m / tile.wg_m)
        gn = math.ceil(self.n / tile.wg_n)
        return (gm, gn)

    def validate(self, tile: TileConfig) -> None:
        """Check that the tile config is compatible with this problem."""
        tile.validate()
        if self.dtype == DataType.F16 and tile.mfma.input_type != "f16":
            raise ValueError(
                f"Data type {self.dtype} requires f16 MFMA, "
                f"got {tile.mfma.input_type}"
            )
        if self.dtype == DataType.MXFP4 and tile.mfma.input_type != "f8f6f4":
            raise ValueError(
                f"Data type {self.dtype} requires f8f6f4 MFMA, "
                f"got {tile.mfma.input_type}"
            )

    # -- performance modelling ----------------------------------------------

    @property
    def total_flops(self) -> int:
        """Total FLOPs for this GEMM: ``2 * M * N * K``."""
        return 2 * self.m * self.n * self.k

    @property
    def bytes_read(self) -> int:
        """Total bytes read (A + B) assuming no reuse."""
        return (self.m * self.k + self.k * self.n) * self.element_bytes

    @property
    def bytes_written(self) -> int:
        """Total bytes written (D)."""
        return self.m * self.n * self.element_bytes

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs / byte  (operational intensity)."""
        total_bytes = self.bytes_read + self.bytes_written
        return self.total_flops / total_bytes if total_bytes > 0 else 0.0
