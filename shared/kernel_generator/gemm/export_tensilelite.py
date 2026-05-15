"""Export kernel generator kernels for TensileLite benchmarking.

Generates a TensileLite-compatible custom kernel .s file that can be
benchmarked through Tensile.sh or tensilelite-client.

Usage example::

    PYTHONPATH=shared python3 -c "
    from kernel_generator.gemm.export_tensilelite import generate_custom_kernel
    open('/tmp/k.s','w').write(generate_custom_kernel(128,128,256,dtype='mxfp4'))
    "
    cp /tmp/k.s <tensilelite>/Tensile/CustomKernels/<name>.s

    # --library-format=msgpack required when client lacks YAML support
    HIP_VISIBLE_DEVICES=6 ./Tensile.sh test.yaml /tmp/out \
      --cxx-compiler /opt/rocm/bin/amdclang++ --library-format=msgpack \
      --prebuilt-client ./tensilelite-client

Note: benchmark YAML must set HighPrecisionAccumulate: true and
Batched: true in the ProblemType section to match the kernel metadata.
"""
from __future__ import annotations

import re
from typing import Optional

from .problem import GemmProblem, DataType, MfmaConfig
from .tiling import GemmTiling
from .kernel import GemmKernel


def _extract_kernel_body(asm: str, old_label: str, new_label: str) -> str:
    """Extract the kernel body from *old_label:* through the first s_endpgm."""
    body_lines: list[str] = []
    started = False
    for line in asm.splitlines():
        if line.startswith(old_label + ":"):
            started = True
            body_lines.append(f"{new_label}:")
            continue
        if started:
            body_lines.append(line)
            if line.strip().startswith("s_endpgm"):
                break
    return "\n".join(body_lines)


def _parse_register_counts(asm: str) -> dict[str, int]:
    """Parse sgpr/vgpr/agpr/lds/kernarg counts from emitted assembly."""

    def _int(pattern: str) -> int:
        m = re.search(pattern, asm)
        assert m, f"pattern not found: {pattern}"
        return int(m.group(1))

    vgpr = _int(r"\.vgpr_count:\s+(\d+)")
    agpr = _int(r"\.agpr_count:\s+(\d+)")
    return {
        "sgpr": _int(r"\.sgpr_count:\s+(\d+)"),
        "vgpr": vgpr,
        "agpr": agpr,
        "lds": _int(r"\.group_segment_fixed_size:\s+(\d+)"),
        "karg": _int(r"\.kernarg_segment_size:\s+(\d+)"),
        "accum_off": vgpr - agpr,
    }


# ---------------------------------------------------------------------------
# MXFP4 custom kernel formatter
# ---------------------------------------------------------------------------

def _format_mxfp4_custom_kernel(
    kernel_name: str,
    body: str,
    sgpr: int,
    vgpr: int,
    agpr: int,
    lds: int,
    accum_off: int,
    unroll_k: int,
    waves_m: int,
    waves_n: int,
    mr: int,
    nr: int,
    pgr: int,
    streamk: bool = False,
) -> str:
    """Format the complete .s file for an MXFP4 custom kernel."""
    sk_val = 3 if streamk else 0
    if streamk:
        return _format_mxfp4_streamk_custom_kernel(
            kernel_name=kernel_name,
            body=body,
            sgpr=sgpr,
            vgpr=vgpr,
            agpr=agpr,
            lds=lds,
            accum_off=accum_off,
            unroll_k=unroll_k,
            waves_m=waves_m,
            waves_n=waves_n,
            mr=mr,
            nr=nr,
            pgr=pgr,
        )
    return f""".amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl {kernel_name}
.p2align 8
.type {kernel_name},@function

{body}

.rodata
.p2align 6
.amdhsa_kernel {kernel_name}
    .amdhsa_group_segment_fixed_size {lds}
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 120
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr {vgpr}
    .amdhsa_next_free_sgpr {sgpr}
    .amdhsa_accum_offset {accum_off}
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
    UseUniversalArgs: false
    SupportUserGSU: false
    SupportCustomWGM: false
    SupportCustomStaggerU: false
  ProblemType:
    OperationType: GEMM
    DataType: F4
    DestDataType: B
    ComputeDataType: S
    HighPrecisionAccumulate: true
    TransposeA: 1
    TransposeB: 0
    UseBeta: true
    Batched: true
    MXBlockA: 32
    MXBlockB: 32
  MatrixInstruction:
  - 16
  - 16
  - 128
  - 1
  MIBlock:
  - 16
  - 16
  - 128
  - 1
  - 1
  - 1
  MIInputPerThread: 32
  MIInputPerThreadA: 32
  MIInputPerThreadB: 32
  WavefrontSize: 64
  WorkGroupMapping: 16
  WorkGroupMappingXCC: 2
  WorkGroupMappingXCCGroup: -1
  StaggerU: 0
  EnableMatrixInstruction: true
  MIWaveGroup:
  - {waves_m}
  - {waves_n}
  MIWaveTile:
  - {mr}
  - {nr}
  DepthU: {unroll_k}
  MacroTile:
  - {waves_m * mr * 16}
  - {waves_n * nr * 16}
  DirectToLds: 1
  LocalReadVectorWidth: -1
  GlobalReadVectorWidthA: 32
  GlobalReadVectorWidthB: 32
  GlobalSplitU: 1
  GlobalSplitUAlgorithm: MultipleBuffer
  GlobalSplitUCoalesced: false
  GlobalSplitUWorkGroupMappingRoundRobin: false
  PrefetchGlobalRead: {pgr}
  PrefetchLocalRead: 1
  StreamK: {sk_val}
  StreamKAtomic: 0
  StreamKXCCMapping: 0
  TransposeLDS: 0
  PreloadKernArgs: False
  NoReject: true
amdhsa.version: [ 1, 1 ]
amdhsa.kernels:
  - .name:            {kernel_name}
    .symbol:          {kernel_name}.kd
    .sgpr_count:      {sgpr}
    .vgpr_count:      {vgpr}
    .agpr_count:      {agpr}
    .kernarg_segment_size: 120
    .kernarg_segment_align: 8
    .group_segment_fixed_size: {lds}
    .private_segment_fixed_size: 0
    .wavefront_size:  64
    .max_flat_workgroup_size: 256
    .args:
      - .name:           SizesFree0
        .offset:         0
        .size:           4
        .value_kind:     by_value
      - .name:           SizesFree1
        .offset:         4
        .size:           4
        .value_kind:     by_value
      - .name:           SizesFree2
        .offset:         8
        .size:           4
        .value_kind:     by_value
      - .name:           SizesSum0
        .offset:         12
        .size:           4
        .value_kind:     by_value
      - .name:           D
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           C
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           A
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           MXSA
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           B
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           MXSB
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           strideD0
        .offset:         64
        .size:           4
        .value_kind:     by_value
      - .name:           strideD1
        .offset:         68
        .size:           4
        .value_kind:     by_value
      - .name:           strideC0
        .offset:         72
        .size:           4
        .value_kind:     by_value
      - .name:           strideC1
        .offset:         76
        .size:           4
        .value_kind:     by_value
      - .name:           strideA0
        .offset:         80
        .size:           4
        .value_kind:     by_value
      - .name:           strideA1
        .offset:         84
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSA0
        .offset:         88
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSA1
        .offset:         92
        .size:           4
        .value_kind:     by_value
      - .name:           strideB0
        .offset:         96
        .size:           4
        .value_kind:     by_value
      - .name:           strideB1
        .offset:         100
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSB0
        .offset:         104
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSB1
        .offset:         108
        .size:           4
        .value_kind:     by_value
      - .name:           alpha
        .offset:         112
        .size:           4
        .value_kind:     by_value
      - .name:           beta
        .offset:         116
        .size:           4
        .value_kind:     by_value
...
.end_amdgpu_metadata
"""


# ---------------------------------------------------------------------------
# FP16 custom kernel formatter
# ---------------------------------------------------------------------------

def _format_mxfp4_streamk_custom_kernel(
    kernel_name: str,
    body: str,
    sgpr: int,
    vgpr: int,
    agpr: int,
    lds: int,
    accum_off: int,
    unroll_k: int,
    waves_m: int,
    waves_n: int,
    mr: int,
    nr: int,
    pgr: int,
    streamk: bool = False,
) -> str:
    """Format the complete .s file for an MXFP4 StreamK custom kernel."""
    return f""".amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl {kernel_name}
.p2align 8
.type {kernel_name},@function

{body}

.rodata
.p2align 6
.amdhsa_kernel {kernel_name}
    .amdhsa_group_segment_fixed_size {lds}
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 156
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr {vgpr}
    .amdhsa_next_free_sgpr {sgpr}
    .amdhsa_accum_offset {accum_off}
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
    UseUniversalArgs: false
    SupportUserGSU: false
    SupportCustomWGM: false
    SupportCustomStaggerU: false
  ProblemType:
    OperationType: GEMM
    DataType: F4
    DestDataType: B
    ComputeDataType: S
    HighPrecisionAccumulate: true
    TransposeA: 1
    TransposeB: 0
    UseBeta: true
    Batched: true
    MXBlockA: 32
    MXBlockB: 32
  MatrixInstruction:
  - 16
  - 16
  - 128
  - 1
  MIBlock:
  - 16
  - 16
  - 128
  - 1
  - 1
  - 1
  MIInputPerThread: 32
  MIInputPerThreadA: 32
  MIInputPerThreadB: 32
  WavefrontSize: 64
  WorkGroupMapping: 16
  WorkGroupMappingXCC: 2
  WorkGroupMappingXCCGroup: -1
  StaggerU: 0
  EnableMatrixInstruction: true
  MIWaveGroup:
  - {waves_m}
  - {waves_n}
  MIWaveTile:
  - {mr}
  - {nr}
  DepthU: {unroll_k}
  MacroTile:
  - {waves_m * mr * 16}
  - {waves_n * nr * 16}
  DirectToLds: 1
  LocalReadVectorWidth: -1
  GlobalReadVectorWidthA: 32
  GlobalReadVectorWidthB: 32
  GlobalSplitU: 1
  GlobalSplitUAlgorithm: MultipleBuffer
  GlobalSplitUCoalesced: false
  GlobalSplitUWorkGroupMappingRoundRobin: false
  PrefetchGlobalRead: {pgr}
  PrefetchLocalRead: 1
  StreamK: 3
  StreamKAtomic: 0
  StreamKXCCMapping: 0
  TransposeLDS: 0
  PreloadKernArgs: False
  NoReject: true
  args:
    - {{ type: address, semantic: AddressWorkspace }}
    - {{ type: address, semantic: AddressFlags }}
    - {{ type: uint32, semantic: NumWorkGroups }}
    - {{ type: uint32, semantic: ItersPerTile }}
    - {{ type: uint32, semantic: SKItersPerWG }}
    - {{ type: uint32, semantic: SKGrid }}
    - {{ type: uint32, semantic: SKTilesAndSplit }}
amdhsa.version: [ 1, 1 ]
amdhsa.kernels:
  - .name:            {kernel_name}
    .symbol:          {kernel_name}.kd
    .sgpr_count:      {sgpr}
    .vgpr_count:      {vgpr}
    .agpr_count:      {agpr}
    .kernarg_segment_size: 152
    .kernarg_segment_align: 8
    .group_segment_fixed_size: {lds}
    .private_segment_fixed_size: 0
    .wavefront_size:  64
    .max_flat_workgroup_size: 256
    .args:
      - .name:           SizesFree0
        .offset:         0
        .size:           4
        .value_kind:     by_value
      - .name:           SizesFree1
        .offset:         4
        .size:           4
        .value_kind:     by_value
      - .name:           SizesFree2
        .offset:         8
        .size:           4
        .value_kind:     by_value
      - .name:           SizesSum0
        .offset:         12
        .size:           4
        .value_kind:     by_value
      - .name:           D
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           C
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           A
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           MXSA
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           B
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           MXSB
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           AddressWorkspace
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           AddressFlags
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           strideD0
        .offset:         80
        .size:           4
        .value_kind:     by_value
      - .name:           strideD1
        .offset:         84
        .size:           4
        .value_kind:     by_value
      - .name:           strideC0
        .offset:         88
        .size:           4
        .value_kind:     by_value
      - .name:           strideC1
        .offset:         92
        .size:           4
        .value_kind:     by_value
      - .name:           strideA0
        .offset:         96
        .size:           4
        .value_kind:     by_value
      - .name:           strideA1
        .offset:         100
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSA0
        .offset:         104
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSA1
        .offset:         108
        .size:           4
        .value_kind:     by_value
      - .name:           strideB0
        .offset:         112
        .size:           4
        .value_kind:     by_value
      - .name:           strideB1
        .offset:         116
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSB0
        .offset:         120
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSB1
        .offset:         124
        .size:           4
        .value_kind:     by_value
      - .name:           alpha
        .offset:         128
        .size:           4
        .value_kind:     by_value
      - .name:           beta
        .offset:         132
        .size:           4
        .value_kind:     by_value
      - .name:           NumWorkGroups
        .offset:         136
        .size:           4
        .value_kind:     by_value
      - .name:           ItersPerTile
        .offset:         140
        .size:           4
        .value_kind:     by_value
      - .name:           SKItersPerWG
        .offset:         144
        .size:           4
        .value_kind:     by_value
      - .name:           SKGrid
        .offset:         148
        .size:           4
        .value_kind:     by_value
      - .name:           SKTilesAndSplit
        .offset:         152
        .size:           4
        .value_kind:     by_value
...
.end_amdgpu_metadata
"""


# ---------------------------------------------------------------------------

def _format_fp16_custom_kernel(
    kernel_name: str,
    body: str,
    sgpr: int,
    vgpr: int,
    agpr: int,
    lds: int,
    karg: int,
    accum_off: int,
    unroll_k: int,
    waves_m: int,
    waves_n: int,
    mr: int,
    nr: int,
    mfma: MfmaConfig,
    pgr: int,
    streamk: bool = False,
    dtype_code: str = "H",
) -> str:
    """Format the complete .s file for an FP16/BF16 custom kernel."""
    sk_val = 3 if streamk else 0
    k_iters = unroll_k // mfma.k
    mi_input = mfma.k // 2

    return f""".amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl {kernel_name}
.p2align 8
.type {kernel_name},@function

{body}

.rodata
.p2align 6
.amdhsa_kernel {kernel_name}
    .amdhsa_group_segment_fixed_size {lds}
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size {karg}
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr {vgpr}
    .amdhsa_next_free_sgpr {sgpr}
    .amdhsa_accum_offset {accum_off}
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
    UseUniversalArgs: false
    SupportUserGSU: false
    SupportCustomWGM: false
    SupportCustomStaggerU: false
  ProblemType:
    OperationType: GEMM
    DataType: {dtype_code}
    DestDataType: {dtype_code}
    ComputeDataType: S
    HighPrecisionAccumulate: true
    TransposeA: 1
    TransposeB: 0
    UseBeta: true
    Batched: true
  MatrixInstruction: [{mfma.m}, {mfma.n}, {mfma.k}, 1]
  MIBlock: [{mfma.m}, {mfma.n}, {mfma.k // k_iters}, 1, 1, {k_iters}]
  MIInputPerThread: {mi_input}
  MIInputPerThreadA: {mi_input}
  MIInputPerThreadB: {mi_input}
  WavefrontSize: 64
  WorkGroupMapping: 16
  WorkGroupMappingXCC: 2
  WorkGroupMappingXCCGroup: -1
  StaggerU: 0
  EnableMatrixInstruction: true
  MIWaveGroup: [{waves_m}, {waves_n}]
  MIWaveTile: [{mr}, {nr}]
  DepthU: {unroll_k}
  DirectToLds: 1
  LocalReadVectorWidth: -1
  GlobalReadVectorWidthA: 32
  GlobalReadVectorWidthB: 32
  GlobalSplitU: 1
  GlobalSplitUAlgorithm: MultipleBuffer
  GlobalSplitUCoalesced: false
  GlobalSplitUWorkGroupMappingRoundRobin: false
  PrefetchGlobalRead: {pgr}
  PrefetchLocalRead: 1
  StreamK: {sk_val}
  StreamKAtomic: 0
  StreamKXCCMapping: 0
  TransposeLDS: 0
  PreloadKernArgs: False
  NoReject: true
amdhsa.version: [ 1, 1 ]
amdhsa.kernels:
  - .name:            {kernel_name}
    .symbol:          {kernel_name}.kd
    .sgpr_count:      {sgpr}
    .vgpr_count:      {vgpr}
    .agpr_count:      {agpr}
    .kernarg_segment_size: {karg}
    .kernarg_segment_align: 8
    .group_segment_fixed_size: {lds}
    .private_segment_fixed_size: 0
    .wavefront_size:  64
    .max_flat_workgroup_size: 256
    .args:
      - .name:           SizesFree0
        .offset:         0
        .size:           4
        .value_kind:     by_value
      - .name:           SizesFree1
        .offset:         4
        .size:           4
        .value_kind:     by_value
      - .name:           SizesFree2
        .offset:         8
        .size:           4
        .value_kind:     by_value
      - .name:           SizesSum0
        .offset:         12
        .size:           4
        .value_kind:     by_value
      - .name:           D
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           C
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           A
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           B
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           strideD0
        .offset:         48
        .size:           4
        .value_kind:     by_value
      - .name:           strideD1
        .offset:         52
        .size:           4
        .value_kind:     by_value
      - .name:           strideC0
        .offset:         56
        .size:           4
        .value_kind:     by_value
      - .name:           strideC1
        .offset:         60
        .size:           4
        .value_kind:     by_value
      - .name:           strideA0
        .offset:         64
        .size:           4
        .value_kind:     by_value
      - .name:           strideA1
        .offset:         68
        .size:           4
        .value_kind:     by_value
      - .name:           strideB0
        .offset:         72
        .size:           4
        .value_kind:     by_value
      - .name:           strideB1
        .offset:         76
        .size:           4
        .value_kind:     by_value
      - .name:           alpha
        .offset:         80
        .size:           4
        .value_kind:     by_value
      - .name:           beta
        .offset:         84
        .size:           4
        .value_kind:     by_value
...
.end_amdgpu_metadata
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_custom_kernel(
    wg_m: int = 256,
    wg_n: int = 256,
    unroll_k: int = 256,
    kernel_name: Optional[str] = None,
    dtype: str = "mxfp4",
    pgr2: bool = False,
    swizzled_scales: bool = True,
    streamk: bool = False,
) -> str:
    """Generate a TensileLite custom kernel .s file contents.

    Args:
        wg_m, wg_n, unroll_k: Tile dimensions.
        kernel_name: Override kernel function name.
        dtype: Data type -- ``"mxfp4"``, ``"fp16"``, or ``"bf16"``.
        pgr2: Enable PGR=2 double-prefetch (FP16 only; MXFP4 always uses 2).
        swizzled_scales: Use pre-swizzled MX scale layout (MXScaleFormat=1).
        streamk: Enable StreamK work distribution.

    Returns:
        The full .s file text including assembly body,
        .amdhsa_kernel descriptor, and .amdgpu_metadata YAML config.
    """
    from .mainloop import mainloop_fp16, mainloop_bf16, mainloop_mxfp4_tensilelite

    # Data type config from registry
    from .config import dtype_config
    dcfg = dtype_config(dtype)
    mfma = dcfg.mfma_factory()
    unroll_k = min(unroll_k, dcfg.max_unroll_k)
    effective_pgr = dcfg.default_pgr if not pgr2 else 2

    # Build swizzle and tiling (same pattern for all types)
    from .memory.swizzle import DTLRotationSwizzle, DataLayout as SwzLayout
    elem = dcfg.element_bytes
    swz_layout = SwzLayout(
        row_stride_bytes=int(unroll_k * elem),
        mfma_k=mfma.k, mfma_m=mfma.m,
        elem_bytes=elem,
        wave_size=64,
    )
    dtl_swz = DTLRotationSwizzle.from_layout(swz_layout)
    t = GemmTiling.high_perf(
        wg_m=wg_m, wg_n=wg_n, unroll_k=unroll_k,
        mfma=mfma, lds_swizzle=True, swizzle=dtl_swz,
    )

    # Problem + mainloop (MX types need special mainloop)
    dtype_enum = {"fp16": DataType.F16, "bf16": DataType.BF16,
                  "mxfp4": DataType.MXFP4}[dtype]
    p = GemmProblem(4096, 4096, 4096, dtype=dtype_enum)

    if dcfg.has_mx_scales:
        ml = mainloop_mxfp4_tensilelite(
            pgr=effective_pgr, wg_mapping_xcc=1, colmajor_output=True,
            swizzled_scales=swizzled_scales, streamk=streamk,
        )
    else:
        # FP16 and BF16 share the same mainloop structure
        mainloop_fn = mainloop_bf16 if dtype == "bf16" else mainloop_fp16
        ml = mainloop_fn(
            pgr=effective_pgr, wg_mapping_xcc=8, colmajor_output=True,
            tensilelite_abi=True,
        )

    k = GemmKernel.build(p, tiling=t, mainloop=ml)

    if kernel_name is None:
        kernel_name = (
            f"Custom_Cijk_Alik_Bljk_{dcfg.kernel_name_fragment}"
            f"_MT{wg_m}x{wg_n}x{unroll_k}_MI16x16x1"
            f"_kgen_gfx950"
        )

    k.kernel_name = kernel_name
    result = k.emit()
    asm = result.asm_text

    regs = _parse_register_counts(asm)
    body = _extract_kernel_body(asm, kernel_name, kernel_name)

    tile_cfg = t.to_tile_config()
    mr = tile_cfg.mfma_m_repeat
    nr = tile_cfg.mfma_n_repeat

    if not dcfg.has_mx_scales:
        return _format_fp16_custom_kernel(
            kernel_name=kernel_name,
            body=body,
            sgpr=regs["sgpr"],
            vgpr=regs["vgpr"],
            agpr=regs["agpr"],
            lds=regs["lds"],
            karg=regs["karg"],
            accum_off=regs["accum_off"],
            unroll_k=unroll_k,
            waves_m=tile_cfg.waves_m,
            waves_n=tile_cfg.waves_n,
            mr=mr,
            nr=nr,
            mfma=mfma,
            pgr=effective_pgr,
            streamk=streamk,
            dtype_code=dcfg.tensile_data_type,
        )
    else:
        return _format_mxfp4_custom_kernel(
            kernel_name=kernel_name,
            body=body,
            sgpr=regs["sgpr"],
            vgpr=regs["vgpr"],
            agpr=regs["agpr"],
            lds=regs["lds"],
            accum_off=regs["accum_off"],
            unroll_k=unroll_k,
            waves_m=tile_cfg.waves_m,
            waves_n=tile_cfg.waves_n,
            mr=mr,
            nr=nr,
            pgr=effective_pgr,
            streamk=streamk,
        )
