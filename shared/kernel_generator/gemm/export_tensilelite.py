"""Export kernel generator kernels for TensileLite benchmarking.

Generates a TensileLite-compatible custom kernel .s file that can be
benchmarked through Tensile.sh or tensilelite-client.
"""
from __future__ import annotations
import re
from .problem import GemmProblem, DataType, MfmaConfig
from .tiling import GemmTiling
from .kernel import GemmKernel


def generate_custom_kernel(
    wg_m: int = 256, wg_n: int = 256, unroll_k: int = 256,
    scheduled: bool = True,
    kernel_name: str = None,
) -> str:
    """Generate a TensileLite custom kernel .s file contents.

    Returns the full .s file text including assembly body,
    .amdhsa_kernel descriptor, and .amdgpu_metadata YAML config.
    """
    mx = MfmaConfig.mxfp4_16x16x128()
    t = GemmTiling.high_perf(wg_m=wg_m, wg_n=wg_n, unroll_k=unroll_k,
                             mfma=mx, lds_swizzle=True)
    # Use a large problem for codegen (actual size comes from TensileLite at runtime)
    p = GemmProblem(4096, 4096, 4096, dtype=DataType.MXFP4)

    if scheduled:
        k = GemmKernel.build(p, tiling=t, scheduled=True)
    else:
        k = GemmKernel.build(p, tiling=t, scheduled=True)

    # Force 1D grid + column-major store for TensileLite compatibility
    k.use_1d_grid = True
    k.swizzled_scales = False  # Use linear scale layout (MXScaleFormat: 0)

    if kernel_name is None:
        kernel_name = (f"Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32"
                       f"_MT{wg_m}x{wg_n}x{unroll_k}_MI16x16x1"
                       f"_kgen_gfx950")

    # Override the kernel name before emitting
    k.kernel_name = kernel_name
    result = k.emit()
    asm = result.asm_text

    # Parse register counts from generated metadata
    sgpr = int(re.search(r'\.sgpr_count:\s+(\d+)', asm).group(1))
    vgpr = int(re.search(r'\.vgpr_count:\s+(\d+)', asm).group(1))
    agpr = int(re.search(r'\.agpr_count:\s+(\d+)', asm).group(1))
    lds = int(re.search(r'\.group_segment_fixed_size:\s+(\d+)', asm).group(1))
    karg = int(re.search(r'\.kernarg_segment_size:\s+(\d+)', asm).group(1))
    accum_off = vgpr - agpr

    mr = t.to_tile_config().mfma_m_repeat
    nr = t.to_tile_config().mfma_n_repeat

    # Extract kernel body (label to s_endpgm)
    old_label = k.kernel_name + ':'
    body_lines = []
    started = False
    for line in asm.splitlines():
        if line.startswith(old_label):
            started = True
            body_lines.append(f'{kernel_name}:')
            continue
        if started:
            body_lines.append(line)
            if line.strip().startswith('s_endpgm'):
                break
    body = '\n'.join(body_lines)

    return f'''.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
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
  MatrixInstruction: [16, 16, 128, 1]
  MIBlock: [16, 16, 32, 1, 1, 4]
  MIInputPerThread: 32
  MIInputPerThreadA: 32
  MIInputPerThreadB: 32
  WavefrontSize: 64
  WorkGroupMapping: 16
  WorkGroupMappingXCC: 2
  WorkGroupMappingXCCGroup: -1
  StaggerU: 0
  EnableMatrixInstruction: true
  MIWaveGroup: [{t.to_tile_config().waves_m}, {t.to_tile_config().waves_n}]
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
  PrefetchGlobalRead: 2
  PrefetchLocalRead: 1
  StreamK: 0
  StreamKAtomic: 0
  StreamKXCCMapping: 0
  TransposeLDS: 0
  NoReject: true
amdhsa.version: [1, 2]
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
      - .name: Gemm info
        .offset: 0
        .size: 4
        .value_kind: by_value
      - .name: kernel info0
        .offset: 4
        .size: 4
        .value_kind: by_value
      - .name: kernel info1
        .offset: 8
        .size: 4
        .value_kind: by_value
      - .name: numWG
        .offset: 12
        .size: 4
        .value_kind: by_value
      - .name: SizesFree0
        .offset: 16
        .size: 4
        .value_kind: by_value
      - .name: SizesFree1
        .offset: 20
        .size: 4
        .value_kind: by_value
      - .name: SizesFree2
        .offset: 24
        .size: 4
        .value_kind: by_value
      - .name: SizesSum0
        .offset: 28
        .size: 4
        .value_kind: by_value
      - .name: D
        .offset: 32
        .size: 8
        .value_kind: global_buffer
        .address_space: generic
      - .name: C
        .offset: 40
        .size: 8
        .value_kind: global_buffer
        .address_space: generic
      - .name: A
        .offset: 48
        .size: 8
        .value_kind: global_buffer
        .address_space: generic
      - .name: MXSA
        .offset: 56
        .size: 8
        .value_kind: global_buffer
        .address_space: generic
      - .name: B
        .offset: 64
        .size: 8
        .value_kind: global_buffer
        .address_space: generic
      - .name: MXSB
        .offset: 72
        .size: 8
        .value_kind: global_buffer
        .address_space: generic
      - .name: strideD0
        .offset: 80
        .size: 4
        .value_kind: by_value
      - .name: strideD1
        .offset: 84
        .size: 4
        .value_kind: by_value
      - .name: strideC0
        .offset: 88
        .size: 4
        .value_kind: by_value
      - .name: strideC1
        .offset: 92
        .size: 4
        .value_kind: by_value
      - .name: strideA0
        .offset: 96
        .size: 4
        .value_kind: by_value
      - .name: strideA1
        .offset: 100
        .size: 4
        .value_kind: by_value
      - .name: strideMXSA0
        .offset: 104
        .size: 4
        .value_kind: by_value
      - .name: strideMXSA1
        .offset: 108
        .size: 4
        .value_kind: by_value
      - .name: strideB0
        .offset: 112
        .size: 4
        .value_kind: by_value
      - .name: strideB1
        .offset: 116
        .size: 4
        .value_kind: by_value
      - .name: strideMXSB0
        .offset: 120
        .size: 4
        .value_kind: by_value
      - .name: strideMXSB1
        .offset: 124
        .size: 4
        .value_kind: by_value
      - .name: alpha
        .offset: 128
        .size: 4
        .value_kind: by_value
      - .name: beta
        .offset: 132
        .size: 4
        .value_kind: by_value
...
.end_amdgpu_metadata
'''
