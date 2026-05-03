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
    kernel_name: str = None,
    dtype: str = "mxfp4",
    pgr2: bool = False,
) -> str:
    """Generate a TensileLite custom kernel .s file contents.

    Args:
        wg_m, wg_n, unroll_k: Tile dimensions.
        kernel_name: Override kernel function name.
        dtype: Data type -- "mxfp4" or "fp16".
        pgr2: Enable PGR=2 double-prefetch.

    Returns the full .s file text including assembly body,
    .amdhsa_kernel descriptor, and .amdgpu_metadata YAML config.
    """
    if dtype == "fp16":
        mfma = MfmaConfig.f16_16x16x16()
        if unroll_k == 256:
            unroll_k = 64  # reasonable default for FP16
        t = GemmTiling.high_perf(wg_m=wg_m, wg_n=wg_n, unroll_k=unroll_k,
                                 mfma=mfma, lds_swizzle=True)
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.F16)
    else:
        mfma = MfmaConfig.mxfp4_16x16x128()
        t = GemmTiling.high_perf(wg_m=wg_m, wg_n=wg_n, unroll_k=unroll_k,
                                 mfma=mfma, lds_swizzle=True)
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.MXFP4)

    k = GemmKernel.build(p, tiling=t, pgr2=pgr2)

    # Force 1D grid + column-major store for TensileLite compatibility
    k.use_1d_grid = True
    if dtype != "fp16":
        k.swizzled_scales = False  # Use linear scale layout (MXScaleFormat: 0)

    if kernel_name is None:
        if dtype == "fp16":
            kernel_name = (f"Custom_Cijk_Ailk_Bjlk_HHS_BH"
                           f"_MT{wg_m}x{wg_n}x{unroll_k}_MI16x16x1"
                           f"_kgen_gfx950")
        else:
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

    # Build metadata based on dtype
    if dtype == "fp16":
        data_type = "H"
        dest_type = "H"
        transpose_a = "0"
        mx_block = ""
        mi_inst = f"[16, 16, {mfma.k}, 1]"
        mi_block = f"[16, 16, {mfma.k // t.to_tile_config().k_iterations}, 1, 1, {t.to_tile_config().k_iterations}]"
        mi_input = str(mfma.k // 2)
        mi_input_a = str(mfma.k // 2)
        mi_input_b = str(mfma.k // 2)
        pgr_val = 2 if pgr2 else 1
        gwvw_d = 4
    else:
        data_type = "F4"
        dest_type = "B"
        transpose_a = "1"
        mx_block = """
    MXBlockA: 32
    MXBlockB: 32"""
        mi_inst = "[16, 16, 128, 1]"
        mi_block = "[16, 16, 32, 1, 1, 4]"
        mi_input = "32"
        mi_input_a = "32"
        mi_input_b = "32"
        pgr_val = 2 if pgr2 else 2  # MXFP4 default PGR=2
        gwvw_d = 4

    tile_cfg = t.to_tile_config()
    mr = tile_cfg.mfma_m_repeat
    nr = tile_cfg.mfma_n_repeat

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
    DataType: {data_type}
    DestDataType: {dest_type}
    ComputeDataType: S
    HighPrecisionAccumulate: true
    TransposeA: {transpose_a}
    TransposeB: 0
    UseBeta: true
    Batched: true{mx_block}
  MatrixInstruction: {mi_inst}
  MIBlock: {mi_block}
  MIInputPerThread: {mi_input}
  MIInputPerThreadA: {mi_input_a}
  MIInputPerThreadB: {mi_input_b}
  WavefrontSize: 64
  WorkGroupMapping: 16
  WorkGroupMappingXCC: 2
  WorkGroupMappingXCCGroup: -1
  StaggerU: 0
  EnableMatrixInstruction: true
  MIWaveGroup: [{tile_cfg.waves_m}, {tile_cfg.waves_n}]
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
  PrefetchGlobalRead: {pgr_val}
  PrefetchLocalRead: 1
  StreamK: 0
  StreamKAtomic: 0
  StreamKXCCMapping: 0
  TransposeLDS: 0
  NonTemporalD: {gwvw_d}
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
        .address_space: global
      - .name: C
        .offset: 40
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .name: A
        .offset: 48
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .name: B
        .offset: {56 if dtype == "fp16" else 64}
        .size: 8
        .value_kind: global_buffer
        .address_space: global
...
'''
