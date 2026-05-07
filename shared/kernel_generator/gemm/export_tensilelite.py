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
) -> str:
    """Format the complete .s file for an FP16 custom kernel."""
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
  ProblemType:
    OperationType: GEMM
    DataType: H
    DestDataType: H
    ComputeDataType: S
    HighPrecisionAccumulate: true
    TransposeA: 0
    TransposeB: 1
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
        .offset: 56
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .name: strideD0
        .offset: 64
        .size: 4
        .value_kind: by_value
      - .name: strideD1
        .offset: 68
        .size: 4
        .value_kind: by_value
      - .name: strideC0
        .offset: 72
        .size: 4
        .value_kind: by_value
      - .name: strideC1
        .offset: 76
        .size: 4
        .value_kind: by_value
      - .name: strideA0
        .offset: 80
        .size: 4
        .value_kind: by_value
      - .name: strideA1
        .offset: 84
        .size: 4
        .value_kind: by_value
      - .name: strideB0
        .offset: 88
        .size: 4
        .value_kind: by_value
      - .name: strideB1
        .offset: 92
        .size: 4
        .value_kind: by_value
      - .name: alpha
        .offset: 96
        .size: 4
        .value_kind: by_value
      - .name: beta
        .offset: 100
        .size: 4
        .value_kind: by_value
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
        dtype: Data type -- ``"mxfp4"`` or ``"fp16"``.
        pgr2: Enable PGR=2 double-prefetch (FP16 only; MXFP4 always uses 2).
        swizzled_scales: Use pre-swizzled MX scale layout (MXScaleFormat=1).
        streamk: Enable StreamK work distribution.

    Returns:
        The full .s file text including assembly body,
        .amdhsa_kernel descriptor, and .amdgpu_metadata YAML config.
    """
    from .mainloop import mainloop_fp16, mainloop_mxfp4_tensilelite

    if dtype == "fp16":
        mfma = MfmaConfig.f16_16x16x16()
        unroll_k = min(unroll_k, 64)
        t = GemmTiling.high_perf(
            wg_m=wg_m, wg_n=wg_n, unroll_k=unroll_k,
            mfma=mfma, lds_swizzle=True,
        )
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.F16)
        effective_pgr = 2 if pgr2 else 1
        ml = mainloop_fp16(
            pgr=effective_pgr, wg_mapping_xcc=8, colmajor_output=True,
        )
    else:
        mfma = MfmaConfig.mxfp4_16x16x128()
        t = GemmTiling.high_perf(
            wg_m=wg_m, wg_n=wg_n, unroll_k=unroll_k,
            mfma=mfma, lds_swizzle=True,
        )
        p = GemmProblem(4096, 4096, 4096, dtype=DataType.MXFP4)
        effective_pgr = 2  # MXFP4 always PGR=2
        ml = mainloop_mxfp4_tensilelite(
            pgr=effective_pgr, wg_mapping_xcc=2, colmajor_output=True,
            swizzled_scales=swizzled_scales,
            streamk=streamk,
        )

    k = GemmKernel.build(p, tiling=t, mainloop=ml)

    if kernel_name is None:
        if dtype == "fp16":
            kernel_name = (
                f"Custom_Cijk_Ailk_Bjlk_HHS_BH"
                f"_MT{wg_m}x{wg_n}x{unroll_k}_MI16x16x1"
                f"_kgen_gfx950"
            )
        else:
            kernel_name = (
                f"Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32"
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

    if dtype == "fp16":
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
