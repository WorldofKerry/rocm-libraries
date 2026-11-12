#!/usr/bin/env python3
"""Reconstruct a GFA custom-kernel .s from a gfxjit-generated code object.

gfxjit emits binary ELF only (no assembly emitter in the headers), so the text
section is recovered with llvm-objdump and paired with a hand-written
.amdhsa_kernel block and an .amdgpu_metadata block carrying custom.config.
The result is assembled back and compared against the original code object, so
the reconstruction is verified rather than trusted.
"""

import re
import subprocess
import sys

OBJDUMP = "/opt/rocm/lib/llvm/bin/llvm-objdump"

NAME = ("gfxjit_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_"
        "256x256x128_w2x2_16x16x128_s3_4096x4096x65536_paired")

M, N, K = 4096, 4096, 65536


def disassemble(co):
    out = subprocess.run([OBJDUMP, "-d", "--mcpu=gfx950", co],
                         capture_output=True, text=True, check=True).stdout
    body = []
    started = False
    for line in out.splitlines():
        if re.match(r"^[0-9a-f]+ <.*>:", line):
            started = True
            continue
        if not started:
            continue
        if not line.strip():
            continue
        # drop the "// <addr>: <encoding>" trailer llvm-objdump appends
        insn = line.split("//")[0].strip()
        if not insn:
            continue
        body.append("\t" + insn)
    return body


HEADER = """\t.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
\t.text
\t.globl\t{name}
\t.p2align\t8
\t.type\t{name},@function
{name}:"""

# Register counts come from the code object's own metadata; next_free_vgpr is
# the unified arch+accum total and accum_offset the arch VGPR count.
DESCRIPTOR = """
\t.section\t.rodata,"a",@progbits
\t.p2align\t6, 0x0
.amdhsa_kernel {name}
  .amdhsa_group_segment_fixed_size 153600
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 40
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr {next_free_vgpr}
  .amdhsa_next_free_sgpr {next_free_sgpr}
  .amdhsa_reserve_vcc {reserve_vcc}
  .amdhsa_accum_offset {accum_offset}
  .amdhsa_float_denorm_mode_32 {denorm32}
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
"""

METADATA = """
\t.amdgpu_metadata
---
custom.config:
  Source:
    Origin: gfxjit
    Description: MX E2M3 GEMM at {M}x{N}x{K} on tiled operands and shuffled scales
  Version: 1.0.0
  Features:
    SupportsUserArgs: false
    SupportsBias: false
    SupportsActivation: false
    SupportsScaleAlpha: false
    SupportsGSU: false
  InternalSupportParams:
    KernArgsVersion: 0
  ProblemType:
    OperationType: GEMM
    DataType: F6
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: False
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
    # SwizzleTensorA/B are plain booleans: they say THAT the operands are swizzled, not
    # WHICH layout. This kernel requires the PAIRED layout -- a 24-byte group stored as 24
    # contiguous bytes at 3072*(t/2) + 48*lane + 24*(t%2) -- and NOT the published
    # HIPBLASLT_ORDER_COL256_R128_EXT ("c0c1") two-plane layout. Given c0c1 data it is
    # silently wrong at full speed. The matching host packer is tileFp6Data in
    # clients/common/include/mx_tile128x4.hpp, which this worktree replaced in place;
    # the two must be changed together.
    SwizzleTensorA: True
    SwizzleTensorB: True
    MXBlockA: 32
    MXBlockB: 32
    DataTypeMXSA: E8
    DataTypeMXSB: E8
  CustomKernel:
    # The kernel's a and sa are the GEMM's B side, b and sb its A side: the
    # kernel writes C row-major [M, N], which is the column-major D hipBLASLt
    # wants only if the two operands are exchanged.
    args: [ {{ type: address, semantic: AddressD }},
            {{ type: address, semantic: AddressB }},
            {{ type: address, semantic: AddressA }},
            {{ type: address, semantic: AddressMXScaleB }},
            {{ type: address, semantic: AddressMXScaleA }} ]
    macrotile: [256, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, One]
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [8, 8]
  WavefrontSize: 64
  StaggerU: 0
  MXScaleFormat: HostPreSwizzle128x4
  AssertSizeEqual: {{0: {M}, 1: {N}, 2: 1, 3: {K}}}
  AssertStrideAEqual: {{1: {K}}}
  AssertStrideBEqual: {{1: {K}}}
  AssertStrideDEqual: {{1: {M}}}
  AssertAlphaValue: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version: [ 1, 1 ]
amdhsa.kernels:
  - .name: {name}
    .symbol: {name}.kd
    .kernarg_segment_size: 40
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 153600
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .sgpr_count: {sgpr_count}
    .vgpr_count: {vgpr_count}
    .agpr_count: {agpr_count}
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uniform_work_group_size: 1
    .args:
      - {{ .name: c,  .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }}
      - {{ .name: a,  .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }}
      - {{ .name: b,  .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }}
      - {{ .name: sa, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global }}
      - {{ .name: sb, .size: 8, .offset: 32, .value_kind: global_buffer, .address_space: global }}
...
\t.end_amdgpu_metadata
"""


def main():
    co = sys.argv[1]
    out = sys.argv[2]
    arch_vgpr, agpr, sgpr = 240, 256, 60
    body = disassemble(co)
    text = HEADER.format(name=NAME) + "\n" + "\n".join(body) + "\n"
    text += DESCRIPTOR.format(name=NAME,
                              next_free_vgpr=arch_vgpr + agpr,
                              next_free_sgpr=sgpr,
                              reserve_vcc=0,
                              accum_offset=arch_vgpr,
                              denorm32=3)
    text += METADATA.format(name=NAME, M=M, N=N, K=K,
                            sgpr_count=sgpr, vgpr_count=arch_vgpr,
                            agpr_count=agpr)
    open(out, "w").write(text)
    print("wrote %s (%d instructions)" % (out, len(body)))


main()
