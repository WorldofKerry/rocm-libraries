"""Generate a BF16 mainloop-only kernel for Triton Gluon inline ASM.

Usage:
    .venv/bin/python generate_bf16_mainloop.py [--wg-m 256] [--wg-n 256] [--unroll-k 64] [--output bf16_mainloop.s]
"""
from kernel_generator.gemm.problem import DataType, GemmProblem, MfmaConfig
from kernel_generator.gemm.tiling import GemmTiling
from kernel_generator.gemm.kernel import GemmKernel


def generate(wg_m=256, wg_n=256, unroll_k=64, output="bf16_mainloop.s"):
    mfma = MfmaConfig.bf16_16x16x32()
    t = GemmTiling.high_perf(wg_m=wg_m, wg_n=wg_n, unroll_k=unroll_k,
                             mfma=mfma, lds_swizzle=True)
    p = GemmProblem(4096, 4096, 4096, dtype=DataType.BF16)
    k = GemmKernel.build(p, tiling=t, skip_store=True)
    result = k.emit()

    with open(output, "w") as f:
        f.write(result.asm_text)

    tile = t.to_tile_config()
    print(f"Wrote: {output}")
    print(f"  Tile: {wg_m}x{wg_n}x{unroll_k}")
    print(f"  MFMA: v_mfma_f32_16x16x32_bf16")
    print(f"  Waves: {tile.waves_m}x{tile.waves_n} ({tile.block_size} threads)")
    print(f"  VGPRs: {result.vgpr_count}, SGPRs: {result.sgpr_count}, Acc: {result.acc_count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wg-m", type=int, default=256)
    parser.add_argument("--wg-n", type=int, default=256)
    parser.add_argument("--unroll-k", type=int, default=64)
    parser.add_argument("--output", default="bf16_mainloop.s")
    args = parser.parse_args()
    generate(args.wg_m, args.wg_n, args.unroll_k, args.output)
