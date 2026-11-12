#!/bin/bash
# Usage: bench.sh [extra hipblaslt-bench args...]
HB=/home/kerrwang/.claude/jobs/89e9b96e/tmp/gfa-wt/projects/hipblaslt
export HIP_VISIBLE_DEVICES=7
export LD_LIBRARY_PATH=$HB/build/release/library:$HOME/.miniforge3/lib
"$HB/build/release/clients/hipblaslt-bench" \
  --a_type f6_r --b_type f6_r \
  --c_type bf16_r --d_type bf16_r --compute_type f32_r --scale_type f32_r \
  -m 4096 -n 4096 -k 65536 --transA T --transB N \
  --scaleA 1018 --scaleB 1018 --swizzleA --swizzleB \
  --alpha 1 --beta 0 --api_method cpp --algo_method heuristic \
  --print_kernel_info "$@"
