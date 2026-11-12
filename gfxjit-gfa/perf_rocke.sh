#!/bin/bash
# Same bench invocation against the prebuilt rocKE worktree (read-only use).
HB=/home/kerrwang/repos/rocm-libraries/wt-mxfp6-rocke/projects/hipblaslt
export HIP_VISIBLE_DEVICES=7
export LD_LIBRARY_PATH=$HB/build/release/library:$HOME/.miniforge3/lib
N=${N:-7}
for i in $(seq 1 $N); do
  "$HB/build/release/clients/hipblaslt-bench" \
    --a_type f6_r --b_type f6_r \
    --c_type bf16_r --d_type bf16_r --compute_type f32_r --scale_type f32_r \
    -m 4096 -n 4096 -k 65536 --transA T --transB N \
    --scaleA 1018 --scaleB 1018 --swizzleA --swizzleB \
    --alpha 1 --beta 0 --api_method cpp --algo_method heuristic \
    --print_kernel_info -i 20 -j 20 2>/dev/null | tee -a rocke_raw.txt \
    | grep -E "^    T,N" | awk -F, '{printf "%.0f\n", $(NF-2)/1000}'
done
