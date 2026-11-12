#!/bin/bash
# Report TFLOP/s per run, from the Gflops column of hipblaslt-bench.
cd /home/kerrwang/.claude/jobs/89e9b96e/tmp/gfa
N=${N:-7}
for i in $(seq 1 $N); do
  bash bench.sh -i 20 -j 20 2>/dev/null | grep -E "^    T,N" \
    | awk -F, '{printf "%.0f\n", $(NF-2)/1000}'
done
