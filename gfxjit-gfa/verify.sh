#!/bin/bash
cd /home/kerrwang/.claude/jobs/89e9b96e/tmp/gfa
bash bench.sh -i 2 -j 2 --verify_strict > verify.out 2>&1
rc=$?
echo "hipblaslt-bench --verify_strict exit code: $rc  (0 = all checks passed, 2 = a check failed)"
grep -E "norm check failed|allclose check failed|verify check" verify.out || echo "no failed-check lines in output"
grep -E "^    T,N" verify.out | awk -F, '{print "norm_error:", $(NF-2), " atol:", $(NF-1), " rtol:", $NF}'
