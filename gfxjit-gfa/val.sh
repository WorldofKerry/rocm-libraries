#!/bin/bash
cd /home/kerrwang/.claude/jobs/89e9b96e/tmp/gfa
export HIP_VISIBLE_DEVICES=7
BIN=${BIN:-./mxfp6_gfa}
for sh in "-M 256 -N 256" "-M 512 -N 512" "-M 768 -N 512" "-M 512 -N 1024"; do
  printf "%-22s K=512 : " "$sh"
  $BIN $sh 2>&1 | grep -E "^PASS|^FAIL|max rel err" | tr '\n' ' '
  echo
done
