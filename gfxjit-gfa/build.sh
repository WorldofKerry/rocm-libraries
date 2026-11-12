#!/bin/bash
cd /home/kerrwang/.claude/jobs/89e9b96e/tmp/gfa-wt/projects/hipblaslt
cmake --build build/release --parallel 8
echo "BUILD_EXIT=$?"
