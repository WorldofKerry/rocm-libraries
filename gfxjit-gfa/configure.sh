#!/bin/bash
set -e
cd /home/kerrwang/.claude/jobs/89e9b96e/tmp/gfa-wt/projects/hipblaslt
cmake -B build/release -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DGPU_TARGETS=gfx950 \
  -DHIPBLASLT_ENABLE_FETCH=ON \
  -DHIPBLASLT_ENABLE_HOST=ON \
  -DHIPBLASLT_ENABLE_DEVICE=ON \
  -DHIPBLASLT_ENABLE_CLIENT=ON \
  -DHIPBLASLT_ENABLE_BLIS=OFF \
  -DPython3_EXECUTABLE=/home/kerrwang/.claude/jobs/89e9b96e/tmp/gfa-wt/projects/hipblaslt/.venv/bin/python
