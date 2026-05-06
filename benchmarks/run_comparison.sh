#!/bin/bash
# Generate our custom kernel and benchmark against TensileLite
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KGEN_DIR="$(dirname "$SCRIPT_DIR")"
TENSILE_DIR="$KGEN_DIR/projects/hipblaslt/tensilelite/Tensile"
CUSTOM_DIR="$TENSILE_DIR/CustomKernels"
BUILD_DIR="${KGEN_DIR}/../agent/projects/hipblaslt/build-tensilelite"
TENSILE_SH="$BUILD_DIR/Tensile.sh"
OUTPUT_DIR="${1:-/tmp/tensile_comparison}"

echo "=== Step 1: Generate custom kernel ==="
cd "$KGEN_DIR"
PYTHONPATH=shared python3 -c "
from kernel_generator.gemm.export_tensilelite import generate_custom_kernel
asm = generate_custom_kernel(256, 256, 256, dtype='mxfp4')
with open('$SCRIPT_DIR/Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32_MT256x256x256_MI16x16x1_kgen_gfx950.s', 'w') as f:
    f.write(asm)
print('Generated custom kernel .s file')
"

echo "=== Step 2: Copy to CustomKernels ==="
cp "$SCRIPT_DIR"/Custom_*.s "$CUSTOM_DIR/"
echo "Copied to $CUSTOM_DIR/"

echo "=== Step 3: Run benchmark ==="
if [ ! -f "$TENSILE_SH" ]; then
    echo "ERROR: Tensile.sh not found at $TENSILE_SH"
    echo "Build TensileLite first (see $tensilelite skill)"
    exit 1
fi

HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-6} "$TENSILE_SH" \
    "$SCRIPT_DIR/mxfp4_comparison.yaml" "$OUTPUT_DIR"

echo ""
echo "=== Results ==="
echo "Output directory: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "(no CSV files found)"
