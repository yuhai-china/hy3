#!/bin/bash
# run_metal.sh - build (if needed) and run hy3 with the Metal backend on
# macOS / Apple Silicon.
#
# Usage:
#   ./run_metal.sh -m /path/to/hy3_q4k_mixed.gguf -p "11+22+33=?" -n 32 -temp 0
#   ./run_metal.sh -m /path/to/model.gguf                # interactive mode
#
# Any extra arguments are passed straight through to hy3-cli (see
# ./hy3-cli -h). This script just makes sure the Metal build exists and
# adds --metal automatically.
#
# Env vars:
#   HY3_METAL_CTX_TOKENS   KV cache context size in tokens (default 8192).
#   HY3_METAL_SHADER       Path to hy3.metal if it's not next to the binary
#                          (the Makefile bakes in an absolute path at build
#                          time, so this is normally only needed if you
#                          moved the binary after building).
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "run_metal.sh: this script is for macOS; on Linux use --gpu-layers (CUDA) instead." >&2
    exit 1
fi

if ! xcrun -sdk macosx --show-sdk-path >/dev/null 2>&1; then
    echo "run_metal.sh: Xcode command line tools not found (xcode-select --install)" >&2
    exit 1
fi

if [[ ! -x ./hy3-cli ]] || [[ hy3_metal.m -nt ./hy3-cli ]] || [[ hy3.c -nt ./hy3-cli ]] || [[ hy3.metal -nt ./hy3-cli ]]; then
    echo "run_metal.sh: building (make clean && make)..." >&2
    make clean
    make -j"$(sysctl -n hw.ncpu)"
fi

if ! file ./hy3-cli | grep -q "Mach-O"; then
    echo "run_metal.sh: ./hy3-cli doesn't look like a macOS binary -- rebuild needed?" >&2
    exit 1
fi

echo "run_metal.sh: launching with --metal (all 80 layers Metal-resident, zero-copy from the mmap'd GGUF)" >&2
exec ./hy3-cli --metal "$@"
