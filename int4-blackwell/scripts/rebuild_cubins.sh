#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/torch-ext/int4_blackwell/cubin"
mkdir -p "$out"

nvcc -cubin -arch=sm_120a -O3 "$root/csrc/probe.cu" -o "$out/probe.cubin"
python3 "$root/scripts/patch_cubin.py" "$out/probe.cubin" "$out/probe_int4a.cubin" a
python3 "$root/scripts/patch_cubin.py" "$out/probe.cubin" "$out/probe_int4b.cubin" b
python3 "$root/scripts/patch_cubin.py" "$out/probe.cubin" "$out/probe_int4.cubin" ab
