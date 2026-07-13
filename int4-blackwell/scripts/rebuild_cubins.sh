#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/torch-ext/int4_blackwell/cubin"

for arch in 120 121; do
  arch_out="$out/sm$arch"
  mkdir -p "$arch_out"
  nvcc -cubin -arch="sm_${arch}a" -O3 \
    "$root/csrc/probe.cu" -o "$arch_out/probe.cubin"
  python3 "$root/scripts/patch_cubin.py" \
    "$arch_out/probe.cubin" "$arch_out/probe_int4a.cubin" a
  python3 "$root/scripts/patch_cubin.py" \
    "$arch_out/probe.cubin" "$arch_out/probe_int4b.cubin" b
  python3 "$root/scripts/patch_cubin.py" \
    "$arch_out/probe.cubin" "$arch_out/probe_int4.cubin" ab
done
