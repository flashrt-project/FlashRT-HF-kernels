# Built Artifact Results

Validated on June 2, 2026 on NVIDIA GeForce RTX 5090.

This document records tests against copied `kernel-builder` artifacts, not
local source-extension builds.

## Release-Candidate Variant

Variant:

```text
torch211-cxx11-cu128-x86_64-linux
```

Artifact source:

- Built with `kernel-builder-docker build --variant torch211-cxx11-cu128-x86_64-linux`.
- Copied from the Docker container Nix store into each package's `build/`
  directory with `scripts/copy_docker_variant_artifacts.py`.

Runtime validation environment:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 580.82.07
- Torch inside HF testshell: 2.11.0+cu128
- CUDA runtime inside Torch: 12.8
- `LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}` was required inside the
  GPU-enabled Docker container so Torch could find `libcuda.so`.

Command:

```bash
PYTHONPATH=flashrt-gemm-epilogues/build/torch211-cxx11-cu128-x86_64-linux:flashrt-vla-video/build/torch211-cxx11-cu128-x86_64-linux:flashrt-nvfp4/build/torch211-cxx11-cu128-x86_64-linux:flashrt-smallm-gemm/build/torch211-cxx11-cu128-x86_64-linux:flashrt-fused-quant/build/torch211-cxx11-cu128-x86_64-linux \
  python scripts/accuracy_sweep.py --backend installed --mode full --package all --smallm-max-ulp 5 --quiet
```

Result:

```text
accuracy sweep passed: 324 checks
```

## Accuracy Notes

- `flashrt-gemm-epilogues`: FP8 quant epilogue byte/exact parity passed over
  the v1 grid.
- `flashrt-vla-video`: BF16 Q/K tolerance and V byte parity passed over the v1
  grid.
- `flashrt-nvfp4`: scale-factor swizzle byte parity passed.
- `flashrt-smallm-gemm`: release artifact BF16 matvec gate is
  `max_ulp <= 5`. The observed worst case is a near-cancellation random output
  at `K=4096,N=12288`: expected `2.953125`, got `3.03125`.
- `flashrt-fused-quant`: split and merged NVFP4 packed bytes and scale bytes
  passed byte parity over the v1 grid.

## Package Tests

Package tests were run separately per package to avoid `tests` package-name
collisions:

| Package | Result |
| --- | --- |
| `flashrt-gemm-epilogues` | 15 passed |
| `flashrt-vla-video` | 8 passed |
| `flashrt-nvfp4` | 18 passed |
| `flashrt-smallm-gemm` | 4 passed |
| `flashrt-fused-quant` | 5 passed |

## Examples

All package examples ran against the copied built artifacts. The HF-style
examples prefer `kernels.get_kernel` when available and fall back to local
artifact imports for release-candidate validation.

Representative output:

| Example | Result |
| --- | --- |
| `flashrt-gemm-epilogues/examples/fp8_quant_epilogue_block.py` | `M=64,N=4096`: 7.652 us vs 24.328 us, 3.18x |
| `flashrt-vla-video/examples/qkv_postprocess_block.py` | `T=256,H=24,D=128`: 8.214 us vs 141.332 us, 17.21x |
| `flashrt-nvfp4/examples/nvfp4_scale_factor_layout.py` | `rows=128,D=4096`: 3.400 us |
| `flashrt-smallm-gemm/examples/nvfp4_w4a4_decode_matvec.py` | output `(1024,)`, BF16 |
| `flashrt-fused-quant/examples/swiglu_nvfp4_quant_block.py` | split and merged packed/scales outputs produced |

## Benchmark Summary

Benchmarks used the local release-candidate runner:

```bash
python scripts/run_built_artifact_benchmarks.py \
  --package all --warmup 10 --iterations 50
```

The runner executes the public `kernels.benchmark.Benchmark` scripts against
the copied built artifacts. It does not replace the official Hub
`kernels benchmark` run after upload.

| Package | Built-artifact benchmark result |
| --- | --- |
| `flashrt-gemm-epilogues` | FP8 quant epilogues verified, 2.60x-4.08x vs PyTorch eager references; BF16 GEMM benchmark is latency-only |
| `flashrt-vla-video` | Q/K/QKV post-processing verified, 10.04x-29.33x vs PyTorch eager references |
| `flashrt-nvfp4` | scale-factor layout helper byte-verified, 67.22x-17408.41x vs Python layout reference |
| `flashrt-smallm-gemm` | W4A4 decode matvec verified, 5.87x-16.23x vs random/dequant PyTorch readability baseline |
| `flashrt-fused-quant` | split and merged fused quant latency grid completed; multi-output byte parity remains covered by accuracy sweep |

## Full Matrix Status

This is not a full HF matrix build. The package variant lists are:

- `flashrt-gemm-epilogues`, `flashrt-vla-video`: six CUDA x86_64 variants
  covering Torch 2.11/2.12 and CUDA 12.6/12.8/13.0/13.2 as applicable.
- `flashrt-nvfp4`, `flashrt-smallm-gemm`, `flashrt-fused-quant`: four CUDA
  x86_64 variants because they require CUDA 12.8+ and SM120.

The full `kernel-builder build-and-copy` matrix remains a release-window job.
After that matrix passes, run hardware validation on the other target machines
before widening public hardware claims. SM120 packages should stay labeled
CUDA 12.8+ SM120 until a non-SM120 source path is added.
