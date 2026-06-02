# Validation: flashrt-smallm-gemm

Validated on June 2, 2026.

This package is still a draft package. The current validation record only
covers the first source slice:

- `nvfp4_w4a4_decode_matvec_bf16out`

## Environment

Runtime smoke environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: 2.9.1+cu128
- CUDA capability: 12.0

## Current Scope

The first draft wrapper exposes the SM120 NVFP4 W4A4 M=1 decode matvec with
BF16 output.

Supported by the synced source slice:

- `M = 1`
- `K in {4096, 12288}`
- `N > 0`
- packed FP4 activation bytes with shape `(K / 2,)` or `(1, K / 2)`
- packed FP4 weight bytes with shape `(N, K / 2)`
- CUTLASS Sm1xx swizzled UE4M3 scale-factor buffers for activation and weight

## Local Source-Extension Smoke

The local source-extension smoke compiled:

```text
torch-ext/torch_binding.cpp
csrc/fp4_w4a4_matvec_sm120.cu
```

The smoke uses the Hugging Face kernel-builder `registration.h` template
include path locally. Full HF builder packaging has not been run yet.

Correctness was checked with deterministic packed FP4 inputs:

- A and B packed bytes: `0x11`, meaning two FP4 `0.5` values per byte.
- SFA and SFB scale bytes: `0x38`, meaning UE4M3 `1.0`.
- Expected output: `K * 0.25 * alpha`.
- `alpha = 0.5`.

Results:

| K | N | Expected | Max error |
| ---: | ---: | ---: | ---: |
| 4096 | 16 | 512.0 | 0.0 |
| 12288 | 16 | 1536.0 | 0.0 |

## Known Gaps

- `build.toml`, `flake.nix`, and `flake.lock` are present.
- `/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker
  check-config .` passed for this package.
- Full `kernel-builder build` has not been run for this package.
- Public benchmark scripts are present, but built-artifact benchmark results
  and fair CUTLASS/cuBLASLt baselines are still pending.
- Runtime validation is currently RTX 5090 only.
- Warpsplit small-M and tiny FP8 source slices are not yet exposed.
