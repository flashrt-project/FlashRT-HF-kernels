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
- Build scope: CUDA 12.8+ and SM120

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

## Source Accuracy Sweep

The local source-extension sweep compiled:

```text
torch-ext/torch_binding.cpp
csrc/fp4_w4a4_matvec_sm120.cu
```

The sweep uses the Hugging Face kernel-builder `registration.h` template
include path locally. Full HF builder packaging has not been run yet.

Command:

```bash
python scripts/accuracy_sweep.py --backend source --mode full --package flashrt-smallm-gemm
```

Result: passed 12 checks.

Correctness was checked with deterministic packed FP4 inputs and random
packed/dequantized references:

- A and B packed bytes: `0x11`, meaning two FP4 `0.5` values per byte.
- SFA and SFB scale bytes: `0x38`, meaning UE4M3 `1.0`.
- Expected output: `K * 0.25 * alpha`.
- `alpha = 0.5`.

Deterministic constant-input results:

| K | N | Expected | Max error |
| ---: | ---: | ---: | ---: |
| 4096 | 16 | 512.0 | 0.0 |
| 12288 | 16 | 1536.0 | 0.0 |

Full random/dequantized source grid:

- K: `4096,12288`
- N: `1024,4096,12288`
- alpha: `0.5`
- Source sweep measured BF16 output: `max_ulp <= 4`
- Built artifact release gate: `max_ulp <= 5`

Worst recorded random source-sweep cases:

| K | N | Max abs | Max rel | Max ULP | Note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 12288 | 0.0625 | 0.0210526 | 4 | near-cancellation output |
| 12288 | 4096 | 4096 | 0.00546448 | 1 | large BF16 output, one ULP |
| 12288 | 12288 | 256 | 0.00877193 | 2 | large BF16 output |

## Built Artifact Accuracy

The `torch211-cxx11-cu128-x86_64-linux` artifact was validated inside an HF
Torch 2.11.0+cu128 testshell on RTX 5090.

Command:

```bash
python scripts/accuracy_sweep.py --backend installed --mode full \
  --package flashrt-smallm-gemm --smallm-max-ulp 5
```

Result: passed 12 checks.

Worst built-artifact case:

| K | N | Max abs | Max rel | Max ULP | Note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 12288 | 0.078125 | 0.026455 | 5 | near-cancellation BF16 output, got `3.03125`, expected `2.953125` |

## Known Gaps

- `build.toml`, `flake.nix`, and `flake.lock` are present.
- `/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker
  check-config .` passed for this package.
- `kernel-builder build --variant torch211-cxx11-cu128-x86_64-linux` passed
  for this package.
- Full `kernel-builder build-and-copy` matrix has not been run for this
  package.
- Public benchmark scripts are present, but built-artifact benchmark results
  and fair CUTLASS/cuBLASLt baselines are still pending.
- Runtime validation is currently RTX 5090 only; this v1 source slice is
  declared SM120-only in `build.toml`.
- Warpsplit small-M and tiny FP8 source slices are not yet exposed.
