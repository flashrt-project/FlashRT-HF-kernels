# Validation: flashrt-nvfp4

Validated on June 2, 2026.

This file records package-level validation that is suitable to keep with the
public kernel package. FlashRT-internal parity notes, private model fixtures,
and machine-specific tuning logs belong under the ignored `internal-tests/`
directory.

## Current Scope

Buildable APIs:

- `nvfp4_sf_swizzled_bytes`
- `nvfp4_sf_linear_to_swizzled`

Planned but not yet buildable in this package:

- `nvfp4_linear_bias_gelu_fp4out_sm120`
- `nvfp4_linear_bias_gelu_bf16out_sm120`
- `nvfp4_linear_streamk_bias_bf16out_sm120`

## Environment

Runtime smoke environment:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 580.82.07
- PyTorch: 2.9.1+cu128
- CUDA runtime reported by PyTorch: 12.8
- CUDA capability: 12.0
- Build scope: CUDA 12.8+ and SM120

Builder tooling:

- `kernel-builder` 0.16.0-dev0
- Docker/Nix build wrapper available under
  `/home/heima/suliang/PI/.hf-kernel-env/bin`

## Commands

From this package directory:

```bash
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker check-config .
```

Host-side source-extension correctness was validated with:

```bash
python scripts/accuracy_sweep.py --backend source --mode full --package flashrt-nvfp4
```

Result: passed 13 checks.

The sweep compiles:

```text
torch-ext/torch_binding.cpp
csrc/nvfp4_sf_reshape_sm120.cu
```

## Correctness Sweep

The source-extension smoke compares byte-for-byte against the Python reference
layout transform.

Covered shapes:

| rows | D | swizzled bytes |
| ---: | ---: | ---: |
| 1 | 4096 | 32768 |
| 2 | 4096 | 32768 |
| 31 | 4096 | 32768 |
| 32 | 4096 | 32768 |
| 33 | 4096 | 32768 |
| 127 | 4096 | 32768 |
| 128 | 4096 | 32768 |
| 129 | 4096 | 65536 |
| 16 | 1024 | 8192 |
| 16 | 2048 | 16384 |
| 16 | 8192 | 65536 |
| 16 | 12288 | 98304 |
| 64 | 16384 | 131072 |

Package tests additionally cover layout boundary rows and invalid shape
rejection:

- rows: `1, 2, 31, 32, 33, 127, 128, 129`
- D: `1024, 4096, 12288, 16384`
- invalid rows: `0`
- invalid D: not divisible by 16
- caller-provided output tensor reuse

## Builder Results

- `check-config` passed for the promoted `build.toml`.
- `kernel-builder build --variant torch211-cxx11-cu128-x86_64-linux` passed.
- The copied `torch211-cxx11-cu128-x86_64-linux` artifact passed package tests,
  examples, installed accuracy sweep, and the local release-candidate benchmark
  runner.
- Full `kernel-builder build-and-copy` matrix has not been run yet.

## Known Gaps

- Official Hub `kernels benchmark` has not been run after upload.
- Runtime validation is currently RTX 5090 only.
- Current public API is a data-layout helper. Fused NVFP4 GEMM epilogues are
  not included in this buildable slice yet.
- The current CUDA implementation is declared CUDA 12.8+ SM120-only in
  `build.toml`; add a separate source path before making non-SM120 claims.
