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

Builder tooling:

- `kernel-builder` 0.16.0-dev0
- Docker/Nix build wrapper available under
  `/home/heima/suliang/PI/.hf-kernel-env/bin`

## Commands

From this package directory:

```bash
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker check-config .
```

Host-side source-extension correctness was validated with a package-local
`torch.utils.cpp_extension.load` smoke that compiled:

```text
torch-ext/torch_binding.cpp
csrc/nvfp4_sf_reshape_sm120.cu
```

## Correctness Smoke

The source-extension smoke compares byte-for-byte against the Python reference
layout transform.

Covered shapes:

| rows | D | swizzled bytes |
| ---: | ---: | ---: |
| 1 | 1024 | 8192 |
| 4 | 4096 | 32768 |
| 33 | 4096 | 32768 |
| 128 | 4096 | 32768 |
| 129 | 4096 | 65536 |
| 16 | 12288 | 98304 |

Package tests additionally cover layout boundary rows and invalid shape
rejection:

- rows: `1, 2, 31, 32, 33, 127, 128, 129`
- D: `1024, 4096, 12288, 16384`
- invalid rows: `0`
- invalid D: not divisible by 16
- caller-provided output tensor reuse

## Builder Results

- `check-config` passed for the promoted `build.toml`.
- Full `kernel-builder build` has not been completed for this package yet.

The first full build attempt generated `flake.lock` correctly, then entered a
large first-time Nix dependency build. That path is intentionally deferred to
the release-validation window instead of being used as the normal development
loop.

## Known Gaps

- Full `kernel-builder build` and `check-builds` are still pending.
- Hub benchmark runner has not been run for
  `benchmarks/benchmark_nvfp4_sf_reshape.py`.
- Runtime validation is currently RTX 5090 only.
- Current public API is a data-layout helper. Fused NVFP4 GEMM epilogues are
  not included in this buildable slice yet.
- The current CUDA implementation is Blackwell-oriented; keep SM120/SM120a
  scope explicit until other architectures are implemented and measured.
