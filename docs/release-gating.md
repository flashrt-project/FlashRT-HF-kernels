# Release Gating

This document is the packaging gate for the first FlashRT HF kernel batch. The
v1 release is a batch release, not a single-package release. Development should
finish the full v1 surface before spending time on full Nix/kernel-builder
builds.

## Gate Definitions

| Gate | Meaning | Required evidence |
| --- | --- | --- |
| G0 | Source slice selected | FlashRT provenance, generic API name, package placement |
| G1 | Local source-extension ready | Tensor binding compiles locally and correctness tests pass |
| G2 | HF builder shape ready | `build.toml`, `flake.nix`, `flake.lock`, and `check-config` pass |
| G3 | Package build ready | `kernel-builder build` and `check-builds` pass for target variants |
| G4 | Benchmark ready | Shape grid, tile policy, fair baseline, and reproducible results |
| G5 | Showcase ready | Downstream HF-style example or model-block benchmark |
| G6 | Community ready | Multi-hardware validation and maintenance ownership are clear |

## Current Package Status

| Package | Current gate | Ready claim | Blocking gaps |
| --- | --- | --- | --- |
| `flashrt-gemm-epilogues` | G5 partial | v1 FP8/GEMM epilogue block with RTX 5090 evidence and an HF-style block example | Run example against built/Hub package, refresh public benchmark runner output, multi-hardware validation |
| `flashrt-vla-video` | G5 partial | v1 VLA/video block with 19-40x local evidence and an HF-style block example | Full builder build, public benchmark runner, run example against built/Hub package, multi-hardware validation |
| `flashrt-nvfp4` | G2 | v1 Blackwell layout helper with benchmark and example paths | Full builder build, benchmark runner, fused GEMM epilogue surfaces, multi-hardware validation |
| `flashrt-smallm-gemm` | G1 | v1 Blackwell small-M slice; first SM120 NVFP4 W4A4 decode matvec compiles locally and passes deterministic correctness | Promote build file, benchmark grid, fair cuBLASLt/CUTLASS baseline, warpsplit/tiny FP8 slices |
| `flashrt-fused-quant` | G1 | First fused SiLU+NVFP4 quantization source slice compiles locally and passes split/merged byte parity | Promote build file, benchmark grid, memory-bandwidth benchmark, residual/RMSNorm slices |

## V1 Batch Blocks

The first public version has four equal blocks:

| Block | Packages | Minimum v1 surface |
| --- | --- | --- |
| FP8/GEMM epilogues | `flashrt-gemm-epilogues` | FP8 quant epilogues plus conservative BF16 GEMM epilogue wrappers |
| VLA/video post-processing | `flashrt-vla-video` | Q/K RMSNorm+RoPE/cache and packed-QKV split+norm+RoPE |
| Blackwell NVFP4/FP4 low-bit | `flashrt-nvfp4`, `flashrt-smallm-gemm` | NVFP4 scale layout helper plus at least one validated small-M/decode W4A4 path |
| Fused quantization | `flashrt-fused-quant` | SiLU/gate or norm/residual low-bit quantization with reference tests and bandwidth benchmark |

## V1 Batch Release Criteria

Before uploading the v1 batch to the Hub:

- `internal-docs/` and `internal-tests/` remain ignored and untracked.
- Every public package has no committed build output, stale `result` symlink, or
  `__pycache__`.
- Every v1 package has `README.md`, `CARD.md`, `VALIDATION.md`, tests,
  benchmarks, and examples or an explicit explanation for why no example is
  meaningful.
- `kernel-builder-docker check-config .` passes for every promoted v1 package.
- Every promoted v1 package completes `kernel-builder build` and
  `check-builds` for the intended torch/CUDA variants during the release
  validation window.
- Package `README.md`, `CARD.md`, and `VALIDATION.md` state the same hardware
  and API scope.
- Benchmarks include the shape grid from `docs/tile-and-shape-coverage.md`.
- Headline speedups use a fair baseline and name the GPU, driver, PyTorch, CUDA
  runtime, warmup count, and measured iterations.

## V1 Showcase Criteria

Before presenting v1 as a strong community update:

- Present the four blocks together. Do not frame this as one package plus
  extras.
- Include a model-block or HF-style call path, not only microbenchmarks.
- State exactly which shapes are fast and which shapes are just compatibility
  coverage.
- Keep SM120-only kernels labeled as SM120/SM120a until other hardware is
  measured.

## Development Loop Policy

Use source-extension tests, package tests, tile sweeps, and benchmarks as the
normal loop. Full Nix/kernel-builder builds are release validation jobs and
should be run after a batch of source and documentation changes has settled.
