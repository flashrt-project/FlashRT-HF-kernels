# Release Gating

This document is the packaging gate for the first FlashRT HF kernel batch. It
tracks what is ready for local development, Hub upload, public showcase, and
possible `kernels-community` promotion.

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
| `flashrt-gemm-epilogues` | G5 partial | First buildable package; FP8 quant epilogue headline on RTX 5090 with an HF-style block example | Torch 2.12 builder variants, run example against built/Hub package, multi-hardware validation |
| `flashrt-vla-video` | G5 partial | Strong VLA/video QKV post-processing showcase candidate with 19-40x local evidence and an HF-style block example | Full builder build, public benchmark runner, run example against built/Hub package, multi-hardware validation |
| `flashrt-nvfp4` | G2 | Buildable NVFP4 scale-factor layout helper | Full builder build, benchmark runner, fused GEMM epilogue surfaces, multi-hardware validation |
| `flashrt-smallm-gemm` | G0 | Decode-latency candidate | Tensor binding, correctness tests, tile sweep, fair cuBLASLt/CUTLASS baseline |
| `flashrt-fused-quant` | G0 | Shared fused quantization utility candidate | Tensor binding, correctness tests, tile sweep, memory-bandwidth benchmark |

## First Package Release Criteria

Before uploading the first package to the Hub:

- `internal-docs/` and `internal-tests/` remain ignored and untracked.
- Every public package has no committed build output or `__pycache__`.
- `kernel-builder-docker check-config .` passes for each promoted package.
- At least one promoted package completes `kernel-builder build` and
  `check-builds` for the intended torch/CUDA variants.
- Package `README.md`, `CARD.md`, and `VALIDATION.md` state the same hardware
  and API scope.
- Benchmarks include the shape grid from `docs/tile-and-shape-coverage.md`.
- Headline speedups use a fair baseline and name the GPU, driver, PyTorch, CUDA
  runtime, warmup count, and measured iterations.

## First Showcase Criteria

Before presenting this as a strong community update:

- Pick one headline package:
  `flashrt-vla-video` or the fused NVFP4 GEMM slice, not a collection of small
  helpers.
- Include a model-block or HF-style call path, not only microbenchmarks.
- State exactly which shapes are fast and which shapes are just compatibility
  coverage.
- Keep SM120-only kernels labeled as SM120/SM120a until other hardware is
  measured.

## Development Loop Policy

Use source-extension tests, package tests, tile sweeps, and benchmarks as the
normal loop. Full Nix/kernel-builder builds are release validation jobs and
should be run after a batch of source and documentation changes has settled.
