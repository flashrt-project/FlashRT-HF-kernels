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
| `flashrt-gemm-epilogues` | G5 RC | v1 FP8/GEMM epilogue block with torch211/cu128 built-artifact correctness, package tests, examples, and verified public benchmark rows on RTX 5090 | Multi-hardware validation and full HF matrix |
| `flashrt-fp8-ffn` | G5 RC | FP8 GEMM and full GELU MLP/FFN torch211/cu128 built-artifact correctness passes on RTX 5090; PI0.5/GROOT FFN benchmark shows 3.83-5.57x vs `torch.compile` source references | Multi-hardware validation, full HF matrix, decide whether CUTLASS/megakernel replaces cuBLASLt path for SM120 headline |
| `flashrt-vla-video` | G5 RC | VLA/video Q/K and QKV post-processing torch211/cu128 built-artifact correctness, package tests, examples, and verified public benchmark rows pass on RTX 5090 | Multi-hardware validation and full HF matrix |
| `flashrt-nvfp4` | G5 RC | v1 Blackwell layout helper torch211/cu128 built-artifact correctness, package tests, examples, and verified public benchmark rows pass on RTX 5090 | Multi-hardware validation, full HF matrix, broader fused GEMM epilogue surfaces |
| `flashrt-smallm-gemm` | G5 RC | SM120 NVFP4 W4A4 decode matvec torch211/cu128 built-artifact correctness, package tests, examples, and verified public benchmark rows pass on RTX 5090 | Multi-hardware validation, full HF matrix, fair cuBLASLt/CUTLASS baseline, warpsplit/tiny FP8 slices |
| `flashrt-fused-quant` | G5 RC | Split and merged SiLU+NVFP4 quantization torch211/cu128 built-artifact correctness, package tests, examples, and benchmark latency grid complete on RTX 5090 | Multi-hardware validation, full HF matrix, memory-bandwidth benchmark, residual/RMSNorm slices |

## V1 Batch Blocks

The first public version has four equal blocks:

| Block | Packages | Minimum v1 surface |
| --- | --- | --- |
| FP8/GEMM and FFN | `flashrt-gemm-epilogues`, `flashrt-fp8-ffn` | FP8 quant epilogues plus full FP8 GELU MLP/FFN sublayers |
| VLA/video post-processing | `flashrt-vla-video` | Q/K RMSNorm+RoPE/cache and packed-QKV split+norm+RoPE |
| Blackwell NVFP4/FP4 low-bit | `flashrt-nvfp4`, `flashrt-smallm-gemm` | NVFP4 scale layout helper plus at least one validated small-M/decode W4A4 path |
| Fused quantization | `flashrt-fused-quant` | SiLU/gate or norm/residual low-bit quantization with reference tests and bandwidth benchmark |

## V1 Batch Release Criteria

Before uploading the v1 batch to the Hub:

- `python scripts/accuracy_sweep.py --backend source --mode full --package all`
  passes. This is the first gate; do not start the full build window before it.
- `python scripts/correctness_audit.py` passes.
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
- The selected release-candidate variant is copied into package `build/`
  directories and passes installed-backend accuracy sweep.
- The full HF matrix uses `kernel-builder build-and-copy`; it is a separate
  release-window job from the single-variant release-candidate build.
- The release window follows `docs/release-runbook.md`.
- Package `README.md`, `CARD.md`, and `VALIDATION.md` state the same hardware
  and API scope.
- `python scripts/prebuild_check.py --check-config` passes before starting the
  full build window.
- Benchmarks include the shape grid from `docs/tile-and-shape-coverage.md`.
- Benchmark baselines follow `docs/benchmark-baselines.md`.
- Package-level comparison coverage follows
  `docs/kernel-comparison-matrix.md`; rows that only beat PyTorch eager are not
  promoted to headline status unless the matrix explicitly allows it.
- Headline speedups use a fair baseline and name the GPU, driver, PyTorch, CUDA
  runtime, warmup count, and measured iterations.

## V1 Showcase Criteria

Before presenting v1 as a strong community update:

- Present the four blocks together. Do not frame this as one package plus
  extras.
- Include a model-block or HF-style call path, not only microbenchmarks.
- State exactly which shapes are fast and which shapes are just compatibility
  coverage.
- Keep SM120-only kernels labeled as CUDA 12.8+ SM120 until a non-SM120 source
  path is added.

## Development Loop Policy

Use source-extension tests, package tests, tile sweeps, and benchmarks as the
normal loop. Full Nix/kernel-builder builds are release validation jobs and
should be run after a batch of source and documentation changes has settled.
Benchmark speedups are recorded only after the corresponding correctness gate
passes.
