# FlashRT HF Kernels

Experimental Hugging Face Kernel Hub packaging for selected FlashRT kernels.

This repository is a distribution and integration layer, not a replacement for
the FlashRT serving runtime. FlashRT remains the upstream source of truth for
model pipelines, CUDA Graph orchestration, private pointer-based bindings, and
hardware-specific serving decisions. This repository exposes stable,
Tensor-based kernel APIs that can be built and loaded by the Hugging Face
`kernels` package.

## Goals

- Package the most reusable FlashRT kernels in the structure expected by
  `kernel-builder`.
- Keep public APIs generic and model-agnostic. Avoid model names in package
  names and exported functions unless a kernel is truly model-specific.
- Provide tests, benchmarks, and kernel cards before publishing to the Hub.
- Keep source synchronization from FlashRT explicit and reviewable.
- Make it easy to promote mature packages to `kernels-community` later.

## Showcase Strategy

The first public surface is intentionally narrow: use `flashrt-gemm-epilogues`
to prove the Hugging Face package format, then make FP8 quantization epilogues
the headline. These kernels are easy to call from Python, easy to benchmark
against PyTorch, and show clear launch and bandwidth wins without depending on
FlashRT serving internals.

The next showcase should not be another generic wrapper. It should target a
visible ecosystem gap where FlashRT has unusually strong kernels:

- VLA, vision, video, and diffusion primitives with clear model-level examples.
- NVFP4/FP4 Blackwell kernels with fused quantization, SFA/SFB layout, and GEMM
  epilogues.
- Decode-oriented small-M GEMM/GEMV kernels where latency dominates.

The bar for a showcase package is higher than the bar for a buildable package:
correctness tests, strong microbenchmarks, shape constraints, hardware scope,
and at least one downstream HF-style calling example should all be documented.

## Package Plan

| Package | Stage | Purpose |
| --- | --- | --- |
| `flashrt-gemm-epilogues` | First package | FP8 quant epilogue helpers plus selected BF16 GEMM epilogues. |
| `flashrt-vla-video` | First showcase candidate | VLA, vision, video, and diffusion kernels that are reusable outside the FlashRT serving engine. |
| `flashrt-nvfp4` | First showcase candidate | NVFP4/FP4 data movement, SFA/SFB layout, low-bit GEMM, and fused epilogues. |
| `flashrt-smallm-gemm` | Second showcase candidate | Decode-oriented small-M GEMM/GEMV and split-K primitives with generic shape-specialized APIs. |
| `flashrt-fused-quant` | Shared utility package | Memory-bound fusion kernels: norm, residual, activation, RoPE/QKV post-processing, and quantization. |

## Repository Status

`flashrt-gemm-epilogues` and `flashrt-vla-video` are the first buildable
packages. The remaining draft package directories now carry concrete first
kernel selections, but intentionally keep `build.toml.draft` until the selected
source slice has Tensor-based bindings, tests, benchmarks, and a passing local
build.

Do not upload a package to the Hub until its draft build file has been promoted
to `build.toml` and the validation checklist in `docs/validation.md` passes.

## Public vs Internal Content

This repository separates Hub-facing package material from local planning and
FlashRT-dependent validation:

- `docs/`: public repository documentation that is suitable for the eventual
  kernel repository.
- `internal-docs/`: local planning notes, package sequencing, and design
  questions that are useful for FlashRT maintainers but do not need to ship
  with a clean public package.
- `internal-tests/`: local tests that may depend on `../official/FlashRT`,
  real model fixtures, private benchmarks, or hardware-specific environments.
  Hub-compatible tests belong in each package's `tests/` directory.

The first-batch tuning grid is documented in
`docs/tile-and-shape-coverage.md`.

## Expected Layout Per Package

```text
flashrt-<area>/
  build.toml              # promoted from build.toml.draft when buildable
  CARD.md
  README.md
  flake.nix
  csrc/
  torch-ext/
    <python_package>/
      __init__.py
    torch_binding.cpp
    torch_binding.h
  tests/
  benchmarks/
  scripts/
```

The public Python package under `torch-ext/` should export Tensor-based
functions. It should not expose FlashRT internal `uintptr_t` or caller-owned
stream APIs.

## Initial Development Loop

1. Pick one package.
2. Sync only the required FlashRT source files into that package.
3. Write `torch-ext/torch_binding.cpp` with `TORCH_LIBRARY_EXPAND`.
4. Write a small Python wrapper that imports `ops` from `._ops`.
5. Add Hub-compatible correctness tests against PyTorch reference output.
6. Add internal FlashRT parity tests under `internal-tests/` when needed.
7. Add benchmarks for representative generic shapes and at least one
   FlashRT-real shape family.
8. Promote `build.toml.draft` to `build.toml`.
9. Run local source-extension tests and benchmarks as the regular development
   loop.
10. Run full `kernel-builder` builds for release validation, not for every
    small source edit.

## References

- Hugging Face kernels docs: https://huggingface.co/docs/kernels/index
- Writing Hub kernels: https://huggingface.co/docs/kernels/builder/writing-kernels
- Kernel requirements: https://huggingface.co/docs/kernels/kernel-requirements
- Community examples: https://github.com/huggingface/kernels-community

See `CONTRIBUTING.md` for public/internal content boundaries.
