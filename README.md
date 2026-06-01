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

## Package Plan

| Package | Stage | Purpose |
| --- | --- | --- |
| `flashrt-gemm-epilogues` | First | GEMM plus fused epilogues such as bias, activation, residual, dequant, and quantized output. |
| `flashrt-fused-quant` | First | Memory-bound fusion kernels: norm, residual, activation, RoPE/QKV post-processing, and quantization. |
| `flashrt-nvfp4` | First/second | NVFP4 data movement and layout primitives: quantize, dequantize, SFA/SFB layout, and low-bit helpers. |
| `flashrt-smallm-gemm` | Second | Decode-oriented small-M GEMM/GEMV and split-K primitives with generic shape-specialized APIs. |
| `flashrt-vla-video` | Second | VLA, vision, video, and diffusion kernels that are reusable outside the FlashRT serving engine. |

## Repository Status

This is a phase-0 scaffold. The package directories intentionally use
`build.toml.draft` rather than `build.toml` until a package has real synced
source files, Tensor-based bindings, tests, and a passing local build.

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
7. Add a benchmark for representative generic shapes and one FlashRT-real
   shape family.
8. Promote `build.toml.draft` to `build.toml`.
9. Run `kernel-builder build`, `kernel-builder check-abi`, and package tests.

## References

- Hugging Face kernels docs: https://huggingface.co/docs/kernels/index
- Writing Hub kernels: https://huggingface.co/docs/kernels/builder/writing-kernels
- Kernel requirements: https://huggingface.co/docs/kernels/kernel-requirements
- Community examples: https://github.com/huggingface/kernels-community

See `CONTRIBUTING.md` for public/internal content boundaries.
