# FlashRT HF Kernels

Experimental Hugging Face Kernel Hub packaging for selected FlashRT kernels.

This repository is a distribution and integration layer, not a replacement for
the FlashRT serving runtime. FlashRT remains the upstream source of truth for
model pipelines, CUDA Graph orchestration, private pointer-based bindings, and
hardware-specific serving decisions. This repository exposes stable,
Tensor-based kernel APIs that can be built and loaded by the Hugging Face
`kernels` package.

For the complete model runtime, serving pipeline, and production FlashRT
frontends, see the upstream repository:
[LiangSu8899/FlashRT](https://github.com/LiangSu8899/FlashRT).

## Current Snapshot

This repository is a release-candidate integration layer for the first FlashRT
kernel batch. The packages are structured for Hugging Face `kernel-builder`,
with public Tensor-based APIs, package tests, examples, and benchmark scripts.

Current local status:

- All v1 packages pass configuration-level prebuild checks.
- Release-candidate full-matrix artifacts have been generated locally and pass
  ABI compatibility plus `get_kernel` load checks.
- The RTX 5090 installed-artifact correctness gate passes 345/345 checks for
  `torch211-cxx11-cu128`.
- Additional hardware validation is in progress.
- Final public artifacts should be regenerated from a clean upstream
  `kernel-builder` revision before upload.

Known builder issue: the current upstream builder path has a
`triton-3.7.0` fixed-output hash mismatch in the Torch 2.12 dependency path.
The FlashRT packages build successfully with a local builder-side hash
correction only; the kernel sources themselves are not involved in that
failure.

## Hub-Style Usage

After upload, packages are intended to be consumed through the Hugging Face
`kernels` API:

```python
from kernels import get_kernel

ops = get_kernel(
    "flashrt/flashrt-vla-video",
    version=1,
    trust_remote_code=True,
)

q, k = ops.qkv_split_norm_rope_bf16(
    packed_qkv,
    norm_q_weight,
    norm_k_weight,
    freqs_re,
    freqs_im,
    heads=24,
    head_dim=128,
)
```

The `flashrt/flashrt-*` namespace is the intended Hub namespace from each
package's `build.toml`. Users can load these names only after the corresponding
kernel repositories and artifacts are uploaded to the Hugging Face Hub. Until
then, this GitHub repository is a source and validation repository, not a live
Kernel Hub distribution.

## Goals

- Package the most reusable FlashRT kernels in the structure expected by
  `kernel-builder`.
- Keep public APIs generic and model-agnostic. Avoid model names in package
  names and exported functions unless a kernel is truly model-specific.
- Provide tests, benchmarks, and kernel cards before publishing to the Hub.
- Keep source synchronization from FlashRT explicit and reviewable.
- Make it easy to promote mature packages to `kernels-community` later.

## Showcase Strategy

The first public surface is a four-block v1 batch, not a single-package pilot:

- FP8/GEMM epilogues.
- VLA, vision, video, and diffusion post-processing primitives.
- Blackwell NVFP4/FP4 layout and low-bit GEMM/decode kernels.
- Fused activation, normalization, residual, and quantization kernels.

The bar for the v1 batch is higher than the bar for one buildable package:
correctness tests, strong microbenchmarks, shape constraints, hardware scope,
and downstream HF-style calling examples should all be documented before the
full builder release window.

## Performance Snapshot

Representative RTX 5090 source-extension results for the first two model-block
demos:

- `demos/wan-qkv-postprocess`: Wan/video-diffusion attention postprocess.
- `demos/pi05-groot-ffn-epilogue`: PI0.5/GROOT-shaped repeated FFN epilogue
  and activation-quant blocks.
- `flashrt-fp8-ffn/benchmarks`: full FP8 GELU MLP sublayers for PI0.5/GROOT
  shapes.

These numbers are math-equivalent comparisons against PyTorch eager and
`torch.compile` tensor code; they do not use cache reuse, sampling-step
reduction, distillation, or quality/performance trade-offs.

Wan/video snapshot uses long-token video/VLA shapes `T in {1024,2520,4096}`.

| Scope | Wan2.2 TI2V-5B vs eager | Wan2.2 TI2V-5B vs compile | Wan A14B family vs eager | Wan A14B family vs compile | What is measured |
| --- | ---: | ---: | ---: | ---: | --- |
| Q/K postprocess only | 17.12x-33.74x | 4.00x-4.66x | 17.15x-24.32x | 2.23x-5.06x | Packed QKV split, Q/K RMSNorm, RoPE. |
| Packed-QKV to attention output | 1.96x-2.36x | 1.06x-1.27x | 2.34x-2.83x | 1.09x-1.46x | Postprocess plus the same SDPA attention on both paths. |
| Self-attention sublayer | 1.41x-1.59x | 1.14x-1.35x | 1.25x-1.45x | 1.06x-1.10x | QKV projection, postprocess, attention, output projection. |

The self-attention sublayer rows are included as an attribution check, not as
the headline for this single kernel. QKV/O projection and SDPA dominate that
wider block, so the fused postprocess kernel is only a fraction of the measured
runtime.

PI0.5/GROOT FFN epilogue snapshot uses repeated model-shaped stacks. Exact
FP8 output matching is required for every row.

| Block | Shape | Layers | vs eager | vs compile | What is measured |
| --- | ---: | ---: | ---: | ---: | --- |
| PI0.5 vision FFN | `512 x 4304` | 27 | 4.16x | 1.33x | SigLIP FFN fc1 bias + GELU + FP8 cast. |
| PI0.5 encoder activation quant | `560 x 2048` | 18 | 6.49x | 2.09x | Encoder activation scale + FP8 cast. |
| GROOT ViT FFN | `512 x 4096` | 24 | 4.23x | 1.53x | ViT FFN fc1 bias + GELU + FP8 cast. |
| GROOT DeepStack merger | `128 x 4096` | 3 | 9.32x | 5.53x | DeepStack merger fc1 bias + GELU + FP8 cast. |
| GROOT VL self-attn FFN | `1024 x 8192` | 4 | 3.77x | 1.27x | Long-sequence VL self-attn FFN fc1 epilogue. |

This PI0.5/GROOT demo is still a reusable model-block benchmark, not a full
model generation throughput claim. A full end-to-end PI0.5/GROOT demo should
ship after the FP8 GEMM/FFN or megakernel path is exported as a Hub-loadable
kernel package.

Full FP8 FFN snapshot uses `flashrt-fp8-ffn`, which exports a Hub-loadable
Tensor API for `FP8 up GEMM -> bias/GELU -> FP8 quant -> FP8 down GEMM -> bias`.

| Block | Shape | Layers | vs eager | vs compile | Precision gate |
| --- | ---: | ---: | ---: | ---: | --- |
| PI0.5 decoder FFN | `10,1024,4096,1024` | 18 | 6.61x | 3.83x | PASS, p99_abs=0 |
| PI0.5 vision FFN | `512,1152,4304,1152` | 27 | 6.42x | 4.95x | PASS, p99_abs=0 |
| GROOT ViT FFN | `512,1024,4096,1024` | 24 | 7.03x | 5.45x | PASS, p99_abs=0 |
| GROOT VL self-attn FFN | `1024,2048,8192,2048` | 4 | 6.66x | 5.62x | PASS, p99_abs=0 |

This is the stronger first-update story than epilogue-only measurement: a full
math-equivalent FFN sublayer remains several times faster than both eager and
`torch.compile` tensor references on RTX 5090.

The expanded source-extension sweep also covers PI0.5 decoder chunk sizes,
PI0.5 vision 1/2/3-view shapes, GROOT ViT 1/2/4-view shapes, GROOT DeepStack,
GROOT VL self-attn sequence lengths up to 2520, and the GROOT action DiT GELU
FFN shape. All rows pass the p99_abs/p99_rel precision gate; built-artifact and
multi-hardware rows remain pending until the full release build is regenerated.

The full FlashRT serving stack combines multiple math-equivalent kernels across
attention, FFN, epilogues, quantization/layout, residual paths, and serving
orchestration. Those gains are designed to stack with community techniques such
as distillation, cache reuse, or fewer denoising steps rather than replace them.

## Package Plan

| Package | Stage | Purpose |
| --- | --- | --- |
| `flashrt-gemm-epilogues` | V1 block | FP8 quant epilogue helpers plus selected BF16 GEMM epilogues. |
| `flashrt-fp8-ffn` | V1 block | Hub-loadable FP8 GEMM and full GELU MLP/FFN sublayers for VLA/VLM shapes. |
| `flashrt-vla-video` | V1 block | VLA, vision, video, and diffusion kernels that are reusable outside the FlashRT serving engine. |
| `flashrt-nvfp4` | V1 block | NVFP4/FP4 data movement, SFA/SFB layout, low-bit GEMM, and fused epilogues. |
| `flashrt-smallm-gemm` | V1 block | Decode-oriented small-M GEMM/GEMV and split-K primitives with generic shape-specialized APIs. |
| `flashrt-fused-quant` | V1 block | Memory-bound fusion kernels: norm, residual, activation, and quantization. |

## Repository Status

All v1 packages have promoted `build.toml`, `flake.nix`, and `flake.lock`
files and pass configuration-level prebuild checks. Some packages are still
draft at the source or benchmark-evidence level; package-specific status is
tracked in `docs/release-gating.md`.

Do not upload a package to the Hub until the validation checklist in
`docs/release-gating.md` passes for the full v1 batch.

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
`docs/tile-and-shape-coverage.md`. Packaging gates and release blockers are
tracked in `docs/release-gating.md`. The four-block v1 release plan is tracked
in `docs/v1-batch-plan.md`. Benchmark baseline rules are defined in
`docs/benchmark-baselines.md`, and package-level comparison requirements are
defined in `docs/kernel-comparison-matrix.md`. Use
`docs/benchmark-results-template.md` when refreshing public or hardware-specific
benchmark reports. The full build procedure is in `docs/release-runbook.md`.
Run `python scripts/prebuild_check.py --check-config` before starting a full
release build window.

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
