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
- Version 1 packages are published under the `flashrt` Kernel Hub namespace.
- Additional hardware validation is in progress.
- Final public hardware claims should wait for the corresponding hardware rows
  in the validation matrix.

## Hub-Style Usage

The v1 packages are published under the `flashrt` Hugging Face Kernel Hub
namespace and can be consumed through the Hugging Face `kernels` API:

Start here:

- `docs/usage.md`: package map, model integration rules, and copy-pasteable
  usage snippets.
- Package cards: each `flashrt-*/CARD.md` explains what that Hub package
  contains and where it should be used.
- `examples/`: runnable top-level examples for direct Hub loading and FFN
  replacement.

For the shortest runnable examples:

- `examples/minimal_fp8_ffn.py`: import one Hub kernel and call it directly.
- `examples/replace_torch_ffn.py`: replace a PyTorch
  `Linear -> GELU(tanh) -> Linear` FFN with FlashRT FP8 kernels.

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

The same Hub-loaded wrappers are registered with fake/meta ops for
`torch.compile` tracing:

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/flashrt-fp8-ffn", version=1, trust_remote_code=True)
compiled_mlp = torch.compile(ops.fp8_gelu_mlp_bf16, fullgraph=True)
y = compiled_mlp(x_fp8, up_w_fp8, up_b, down_w_fp8, down_b, x_s, up_s, h_s, d_s)
```

Run the smoke check with:

```bash
python scripts/torch_compile_smoke.py --version 1
```

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
- `fp8-gemm/benchmarks`: native Blackwell FP8 decode GEMV and small-M GEMM
  rows for low-latency `Linear` replacements.
- `fp4-fused-ops/benchmarks`: native Blackwell FP16-to-NVFP4 producer and
  FP4-to-FP4 combiner rows for keeping low-bit runtime islands continuous.
- `fp4-gemm/benchmarks`: native Blackwell NVFP4 W4A16 GEMM rows with BF16
  output and schedule-specific validation.
- `demos/pi05-hf-runtime`: HF Kernel Hub runtime-overhead prototype with
  preallocated buffers and CUDA Graph replay for PI0.5/GROOT-shaped FFN chains.
- `demos/runtime-demo`: multi-package PI0.5-shaped runtime prototype using
  Hub-loaded kernels, persistent buffers, and CUDA Graph replay.

These numbers are math-equivalent comparisons against validated PyTorch
references. `torch.compile` speedups are shown only when the compiled reference
is verified equivalent to the eager reference; otherwise the compiled baseline
is marked unsupported. The measurements do not use cache reuse, sampling-step
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

| Block | Shape | Layers | vs eager | vs compile-stable reference | Precision gate |
| --- | ---: | ---: | ---: | ---: | --- |
| PI0.5 decoder FFN | `10,1024,4096,1024` | 18 | 6.61x | 6.10x | PASS, p99_abs=0 |
| PI0.5 vision FFN | `512,1152,4304,1152` | 27 | 6.49x | 6.04x | PASS, p99_abs=0 |
| GROOT ViT FFN | `512,1024,4096,1024` | 24 | 7.19x | 6.66x | PASS, p99_abs=0 |
| GROOT VL self-attn FFN | `1024,2048,8192,2048` | 4 | 6.51x | 5.89x | PASS, p99_abs=0 |

This is the stronger first-update story than epilogue-only measurement: a full
math-equivalent FFN sublayer remains several times faster than both the eager
PyTorch reference and the compile-stable `torch.compile` reference on RTX 5090.

The compile-stable reference intentionally graph-breaks the numerically
sensitive `GELU -> FP8 requant` and final BF16 bias/cast boundaries, while
leaving the FP8 dequant GEMMs in the compiled graph. This is required because a
raw default-Inductor compile of the whole FP8 fake-quant chain is not
bit-equivalent to eager at the FP8 rounding boundary. The benchmark verifies
compiled-reference output against eager output before reporting `vs compile`.

The expanded source-extension sweep also covers PI0.5 decoder chunk sizes,
PI0.5 vision 1/2/3-view shapes, GROOT ViT 1/2/4-view shapes, GROOT DeepStack,
GROOT VL self-attn sequence lengths up to 2520, and the GROOT action DiT GELU
FFN shape. All rows pass the p99_abs/p99_rel precision gate; built-artifact and
multi-hardware rows remain pending until the full release build is regenerated.

Native FP8 GEMM snapshot uses `fp8-gemm`, which exports
`fp8_linear_bf16` and `fp8_linear_residual_bf16` for FP8 E4M3 inputs and BF16
outputs. Public v1 scope is Blackwell `sm_120a`, `M=1` decode and
`2 <= M <= 64` small-M rows.

| Shape | Tile | vs eager | vs compile | Precision gate |
| --- | --- | ---: | ---: | --- |
| `M=1,K=4096,N=2048` | `gemv_fp8_m1_w4` | 5.30x | 6.74x | PASS, p99_abs=0 |
| `M=1,K=4096,N=8192` | `gemv_fp8_m1_w8` | 15.78x | 15.16x | PASS, p99_abs=0 |
| `M=16,K=4096,N=4096` | `ld_fp8_gemm_16x128x256_w4` | 7.38x | 6.68x | PASS, p99_abs=0 |
| `M=32,K=4096,N=8192` | `ld_fp8_gemm_32x128x256_w4` | 8.90x | 8.38x | PASS, p99_abs=0 |
| `M=64,K=512,N=1024` | `ld_fp8_gemm_64x128x256_w4` | 2.19x | 6.05x | PASS, p99_abs=0 |

PI0.5 HF-kernel runtime milestone:

- Real LIBERO rollout frame -> normalized image/prompt/state/noise bundle.
- HF-kernel SigLIP vision/projector -> HF-kernel Gemma encoder -> HF-kernel
  PI0.5 decoder -> 10-step denoise -> action.
- Timed hot path has `torch_gaps=[]` and CUDA Graph replay enabled.
- RTX 5090 graph latency: `~21.6 ms` (`~46.3 Hz`, `11.9x` over the baseline) —
  default path runs QKV/O/vision projection GEMMs in FP8 (published Hub kernels
  only), action `cosine ~0.99986` vs official FlashRT. `--no-fp8-projections`
  gives the BF16-projection path (`~22.5 ms`, `cosine ~0.99996`).
- Current conservative OpenPI/PyTorch BF16 first-call baseline:
  `257.078 ms` (`3.89 Hz`).
- Action correctness vs HF reference: `p99_abs=0.007812`,
  `cosine=0.999965`.
- Action correctness vs official FlashRT decoder output:
  `p99_abs=0.011719`, `cosine=0.999947`.

The OpenPI/PyTorch baseline above is a first-call model path baseline. The
complete OpenPI policy wrapper, including input preprocessing and observation
capture, should be reported as a separate future benchmark.

This composed Hub-kernel runtime and the bare-metal FlashRT runtime share the
same white-box philosophy but are two ways of using the same kernels: a
portable one-kernel-per-operation composition here, versus a fully-fused
bare-metal path in FlashRT. On the same model and GPU FlashRT runs in roughly
`18.7 ms`; the composed path is `~21.6 ms` with `--fp8-projections`, within
about 10–15%. The difference comes mainly from attention implementation and
epilogue/quantization fusion. The same Hub kernels driven with deeper FP8
quantization close most of that margin losslessly (`cosine ~0.99986`),
confirming the kernels are equivalent and the residual is the inherent
advantage of a fully-fused runtime. See
`demos/runtime-demo/README.md` for the full comparison.

Second-batch VLA/runtime packages target the model-demo hot path:

- `flashrt-fp8-swiglu-ffn`: true SwiGLU package for Gemma-style FFN islands.
- `flashrt-residual-norm-quant`: residual/RMSNorm/static-FP8 runtime glue for
  feeding adjacent FP8 blocks without returning to PyTorch ops.
- `flashrt-qkv-cache-rope`: packed-QKV split, Q/K RMSNorm, and RoPE staging for
  VLA/VLM/video attention inputs, plus decode Q staging and KV cache-write.
- `flashrt-vla-residual-gates`: video/action/und joint gated residual updates
  for VLA block glue.
- `flashrt-adaptive-norms`: AdaRMSNorm/style modulation and fused
  residual/AdaRMSNorm/static-FP8 output for DiT/VLA/world-model blocks.
- `flashrt-spatiotemporal-layout`: NCDHW/BLC layout, temporal unshuffle,
  channel-bias, and short-cache helpers for VLA/video/world-model pipelines.
- `vl-transformer-primitives`: Q/K norm + RoPE + KV-write staging and vision
  token pooling primitives for VLM transformer blocks.
- `linear-attention-primitives`: small-M BF16 linear, QKV broadcast split,
  partial RoPE, and gated-delta preparation primitives for linear-attention
  transformer blocks.
- `diffusion-step-ops`: diffusion scheduler, CFG, first-frame forcing, and
  decode-postprocess CUDA helpers for diffusion/video runtime glue.
- `turboquant-kv`: TurboQuant-style KV unpack/combine helpers for
  serving/cache-compression demos.
- `world-model-conv`: Blackwell FP8 3D causal conv primitive for
  video/world-model/VAE-style blocks.

```text
FP8 input -> FP8 gate/up GEMM -> SiLU(gate) * up -> FP8 requant -> FP8 down GEMM -> BF16 output
BF16 residual/x -> residual add -> RMSNorm -> static-scale FP8 E4M3 activation
packed QKV -> split Q/K -> RMSNorm Q/K -> RoPE Q/K
decode Q/K/V -> RMSNorm Q/K -> rotate-half RoPE Q/K -> Q stage / KV cache write
video/action/und residuals -> gated residual updates -> BF16 segment outputs
style -> AdaRMSNorm/style gate -> BF16 or static-FP8 activation
NCDHW latent -> BLC tokens / temporal unshuffle / channel-bias / cache update
```

Current RTX 5090 source-extension results cover PI0.5 decoder/vision,
GROOT/VL, action/DiT-shaped FFN rows, and video prefill rows. The strict
SwiGLU/GeGLU package gate compares the fused API against staged FlashRT
primitives and passes with `staged_p99=0` for all rows; the residual/norm,
QKV/RoPE, residual-gates, adaptive-norms, and spatiotemporal-layout packages
have source correctness and config checks in place. Final public package
claims should use the corresponding built-artifact and multi-hardware rows.

The full FlashRT serving stack combines multiple math-equivalent kernels across
attention, FFN, epilogues, quantization/layout, residual paths, and serving
orchestration. Those gains are designed to stack with community techniques such
as distillation, cache reuse, or fewer denoising steps rather than replace them.

## Package Plan

| Package | Stage | Purpose |
| --- | --- | --- |
| `flashrt-gemm-epilogues` | V1 block | FP8 quant epilogue helpers plus selected BF16 GEMM epilogues. |
| `flashrt-fp8-ffn` | V1 block | Hub-loadable FP8 GEMM and full GELU MLP/FFN sublayers for VLA/VLM shapes. |
| `flashrt-fp8-swiglu-ffn` | Runtime package | True SwiGLU/GeGLU FP8 FFN block for Gemma-style VLA/VLM language paths. |
| `flashrt-residual-norm-quant` | Runtime package | Residual add, RMSNorm, and static FP8 activation producer kernels. |
| `flashrt-qkv-cache-rope` | Runtime package | Packed-QKV split, Q/K RMSNorm, RoPE staging, decode Q staging, and KV cache-write for VLA/VLM/video attention inputs. |
| `flashrt-vla-residual-gates` | Runtime package | Video/action/und joint gated residual updates for VLA block glue. |
| `flashrt-adaptive-norms` | Runtime package | AdaRMSNorm/style modulation and fused residual/AdaRMSNorm/static-FP8 activation output for DiT/VLA/world-model blocks. |
| `flashrt-spatiotemporal-layout` | Runtime package | NCDHW/BLC layout, temporal unshuffle, channel-bias, and short-cache helpers for VLA/video/world-model pipelines. |
| `flashrt-vla-video` | V1 block | VLA, vision, video, and diffusion attention postprocess kernels that are reusable outside the FlashRT serving engine. |
| `flashrt-nvfp4` | V1 block | NVFP4/FP4 data movement, SFA/SFB layout, low-bit GEMM, and fused epilogues. |
| `flashrt-smallm-gemm` | V1 block | Decode-oriented small-M GEMM/GEMV and split-K primitives with generic shape-specialized APIs. |
| `flashrt-fused-quant` | V1 block | Memory-bound fusion kernels: norm, residual, activation, and quantization. |
| `MiniMaxAI-msa-blackwell` | Partner extension | MiniMax MSA sparse attention extension for Blackwell hardware. |
| `vl-transformer-primitives` | Transformers/diffusers package | VLM Q/K norm + RoPE + KV-write staging and vision token pooling primitives. |
| `linear-attention-primitives` | Transformers package | Small-M BF16 linear, QKV broadcast split, partial RoPE, and gated-delta preparation primitives. |
| `diffusion-step-ops` | Diffusers package | Scheduler, CFG, first-frame forcing, and decode-postprocess helpers for diffusion/video runtime glue. |
| `turboquant-kv` | Transformers package | KV unpack/combine helpers for TurboQuant-style serving and cache-compression demos. |
| `world-model-conv` | Diffusers package | Blackwell FP8 3D causal conv primitive for video/world-model/VAE-style blocks. |
| `fp4-fused-ops` | Native FP4 package | FP16-to-NVFP4 producer and FP4-to-FP4 combiner kernels for continuous low-bit transformer/diffuser paths. |
| `fp4-gemm` | Native FP4 package | NVFP4 W4A16 GEMM with BF16 output for Blackwell low-bit linear layers. |

## Repository Status

All v1 packages have promoted `build.toml`, `flake.nix`, and `flake.lock`
files and pass configuration-level prebuild checks. Package-specific scope,
supported shapes, validation records, and benchmark evidence are tracked in
each package directory and in `docs/release-gating.md`.

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
