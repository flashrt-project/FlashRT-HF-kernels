# Agent Instructions

This repository packages selected FlashRT kernels for Hugging Face Kernel Hub.
Keep changes scoped to packaging, bindings, tests, benchmarks, and copied kernel
source needed by one package.

## Global Rules

- FlashRT upstream source lives in `../official/FlashRT`.
- Do not modify FlashRT upstream from this repository unless explicitly asked.
- Do not expose FlashRT internal `uintptr_t` or raw stream APIs as public Hub
  APIs. Public functions should accept and return `torch.Tensor` objects.
- Native ops must be registered with `TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)`.
- Python wrappers must import `ops` from `._ops`.
- If Python code defines Torch custom ops, prefix op names with
  `add_op_namespace_prefix` from `._ops`.
- Package names and public functions should be generic. Avoid names such as
  `qwen`, `pi05`, `groot`, or `motus` unless the package is explicitly a
  model-specific compatibility layer.
- Do not commit build outputs, `result/`, `build/`, generated wheels, or Hub
  upload artifacts.
- Keep dependency declarations in `build.toml`; do not depend on FlashRT local
  `third_party/` paths.
- If CUTLASS is required, use a `kernel-builder` dependency such as
  `cutlass_4_0` and include only package-local headers.
- Keep public package docs clean. Put local planning notes in `internal-docs/`
  and FlashRT-dependent tests in `internal-tests/`.

## Package Promotion Criteria

A package may be promoted from draft to buildable when:

- `build.toml.draft` has been renamed to `build.toml`.
- `torch-ext/torch_binding.cpp` and headers exist.
- All source files listed in `build.toml` exist.
- Correctness tests cover dtype, shape, device, and stride expectations.
- Benchmarks include at least one generic shape set and one FlashRT-real shape
  set.
- Hub-compatible tests do not depend on `../official/FlashRT`; FlashRT parity
  checks live under `internal-tests/`.
- `kernel-builder build <package>` succeeds locally.
- `kernel-builder check-abi <package>` succeeds for produced native modules.

## Release / HF Jobs Workflow

Before pushing a package change that should rebuild Hub artifacts:

1. Identify every package with changed CUDA/C++ bindings, Python wrappers,
   tests, benchmarks, `build.toml`, `flake.nix`, or `flake.lock`.
2. Run source correctness for those packages with the strictest available
   shape mode. Use package tests rather than model demos as the package gate.
3. Run representative benchmarks for changed performance-sensitive APIs and
   record shape, dtype, tolerance, and hardware in package docs or internal
   notes.
4. Run `kernel-builder-docker check-config .` from each changed package
   directory.
5. Keep `build/`, `result/`, `dist/`, wheels, `__pycache__/`, and
   `internal-tests/` outputs untracked. `scripts/prebuild_check.py` is expected
   to fail until ignored local build artifacts are removed or the check is run
   in a clean clone.
6. Use `.github/workflows/build-kernels-hf-jobs.yml` for release packaging and
   upload. Add any newly changed package to the workflow path filters and
   matrix before relying on push-triggered builds.
7. After HF Jobs uploads artifacts, verify by loading through
   `get_kernel("flashrt/<package>", version=1, trust_remote_code=True)` in a
   matching PyTorch/CUDA environment and rerun installed-artifact correctness.

For the PI0.5 runtime demo, do not use random-input OpenPI smoke rows as public
baselines. Public runtime comparisons should use the real LIBERO bundle path
or a clearly labeled full policy-wrapper benchmark.

## Source Sync Rules

For each package, document source provenance in `SYNC.md` before copying code.
Include:

- Upstream FlashRT commit.
- Exact source files copied.
- Local edits made after copying.
- Required compile flags and architecture assumptions.
- Runtime constraints and unsupported shapes.

When syncing source from FlashRT:

1. Copy only the files needed by the package.
2. Rewrite includes to be package-local.
3. Remove serving-runtime dependencies.
4. Replace pointer-only binding assumptions with Tensor validation in
   `torch-ext/torch_binding.cpp`.
5. Keep CUDA launchers graph-safe: no dynamic allocation inside hot kernels.

## Package-Specific Notes

### flashrt-gemm-epilogues

Focus on fused GEMM outputs that remove bandwidth and launch overhead:

- GEMM + bias.
- GEMM + activation.
- GEMM + residual.
- GEMM + dequant/scale.
- GEMM + quantized output.

Likely upstream areas:

- `csrc/gemm/`
- `csrc/gemm/fp4/`
- `csrc/quantize/bias_gelu_quantize_fp8.*`
- `csrc/quantize/awq_quant_fp8_static_bf16.*`

### flashrt-fused-quant

Focus on non-GEMM memory-bound fusion:

- RMSNorm/LayerNorm plus quantization.
- Residual plus norm plus quantization.
- SiLU/GEGLU/SwiGLU plus quantization.
- QKV split, RoPE, and KV write helpers when generic.

Likely upstream areas:

- `csrc/kernels/norm.*`
- `csrc/kernels/fusion.*`
- `csrc/kernels/quantize.*`
- `csrc/kernels/rope.*`
- `csrc/quantize/qkv_split_norm_rope_bf16.*`

### flashrt-nvfp4

Focus on NVFP4 primitives and layouts:

- BF16/FP16 to NVFP4.
- NVFP4 dequantization.
- SFA/SFB layout reshape.
- CUTLASS-compatible scale layout helpers.

Likely upstream areas:

- `csrc/quantize/quantize_fp4_dynamic.*`
- `csrc/quantize/quantize_fp4_sfa.*`
- `csrc/quantize/reshape_scales_sfa.*`
- `csrc/fused_fp4/`
- `csrc/gemm/fp4/`

### flashrt-smallm-gemm

Focus on generic decode-oriented small-M kernels. Avoid model names in public
API. Public APIs should describe shape and dtype behavior, not model origin.

Likely upstream areas:

- `csrc/gemm/fp8_smallM_handtuned*`
- `csrc/kernels/bf16_matvec_*`
- `csrc/kernels/bf16_matmul_*`
- `csrc/kernels/fp4_w4a4_*`

### flashrt-vla-video

Focus on reusable VLA, vision, video, and diffusion primitives:

- Patch embedding and vision data movement.
- Video/3D convolution low-bit paths.
- DiT/VAE helper fusions.
- Vision attention post-processing if generic.

Likely upstream areas:

- `csrc/kernels/patch_embed.*`
- `csrc/kernels/dit_bf16.*`
- `csrc/conv/`
- `csrc/quantize/bf16_*ncdhw*`
