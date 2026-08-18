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
- If CUTLASS is required, use a dependency accepted by the package's pinned
  upstream builder revision (for example `cutlass_4_0` or `cutlass_4_5`) and
  include only package-local overrides.
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
4. Run `check-config` from each changed package directory using the exact
   upstream builder revision in that package's `flake.lock`. A stale local
   builder that rejects a newly supported dependency is not authoritative.
5. Native packages must pass the local artifact ladder before a paid HF Job.
   Exact Torch/CUDA source compile, strong-symbol and cubin inspection,
   installed-layout correctness, `torch.compile`, CUDA Graph, and the
   package-specific performance gate are mandatory. Run an exact target-variant
   local `kernel-builder build` when its external dependencies are cached. If
   Nix instead starts rebuilding PyTorch/NCCL, stop that dependency build and
   stage the exact-environment source extension in installed-artifact layout;
   then use one targeted HF Job as the release artifact build. Never repeat
   paid Jobs to discover source compile, link, missing-symbol, architecture, or
   known performance failures. In either route, run
   `scripts/native_artifact_preflight.py` and the full installed tests again on
   the downloaded Hub artifact before promotion. A Python wrapper, a weak
   unresolved symbol, or a source-RC artifact alone is not release evidence.
6. For every `[torch-noarch]` package, reproduce the builder's final copied
   layout and import it before using HF Jobs:
   `python scripts/validate_noarch_layout.py <package> --backend cuda` (and
   `--backend rocm` when applicable). This is a mandatory paid-build gate. It
   catches sibling Python packages and other files that exist under
   `torch-ext/` but are omitted because the noarch builder copies only
   `torch-ext/<general.name as module>/`.
7. Keep `build/`, `result/`, `dist/`, wheels, `__pycache__/`, and
   `internal-tests/` outputs untracked. `scripts/prebuild_check.py` is expected
   to fail until ignored local build artifacts are removed or the check is run
   in a clean clone.
8. Use `.github/workflows/build-kernels-hf-jobs.yml` for release packaging and
   upload. Add any newly changed package to the workflow path filters,
   dispatch options, and normal matrix before relying on push-triggered builds.
   These lists must match except for packages such as FA2 with a dedicated job.
9. HF Jobs must run under the `liangsu9988` HF account/namespace. Before
   pushing release-triggering package changes, confirm the repository secret
   `HF_TOKEN` is a valid token for `liangsu9988` or another token with both HF
   Jobs access and `flashrt/<package>` kernel publishing rights. If the token is
   missing, expired, or belongs to the wrong account, the workflow fails before
   compilation with `HTTP 401: {"error":"Invalid username or password."}`.
10. After HF Jobs uploads artifacts, read `general.version` from `build.toml`
   and verify by loading through `get_kernel("flashrt/<package>", version=N,
   trust_remote_code=True)` in a matching PyTorch/CUDA environment. Rerun
   installed-artifact correctness; never assume every package is v1.
   Audit every existing `build/<variant>/metadata.json` provenance SHA as well.
   A default bundle rebuild can leave retired Torch variants untouched. When a
   compatible API is added without a major-version bump, use the workflow's
   comma-separated `variants` input for variants still listed by the selected
   builder. A retired variant that the current builder no longer lists must be
   rebuilt with its pinned historical builder and matching toolchain, or
   removed from the version branch. Loading only the newest variant is not a
   valid release gate.
11. Run `.github/workflows/mirror-kernels-legacy-model.yml` for every new or
   rebuilt package, then verify both the Kernel Hub `v1` ref and the legacy
   model-repo `v1` ref. A successful source upload without compiled build
   variants is not a release.
12. Use the current upstream builder matrix for normal releases. Do not claim
    compatibility with retired Torch versions by renaming artifact directories.
    `torch.stable-abi` is valid only after `kernel-builder check-abi
    --torch-stable-abi` passes and the binding uses only stable ABI headers.
13. Treat every CUDA capability as a source-level contract. A release artifact
    for a new capability must come from a matching target in `build.toml` and a
    native backend that has passed correctness and performance gates on that
    device. Never use a builder `--force-capability` option, rename an artifact
    directory, or compile an SM120-only implementation for SM110 as release
    evidence. One package may contain separate architecture-specific targets
    and source lists behind the same stable Tensor API.
14. For architecture-specific additions, compare against the strongest existing
    backend on that device. Keep a correct but losing implementation explicit
    and opt-in; do not let it replace cuDNN, cuBLASLt, or an already-qualified
    FlashRT package in automatic dispatch.

## Performance Qualification

- Tile every public production path over representative boundary and real-model
  shapes. Compare against warmed PyTorch eager and an equivalent warmed
  `torch.compile` reference when compilation is numerically stable.
- Keep diagnostic tile variants separate from production auto dispatch.
- Production auto must reject a shape when no measured variant meets the
  package's performance threshold. Never route a known-slow compatibility path
  silently.
- For packages exposing multiple variants, the benchmark must fail when auto
  is materially slower than the fastest valid diagnostic variant on an
  accepted shape.

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
