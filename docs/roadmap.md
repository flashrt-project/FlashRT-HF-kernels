# Roadmap

## North Star

This repository is the HF-facing distribution layer for selected FlashRT
kernels. The first goal is not to mirror every FlashRT kernel. The first goal
is to publish a small number of kernels that are:

- Generic enough for Hugging Face users to call.
- Fast enough to be worth external attention.
- Documented with shape constraints, baselines, and hardware scope.
- Connected to a concrete model or workload story where possible.

## Phase 0: Shape the Repository

Status: complete.

Deliverables:

- Monorepo skeleton with five package directories.
- Package-level README files with scope and non-goals.
- Draft `build.toml` files.
- Validation checklist.
- Source synchronization rules.

Exit criteria:

- HF collaborators can review package boundaries and naming.
- FlashRT maintainers agree on the first source files to sync.

## Phase 1: First Buildable Package

Recommended first package: `flashrt-gemm-epilogues`.

Status: buildable source slice implemented and builder-validated for the
selected CUDA/PyTorch matrix.

Why:

- FP8 quantization epilogues are easy to explain to Transformers, Diffusers,
  and VLA/video users.
- The value proposition is clear: remove launches and memory round-trips around
  activation, bias, channel scaling, and quantized output.
- Tests can compare against PyTorch reference expressions.
- Benchmarks can use generic Linear plus activation/residual/quant patterns.

Deliverables:

- One buildable CUDA package.
- Tensor-based public API.
- Correctness tests.
- Microbenchmarks with generic and FlashRT-real shape families.
- `kernel-builder` build/check validation.

Exit criteria:

- Package can be loaded locally through `kernels.get_local_kernel`.
- FP8 quant epilogue results are strong enough to be the first public headline.
- BF16 GEMM epilogue results are documented conservatively by shape.

## Phase 2: First Showcase Package

Recommended candidates:

- `flashrt-vla-video`
- `flashrt-nvfp4`
- `flashrt-smallm-gemm`

Selection rule:

Pick the package with the strongest combination of microbenchmark speedup,
model-level relevance, and ecosystem gap. A package should not be presented as
a showcase only because it builds.

Preferred story:

- VLA, vision, video, or diffusion kernels that current HF kernel examples do
  not already cover deeply.
- A direct HF-style call path showing how a downstream model or model block
  benefits.
- A benchmark table that includes PyTorch eager, a strong library baseline when
  applicable, and FlashRT.

Exit criteria:

- At least one public API has a clear model-block use case.
- Benchmarks include broad generic shapes plus at least one real FlashRT shape
  family.
- Hardware scope is explicit: for example SM120-only for Blackwell NVFP4
  kernels versus broader CUDA support for simpler epilogue kernels.

## Phase 3: Fused Quant and NVFP4

Recommended packages:

- `flashrt-fused-quant`
- `flashrt-nvfp4`

Deliverables:

- Fused norm/residual/activation quantization APIs.
- NVFP4 quant/dequant/SFA helpers.
- Representative benchmarks for LLM, VLA, and diffusion-style shapes.

Exit criteria:

- APIs are generic enough to discuss with Transformers maintainers.
- Benchmarks show clear launch and bandwidth wins versus PyTorch eager.

## Phase 4: Small-M Decode Kernels

Recommended package: `flashrt-smallm-gemm`.

Deliverables:

- Generic small-M GEMM/GEMV APIs.
- Decode-oriented benchmark set.
- Shape constraints documented explicitly.

Exit criteria:

- Package is useful without model-specific naming.
- Benchmarks isolate small-M latency wins.

## Phase 5: VLA and Video Kernels

Recommended package: `flashrt-vla-video`.

Deliverables:

- Reusable VLA, vision, video, and diffusion primitives.
- Benchmarks covering patch, DiT, video convolution, and quantized layout paths.
- Integration notes for downstream model libraries.

Exit criteria:

- Package demonstrates a gap not already covered by current
  `kernels-community` attention, MoE, or quantization packages.

## Phase 6: Community Promotion

Promotion candidates:

- `flashrt-gemm-epilogues`
- `flashrt-fused-quant`
- `flashrt-nvfp4`

Paths:

- Keep packages under the FlashRT Hub namespace if we own fast iteration.
- Propose selected stable packages to `kernels-community` if Transformers,
  Diffusers, or another HF project wants to consume them by default.
- Ask for trusted publisher status once APIs and security posture are stable.

Promotion bar:

- Stable public API.
- Clean builder matrix.
- Correctness tests in the package.
- Benchmark evidence against fair baselines.
- Clear ownership for maintenance.
- A downstream maintainer or HF project has a reason to consume the package.
