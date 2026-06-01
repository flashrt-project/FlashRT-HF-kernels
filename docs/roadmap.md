# Roadmap

## Phase 0: Shape the Repository

Status: current phase.

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

Status: initial source slice implemented.

Why:

- GEMM epilogues are easy to explain to Transformers and Diffusers users.
- The value proposition is clear: remove launches and memory round-trips after
  linear projections.
- Tests can compare against PyTorch reference expressions.
- Benchmarks can use generic Linear plus activation/residual/quant patterns.

Deliverables:

- One buildable CUDA package. Initial source slice present.
- Tensor-based public API. Initial source slice present.
- Correctness tests. Initial source slice present.
- Microbenchmarks. Initial source slice present.
- `kernel-builder build` and `check-abi` pass. Pending local environment with
  `kernel-builder` and Nix.

Exit criteria:

- Package can be loaded locally through `kernels.get_local_kernel`.
- Public API is stable enough for an initial Hub upload under the FlashRT
  namespace.

## Phase 2: Fused Quant and NVFP4

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

## Phase 3: Small-M Decode Kernels

Recommended package: `flashrt-smallm-gemm`.

Deliverables:

- Generic small-M GEMM/GEMV APIs.
- Decode-oriented benchmark set.
- Shape constraints documented explicitly.

Exit criteria:

- Package is useful without model-specific naming.
- Benchmarks isolate small-M latency wins.

## Phase 4: VLA and Video Kernels

Recommended package: `flashrt-vla-video`.

Deliverables:

- Reusable VLA, vision, video, and diffusion primitives.
- Benchmarks covering patch, DiT, video convolution, and quantized layout paths.
- Integration notes for downstream model libraries.

Exit criteria:

- Package demonstrates a gap not already covered by current
  `kernels-community` attention, MoE, or quantization packages.

## Phase 5: Community Promotion

Promotion candidates:

- `flashrt-gemm-epilogues`
- `flashrt-fused-quant`
- `flashrt-nvfp4`

Paths:

- Keep packages under the FlashRT Hub namespace if we own fast iteration.
- Propose selected stable packages to `kernels-community` if Transformers,
  Diffusers, or another HF project wants to consume them by default.
- Ask for trusted publisher status once APIs and security posture are stable.
