# Roadmap

## North Star

This repository is the HF-facing distribution layer for selected FlashRT
kernels. The first public version is a batch of four peer blocks:

- FP8/GEMM epilogues.
- VLA/video post-processing.
- Blackwell NVFP4/FP4 low-bit kernels.
- Fused quantization.

The goal is not to upload one small package as soon as it builds. The goal is
to finish enough source, tests, benchmarks, and examples across all four blocks
that the first update looks like a coherent FlashRT kernel release.

## Phase 0: Repository Shape

Status: complete.

Deliverables:

- Monorepo skeleton with package directories.
- Package-level README and CARD files with scope and non-goals.
- Draft or promoted `build.toml` files.
- Validation checklist and release-gating docs.
- Source synchronization rules.

Exit criteria:

- HF collaborators can review package boundaries and naming.
- FlashRT maintainers agree on the first source files to sync.

## Phase 1: V1 Source And API Readiness

Status: in progress.

Deliverables:

- Tensor-based public APIs for every v1 block.
- Synced source slices with minimal dependency surfaces.
- Python wrappers that expose tensors, not raw pointers or stream handles.
- Runtime guards for dtype, device, shape, layout, and hardware scope.

Exit criteria:

- Every v1 package reaches at least G1 in `docs/release-gating.md`.
- Draft packages are not promoted until their Tensor bindings and correctness
  tests pass locally.

## Phase 2: V1 Correctness And Benchmark Readiness

Status: in progress.

Deliverables:

- Package-local correctness tests.
- Internal FlashRT parity tests only where public references are not enough.
- Benchmark scripts covering `docs/tile-and-shape-coverage.md`.
- Public `RESULTS.md` summaries with GPU, driver, PyTorch, CUDA runtime,
  warmup, measured iterations, and fair baselines.

Exit criteria:

- Every v1 block has a documented shape grid and benchmark baseline.
- Every headline claim is tied to exact shapes and hardware.

## Phase 3: V1 Examples And Model-Block Story

Status: in progress.

Deliverables:

- HF-style examples for each stable v1 public API surface.
- At least one model-block note for the VLA/video path.
- Clear distinction between helper APIs, showcase APIs, and SM120-only APIs.

Exit criteria:

- Users can see which PyTorch/HF op sequence each fused kernel replaces.
- The four blocks can be presented together without over-claiming any one
  package.

## Phase 4: V1 Batch Build Window

Status: pending.

Deliverables:

- Clean build tree with no stale `result` symlinks or committed outputs.
- `kernel-builder-docker check-config .` passes for every promoted v1 package.
- Full `kernel-builder build` and `check-builds` run for the intended matrix.
- Package tests, benchmark CLIs, and examples run against built artifacts.
- `VALIDATION.md` files updated with exact build variants and failures.

Exit criteria:

- The whole v1 batch can be uploaded together.
- Known gaps are explicit and do not contradict public package cards.

## Phase 5: Hub Upload And Community Path

Status: pending.

Paths:

- Keep packages under the FlashRT Hub namespace while APIs iterate quickly.
- Propose stable, high-impact packages to `kernels-community` only if HF
  projects want to consume or maintain them by default.
- Ask for trusted publisher status once APIs and security posture are stable.

Promotion bar:

- Stable public API.
- Clean builder matrix.
- Correctness tests in the package.
- Benchmark evidence against fair baselines.
- Clear ownership for maintenance.
- A downstream maintainer or HF project has a reason to consume the package.
