# Release Runbook

This runbook is for the v1 batch release window. It intentionally separates
cheap prebuild checks from full `kernel-builder` builds.

## Prebuild Checks

Run these during normal development:

```bash
python scripts/prebuild_check.py --check-config
python -m py_compile scripts/prebuild_check.py
git diff --check
```

Expected result:

- no tracked files under `internal-docs/` or `internal-tests/`;
- no `result`, `build`, or `dist` artifacts in package directories;
- every v1 package has `build.toml`, `flake.nix`, `flake.lock`, tests,
  benchmarks, examples, and `benchmarks/RESULTS.md`;
- `kernel-builder-docker check-config` passes for every v1 package.

## Full Build Window

Run this only after source, tests, benchmark scripts, and docs have settled:

```bash
python scripts/prebuild_check.py --check-config
```

Then build all promoted v1 packages:

```bash
cd flashrt-gemm-epilogues
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker build .
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker check-builds .

cd ../flashrt-vla-video
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker build .
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker check-builds .

cd ../flashrt-nvfp4
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker build .
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker check-builds .

cd ../flashrt-smallm-gemm
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker build .
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker check-builds .

cd ../flashrt-fused-quant
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker build .
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker check-builds .
```

Do not change source while this build window is running unless the build fails.
Record failures in the corresponding `VALIDATION.md` before making fixes.

## Built-Artifact Validation

After each package builds:

- run package tests against the built artifact;
- run the package benchmark script through the HF benchmark runner;
- run the package example;
- update `benchmarks/RESULTS.md`;
- update `VALIDATION.md` with build variants, hardware, PyTorch/CUDA versions,
  and failures.

## Benchmark Order

1. `flashrt-gemm-epilogues`: FP8 quant epilogue benchmarks first; BF16 GEMM
   epilogue claims stay shape-specific.
2. `flashrt-vla-video`: `qkv_split_norm_rope_bf16` is the strongest showcase
   path and should get the first model-block note.
3. `flashrt-nvfp4`: layout helper first; fused NVFP4 GEMM epilogue only after
   a fair CUTLASS/cuBLASLt or unfused CUDA-chain baseline is ready.
4. `flashrt-smallm-gemm`: run decode grid and record SM120-only scope.
5. `flashrt-fused-quant`: run split and merged SwiGLU quant grids and report
   effective memory bandwidth.

## Failure Policy

- Config failure: fix `build.toml`, `flake`, or missing source first.
- Compile failure: reproduce with local source-extension when possible before
  rerunning full builder.
- Benchmark failure: keep the package built, fix benchmark/test logic, and do
  not rerun full build unless source changed.
- Weak speedup: keep the API as compatibility coverage, but do not promote the
  shape as a headline.

## Final Push Before Upload

Before upload or sharing with Hugging Face:

```bash
python scripts/prebuild_check.py --check-config
git status --short --ignored
```

Only ignored `internal-docs/` and `internal-tests/` should remain.
