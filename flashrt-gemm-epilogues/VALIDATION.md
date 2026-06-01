# Validation: flashrt-gemm-epilogues

This file records package-level validation that is suitable to keep with the
public kernel package. Local FlashRT parity notes and machine-specific scratch
tests belong under the ignored `internal-tests/` directory.

## Current Status

Validated on June 1, 2026.

Build target:

- `torch211-cxx11-cu128-x86_64-linux`

Build environment:

- `kernel-builder` 0.16.0-dev0
- Docker/Nix build wrapper
- CUDA 12.8 variant
- Python ABI: abi3, manylinux_2_28 check

Runtime smoke environment:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 580.82.07
- PyTorch: 2.9.1+cu128
- CUDA runtime reported by PyTorch: 12.8
- CUDA capability: 12.0

## Commands

From this package directory:

```bash
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker check-config .
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker build --variant torch211-cxx11-cu128-x86_64-linux --max-jobs 1 --cores 8 -L .
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker check-builds .
```

Host-side correctness smoke from the repository root:

```bash
python internal-tests/flashrt-gemm-epilogues/manual_torch_extension_smoke.py --large
```

## Builder Results

- `check-config` passed.
- `build` passed for `torch211-cxx11-cu128-x86_64-linux`.
- Native extension linked as `_flashrt_gemm_epilogues_cuda_*.abi3.so`.
- ABI compatibility check passed for manylinux_2_28 and Python ABI 3.9.
- `get_kernel` loading check passed for `flashrt_gemm_epilogues`.
- `check-builds` passed.

Installed build layout:

```text
torch211-cxx11-cu128-x86_64-linux/
  __init__.py
  _flashrt_gemm_epilogues_cuda_<git>.abi3.so
  _ops.py
  flashrt_gemm_epilogues/__init__.py
  metadata.json
```

## Correctness Smoke

The host-side smoke test compares against PyTorch references and synchronizes
after each kernel call.

Covered APIs:

- `bf16_gemm_bias_gelu`
- `bf16_gemm_bias`
- `bias_gelu_quantize_fp8_static_bf16`
- `gelu_quantize_fp8_static_bf16`
- `channel_scale_quantize_fp8_static_bf16`

Covered shapes:

- BF16 GEMM: `(M, N, K) = (16, 64, 32)`
- BF16 GEMM: `(M, N, K) = (32, 128, 64)`
- BF16 GEMM large smoke: `(M, N, K) = (64, 4096, 4096)`
- FP8 bias/GELU quantize: `(4, 16)`
- FP8 GELU quantize without bias: `(4, 16)`
- FP8 channel-scale quantize: `(2, 3, 32)`

## Known Gaps

- Only one HF build variant has been built so far.
- Docker does not currently expose the NVIDIA runtime, so GPU execution tests
  are run on the host instead of inside the Docker/Nix builder.
- Package benchmarks are scaffolded but not yet recorded.
- FP8 GEMM wrappers are intentionally not exposed yet; the first attempted
  cuBLASLt FP8 route compiled but did not return a supported heuristic on the
  local RTX 5090 environment.
