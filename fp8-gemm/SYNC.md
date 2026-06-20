# Source Sync

- Upstream FlashRT source: `../official/FlashRT`
- Initial package date: June 20, 2026

Copied source files:

- `csrc/gemm/fp8_gemv_m1_sm120.cu`
- `csrc/gemm/fp8_gemv_m1_sm120.cuh`
- `csrc/gemm/fp8_smallM_handtuned_sm120.cu`
- `csrc/gemm/fp8_smallM_handtuned_sm120.cuh`
- `csrc/gemm/fp8_smallM_handtuned_ldmatrix_sm120.cu`
- `csrc/gemm/fp8_smallM_handtuned_ldmatrix_sm120.cuh`

Local packaging edits:

- Added Tensor-facing PyTorch custom ops in `torch-ext/torch_binding.cpp`.
- Added Python wrappers and fake registrations in `torch-ext/fp8_gemm`.
- Kept public APIs model-agnostic; no raw pointer or stream arguments.

Architecture assumptions:

- CUDA 12.8+
- NVIDIA Blackwell SM120a local validation target. The FP8 MMA path uses
  `.kind::f8f6f4` instructions and must be compiled for `sm_120a`, not plain
  `sm_120`.

Runtime constraints:

- Inputs are FP8 E4M3 tensors with layout `input[M, K]` and `weight[N, K]`.
- Output is BF16 `out[M, N]`.
- `K` must be divisible by 32.
- `M` must be `1` or in `2..64` for v1. M=128 remains an internal tuning
  item because the validated correct tile is not performance-positive enough
  for public release.
- `alpha` is a host float scale multiplier, normally
  `input_scale * weight_scale`.
