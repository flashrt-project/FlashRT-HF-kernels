# Source Sync

- Upstream FlashRT source: `../official/FlashRT`
- Initial package date: June 20, 2026
- SM89 source revision: `70b8eac4b05e9193bd99631cf872c5a971b59f5d`
- SM110 sync revision: `132049d7c3a3534fb7d35676cd726f39408b1af6`

Copied source files:

- `csrc/gemm/fp8_gemv_m1_sm120.cu`
- `csrc/gemm/fp8_gemv_m1_sm120.cuh`
- `csrc/gemm/fp8_smallM_handtuned_sm120.cu`
- `csrc/gemm/fp8_smallM_handtuned_sm120.cuh`
- `csrc/gemm/fp8_smallM_handtuned_ldmatrix_sm120.cu`
- `csrc/gemm/fp8_smallM_handtuned_ldmatrix_sm120.cuh`
- `csrc/gemm/cutlass_sm120_block128_fp8_gemm.cu`
- `csrc/gemm/cutlass_sm120_block128_fp8_gemm.cuh`
- `csrc/gemm/fp8_block128_gemm_mma_sm89.cu`
- `csrc/gemm/fp8_block128_gemm_mma_sm89.cuh`
- `csrc/gemm/fp8_bs_gemm_device.cuh`
- `csrc/gemm/fp8_gemv_m1_sm89.cu`
- `csrc/gemm/fp8_gemv_m1_sm89.cuh`
- `csrc/gemm/gemm_types_sm100.h`
- `csrc/gemm/cutlass_sm100.cu`

The SM110 copies are package-local as `csrc/gemm_types_sm110.h` and
`csrc/cutlass_sm110_fp8_gemm.cu`. The C declarations in
`csrc/cutlass_sm110_fp8_gemm.cuh` are packaging glue; the upstream pointer API
declares them in its aggregate binding instead.

Local packaging edits:

- Added Tensor-facing PyTorch custom ops in `torch-ext/torch_binding.cpp`.
- Added Python wrappers and fake registrations in `torch-ext/fp8_gemm`.
- Kept public APIs model-agnostic; no raw pointer or stream arguments.
- Bound the upstream measured `32x128-w4-s1` fused SwiGLU producer without
  changing its CUDA tile or arithmetic.
- Added a Tensor-facing SM110 dispatcher over the upstream BF16-output Sq, T1,
  and Wide tactics. The public dispatcher and diagnostic variants do not alter
  the copied GEMM templates or arithmetic.
- Renamed SM100 source filenames locally to make their SM110 package role
  explicit; CUTLASS still uses the SM100-family architecture templates when
  compiling for `sm_110a`.
- The SM110 build uses `-O3`, `--expt-relaxed-constexpr`, and
  `--use_fast_math`, matching the validated native path.

Architecture assumptions:

- CUDA 12.8+ for SM89/SM120; CUDA 13.0+ for SM110.
- NVIDIA Ada SM89 for block-128 scaled GEMM/GEMV.
- NVIDIA Blackwell SM110a for per-tensor Sq/T1/Wide FP8 GEMM with BF16 output.
- NVIDIA Blackwell SM120a for the original public APIs. The per-tensor FP8 MMA path uses
  `.kind::f8f6f4` instructions and must be compiled for `sm_120a`, not plain
  `sm_120`.
- The SM110 kernel depends on the builder-provided CUTLASS 4.5 package. The
  package flake is pinned to a builder revision that exports `cutlass_4_5`.

Runtime constraints:

- Inputs are FP8 E4M3 tensors with layout `input[M, K]` and `weight[N, K]`.
- Output is BF16 `out[M, N]`.
- `K` must be divisible by 32.
- On SM120, `M` must be `1` or in `2..64`. M=128 remains an internal tuning
  item because the validated correct SM120 tile is not performance-positive
  enough for public release.
- On SM110, `N` and `K` must be divisible by 16. The full-row Sq/T1/Wide path
  has been validated on `M` from 1 through 1024 across PI0.5, GROOT,
  Cosmos Edge, and LingBot projection families.
- `alpha` is a host float scale multiplier, normally
  `input_scale * weight_scale`.
- The blockwise path consumes FP32 scales with layouts `(M, K/128)` and
  `(N/128, K/128)`. It is the same CUTLASS kernel and schedule dispatcher used
  by the upstream FlashRT pointer API.
- Blockwise scaling is not exposed on SM110 in this increment.
