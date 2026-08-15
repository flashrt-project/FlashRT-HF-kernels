# Source Sync

- Upstream FlashRT source: `../official/FlashRT`
- Original SM110 sync commit: `132049d7c3a3534fb7d35676cd726f39408b1af6`
- GROOT N1.7 fused-epilogue sync commit:
  `24df793f4fa2d50780aea03b644208c6e0cb4162`
- Qwen3.8 SM120 decode/prefill tier sync commit:
  `e2f4b16cea32bd520c93119b142758693793dfeb`
- Initial package date: June 20, 2026

Copied source files:

- `csrc/gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm120.cu/.cuh`
- `csrc/gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm100.cu/.cuh`
- `csrc/gemm/fp4/cutlass_fp4_gemm_bias_bf16_sm100.cu/.cuh`
- `csrc/quantize/quantize_fp4_sfa.cu/.cuh`
- `csrc/quantize/quantize_fp4_sfa_bf16.cu/.cuh`
- `csrc/gemm/fp4/fp4_w4a4_mma_warpsplit_ilv_sm120.cu/.cuh`
- `csrc/gemm/fp4/cutlass_nvfp4_gemm_m256_sm120.cu/.cuh`
- `cutlass/util/packed_stride.hpp`, copied from CUTLASS tools util headers
  into `csrc/cutlass/util/packed_stride.hpp` so the Hub package does not
  depend on a local `third_party/cutlass/tools/util/include` path.

Packaging helper:

- `csrc/dequantize_fp4_sfa.cu/.cuh` derived from the SFA dequant validation
  helper used in `fp4-fused-ops`; this package adds `is_sfb` support so tests
  can dequant both A/SFA and B/SFB.

Local packaging edits:

- Added Tensor-facing PyTorch custom ops in `torch-ext/torch_binding.cpp`.
- Added Python wrappers and fake registrations in `torch-ext/fp4_gemm`.
- Added the BF16 direct SFA/SFB producer as an input-type specialization of
  the existing FP16 producer. Its E2M1 encoding and CUTLASS scale layout are
  unchanged; the additive entry removes a standalone activation cast.
- Public APIs accept CUDA tensors only; no raw pointers or stream arguments.
- CUTLASS SM100/SM120 block-scaled layout support is treated as package scope,
  not as a test-only compiler define.
- The Tensor binding dispatches the canonical BF16-output GEMM by runtime
  compute capability. CUDA 12.8 artifacts link only the SM120 implementation;
  CUDA 13 artifacts link both SM110 and SM120 implementations.
- SM110 production auto-dispatch was tiled against PI0.5, GROOT, Cosmos Edge,
  and LingBot VLA projection shapes. Explicit schedule IDs remain diagnostic.

Architecture limits:

- The canonical BF16-output GEMM and SFA/SFB helpers support SM110 and SM120.
- Bias, residual, and bias/GELU-to-FP4 epilogues have an independent SM110
  backend copied from the production GROOT N1.7 path. Stream-K remains an
  SM120-only API and rejects on SM110.
- SM110 requires CUDA 13 and the package's pinned CUTLASS 4.4 target; SM120
  requires CUDA 12.8 and CUTLASS 4.5 after the M256 tier was added.
- The M256 source include was rewritten to the package-local
  `gemm/fp4/...` path. Kernel arithmetic is otherwise unchanged.
- The interleaved GEMV accepts only M=1, N divisible by 8, K divisible by 64,
  and a K/64 tile count divisible by the selected warp count. Repacking is a
  bind-time byte permutation and is not part of the decode hot path.
- The M256 entry is explicit rather than automatic. RTX 5090 qualification
  covers `(N,K)=(17408,5120),(5120,17408),(12288,5120)` at M=2044. The
  `(16384,5120)` row remains diagnostic because it did not beat the existing
  128-tile path in min-of-N testing.
