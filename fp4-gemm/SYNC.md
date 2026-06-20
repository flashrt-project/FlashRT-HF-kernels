# Source Sync

- Upstream FlashRT source: `../official/FlashRT`
- Initial package date: June 20, 2026

Copied source files:

- `csrc/gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm120.cu/.cuh`
- `csrc/quantize/quantize_fp4_sfa.cu/.cuh`

Packaging helper:

- `csrc/dequantize_fp4_sfa.cu/.cuh` derived from the SFA dequant validation
  helper used in `fp4-fused-ops`; this package adds `is_sfb` support so tests
  can dequant both A/SFA and B/SFB.

Local packaging edits:

- Added Tensor-facing PyTorch custom ops in `torch-ext/torch_binding.cpp`.
- Added Python wrappers and fake registrations in `torch-ext/fp4_gemm`.
- Public APIs accept CUDA tensors only; no raw pointers or stream arguments.

Not included in v1:

- FP4-output GEMM. The copied CUTLASS FP4out path returned `can_implement`
  failures under the current local source-test CUTLASS include, including
  production-sized shapes. It is intentionally not exposed until revalidated
  under the official builder/CUTLASS environment.
