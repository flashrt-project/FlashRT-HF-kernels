# Source Sync

- Upstream FlashRT source: `../official/FlashRT`
- SM110 sync commit: `132049d7c3a3534fb7d35676cd726f39408b1af6`
- Initial package date: June 20, 2026

Copied source files:

- `csrc/gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm120.cu/.cuh`
- `csrc/gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm100.cu/.cuh`
- `csrc/quantize/quantize_fp4_sfa.cu/.cuh`
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
- Residual, bias/GELU, FP4-output, and Stream-K epilogues use the existing
  SM120 implementation. They reject on SM110 rather than silently falling
  back or calling an incompatible image.
- SM110 requires CUDA 13 and CUTLASS 4.5; SM120 requires CUDA 12.8 and
  CUTLASS 4.0.
