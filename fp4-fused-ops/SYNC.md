# Source Sync

- Upstream FlashRT source: `../official/FlashRT`
- Initial package date: June 20, 2026

Copied source files:

- `csrc/fused_fp4/norm_silu_fp4_sfa.cu/.cuh`
- `csrc/fused_fp4/dequantize_fp4_sfa.cu/.cuh`
- `csrc/fused_fp4/res_rms_fp4_sfa_v2.cu`
- `csrc/fused_fp4/res_rms_mul_fp4_sfa.cu`
- `csrc/fused_fp4/silu_mul_fp4_sfa_v2.cu`
- `csrc/fused_fp4/silu_mul_mul_fp4_sfa_v2.cu`
- `csrc/fused_fp4/silu_mul_two_fp4_to_fp4.cu/.cuh`
- `csrc/quantize/reshape_scales_sfa.cu/.cuh`

Local packaging edits:

- Added Tensor-facing PyTorch custom ops in `torch-ext/torch_binding.cpp`.
- Added Python wrappers and fake registrations in `torch-ext/fp4_fused_ops`.
- Public APIs accept CUDA tensors only; no raw pointers or stream arguments.

Architecture and dependency assumptions:

- CUDA 12.8+
- NVIDIA Blackwell `sm_120a`
- `cutlass_4_0` from `kernel-builder` for CUTE/CUTLASS SFA layout helpers.

Runtime constraints:

- Inputs are FP16 (`torch.float16`) matrices.
- Packed outputs are `torch.uint8`.
- SFA buffers are `torch.uint8` with byte length from `sfa_size_bytes`.
- Dimensions must be multiples of 16.
- v1 RMS producer paths are explicitly capped at `dim <= 2048` because the
  upstream shared-memory kernels only cover that range safely.
- The added `dequantize_fp4_sfa_fp16` utility is packaging glue for validation
  and debugging of CUTLASS SFA outputs.
