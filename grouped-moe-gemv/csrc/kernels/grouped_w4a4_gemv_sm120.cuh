// SPDX-License-Identifier: Apache-2.0
//
// Grouped NVFP4 W4A4 GEMV for dynamic top-k MoE routing on SM120.
//
// Decode routes tokens to device-selected experts. This kernel runs all
// M*top_k selected projections in one launch (grid.y = routed pair),
// indexing the weight stack by a device-side expert-id buffer so the
// dynamic top-k routing can drive a captured CUDA graph (the buffer is
// re-read each replay). Token-major indexing lets gate_up reuse each token's
// activation across top-k; down uses M=routed_pairs and top_k=1.
//
// Inner block-scaled mma + swizzled-SF decode are identical to
// fp4_w4a4_mma_sm120_full_n_bf16out (validated cos=1.0); only the per-slot
// base-pointer arithmetic is added on top.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// Grouped NVFP4 W4A4 GEMV, BF16 output, SM120.
//
//   A_packed   : (M,K/2) packed activation; quantized once per token.
//   B_stack    : (num_experts, N, K/2) packed weight stack.
//   D          : (M,top_k,N) bf16 output.
//   SFA        : batched CUTLASS SFA layout for (M,K).
//   SFB_stack  : (num_experts, sf_bytes) swizzled SF stack.
//   alpha_stack: (num_experts,) fp32 per-expert GEMM alpha.
//   expert_idx : (M,top_k) contiguous device int, flattened token-major.
//   strides    : byte strides into the stacks (w_stride = N*K/2,
//                sfb_stride = swizzled SF bytes for (N,K)).
//   force_simt : route every shape to the portable SIMT kernel. The default
//                block-scaled mma path is SM120-only
//                (SM120_16x8x64_TN_VS); on SM110 (Thor) that atom asserts, so
//                callers on non-SM120 devices must pass true.
//
// Returns 0 on success, nonzero on caller-side argument error.
int grouped_w4a4_gemv_sm120_bf16(
    const void*  A_packed,
    const void*  B_stack,
    void*        D,
    const void*  SFA,
    const void*  SFB_stack,
    const void*  alpha_stack,
    const void*  expert_idx,
    int          M,
    int          top_k,
    int          N,
    int          K,
    long         w_stride,
    long         sfb_stride,
    cudaStream_t stream,
    bool         force_simt);

}  // namespace gemm
}  // namespace flash_rt
