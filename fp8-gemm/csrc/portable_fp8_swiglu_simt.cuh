// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

// Portable SIMT implementation of the block-128 FP8 SwiGLU + FP8-quantize
// fusion. The sm_89 fused producer is unavailable on pre-sm89 devices; this
// reference kernel computes the same math in pure SIMT FMA so the op stays
// usable (slowly) on sm_110 Thor. sm_89 keeps the MMA path.

namespace flash_rt {
namespace gemm {

// output[M,N] fp8, out_scale[M,N/128] fp32 =
//   quant_fp8( silu_f32(A@B_gate^T) * (A@B_up^T) )
// where B = gate_up_weight is (2N, K); rows [0,N) = gate, [N,2N) = up.
int fp8_blockwise_swiglu_quantize_simt(
    const void* A_fp8, const void* gate_up_fp8, const float* act_scale,
    const float* w_scale, void* output_fp8, float* out_scale,
    int M, int N, int K, cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
