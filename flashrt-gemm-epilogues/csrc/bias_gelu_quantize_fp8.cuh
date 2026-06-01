// SPDX-License-Identifier: Apache-2.0
//
// Fused bias + GELU(tanh) + per-tensor FP8 e4m3 quantize.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

// in_bf16  : (..., N) bf16 row-major contiguous
// bias     : (N,) bf16, or nullptr for no bias
// out_fp8  : (..., N) fp8_e4m3 row-major contiguous
// act_scale: device fp32 scalar
//   out = clamp(gelu(in + bias) / scale, +/-448).to(fp8_e4m3)
void bias_gelu_quantize_fp8_static_bf16(
    const void* in_bf16,
    const void* bias_bf16,
    void* out_fp8,
    const float* act_scale,
    long long M,
    int N,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
