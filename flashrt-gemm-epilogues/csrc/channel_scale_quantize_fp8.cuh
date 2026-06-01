// SPDX-License-Identifier: Apache-2.0
//
// Fused per-channel BF16 scale + per-tensor FP8 e4m3 quantize.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

// in_bf16      : (..., K) bf16 row-major contiguous
// channel_scale: (K,) bf16, broadcast over leading dimensions
// out_fp8      : (..., K) fp8_e4m3 row-major contiguous
// act_scale    : device fp32 scalar
//   out[..., k] = clamp((in[..., k] * channel_scale[k]) / scale, +/-448)
//                 .to(fp8_e4m3)
void channel_scale_quantize_fp8_static_bf16(
    const void* in_bf16,
    const void* channel_scale_bf16,
    void* out_fp8,
    const float* act_scale,
    long long M,
    int K,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
