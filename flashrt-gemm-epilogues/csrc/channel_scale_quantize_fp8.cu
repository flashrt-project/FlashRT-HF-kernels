// SPDX-License-Identifier: Apache-2.0

#include "channel_scale_quantize_fp8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {
namespace {

constexpr float kFp8Max = 448.0f;

__global__ void channel_scale_quantize_fp8_static_bf16_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ channel_scale,
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ act_scale_ptr,
    long long total,
    int K) {
  const long long idx =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int k = static_cast<int>(idx % static_cast<long long>(K));
  const float v = __bfloat162float(in[idx]);
  const float s = __bfloat162float(channel_scale[k]);
  float q = v * s * (1.0f / *act_scale_ptr);
  q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
  out[idx] = __nv_fp8_e4m3(q);
}

}  // namespace

void channel_scale_quantize_fp8_static_bf16(
    const void* in_bf16,
    const void* channel_scale_bf16,
    void* out_fp8,
    const float* act_scale,
    long long M,
    int K,
    cudaStream_t stream) {
  const long long total = M * static_cast<long long>(K);
  if (total <= 0) {
    return;
  }

  constexpr int block_sz = 256;
  const unsigned grid =
      static_cast<unsigned>((total + block_sz - 1) / block_sz);

  channel_scale_quantize_fp8_static_bf16_kernel<<<grid, block_sz, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(in_bf16),
      reinterpret_cast<const __nv_bfloat16*>(channel_scale_bf16),
      reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
      act_scale,
      total,
      K);
}

}  // namespace quantize
}  // namespace flash_rt
