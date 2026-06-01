// SPDX-License-Identifier: Apache-2.0

#include "bias_gelu_quantize_fp8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {
namespace {

constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float gelu_tanh(float x) {
  return 0.5f * x *
         (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void bias_gelu_quantize_fp8_static_bf16_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ bias,
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ act_scale_ptr,
    long long total,
    int N,
    int has_bias) {
  const long long idx =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  float v = __bfloat162float(in[idx]);
  if (has_bias) {
    const int n = static_cast<int>(idx % N);
    v += __bfloat162float(bias[n]);
  }

  const float g = gelu_tanh(v);
  float q = g * (1.0f / *act_scale_ptr);
  q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
  out[idx] = __nv_fp8_e4m3(q);
}

}  // namespace

void bias_gelu_quantize_fp8_static_bf16(
    const void* in_bf16,
    const void* bias_bf16,
    void* out_fp8,
    const float* act_scale,
    long long M,
    int N,
    cudaStream_t stream) {
  const long long total = M * static_cast<long long>(N);
  if (total <= 0) {
    return;
  }

  constexpr int block_sz = 256;
  const unsigned grid =
      static_cast<unsigned>((total + block_sz - 1) / block_sz);
  const int has_bias = bias_bf16 != nullptr ? 1 : 0;

  bias_gelu_quantize_fp8_static_bf16_kernel<<<grid, block_sz, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(in_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
      act_scale,
      total,
      N,
      has_bias);
}

}  // namespace quantize
}  // namespace flash_rt
