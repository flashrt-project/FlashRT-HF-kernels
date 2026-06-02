// SPDX-License-Identifier: Apache-2.0

#include "bias_gelu_quantize_fp8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdlib>

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
    long long tiles_per_row,
    int N,
    int has_bias) {
  const long long tile = static_cast<long long>(blockIdx.x);
  const long long row = tile / tiles_per_row;
  const long long col_tile = tile - row * tiles_per_row;
  const int col = static_cast<int>(col_tile * blockDim.x + threadIdx.x);
  if (col >= N) {
    return;
  }

  const long long idx = row * static_cast<long long>(N) + col;
  float v = __bfloat162float(in[idx]);
  if (has_bias) {
    v += __bfloat162float(bias[col]);
  }

  const float g = gelu_tanh(v);
  float q = g * (1.0f / *act_scale_ptr);
  q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
  out[idx] = __nv_fp8_e4m3(q);
}

int quant_block_size(long long M, int N, bool has_bias) {
  const char* value = std::getenv("FLASHRT_QUANT_BLOCK_SIZE");
  if (value != nullptr) {
    const int block_size = std::atoi(value);
    if (block_size == 128 || block_size == 256 || block_size == 512 ||
        block_size == 1024) {
      return block_size;
    }
  }

  if (N >= 12288) {
    return (has_bias && M <= 32) ? 512 : 256;
  }
  if (M == 1) {
    return has_bias ? 512 : 256;
  }
  if (M <= 2) {
    return has_bias ? 512 : 1024;
  }
  if (M <= 32) {
    return 1024;
  }
  return 256;
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

  const int has_bias = bias_bf16 != nullptr ? 1 : 0;
  const int block_sz = quant_block_size(M, N, has_bias != 0);
  const long long tiles_per_row =
      (static_cast<long long>(N) + block_sz - 1) / block_sz;
  const unsigned grid = static_cast<unsigned>(M * tiles_per_row);

  bias_gelu_quantize_fp8_static_bf16_kernel<<<grid, block_sz, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(in_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
      act_scale,
      tiles_per_row,
      N,
      has_bias);
}

}  // namespace quantize
}  // namespace flash_rt
