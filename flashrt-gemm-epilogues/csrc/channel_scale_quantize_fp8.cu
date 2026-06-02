// SPDX-License-Identifier: Apache-2.0

#include "channel_scale_quantize_fp8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdlib>

namespace flash_rt {
namespace quantize {
namespace {

constexpr float kFp8Max = 448.0f;

__global__ void channel_scale_quantize_fp8_static_bf16_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ channel_scale,
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ act_scale_ptr,
    long long tiles_per_row,
    int K) {
  const long long tile = static_cast<long long>(blockIdx.x);
  const long long row = tile / tiles_per_row;
  const long long col_tile = tile - row * tiles_per_row;
  const int col = static_cast<int>(col_tile * blockDim.x + threadIdx.x);
  if (col >= K) {
    return;
  }

  const long long idx = row * static_cast<long long>(K) + col;
  const float v = __bfloat162float(in[idx]);
  const float s = __bfloat162float(channel_scale[col]);
  float q = v * s * (1.0f / *act_scale_ptr);
  q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
  out[idx] = __nv_fp8_e4m3(q);
}

int quant_block_size(long long M, int K) {
  const char* value = std::getenv("FLASHRT_QUANT_BLOCK_SIZE");
  if (value != nullptr) {
    const int block_size = std::atoi(value);
    if (block_size == 128 || block_size == 256 || block_size == 512 ||
        block_size == 1024) {
      return block_size;
    }
  }

  if (K >= 8192) {
    return M >= 128 ? 256 : 512;
  }
  if (M <= 16) {
    return 128;
  }
  return 256;
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

  const int block_sz = quant_block_size(M, K);
  const long long tiles_per_row =
      (static_cast<long long>(K) + block_sz - 1) / block_sz;
  const unsigned grid = static_cast<unsigned>(M * tiles_per_row);

  channel_scale_quantize_fp8_static_bf16_kernel<<<grid, block_sz, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(in_bf16),
      reinterpret_cast<const __nv_bfloat16*>(channel_scale_bf16),
      reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
      act_scale,
      tiles_per_row,
      K);
}

}  // namespace quantize
}  // namespace flash_rt
