// SPDX-License-Identifier: Apache-2.0
#include "fused_fp4/adarms_fp8_static_fp16.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace flash_rt::fused_fp4 {
namespace {

__device__ __forceinline__ float block_sum(float value, float* warp_sums) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < 8 ? warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) warp_sums[0] = value;
  }
  __syncthreads();
  return warp_sums[0];
}

template <bool AddResidual>
__global__ void adarms_fp8_static_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ previous_gate,
    __half* __restrict__ residual,
    const __half* __restrict__ style,
    __nv_fp8_e4m3* __restrict__ out,
    __half* __restrict__ gate,
    int rows,
    int dim,
    const float* __restrict__ scale) {
  const int row_idx = blockIdx.x;
  if (row_idx >= rows) return;
  const __half* x_row = x + static_cast<long long>(row_idx) * dim;
  const __half* style_row = style + static_cast<long long>(row_idx) * 3 * dim;
  __half* residual_row = AddResidual
      ? residual + static_cast<long long>(row_idx) * dim
      : nullptr;

  __shared__ float warp_sums[8];
  float sum_sq = 0.0f;
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    float value = __half2float(x_row[col]);
    if constexpr (AddResidual) {
      value = __half2float(residual_row[col]) +
          value * __half2float(previous_gate[static_cast<long long>(row_idx) * dim + col]);
      residual_row[col] = __float2half_rn(value);
      // Match the native runtime contract: RMS is evaluated from the FP32
      // residual expression before its FP16 store.
    }
    sum_sq += value * value;
  }
  const float total = block_sum(sum_sq, warp_sums);
  const float rstd = rsqrtf(total / static_cast<float>(dim) + 1.0e-6f);
  const float inv_scale = 1.0f / fmaxf(scale[0], 1.0e-12f);
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    float value = AddResidual
        ? __half2float(residual_row[col])
        : __half2float(x_row[col]);
    const float normalized = value * rstd *
        (1.0f + __half2float(style_row[col])) +
        __half2float(style_row[dim + col]);
    out[static_cast<long long>(row_idx) * dim + col] = __nv_fp8_e4m3(
        fminf(fmaxf(normalized * inv_scale, -448.0f), 448.0f));
    gate[static_cast<long long>(row_idx) * dim + col] = style_row[2 * dim + col];
  }
}

}  // namespace

void adaptive_rms_norm_fp8_static_fp16(
    const void* x, const void* style, void* out, void* gate,
    int rows, int dim, const float* scale, cudaStream_t stream) {
  adarms_fp8_static_kernel<false><<<rows, 256, 0, stream>>>(
      static_cast<const __half*>(x), nullptr, nullptr,
      static_cast<const __half*>(style), static_cast<__nv_fp8_e4m3*>(out),
      static_cast<__half*>(gate), rows, dim, scale);
}

void gated_residual_adaptive_rms_norm_fp8_static_fp16(
    const void* x, const void* previous_gate, void* residual,
    const void* style, void* out, void* gate, int rows, int dim,
    const float* scale, cudaStream_t stream) {
  adarms_fp8_static_kernel<true><<<rows, 256, 0, stream>>>(
      static_cast<const __half*>(x), static_cast<const __half*>(previous_gate),
      static_cast<__half*>(residual), static_cast<const __half*>(style),
      static_cast<__nv_fp8_e4m3*>(out), static_cast<__half*>(gate),
      rows, dim, scale);
}

}  // namespace flash_rt::fused_fp4
