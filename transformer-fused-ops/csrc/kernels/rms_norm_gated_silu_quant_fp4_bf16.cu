// SPDX-License-Identifier: Apache-2.0
#include "rms_norm_gated_silu_quant_fp4_bf16.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace {

constexpr int kDim = 128;

__device__ __forceinline__ int sfa_offset_128x64(int row, int column, int dim) {
  const int row_block = row >> 7;
  const int row_in_block = row & 127;
  const int column_block = column >> 6;
  const int column_in_block = column & 63;
  const int column_blocks = (dim + 63) >> 6;
  return row_block * column_blocks * 512 + column_block * 512 +
      (row_in_block & 31) * 16 + (row_in_block >> 5) * 4 +
      (column_in_block >> 4);
}

__device__ __forceinline__ uint8_t fp32_to_e2m1(float value) {
  const uint8_t sign = value < 0.0f ? 0x8u : 0x0u;
  const float absolute = fabsf(value);
  uint8_t magnitude;
  if (absolute <= 0.25f) magnitude = 0;
  else if (absolute <= 0.75f) magnitude = 1;
  else if (absolute <= 1.25f) magnitude = 2;
  else if (absolute <= 1.75f) magnitude = 3;
  else if (absolute <= 2.5f) magnitude = 4;
  else if (absolute <= 3.5f) magnitude = 5;
  else if (absolute <= 5.0f) magnitude = 6;
  else magnitude = 7;
  return sign | magnitude;
}

__global__ void rms_norm_gated_silu_quant_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    uint2* __restrict__ packed,
    uint8_t* __restrict__ sfa,
    int rows, int flattened_dim, float eps) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  if (row >= rows || lane >= kDim) return;

  const size_t index = static_cast<size_t>(row) * kDim + lane;
  const float x_value = __bfloat162float(x[index]);
  const float gate_value = __bfloat162float(gate[index]);
  float square = x_value * x_value;
  for (int offset = 16; offset > 0; offset >>= 1) {
    square += __shfl_xor_sync(0xffffffffu, square, offset);
  }
  __shared__ float warp_square[4];
  __shared__ float reduced;
  const int warp_lane = lane & 31;
  const int warp = lane >> 5;
  if (warp_lane == 0) warp_square[warp] = square;
  __syncthreads();
  if (warp == 0) {
    float value = warp_lane < 4 ? warp_square[warp_lane] : 0.0f;
    value += __shfl_xor_sync(0xffffffffu, value, 1);
    value += __shfl_xor_sync(0xffffffffu, value, 2);
    if (warp_lane == 0) reduced = value;
  }
  __syncthreads();

  const float inverse_rms = rsqrtf(reduced / static_cast<float>(kDim) + eps);
  const __nv_bfloat16 normalized = __float2bfloat16(x_value * inverse_rms);
  const __nv_bfloat16 weighted = __float2bfloat16(
      __bfloat162float(weight[lane]) * __bfloat162float(normalized));
  const float silu_gate = gate_value / (1.0f + __expf(-gate_value));
  const __nv_bfloat16 result = __float2bfloat16(
      __bfloat162float(weighted) * silu_gate);
  out[index] = result;

  __shared__ float values[kDim];
  values[lane] = __bfloat162float(result);
  __syncthreads();
  if (lane >= kDim / 16) return;

  const int base = lane * 16;
  float amax = 0.0f;
  #pragma unroll
  for (int element = 0; element < 16; ++element) {
    amax = fmaxf(amax, fabsf(values[base + element]));
  }
  const float desired = fmaxf(amax / 6.0f, 1e-12f);
  const __nv_fp8_e4m3 scale_quantized = __nv_fp8_e4m3(desired);
  const float inverse_scale = 1.0f / static_cast<float>(scale_quantized);
  const int column = row * kDim + base;
  sfa[sfa_offset_128x64(0, column, flattened_dim)] =
      *reinterpret_cast<const uint8_t*>(&scale_quantized);

  uint2 output;
  uint8_t* bytes = reinterpret_cast<uint8_t*>(&output);
  #pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const uint8_t low = fp32_to_e2m1(values[base + 2 * pair] * inverse_scale);
    const uint8_t high = fp32_to_e2m1(
        values[base + 2 * pair + 1] * inverse_scale);
    bytes[pair] = static_cast<uint8_t>(low | (high << 4));
  }
  packed[static_cast<size_t>(row) * (kDim / 16) + lane] = output;
}

}  // namespace

int rms_norm_gated_silu_quant_fp4_bf16(
    const void* x, const void* gate, const void* weight, void* out,
    void* packed, void* sfa, int rows, int dim, float eps,
    cudaStream_t stream) {
  if (!x || !gate || !weight || !out || !packed || !sfa) return 1;
  if (rows <= 0 || dim != kDim) return 2;
  rms_norm_gated_silu_quant_kernel<<<rows, kDim, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(out), static_cast<uint2*>(packed),
      static_cast<uint8_t*>(sfa), rows, rows * kDim, eps);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? 0 : -static_cast<int>(error);
}

}  // namespace kernels
}  // namespace flash_rt
