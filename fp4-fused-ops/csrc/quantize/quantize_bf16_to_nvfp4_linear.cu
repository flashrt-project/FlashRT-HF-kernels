// SPDX-License-Identifier: Apache-2.0
#include "quantize_bf16_to_nvfp4_linear.cuh"

#include <cmath>

namespace flash_rt::quantize {
namespace {

__device__ __forceinline__ uint8_t fp32_to_e2m1(float value) {
  const uint8_t sign = value < 0.0f ? 0x8u : 0x0u;
  const float magnitude = fabsf(value);
  uint8_t encoded;
  if (magnitude < 0.25f) encoded = 0;
  else if (magnitude < 0.75f) encoded = 1;
  else if (magnitude < 1.25f) encoded = 2;
  else if (magnitude < 1.75f) encoded = 3;
  else if (magnitude < 2.5f) encoded = 4;
  else if (magnitude < 3.5f) encoded = 5;
  else if (magnitude < 5.0f) encoded = 6;
  else encoded = 7;
  return sign | encoded;
}

__device__ __forceinline__ uint8_t fp32_to_ue4m3_ceil(float value) {
  if (value <= 0.0f) return 0;
  if (value > 240.0f) return 0xFE;

  const uint32_t bits = __float_as_uint(value);
  const int float_exp = static_cast<int>((bits >> 23) & 0xFF) - 127;
  const uint32_t fraction = bits & 0x7FFFFF;
  int ue_exp = float_exp + 7;
  if (ue_exp <= 0) {
    int mantissa = static_cast<int>(ceilf(value * 512.0f));
    if (mantissa > 7) return 1 << 3;
    if (mantissa < 1) mantissa = 1;
    return static_cast<uint8_t>(mantissa);
  }
  if (ue_exp >= 15) return 0xFE;

  int mantissa = static_cast<int>(fraction >> 20);
  if (fraction & 0xFFFFF) ++mantissa;
  if (mantissa >= 8) {
    mantissa = 0;
    ++ue_exp;
  }
  if (ue_exp >= 15) return 0xFE;
  return static_cast<uint8_t>((ue_exp << 3) | mantissa);
}

__device__ __forceinline__ float ue4m3_to_fp32(uint8_t value) {
  const int exponent = (value >> 3) & 0xF;
  const int mantissa = value & 0x7;
  if (exponent == 0) return ldexpf(static_cast<float>(mantissa) / 8.0f, -6);
  return ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

__global__ void quantize_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ scale_factors,
    int cols,
    int blocks_per_row) {
  const int row = blockIdx.x;
  const auto* row_input = input + static_cast<size_t>(row) * cols;
  auto* row_packed = packed + static_cast<size_t>(row) * (cols / 2);
  auto* row_scales =
      scale_factors + static_cast<size_t>(row) * blocks_per_row;
  extern __shared__ float shared_scales[];

  for (int block = threadIdx.x; block < blocks_per_row;
       block += blockDim.x) {
    shared_scales[block] = 0.0f;
  }
  __syncthreads();

  for (int index = threadIdx.x; index < cols; index += blockDim.x) {
    const float value = fabsf(__bfloat162float(row_input[index]));
    atomicMax(
        reinterpret_cast<int*>(&shared_scales[index >> 4]),
        __float_as_int(value));
  }
  __syncthreads();

  for (int block = threadIdx.x; block < blocks_per_row;
       block += blockDim.x) {
    const float amax =
        __int_as_float(*reinterpret_cast<int*>(&shared_scales[block]));
    const uint8_t encoded = fp32_to_ue4m3_ceil(amax / 6.0f);
    row_scales[block] = encoded;
    shared_scales[block] = ue4m3_to_fp32(encoded);
  }
  __syncthreads();

  for (int pair = threadIdx.x; pair < cols / 2; pair += blockDim.x) {
    const int index = pair * 2;
    const float scale = shared_scales[index >> 4];
    const float inverse_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    const uint8_t low =
        fp32_to_e2m1(__bfloat162float(row_input[index]) * inverse_scale);
    const uint8_t high =
        fp32_to_e2m1(__bfloat162float(row_input[index + 1]) * inverse_scale);
    row_packed[pair] = static_cast<uint8_t>((high << 4) | low);
  }
}

}  // namespace

void quantize_bf16_to_nvfp4_linear(
    const __nv_bfloat16* input,
    uint8_t* packed,
    uint8_t* scale_factors,
    int rows,
    int cols,
    cudaStream_t stream) {
  const int blocks_per_row = cols / 16;
  quantize_kernel<<<rows, 256, blocks_per_row * sizeof(float), stream>>>(
      input, packed, scale_factors, cols, blocks_per_row);
}

}  // namespace flash_rt::quantize
