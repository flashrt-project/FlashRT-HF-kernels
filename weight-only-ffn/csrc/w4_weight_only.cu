#include "weight_only_ffn.cuh"

#include "w4a16_gemm_sm120.cuh"
#include "w4a16_matvec_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flashrt_weight_only {
namespace {

__device__ __forceinline__ uint8_t fp32_to_e2m1(float x) {
  const uint8_t sign = x < 0.0f ? 0x8u : 0u;
  const float ax = fabsf(x);
  uint8_t mag;
  if (ax <= 0.25f) mag = 0;
  else if (ax <= 0.75f) mag = 1;
  else if (ax <= 1.25f) mag = 2;
  else if (ax <= 1.75f) mag = 3;
  else if (ax <= 2.5f) mag = 4;
  else if (ax <= 3.5f) mag = 5;
  else if (ax <= 5.0f) mag = 6;
  else mag = 7;
  return sign | mag;
}

__device__ __forceinline__ float e2m1_to_fp32(uint8_t value) {
  constexpr float mags[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float mag = mags[value & 0x7u];
  return (value & 0x8u) ? -mag : mag;
}

__device__ __forceinline__ int sfb_offset(int row, int block, int n_col_super) {
  const int row_block = row >> 7;
  const int row_inner = row & 127;
  return (row_block * n_col_super + (block >> 2)) * 512
      + (row_inner & 31) * 16 + ((row_inner >> 5) & 3) * 4
      + (block & 3);
}

__device__ __forceinline__ float decode_scale(uint8_t byte) {
  __nv_fp8_e4m3 value;
  *reinterpret_cast<uint8_t*>(&value) = byte;
  return static_cast<float>(value);
}

__global__ void quantize_w4_kernel(const __nv_bfloat16* weight,
                                   uint8_t* packed, uint8_t* sfb,
                                   int rows, int cols, int n_col_super) {
  const int block = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  const int blocks = cols >> 4;
  if (row >= rows || block >= blocks) return;

  const int base = row * cols + block * 16;
  float values[16];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    values[i] = __bfloat162float(weight[base + i]);
    amax = fmaxf(amax, fabsf(values[i]));
  }
  const float desired = fmaxf(amax / 6.0f, 1.0e-12f);
  const __nv_fp8_e4m3 scale_q(desired);
  const float scale = static_cast<float>(scale_q);
  sfb[sfb_offset(row, block, n_col_super)] =
      *reinterpret_cast<const uint8_t*>(&scale_q);

  const int out_base = row * (cols >> 1) + block * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint8_t lo = fp32_to_e2m1(values[2 * i] / scale);
    const uint8_t hi = fp32_to_e2m1(values[2 * i + 1] / scale);
    packed[out_base + i] = lo | static_cast<uint8_t>(hi << 4);
  }
}

__global__ void dequantize_w4_kernel(const uint8_t* packed,
                                     const uint8_t* sfb,
                                     __nv_bfloat16* weight,
                                     int rows, int cols, int n_col_super) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) return;
  const int row = idx / cols;
  const int col = idx - row * cols;
  const int block = col >> 4;
  const uint8_t byte = packed[row * (cols >> 1) + (col >> 1)];
  const uint8_t nibble = (col & 1) ? byte >> 4 : byte & 0x0f;
  const float scale = decode_scale(sfb[sfb_offset(row, block, n_col_super)]);
  weight[idx] = __float2bfloat16(e2m1_to_fp32(nibble) * scale);
}

}  // namespace

int quantize_w4_weight_bf16(const void* weight, void* packed, void* sfb,
                            int rows, int cols, cudaStream_t stream) {
  if (!weight || !packed || !sfb || rows <= 0 || cols <= 0 || cols % 64 != 0) return 1;
  const int blocks = cols / 16;
  const int n_col_super = (blocks + 3) / 4;
  dim3 block(128);
  dim3 grid((blocks + block.x - 1) / block.x, rows);
  quantize_w4_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(weight), static_cast<uint8_t*>(packed),
      static_cast<uint8_t*>(sfb), rows, cols, n_col_super);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : -static_cast<int>(error);
}

int dequantize_w4_weight_bf16(const void* packed, const void* sfb, void* weight,
                              int rows, int cols, cudaStream_t stream) {
  if (!packed || !sfb || !weight || rows <= 0 || cols <= 0 || cols % 64 != 0) return 1;
  const int n_col_super = ((cols / 16) + 3) / 4;
  const int total = rows * cols;
  dequantize_w4_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      static_cast<const uint8_t*>(packed), static_cast<const uint8_t*>(sfb),
      static_cast<__nv_bfloat16*>(weight), rows, cols, n_col_super);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : -static_cast<int>(error);
}

int w4a16_linear_bf16(const void* x, const void* packed, const void* sfb,
                      void* out, int m, int n, int k, float alpha, int variant,
                      cudaStream_t stream) {
  if (!x || !packed || !sfb || !out || m <= 0 || n <= 0 || k <= 0 || k % 64 != 0) return 1;
  if (variant < 0 || variant > 3) return 2;
  if (variant == 0 && m > 4) return 3;
  const bool row_gemv = variant >= 2 || variant == 0;
  if (row_gemv) {
    if (m > 8) return 3;
    return flash_rt::kernels::w4a16_smallm_sm120_bf16(
        x, packed, sfb, out, m, n, k, alpha,
        variant == 2 ? 2 : 3, stream);
  }
  return flash_rt::gemm::w4a16_gemm_sm120_bf16(
      x, packed, sfb, out, m, n, k, alpha, stream);
}

}  // namespace flashrt_weight_only
