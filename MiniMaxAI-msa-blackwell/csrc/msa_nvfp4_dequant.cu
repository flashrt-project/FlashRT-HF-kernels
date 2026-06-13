// SPDX-License-Identifier: Apache-2.0
//
// NVFP4 swizzled -> BF16 dequantization for Blackwell.
// This mirrors FlashRT's MiniMax-Spark W4A16 quality path: packed e2m1 FP4
// values plus UE4M3 block scales in the 128x4 swizzled layout are expanded to
// dense BF16 so BF16 activations can be used without activation quantization.

#include "msa_nvfp4_dequant.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flashrt_minimax_msa {
namespace {

__device__ __constant__ float c_fp4_codebook[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float ue4m3_decode(uint8_t value) {
  const int exponent = (value >> 3) & 0xF;
  const int mantissa = value & 0x7;
  if (exponent == 0) {
    return static_cast<float>(mantissa) * 0.001953125f;
  }
  if (exponent == 0xF && mantissa == 7) {
    return 0.0f;
  }
  return (1.0f + static_cast<float>(mantissa) * 0.125f) *
         exp2f(static_cast<float>(exponent - 7));
}

__device__ __forceinline__ int scale_128x4_offset(int row,
                                                  int scale_col,
                                                  int scale_cols) {
  const int tiles_n = (scale_cols + 3) / 4;
  const int tile_m = row >> 7;
  const int tile_n = scale_col >> 2;
  const int outer = row & 127;
  const int inner = scale_col & 3;
  return (tile_m * tiles_n + tile_n) * 512 +
         (outer & 31) * 16 +
         ((outer >> 5) & 3) * 4 +
         inner;
}

__global__ void nvfp4_dequant_kernel(const uint8_t* __restrict__ packed,
                                     const uint8_t* __restrict__ scale_128x4,
                                     __nv_bfloat16* __restrict__ out,
                                     long total_bytes,
                                     int cols,
                                     int scale_cols,
                                     float global_scale) {
  const int packed_cols = cols >> 1;
  for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total_bytes;
       idx += static_cast<long>(gridDim.x) * blockDim.x) {
    const int row = static_cast<int>(idx / packed_cols);
    const int byte_col = static_cast<int>(idx - static_cast<long>(row) * packed_cols);
    const int col0 = byte_col << 1;
    const int scale_col = col0 >> 4;
    const float scale =
        ue4m3_decode(__ldg(scale_128x4 + scale_128x4_offset(row, scale_col, scale_cols))) *
        global_scale;
    const uint8_t byte = __ldg(packed + idx);
    const long base = static_cast<long>(row) * cols + col0;
    out[base] = __float2bfloat16(c_fp4_codebook[byte & 0xF] * scale);
    out[base + 1] = __float2bfloat16(c_fp4_codebook[(byte >> 4) & 0xF] * scale);
  }
}

}  // namespace

void nvfp4_dequant_swizzled_to_bf16_cuda(const uint8_t* packed,
                                         const uint8_t* scale_128x4,
                                         void* out_bf16,
                                         int rows,
                                         int cols,
                                         float global_scale,
                                         void* stream) {
  const long total_bytes = static_cast<long>(rows) * (cols >> 1);
  const int scale_cols = cols >> 4;
  const int threads = 256;
  long blocks = (total_bytes + threads - 1) / threads;
  if (blocks > 65535) {
    blocks = 65535;
  }
  nvfp4_dequant_kernel<<<static_cast<int>(blocks), threads, 0,
                         reinterpret_cast<cudaStream_t>(stream)>>>(
      packed,
      scale_128x4,
      reinterpret_cast<__nv_bfloat16*>(out_bf16),
      total_bytes,
      cols,
      scale_cols,
      global_scale);
}

}  // namespace flashrt_minimax_msa
