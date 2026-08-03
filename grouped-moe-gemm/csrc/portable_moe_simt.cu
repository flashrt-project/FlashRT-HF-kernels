// SPDX-License-Identifier: Apache-2.0
//
// Portable SIMT reference for the grouped NVFP4 block-scaled GEMM.
//
// The sm_120a kernels use cute's `SM120_16x8x64_TN_VS` block-scaled MMA which
// traps on pre-sm120 devices (CUTE_INVALID_CONTROL_PATH). This reference
// computes the same grouped FP4 x FP4 -> BF16 GEMM in pure SIMT FMA so the
// package is usable (slowly) on sm_110 Thor. sm_120 keeps the MMA path.
//
// Semantics (matches moe_m16/moe_m64/moe_blocktile):
//   for tile t, row r (global row grow = t*tile_rows + r), expert e = tile_expert[t]:
//     D[grow, n] = alpha[e] * sum_k  fp4(A[grow,k]) * ue4m3(SFA[swz(grow,k)])
//                                   * fp4(B[e,n,k]) * ue4m3(SFB[e][swz(n,k)])
//   SFA/SFB use the NVFP4 128-row super-block swizzle:
//     byte = (super_row * n_col_super + k/64) * 512
//            + (row & 31) * 16 + ((row >> 5) & 3) * 4 + ((k % 64) / 16)

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {

namespace {

constexpr int THREADS = 256;

__device__ __forceinline__ float fp4_to_float(uint8_t v) {
  int sign = (v & 8) ? -1 : 1;
  int exp = (v >> 1) & 3;
  int man = v & 1;
  if (exp == 0 && man == 0) return sign * 0.0f;
  float val = (exp == 0) ? (0.5f * man)
                         : (float)(1 << (exp - 1)) * (1.0f + 0.5f * man);
  return sign * val;
}

__device__ __forceinline__ float ue4m3_to_float(uint8_t v) {
  int e = (v >> 3) & 0xF;
  int m = v & 7;
  if (e == 0) return m * (1.0f / 512.0f);
  return (1.0f + m * (1.0f / 8.0f)) * exp2f(static_cast<float>(e - 7));
}

__device__ __forceinline__ float fp4_read(const uint8_t* p, int idx) {
  uint8_t byte = p[idx >> 1];
  return fp4_to_float((idx & 1) ? (byte >> 4) : (byte & 0xF));
}

// NVFP4 super-block swizzle byte offset for (row, k) within a flat SFA/SFB buf.
__device__ __forceinline__ int sf_off(int row, int k, int n_col_super) {
  int rb = row >> 7;
  int ri = row & 127;
  int kt = k >> 6;              // 64-element K-tile
  int cb = (k >> 4) & 3;        // 16-element block within the tile
  return (rb * n_col_super + kt) * 512 + (ri & 31) * 16 + ((ri >> 5) & 3) * 4 + cb;
}

__global__ void moe_gemm_simt_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ SFA, const uint8_t* __restrict__ SFB,
    const float* __restrict__ alpha, const int* __restrict__ tile_expert,
    __nv_bfloat16* __restrict__ D,
    int num_tiles, int tile_rows, int N, int K,
    long w_stride, long sfb_stride) {
  int M = num_tiles * tile_rows;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int m = idx / N, n = idx - m * N;
  int e = tile_expert[m / tile_rows];
  if (e < 0) return;  // padded/empty tile sentinel
  int K_half = K / 2;
  const uint8_t* arow = A + (size_t)m * K_half;
  const uint8_t* brow = B + (size_t)e * w_stride + (size_t)n * K_half;
  const uint8_t* sfb = SFB + (size_t)e * sfb_stride;
  int n_col_super = (K / 16 + 3) / 4;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    float av = fp4_read(arow, k);
    av *= ue4m3_to_float(SFA[sf_off(m, k, n_col_super)]);
    float bv = fp4_read(brow, k);
    bv *= ue4m3_to_float(sfb[sf_off(n, k, n_col_super)]);
    acc += av * bv;
  }
  D[idx] = __float2bfloat16(acc * alpha[e]);
}

}  // namespace

int moe_gemm_bf16_simt(
    const void* A_tiled, const void* B_stack, const void* SFA_tiled,
    const void* SFB_stack, void* D, const void* alpha_stack,
    const void* tile_expert, int num_tiles, int tile_rows, int N, int K,
    long input_scale_stride, long w_stride, long sfb_stride,
    cudaStream_t stream) {
  (void)input_scale_stride;
  if (num_tiles <= 0 || N <= 0 || K <= 0) return 1;
  int M = num_tiles * tile_rows;
  int total = M * N;
  moe_gemm_simt_kernel<<<(total + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(A_tiled),
      reinterpret_cast<const uint8_t*>(B_stack),
      reinterpret_cast<const uint8_t*>(SFA_tiled),
      reinterpret_cast<const uint8_t*>(SFB_stack),
      reinterpret_cast<const float*>(alpha_stack),
      reinterpret_cast<const int*>(tile_expert),
      reinterpret_cast<__nv_bfloat16*>(D),
      num_tiles, tile_rows, N, K, w_stride, sfb_stride);
  return (cudaGetLastError() == cudaSuccess) ? 0 : 1;
}

}  // namespace gemm
}  // namespace flash_rt
