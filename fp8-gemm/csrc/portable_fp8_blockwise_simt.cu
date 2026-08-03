// SPDX-License-Identifier: Apache-2.0
//
// Portable SIMT implementation of the block-128 FP8 GEMM.
//
// Computes D[m, n] = sum_k  A[m, k] * act_scale[m, k/128]
//                         * B[n, k] * w_scale[n/128, k/128]
// in pure SIMT FMA so it runs on sm_110 (Thor). One thread per output
// element keeps the kernel trivially correct; this is a compatibility
// path, not a performance kernel (sm_120 keeps the CUTLASS path).

#include "portable_fp8_blockwise_simt.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {

namespace {

constexpr int THREADS = 256;

__global__ void fp8_block128_gemm_simt_kernel(
    const __nv_fp8_e4m3* __restrict__ A,   // (M, K) row-major
    const __nv_fp8_e4m3* __restrict__ B,   // (N, K) row-major
    const float* __restrict__ act_scale,   // (M, K/128)
    const float* __restrict__ w_scale,     // (N/128, K/128)
    __nv_bfloat16* __restrict__ D,         // (M, N) row-major
    int M, int N, int K) {
  const int total = M * N;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const int m = idx / N;
    const int n = idx - m * N;
    const int k_blocks = K >> 7;
    const __nv_fp8_e4m3* Arow = A + (size_t)m * K;
    const __nv_fp8_e4m3* Brow = B + (size_t)n * K;
    const float* ais = act_scale + (size_t)m * k_blocks;
    const float* nws = w_scale + (size_t)(n >> 7) * k_blocks;
    float acc = 0.0f;
    for (int kb = 0; kb < k_blocks; ++kb) {
      const float sa = ais[kb];
      const float sb = nws[kb];
      const int k0 = kb << 7;
      const __nv_fp8_e4m3* ap = Arow + k0;
      const __nv_fp8_e4m3* bp = Brow + k0;
      #pragma unroll 4
      for (int k = 0; k < 128; ++k) {
        acc += (float(ap[k]) * sa) * (float(bp[k]) * sb);
      }
    }
    D[idx] = __float2bfloat16(acc);
  }
}

}  // namespace

void fp8_block128_gemm_simt_bf16out(
    const void* A_fp8,
    const void* B_fp8,
    void*       D_bf16,
    int M, int N, int K,
    const float* act_scale,
    const float* w_scale,
    cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0 || K % 128 != 0 || N % 128 != 0) return;
  const int total = M * N;
  const int blocks = (total + THREADS - 1) / THREADS;
  fp8_block128_gemm_simt_kernel<<<blocks, THREADS, 0, stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(A_fp8),
      reinterpret_cast<const __nv_fp8_e4m3*>(B_fp8),
      act_scale,
      w_scale,
      reinterpret_cast<__nv_bfloat16*>(D_bf16),
      M, N, K);
}

}  // namespace gemm
}  // namespace flash_rt
