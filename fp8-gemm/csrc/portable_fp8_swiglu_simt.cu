// SPDX-License-Identifier: Apache-2.0
//
// Portable SIMT implementation of the block-128 FP8 SwiGLU producer.
//
// The sm_89 MMA path is unavailable on pre-sm89 devices; this reference
// computes the same fusion in pure SIMT FMA so it runs on sm_110 (Thor):
//
//   gate[m, n] = sum_k  A[m,k] * act_scale[m, k/128] * B_gate[n,k] * w_scale[n/128, k/128]
//   up[m, n]   = sum_k  A[m,k] * act_scale[m, k/128] * B_up[n,k]   * w_scale[(N+n)/128, k/128]
//   v[m, n]    = bf16( bf16(silu_f32(gate)) * up )        // bf16 roundings match the fused kernel
//   out_scale[m, n/128] = max(amax/448, 1e-12)            // amax = max_{n in block} |v|
//   output[m, n]        = fp8_e4m3( clamp(v / out_scale, -448, 448) )
//
// B is the (2N, K) gate_up_weight; rows [0,N) are the gate, [N,2N) the up
// projection. sm_89 keeps the MMA path; this is a compatibility path only.

#include "portable_fp8_swiglu_simt.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {

namespace {

constexpr int THREADS = 256;
constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float silu_f32(float x) {
  return x / (1.0f + expf(-x));
}

__global__ void fp8_swiglu_quantize_simt_kernel(
    const __nv_fp8_e4m3* __restrict__ A,        // (M, K) row-major
    const __nv_fp8_e4m3* __restrict__ B,        // (2N, K) row-major
    const float* __restrict__ act_scale,        // (M, K/128)
    const float* __restrict__ w_scale,          // (2N/128, K/128)
    __nv_fp8_e4m3* __restrict__ output,         // (M, N) row-major
    float* __restrict__ out_scale,              // (M, N/128)
    int M, int N, int K) {
  const int n_blocks = N >> 7;
  const int k_blocks = K >> 7;
  const int total = M * n_blocks;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const int m = idx / n_blocks;
    const int nb = idx - m * n_blocks;
    const int n0 = nb << 7;
    const __nv_fp8_e4m3* Arow = A + (size_t)m * K;
    const float* ais = act_scale + (size_t)m * k_blocks;
    float v[128];
    float amax = 0.0f;
    for (int c = 0; c < 128; ++c) {
      const int n = n0 + c;
      const __nv_fp8_e4m3* gateB = B + (size_t)n * K;
      const __nv_fp8_e4m3* upB = B + (size_t)(n + N) * K;
      const float* sbg = w_scale + (size_t)(n >> 7) * k_blocks;
      const float* sbu = w_scale + (size_t)((n + N) >> 7) * k_blocks;
      float gate = 0.0f, up = 0.0f;
      for (int kb = 0; kb < k_blocks; ++kb) {
        const float sa = ais[kb];
        const float sb_g = sbg[kb];
        const float sb_u = sbu[kb];
        const int k0 = kb << 7;
        #pragma unroll 4
        for (int k = 0; k < 128; ++k) {
          const float av = float(Arow[k0 + k]) * sa;
          gate += av * (float(gateB[k0 + k]) * sb_g);
          up += av * (float(upB[k0 + k]) * sb_u);
        }
      }
      const float g_bf16 = __bfloat162float(__float2bfloat16(silu_f32(gate)));
      v[c] = __bfloat162float(__float2bfloat16(g_bf16 * up));
      amax = fmaxf(amax, fabsf(v[c]));
    }
    const float sc = fmaxf(amax / kFp8Max, 1.0e-12f);
    out_scale[(size_t)m * n_blocks + nb] = sc;
    const float inv = 1.0f / sc;
    __nv_fp8_e4m3* orow = output + (size_t)m * N;
    for (int c = 0; c < 128; ++c) {
      const float q = fminf(fmaxf(v[c] * inv, -kFp8Max), kFp8Max);
      orow[n0 + c] = __nv_fp8_e4m3(q);
    }
  }
}

}  // namespace

int fp8_blockwise_swiglu_quantize_simt(
    const void* A_fp8, const void* gate_up_fp8, const float* act_scale,
    const float* w_scale, void* output_fp8, float* out_scale,
    int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0 || N % 128 != 0 || K % 128 != 0) return 1;
  const int total = M * (N >> 7);
  const int blocks = (total + THREADS - 1) / THREADS;
  fp8_swiglu_quantize_simt_kernel<<<blocks, THREADS, 0, stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(A_fp8),
      reinterpret_cast<const __nv_fp8_e4m3*>(gate_up_fp8), act_scale, w_scale,
      reinterpret_cast<__nv_fp8_e4m3*>(output_fp8), out_scale, M, N, K);
  return (cudaGetLastError() == cudaSuccess) ? 0 : 1;
}

}  // namespace gemm
}  // namespace flash_rt
