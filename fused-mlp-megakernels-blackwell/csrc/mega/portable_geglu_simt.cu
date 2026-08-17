// SPDX-License-Identifier: Apache-2.0
//
// Portable SIMT implementation of the fused FP16 GeGLU megakernel for sm_110a
// (Thor). Block-tiled over shared memory: a 32x32 output tile reuses each X
// row / weight column across the whole tile, so global traffic drops ~32x
// versus a one-thread-per-output reference. sm_100/sm_103 keep the CUTLASS
// megakernel path; this is a compatibility path, not a performance kernel.

#include "portable_geglu_simt.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flashrt {
namespace megakernel {

namespace {

constexpr int THREADS = 256;
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 16;               // per-chunk K depth (fp16 tile fits SMEM)
constexpr int ELEMS_PER_THREAD = (BM * BN) / THREADS;  // 4

__device__ __forceinline__ float gelu_tanh_f32(float x) {
  // Matches cutlass::epilogue::thread::GELU_taylor (the tanh approximation).
  const float t = 0.5f * x;
  return t * (1.0f + tanhf(0.7978845608028654f *
                           (x + 0.044715f * x * x * x)));
}

__global__ void geglu_fused_tiled_kernel(
    const __half* __restrict__ X,      // (M, K) row-major
    const __half* __restrict__ W_gate, // (N, K) row-major
    const __half* __restrict__ W_up,   // (N, K) row-major
    __half* __restrict__ gate_scratch, // (M, N) row-major
    __half* __restrict__ hidden,       // (M, N) row-major
    int M, int N, int K) {
  __shared__ __half xs[BM][BK];
  __shared__ __half wgs[BN][BK];
  __shared__ __half wus[BN][BK];

  const int tile_m = blockIdx.y * BM;
  const int tile_n = blockIdx.x * BN;
  const int nk = (K + BK - 1) / BK;

  float gate_acc[ELEMS_PER_THREAD];
  float up_acc[ELEMS_PER_THREAD];
  int rows[ELEMS_PER_THREAD];
  int cols[ELEMS_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    const int lin = threadIdx.x + i * THREADS;
    const int r = lin >> 5;           // / BN
    const int c = lin & 31;           // % BN
    rows[i] = r;
    cols[i] = c;
    gate_acc[i] = 0.0f;
    up_acc[i] = 0.0f;
  }

  for (int kb = 0; kb < nk; ++kb) {
    const int k0 = kb * BK;
    // cooperative tile load with edge masking
    const int tid = threadIdx.x;
    const int tiles_per_chunk = BM + 2 * BN;  // X tile + gate/up weight tiles
    const int load_total = tiles_per_chunk * BK;
    for (int e = tid; e < load_total; e += THREADS) {
      const int row_sel = e / BK;
      const int j = e % BK;
      const int kk = k0 + j;
      if (row_sel < BM) {
        const int m = tile_m + row_sel;
        xs[row_sel][j] = (m < M && kk < K) ? X[(size_t)m * K + kk] : __float2half(0.0f);
      } else if (row_sel < BM + BN) {
        const int c = row_sel - BM;
        const int n = tile_n + c;
        const bool ok = (n < N && kk < K);
        wgs[c][j] = ok ? W_gate[(size_t)n * K + kk] : __float2half(0.0f);
      } else {
        const int c = row_sel - BM - BN;
        const int n = tile_n + c;
        const bool ok = (n < N && kk < K);
        wus[c][j] = ok ? W_up[(size_t)n * K + kk] : __float2half(0.0f);
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
      const int r = rows[i];
      const int c = cols[i];
      float g = gate_acc[i];
      float u = up_acc[i];
      #pragma unroll
      for (int j = 0; j < BK; ++j) {
        const float xv = __half2float(xs[r][j]);
        g += xv * __half2float(wgs[c][j]);
        u += xv * __half2float(wus[c][j]);
      }
      gate_acc[i] = g;
      up_acc[i] = u;
    }
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    const int m = tile_m + rows[i];
    const int n = tile_n + cols[i];
    if (m >= M || n >= N) continue;
    const __half g_h = __float2half(gelu_tanh_f32(gate_acc[i]));
    const size_t idx = (size_t)m * N + n;
    gate_scratch[idx] = g_h;
    hidden[idx] = __float2half(__half2float(g_h) * up_acc[i]);
  }
}

}  // namespace

int geglu_fused_fp16_simt(
    const void* X, const void* W_gate, const void* W_up,
    void* gate_scratch, void* hidden,
    int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) return 1;
  dim3 block(THREADS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  geglu_fused_tiled_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(X),
      reinterpret_cast<const __half*>(W_gate),
      reinterpret_cast<const __half*>(W_up),
      reinterpret_cast<__half*>(gate_scratch),
      reinterpret_cast<__half*>(hidden),
      M, N, K);
  return (cudaGetLastError() == cudaSuccess) ? 0 : 1;
}

}  // namespace megakernel
}  // namespace flashrt
