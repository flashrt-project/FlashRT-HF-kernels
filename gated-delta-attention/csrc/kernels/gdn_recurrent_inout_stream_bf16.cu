// SPDX-License-Identifier: Apache-2.0
#include "kernels/gdn_recurrent_inout_stream_bf16.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {
namespace {

constexpr int kHeadDim = 128;
constexpr float kEps = 1e-6f;

__device__ __forceinline__ float block_reduce_sum(float value, float* smem) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < (kHeadDim / 32) ? smem[lane] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) smem[0] = value;
  }
  __syncthreads();
  return smem[0];
}

__global__ void recurrent_stream_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    const __nv_bfloat16* __restrict__ state_in,
    __nv_bfloat16* __restrict__ state_out,
    __nv_bfloat16* __restrict__ out,
    int num_v_heads, bool use_qk_l2norm) {
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int column = threadIdx.x;
  if (column >= kHeadDim) return;

  const size_t head_offset =
      (static_cast<size_t>(batch) * num_v_heads + head) * kHeadDim;
  __shared__ float smem[2 * kHeadDim + 32];
  float* q = smem;
  float* k = smem + kHeadDim;
  float* scratch = smem + 2 * kHeadDim;
  q[column] = static_cast<float>(q_in[head_offset + column]);
  k[column] = static_cast<float>(k_in[head_offset + column]);
  __syncthreads();

  if (use_qk_l2norm) {
    const float q_sq = block_reduce_sum(q[column] * q[column], scratch);
    __syncthreads();
    const float k_sq = block_reduce_sum(k[column] * k[column], scratch);
    q[column] *= rsqrtf(q_sq + kEps);
    k[column] *= rsqrtf(k_sq + kEps);
    __syncthreads();
  }
  q[column] *= rsqrtf(static_cast<float>(kHeadDim));
  __syncthreads();

  const float decay = __expf(static_cast<float>(
      g_in[batch * num_v_heads + head]));
  const float beta = static_cast<float>(
      beta_in[batch * num_v_heads + head]);
  const size_t state_offset = head_offset * kHeadDim;

  // Streaming two passes avoids the 128-float per-thread array used by the
  // old implementation. That array spills to local memory; the second read
  // below instead hits the cache for the 32 KiB head slice.
  float kv = 0.0f;
  #pragma unroll 16
  for (int row = 0; row < kHeadDim; ++row) {
    const float state_value = static_cast<float>(
        state_in[state_offset + static_cast<size_t>(row) * kHeadDim + column]) * decay;
    kv = fmaf(state_value, k[row], kv);
  }

  const float value = static_cast<float>(v_in[head_offset + column]);
  const float delta = (value - kv) * beta;
  float output = 0.0f;
  #pragma unroll 16
  for (int row = 0; row < kHeadDim; ++row) {
    const size_t index =
        state_offset + static_cast<size_t>(row) * kHeadDim + column;
    const float updated = fmaf(
        k[row], delta, static_cast<float>(state_in[index]) * decay);
    state_out[index] = __float2bfloat16(updated);
    output = fmaf(updated, q[row], output);
  }
  out[head_offset + column] = __float2bfloat16(output);
}

}  // namespace

int gdn_recurrent_inout_stream_bf16(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, const void* state_in, void* state_out, void* out,
    int B, int num_v_heads, int head_dim, bool use_qk_l2norm,
    cudaStream_t stream) {
  if (!q || !k || !v || !g || !beta || !state_in || !state_out || !out) {
    return 1;
  }
  if (B <= 0 || num_v_heads <= 0 || head_dim != kHeadDim) return 2;
  recurrent_stream_kernel<<<dim3(num_v_heads, B), kHeadDim, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v),
      static_cast<const __nv_bfloat16*>(g),
      static_cast<const __nv_bfloat16*>(beta),
      static_cast<const __nv_bfloat16*>(state_in),
      static_cast<__nv_bfloat16*>(state_out),
      static_cast<__nv_bfloat16*>(out), num_v_heads, use_qk_l2norm);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? 0 : -static_cast<int>(error);
}

}  // namespace kernels
}  // namespace flash_rt
