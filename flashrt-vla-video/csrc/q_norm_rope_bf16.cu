#include "q_norm_rope_bf16.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace vla_video {
namespace {

constexpr int kHeadDim = 128;
constexpr int kHalfDim = kHeadDim / 2;
constexpr int kThreads = kHeadDim;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float block_sum_4warp(float value, float* scratch) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    value = (lane < kWarps) ? scratch[lane] : 0.0f;
    #pragma unroll
    for (int offset = 2; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    if (lane == 0) {
      scratch[0] = value;
    }
  }
  __syncthreads();
  return scratch[0];
}

__device__ __forceinline__ void write_norm_rope_row(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* out,
    float eps) {
  const int tid = threadIdx.x;
  __shared__ float normed[kHeadDim];
  __shared__ float scratch[kWarps];

  const float value = __bfloat162float(input[tid]);
  const float square = value * value;
  const float sum_square = block_sum_4warp(square, scratch);
  const float rstd = rsqrtf(sum_square / static_cast<float>(kHeadDim) + eps);
  normed[tid] = value * rstd * __bfloat162float(weight[tid]);
  __syncthreads();

  if (tid < kHalfDim) {
    const float lo = normed[tid];
    const float hi = normed[tid + kHalfDim];
    const float c = __bfloat162float(cos[tid]);
    const float s = __bfloat162float(sin[tid]);
    out[tid] = __float2bfloat16(lo * c - hi * s);
  } else {
    const int idx = tid - kHalfDim;
    const float lo = normed[idx];
    const float hi = normed[tid];
    const float c = __bfloat162float(cos[idx]);
    const float s = __bfloat162float(sin[idx]);
    out[tid] = __float2bfloat16(hi * c + lo * s);
  }
}

__global__ void q_norm_rope_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ out,
    int rows,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  write_norm_rope_row(
      q + row * kHeadDim,
      weight,
      cos,
      sin,
      out + row * kHeadDim,
      eps);
}

__global__ void k_norm_rope_v_cache_kernel(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int rows,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int offset = row * kHeadDim;
  write_norm_rope_row(
      k + offset,
      weight,
      cos,
      sin,
      k_out + offset,
      eps);
  v_out[offset + threadIdx.x] = v[offset + threadIdx.x];
}

}  // namespace

void q_norm_rope_bf16(
    const void* q,
    const void* weight,
    const void* cos,
    const void* sin,
    void* out,
    int rows,
    float eps,
    cudaStream_t stream) {
  q_norm_rope_kernel<<<rows, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(out),
      rows,
      eps);
}

void k_norm_rope_v_cache_bf16(
    const void* k,
    const void* v,
    const void* weight,
    const void* cos,
    const void* sin,
    void* k_out,
    void* v_out,
    int rows,
    float eps,
    cudaStream_t stream) {
  k_norm_rope_v_cache_kernel<<<rows, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(k_out),
      reinterpret_cast<__nv_bfloat16*>(v_out),
      rows,
      eps);
}

}  // namespace vla_video
}  // namespace flash_rt
