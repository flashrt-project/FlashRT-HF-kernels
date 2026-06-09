// SPDX-License-Identifier: Apache-2.0
//
// Tensor-facing BF16 RMSNorm / residual RMSNorm / static FP8 quant kernels.
// Logic follows official/FlashRT csrc/kernels/norm.cu for the BF16 paths.

#include "residual_norm_quant.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace residual_norm_quant {
namespace {

constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float to_f32(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 from_f32(float x) {
  return __float2bfloat16(x);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
  val += __shfl_down_sync(0xffffffff, val, 16);
  val += __shfl_down_sync(0xffffffff, val, 8);
  val += __shfl_down_sync(0xffffffff, val, 4);
  val += __shfl_down_sync(0xffffffff, val, 2);
  val += __shfl_down_sync(0xffffffff, val, 1);
  return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  val = warp_reduce_sum(val);
  if (lane == 0) {
    shared[warp_id] = val;
  }
  __syncthreads();
  const int num_warps = blockDim.x >> 5;
  val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
  if (warp_id == 0) {
    val = warp_reduce_sum(val);
  }
  if (threadIdx.x == 0) {
    shared[0] = val;
  }
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ __nv_fp8_e4m3 quant_fp8(float x, float inv_scale) {
  const float q = fminf(fmaxf(x * inv_scale, -kFp8Max), kFp8Max);
  return __nv_fp8_e4m3(q);
}

__global__ void rms_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int dim,
    float eps) {
  const int row = blockIdx.x;
  const __nv_bfloat162* x2 =
      reinterpret_cast<const __nv_bfloat162*>(x + row * dim);
  const __nv_bfloat162* w2 =
      reinterpret_cast<const __nv_bfloat162*>(weight);
  __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out + row * dim);
  const int dim2 = dim >> 1;

  extern __shared__ float shared[];
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 val = x2[i];
    const float v0 = to_f32(val.x);
    const float v1 = to_f32(val.y);
    local_sum += v0 * v0 + v1 * v1;
  }
  const float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 xv = x2[i];
    const __nv_bfloat162 wv = w2[i];
    const float v0 = to_f32(xv.x) * rms * to_f32(wv.x);
    const float v1 = to_f32(xv.y) * rms * to_f32(wv.y);
    out2[i] = __halves2bfloat162(from_f32(v0), from_f32(v1));
  }
}

__global__ void layer_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int dim,
    float eps) {
  const int row = blockIdx.x;
  const __nv_bfloat162* x2 =
      reinterpret_cast<const __nv_bfloat162*>(x + row * dim);
  const __nv_bfloat162* w2 =
      reinterpret_cast<const __nv_bfloat162*>(weight);
  const __nv_bfloat162* b2 =
      reinterpret_cast<const __nv_bfloat162*>(bias);
  __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out + row * dim);
  const int dim2 = dim >> 1;

  extern __shared__ float shared[];
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 val = x2[i];
    local_sum += to_f32(val.x) + to_f32(val.y);
  }
  const float mean = block_reduce_sum(local_sum, shared) / dim;

  float local_var = 0.0f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 val = x2[i];
    const float d0 = to_f32(val.x) - mean;
    const float d1 = to_f32(val.y) - mean;
    local_var += d0 * d0 + d1 * d1;
  }
  const float inv_std = rsqrtf(block_reduce_sum(local_var, shared) / dim + eps);

  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 xv = x2[i];
    const __nv_bfloat162 wv = w2[i];
    const __nv_bfloat162 bv = b2[i];
    const float v0 = (to_f32(xv.x) - mean) * inv_std * to_f32(wv.x) + to_f32(bv.x);
    const float v1 = (to_f32(xv.y) - mean) * inv_std * to_f32(wv.y) + to_f32(bv.y);
    out2[i] = __halves2bfloat162(from_f32(v0), from_f32(v1));
  }
}

__global__ void rms_norm_quant_fp8_static_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_fp8_e4m3* __restrict__ out,
    int dim,
    float eps,
    const float* __restrict__ scale) {
  const int row = blockIdx.x;
  const __nv_bfloat162* x2 =
      reinterpret_cast<const __nv_bfloat162*>(x + row * dim);
  const __nv_bfloat162* w2 =
      reinterpret_cast<const __nv_bfloat162*>(weight);
  __nv_fp8_e4m3* out_row = out + row * dim;
  const int dim2 = dim >> 1;

  extern __shared__ float shared[];
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 val = x2[i];
    const float v0 = to_f32(val.x);
    const float v1 = to_f32(val.y);
    local_sum += v0 * v0 + v1 * v1;
  }
  const float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
  const float inv_scale = 1.0f / (*scale);

  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 xv = x2[i];
    const __nv_bfloat162 wv = w2[i];
    const float v0 = to_f32(xv.x) * rms * to_f32(wv.x);
    const float v1 = to_f32(xv.y) * rms * to_f32(wv.y);
    out_row[2 * i] = quant_fp8(v0, inv_scale);
    out_row[2 * i + 1] = quant_fp8(v1, inv_scale);
  }
}

__global__ void residual_add_rms_norm_quant_fp8_static_bf16_kernel(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_fp8_e4m3* __restrict__ out,
    int dim,
    float eps,
    const float* __restrict__ scale) {
  const int row = blockIdx.x;
  __nv_bfloat162* res2 =
      reinterpret_cast<__nv_bfloat162*>(residual + row * dim);
  const __nv_bfloat162* x2 =
      reinterpret_cast<const __nv_bfloat162*>(x + row * dim);
  const __nv_bfloat162* w2 =
      reinterpret_cast<const __nv_bfloat162*>(weight);
  __nv_fp8_e4m3* out_row = out + row * dim;
  const int dim2 = dim >> 1;

  extern __shared__ float shared[];
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 rv = res2[i];
    const __nv_bfloat162 xv = x2[i];
    const float r0 = to_f32(rv.x) + to_f32(xv.x);
    const float r1 = to_f32(rv.y) + to_f32(xv.y);
    res2[i] = __halves2bfloat162(from_f32(r0), from_f32(r1));
    local_sum += r0 * r0 + r1 * r1;
  }
  const float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
  const float inv_scale = 1.0f / (*scale);

  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 rv = res2[i];
    const __nv_bfloat162 wv = w2[i];
    const float v0 = to_f32(rv.x) * rms * to_f32(wv.x);
    const float v1 = to_f32(rv.y) * rms * to_f32(wv.y);
    out_row[2 * i] = quant_fp8(v0, inv_scale);
    out_row[2 * i + 1] = quant_fp8(v1, inv_scale);
  }
}

}  // namespace

void rms_norm_bf16(
    const void* x_bf16,
    const void* weight_bf16,
    void* out_bf16,
    int rows,
    int dim,
    float eps,
    cudaStream_t stream) {
  rms_norm_bf16_kernel<<<rows, 256, 256 * sizeof(float), stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(weight_bf16),
      reinterpret_cast<__nv_bfloat16*>(out_bf16),
      dim,
      eps);
}

void layer_norm_bf16(
    const void* x_bf16,
    const void* weight_bf16,
    const void* bias_bf16,
    void* out_bf16,
    int rows,
    int dim,
    float eps,
    cudaStream_t stream) {
  layer_norm_bf16_kernel<<<rows, 256, 256 * sizeof(float), stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(weight_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      reinterpret_cast<__nv_bfloat16*>(out_bf16),
      dim,
      eps);
}

void rms_norm_quant_fp8_static_bf16(
    const void* x_bf16,
    const void* weight_bf16,
    void* out_fp8,
    int rows,
    int dim,
    float eps,
    const float* scale,
    cudaStream_t stream) {
  rms_norm_quant_fp8_static_bf16_kernel<<<
      rows,
      256,
      256 * sizeof(float),
      stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(weight_bf16),
      reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
      dim,
      eps,
      scale);
}

void residual_add_rms_norm_quant_fp8_static_bf16(
    void* residual_bf16,
    const void* x_bf16,
    const void* weight_bf16,
    void* out_fp8,
    int rows,
    int dim,
    float eps,
    const float* scale,
    cudaStream_t stream) {
  residual_add_rms_norm_quant_fp8_static_bf16_kernel<<<
      rows,
      256,
      256 * sizeof(float),
      stream>>>(
      reinterpret_cast<__nv_bfloat16*>(residual_bf16),
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(weight_bf16),
      reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
      dim,
      eps,
      scale);
}

}  // namespace residual_norm_quant
}  // namespace flash_rt
