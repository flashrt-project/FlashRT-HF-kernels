// SPDX-License-Identifier: Apache-2.0

#include "int8_transformer_primitives.cuh"

#include <cmath>

namespace flashrt_hub {
namespace int8_transformer {

namespace {

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, off);
  }
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
  }
  return v;
}

__device__ __forceinline__ float block_sum(float v, float* shared) {
  v = warp_sum(v);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  v = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
  if (wid == 0) v = warp_sum(v);
  if (threadIdx.x == 0) shared[0] = v;
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ float block_max(float v, float* shared) {
  v = warp_max(v);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  v = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
  if (wid == 0) v = warp_max(v);
  if (threadIdx.x == 0) shared[0] = v;
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ int8_t quant_i8(float v) {
  int q = __float2int_rn(v);
  q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
  return static_cast<int8_t>(q);
}

__global__ void quantize_int8_static_kernel(
    const __nv_bfloat16* __restrict__ input,
    int8_t* __restrict__ output,
    const float* __restrict__ scale,
    int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const float inv_s = 1.0f / fmaxf(*scale, 1e-12f);
  output[idx] = quant_i8(__bfloat162float(input[idx]) * inv_s);
}

__global__ void quantize_int8_rowwise_kernel(
    const __nv_bfloat16* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const __nv_bfloat16* in = input + static_cast<int64_t>(row) * cols;
  int8_t* out = output + static_cast<int64_t>(row) * cols;

  float local_max = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(__bfloat162float(in[col])));
  }
  __shared__ float partial[8];
  const float max_abs = block_max(local_max, partial);
  __shared__ float scale_s;
  if (threadIdx.x == 0) {
    scale_s = fmaxf(max_abs / 127.0f, 1e-10f);
    scales[row] = scale_s;
  }
  __syncthreads();
  const float inv_s = 1.0f / scale_s;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    out[col] = quant_i8(__bfloat162float(in[col]) * inv_s);
  }
}

__global__ void quantize_int8_rowwise_static_kernel(
    const __nv_bfloat16* __restrict__ input,
    int8_t* __restrict__ output,
    const float* __restrict__ scales,
    int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const __nv_bfloat16* in = input + static_cast<int64_t>(row) * cols;
  int8_t* out = output + static_cast<int64_t>(row) * cols;
  const float inv_s = 1.0f / fmaxf(__ldg(&scales[row]), 1e-12f);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    out[col] = quant_i8(__bfloat162float(in[col]) * inv_s);
  }
}

__global__ void dequant_int32_to_bf16_kernel(
    const int32_t* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ act_scale,
    const float* __restrict__ weight_scale,
    int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  output[idx] = __float2bfloat16(static_cast<float>(input[idx]) * (*act_scale) * (*weight_scale));
}

__global__ void rms_norm_quantize_int8_rowwise_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    int8_t* __restrict__ out,
    float* __restrict__ scales,
    int rows, int cols, float eps) {
  extern __shared__ float smem[];
  float* partial = smem + cols;
  const int row = blockIdx.x;
  if (row >= rows) return;
  const __nv_bfloat16* xr = x + static_cast<int64_t>(row) * cols;
  int8_t* outr = out + static_cast<int64_t>(row) * cols;

  float sum_sq = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float v = __bfloat162float(xr[col]);
    smem[col] = v;
    sum_sq += v * v;
  }
  const float rms = rsqrtf(block_sum(sum_sq, partial) / cols + eps);

  float local_max = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float v = smem[col] * rms * __bfloat162float(weight[col]);
    smem[col] = v;
    local_max = fmaxf(local_max, fabsf(v));
  }
  const float max_abs = block_max(local_max, partial);
  __shared__ float scale_s;
  if (threadIdx.x == 0) {
    scale_s = fmaxf(max_abs / 127.0f, 1e-12f);
    scales[row] = scale_s;
  }
  __syncthreads();
  const float inv_s = 1.0f / scale_s;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    outr[col] = quant_i8(smem[col] * inv_s);
  }
}

__global__ void residual_add_rms_norm_quantize_int8_rowwise_kernel(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    int8_t* __restrict__ out,
    float* __restrict__ scales,
    int rows, int cols, float eps) {
  extern __shared__ float smem[];
  float* partial = smem + cols;
  const int row = blockIdx.x;
  if (row >= rows) return;
  __nv_bfloat16* rr = residual + static_cast<int64_t>(row) * cols;
  const __nv_bfloat16* xr = x + static_cast<int64_t>(row) * cols;
  int8_t* outr = out + static_cast<int64_t>(row) * cols;

  float sum_sq = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float v = __bfloat162float(rr[col]) + __bfloat162float(xr[col]);
    rr[col] = __float2bfloat16(v);
    smem[col] = v;
    sum_sq += v * v;
  }
  const float rms = rsqrtf(block_sum(sum_sq, partial) / cols + eps);

  float local_max = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float v = smem[col] * rms * __bfloat162float(weight[col]);
    smem[col] = v;
    local_max = fmaxf(local_max, fabsf(v));
  }
  const float max_abs = block_max(local_max, partial);
  __shared__ float scale_s;
  if (threadIdx.x == 0) {
    scale_s = fmaxf(max_abs / 127.0f, 1e-12f);
    scales[row] = scale_s;
  }
  __syncthreads();
  const float inv_s = 1.0f / scale_s;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    outr[col] = quant_i8(smem[col] * inv_s);
  }
}

int row_threads(int cols) {
  int threads = cols < 256 ? cols : 256;
  threads = ((threads + 31) / 32) * 32;
  return threads < 32 ? 32 : threads;
}

}  // namespace

void quantize_int8_static_bf16(const __nv_bfloat16* input, int8_t* output,
                               const float* scale, int n,
                               cudaStream_t stream) {
  const int threads = 256;
  int blocks = (n + threads - 1) / threads;
  if (blocks > 65535) blocks = 65535;
  quantize_int8_static_kernel<<<blocks, threads, 0, stream>>>(input, output, scale, n);
}

void quantize_int8_rowwise_bf16(const __nv_bfloat16* input, int8_t* output,
                                float* scales, int rows, int cols,
                                cudaStream_t stream) {
  quantize_int8_rowwise_kernel<<<rows, row_threads(cols), 0, stream>>>(input, output, scales, rows, cols);
}

void quantize_int8_rowwise_static_bf16(const __nv_bfloat16* input, int8_t* output,
                                       const float* scales, int rows, int cols,
                                       cudaStream_t stream) {
  quantize_int8_rowwise_static_kernel<<<rows, row_threads(cols), 0, stream>>>(
      input, output, scales, rows, cols);
}

void dequant_int32_to_bf16(const int32_t* input, __nv_bfloat16* output,
                           const float* act_scale, const float* weight_scale,
                           int n, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  dequant_int32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
      input, output, act_scale, weight_scale, n);
}

void rms_norm_quantize_int8_rowwise_bf16(const __nv_bfloat16* x,
                                         const __nv_bfloat16* weight,
                                         int8_t* out, float* scales,
                                         int rows, int cols, float eps,
                                         cudaStream_t stream) {
  const int smem = (cols + 32) * sizeof(float);
  rms_norm_quantize_int8_rowwise_kernel<<<rows, 256, smem, stream>>>(
      x, weight, out, scales, rows, cols, eps);
}

void residual_add_rms_norm_quantize_int8_rowwise_bf16(
    __nv_bfloat16* residual, const __nv_bfloat16* x,
    const __nv_bfloat16* weight, int8_t* out, float* scales,
    int rows, int cols, float eps, cudaStream_t stream) {
  const int smem = (cols + 32) * sizeof(float);
  residual_add_rms_norm_quantize_int8_rowwise_kernel<<<rows, 256, smem, stream>>>(
      residual, x, weight, out, scales, rows, cols, eps);
}

}  // namespace int8_transformer
}  // namespace flashrt_hub
