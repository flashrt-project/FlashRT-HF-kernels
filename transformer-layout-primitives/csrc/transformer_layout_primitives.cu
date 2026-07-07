// SPDX-License-Identifier: Apache-2.0

#include "transformer_layout_primitives.cuh"

#include <cmath>

namespace flashrt_hub {
namespace transformer_layout {

namespace {

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
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

__global__ void fill_neginf_kernel(__nv_bfloat16* dst, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) dst[idx] = __float2bfloat16(-1e30f);
}

__global__ void add_bias_kernel(__nv_bfloat16* data, const __nv_bfloat16* bias,
                                int rows, int cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * cols) return;
  const int col = idx % cols;
  data[idx] = __float2bfloat16(__bfloat162float(data[idx]) + __bfloat162float(bias[col]));
}

__global__ void repeat_heads_kernel(const __nv_bfloat16* src, __nv_bfloat16* dst,
                                    int seq, int src_heads, int head_dim, int repeat) {
  const int total = seq * src_heads * head_dim;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int d = idx % head_dim;
  const int h_src = (idx / head_dim) % src_heads;
  const int s = idx / (head_dim * src_heads);
  const __nv_bfloat16 v = src[idx];
  const int dst_base = (s * src_heads * repeat + h_src * repeat) * head_dim + d;
  for (int r = 0; r < repeat; ++r) {
    dst[dst_base + r * head_dim] = v;
  }
}

__global__ void text_gather_kernel(const __nv_bfloat16* src, __nv_bfloat16* dst,
                                   int seq, int dim) {
  const int row = blockIdx.x;
  const int batch = row >> 1;
  const int which = row & 1;
  const int src_row = batch * seq + (which ? (seq - 1) : 0);
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    dst[row * dim + col] = src[src_row * dim + col];
  }
}

__global__ void text_scatter_kernel(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                    int seq, int dim) {
  const int row = blockIdx.x;
  const int batch = row >> 1;
  const int which = row & 1;
  const int dst_row = batch * seq + (which ? (seq - 1) : 0);
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    dst[dst_row * dim + col] = src[row * dim + col];
  }
}

__global__ void rope_rotate_half_kernel(__nv_bfloat16* x, const __nv_bfloat16* cos,
                                        const __nv_bfloat16* sin, int seq,
                                        int heads, int head_dim) {
  const int half = head_dim >> 1;
  const int total = seq * heads * half;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int d = idx % half;
  const int t = idx / half;
  const int h = t % heads;
  const int s = t / heads;
  const int base = (s * heads + h) * head_dim;
  const float c = __bfloat162float(cos[s * head_dim + d]);
  const float si = __bfloat162float(sin[s * head_dim + d]);
  const float lo = __bfloat162float(x[base + d]);
  const float hi = __bfloat162float(x[base + half + d]);
  x[base + d] = __float2bfloat16(lo * c - hi * si);
  x[base + half + d] = __float2bfloat16(hi * c + lo * si);
}

__global__ void qk_rmsnorm_rope_kernel(__nv_bfloat16* qk, const __nv_bfloat16* weight,
                                       const __nv_bfloat16* cos, const __nv_bfloat16* sin,
                                       int rows, int heads, int head_dim, float eps) {
  extern __shared__ float partial[];
  const int row_head = blockIdx.x;
  if (row_head >= rows * heads) return;
  const int row = row_head / heads;
  const int base = row_head * head_dim;

  float ss = 0.0f;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const float v = __bfloat162float(qk[base + d]);
    ss += v * v;
  }
  const float rms = rsqrtf(block_sum(ss, partial) / head_dim + eps);
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    qk[base + d] = __float2bfloat16(__bfloat162float(qk[base + d]) * rms * __bfloat162float(weight[d]));
  }
  __syncthreads();

  const int half = head_dim >> 1;
  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    const float c = __bfloat162float(cos[row * head_dim + d]);
    const float si = __bfloat162float(sin[row * head_dim + d]);
    const float lo = __bfloat162float(qk[base + d]);
    const float hi = __bfloat162float(qk[base + half + d]);
    qk[base + d] = __float2bfloat16(lo * c - hi * si);
    qk[base + half + d] = __float2bfloat16(hi * c + lo * si);
  }
}

}  // namespace

void fill_neginf_bf16(__nv_bfloat16* dst, int n, cudaStream_t stream) {
  fill_neginf_kernel<<<(n + 255) / 256, 256, 0, stream>>>(dst, n);
}

void add_bias_bf16(__nv_bfloat16* data, const __nv_bfloat16* bias,
                   int rows, int cols, cudaStream_t stream) {
  add_bias_kernel<<<(rows * cols + 255) / 256, 256, 0, stream>>>(data, bias, rows, cols);
}

void repeat_interleave_heads_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst,
                                  int seq, int src_heads, int head_dim, int repeat,
                                  cudaStream_t stream) {
  const int src_total = seq * src_heads * head_dim;
  repeat_heads_kernel<<<(src_total + 255) / 256, 256, 0, stream>>>(src, dst, seq, src_heads, head_dim, repeat);
}

void text_gather_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst,
                      int batch, int seq, int dim, cudaStream_t stream) {
  text_gather_kernel<<<2 * batch, 256, 0, stream>>>(src, dst, seq, dim);
}

void text_scatter_bf16(__nv_bfloat16* dst, const __nv_bfloat16* src,
                       int batch, int seq, int dim, cudaStream_t stream) {
  text_scatter_kernel<<<2 * batch, 256, 0, stream>>>(dst, src, seq, dim);
}

void rope_rotate_half_bf16(__nv_bfloat16* x, const __nv_bfloat16* cos,
                           const __nv_bfloat16* sin, int seq, int heads,
                           int head_dim, cudaStream_t stream) {
  const int total = seq * heads * (head_dim / 2);
  rope_rotate_half_kernel<<<(total + 255) / 256, 256, 0, stream>>>(x, cos, sin, seq, heads, head_dim);
}

void qk_rmsnorm_rope_bf16(__nv_bfloat16* qk, const __nv_bfloat16* weight,
                          const __nv_bfloat16* cos, const __nv_bfloat16* sin,
                          int rows, int heads, int head_dim, float eps,
                          cudaStream_t stream) {
  qk_rmsnorm_rope_kernel<<<rows * heads, 256, 8 * sizeof(float), stream>>>(
      qk, weight, cos, sin, rows, heads, head_dim, eps);
}

}  // namespace transformer_layout
}  // namespace flashrt_hub
