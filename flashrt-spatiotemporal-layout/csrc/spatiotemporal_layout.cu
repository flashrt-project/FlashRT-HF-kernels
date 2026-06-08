// SPDX-License-Identifier: Apache-2.0
//
// Tensor-facing spatiotemporal layout kernels. Math follows FlashRT
// csrc/kernels/elementwise.cu NCDHW/BLC and cache helpers.

#include "spatiotemporal_layout.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace spatiotemporal_layout {
namespace {

__global__ void ncdhw_to_blc_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int channels,
    int frames,
    int height,
    int width,
    long long total) {
  const int spatial = frames * height * width;
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  for (; idx < total; idx += stride) {
    const int c = static_cast<int>(idx % channels);
    const int s = static_cast<int>((idx / channels) % spatial);
    const long long b = idx / (static_cast<long long>(spatial) * channels);
    const int w = s % width;
    const int h = (s / width) % height;
    const int t = s / (height * width);
    const long long src =
        (((b * channels + c) * static_cast<long long>(frames) + t) * height + h) * width + w;
    out[idx] = x[src];
  }
}

__global__ void time_unshuffle2_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int channels,
    int frames,
    int height,
    int width,
    long long total) {
  const long long hw = static_cast<long long>(height) * width;
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int ow = static_cast<int>(idx % width);
  long long q = idx / width;
  const int oh = static_cast<int>(q % height);
  q /= height;
  const int ot = static_cast<int>(q % (2 * frames));
  q /= (2 * frames);
  const int c = static_cast<int>(q % channels);
  const long long b = q / channels;
  const int src_group = ot & 1;
  const int src_t = ot >> 1;
  const int src_c = src_group * channels + c;
  const long long src =
      (((b * (2LL * channels) + src_c) * frames + src_t) * hw) +
      static_cast<long long>(oh) * width + ow;
  out[idx] = x[src];
}

__global__ void add_bias_ncdhw_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    int channels,
    int frames,
    int height,
    int width,
    long long total) {
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const long long inner = idx % (static_cast<long long>(channels) * frames * height * width);
  const int c = static_cast<int>(inner / (static_cast<long long>(frames) * height * width));
  const float v = __bfloat162float(x[idx]) + __bfloat162float(bias[c]);
  x[idx] = __float2bfloat16(v);
}

__global__ void update_cache2_ncdhw_bf16_kernel(
    const __nv_bfloat16* __restrict__ cur,
    const __nv_bfloat16* __restrict__ prev,
    __nv_bfloat16* __restrict__ out,
    int channels,
    int frames,
    int height,
    int width,
    long long total_out) {
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_out) return;
  const long long hw = static_cast<long long>(height) * width;
  const long long cache_stride_c = 2LL * hw;
  const long long block = idx / hw;
  const int hw_idx = static_cast<int>(idx - block * hw);
  const int t_cache = static_cast<int>(block % 2);
  const long long bc = block / 2;
  const int c = static_cast<int>(bc % channels);
  const long long b = bc / channels;

  __nv_bfloat16 value = __float2bfloat16(0.0f);
  if (frames >= 2) {
    const int src_t = frames - 2 + t_cache;
    const long long src = (((b * channels + c) * static_cast<long long>(frames) + src_t) * hw) + hw_idx;
    value = cur[src];
  } else if (frames == 1) {
    if (t_cache == 1) {
      const long long src = ((b * channels + c) * static_cast<long long>(frames) * hw) + hw_idx;
      value = cur[src];
    } else if (prev != nullptr) {
      const long long src = ((b * channels + c) * cache_stride_c + hw) + hw_idx;
      value = prev[src];
    }
  }
  out[idx] = value;
}

}  // namespace

void ncdhw_to_blc_bf16(
    const void* x,
    void* out,
    int batch,
    int channels,
    int frames,
    int height,
    int width,
    cudaStream_t stream) {
  const long long total = static_cast<long long>(batch) * channels * frames * height * width;
  int blocks = static_cast<int>((total + 255) / 256);
  if (blocks > 4096) blocks = 4096;
  ncdhw_to_blc_bf16_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<__nv_bfloat16*>(out),
      channels,
      frames,
      height,
      width,
      total);
}

void time_unshuffle2_bf16(
    const void* x,
    void* out,
    int batch,
    int channels,
    int frames,
    int height,
    int width,
    cudaStream_t stream) {
  const long long total = static_cast<long long>(batch) * channels * 2LL * frames * height * width;
  time_unshuffle2_bf16_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<__nv_bfloat16*>(out),
      channels,
      frames,
      height,
      width,
      total);
}

void add_bias_ncdhw_bf16(
    void* x,
    const void* bias,
    int batch,
    int channels,
    int frames,
    int height,
    int width,
    cudaStream_t stream) {
  const long long total = static_cast<long long>(batch) * channels * frames * height * width;
  add_bias_ncdhw_bf16_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(x),
      reinterpret_cast<const __nv_bfloat16*>(bias),
      channels,
      frames,
      height,
      width,
      total);
}

void update_cache2_ncdhw_bf16(
    const void* cur,
    const void* prev,
    void* out,
    int batch,
    int channels,
    int frames,
    int height,
    int width,
    cudaStream_t stream) {
  const long long total = static_cast<long long>(batch) * channels * 2LL * height * width;
  update_cache2_ncdhw_bf16_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(cur),
      reinterpret_cast<const __nv_bfloat16*>(prev),
      reinterpret_cast<__nv_bfloat16*>(out),
      channels,
      frames,
      height,
      width,
      total);
}

}  // namespace spatiotemporal_layout
}  // namespace flash_rt
