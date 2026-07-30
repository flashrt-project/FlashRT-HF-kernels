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

__global__ void patch_im2col_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int num_views,
    int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int patch_idx = idx / 588;
  const int feat_idx = idx - patch_idx * 588;
  const int view = patch_idx / 256;
  const int local_patch = patch_idx - view * 256;
  const int patch_h = local_patch / 16;
  const int patch_w = local_patch - patch_h * 16;
  const int pixel_h = feat_idx / 42;
  const int pixel_w = (feat_idx - pixel_h * 42) / 3;
  const int channel = feat_idx - pixel_h * 42 - pixel_w * 3;
  const int row = patch_h * 14 + pixel_h;
  const int col = patch_w * 14 + pixel_w;
  const int src =
      view * (224 * 224 * 3) + row * (224 * 3) + col * 3 + channel;
  output[idx] = input[src];
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

__global__ void avg_pool3d_channels_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int total,
    int channels,
    int frames,
    int height,
    int width,
    int out_channels,
    int out_frames,
    int out_height,
    int out_width,
    int factor_t,
    int factor_s,
    int group_size,
    int pad_t) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  int rem = idx;
  const int ow = rem % out_width;
  rem /= out_width;
  const int oh = rem % out_height;
  rem /= out_height;
  const int ot = rem % out_frames;
  rem /= out_frames;
  const int oc = rem % out_channels;
  const int batch = rem / out_channels;

  const int factor = factor_t * factor_s * factor_s;
  float acc = 0.0f;
  #pragma unroll
  for (int group = 0; group < 8; ++group) {
    if (group >= group_size) break;
    int channel_factor = oc * group_size + group;
    const int ws = channel_factor % factor_s;
    channel_factor /= factor_s;
    const int hs = channel_factor % factor_s;
    channel_factor /= factor_s;
    const int tt = channel_factor % factor_t;
    const int channel = channel_factor / factor_t;
    if (channel >= channels || oc * group_size + group >= channels * factor) {
      continue;
    }
    const int padded_t = ot * factor_t + tt;
    if (padded_t < pad_t) continue;
    const int input_t = padded_t - pad_t;
    if (input_t >= frames) continue;
    const int input_h = oh * factor_s + hs;
    const int input_w = ow * factor_s + ws;
    const long long input_idx =
        (((static_cast<long long>(batch) * channels + channel) * frames +
          input_t) *
             height +
         input_h) *
            width +
        input_w;
    acc += __bfloat162float(x[input_idx]);
  }
  out[idx] = __float2bfloat16(acc / static_cast<float>(group_size));
}

__global__ void channel_to_space3d_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int in_channels,
    int out_channels,
    int frames,
    int height,
    int width,
    int temporal_factor,
    int spatial_factor,
    int repeats,
    int out_frames,
    long long total) {
  const long long index =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int out_width = width * spatial_factor;
  const int out_height = height * spatial_factor;
  const int out_w = index % out_width;
  long long rem = index / out_width;
  const int out_h = rem % out_height;
  rem /= out_height;
  const int out_t = rem % out_frames;
  rem /= out_frames;
  const int out_c = rem % out_channels;
  const int batch = rem / out_channels;
  const int full_t =
      out_t + (frames * temporal_factor - out_frames);
  const int dt = full_t % temporal_factor;
  const int in_t = full_t / temporal_factor;
  const int dh = out_h % spatial_factor;
  const int in_h = out_h / spatial_factor;
  const int dw = out_w % spatial_factor;
  const int in_w = out_w / spatial_factor;
  const int subpixel =
      ((dt * spatial_factor) + dh) * spatial_factor + dw;
  const int expanded_channel =
      out_c * temporal_factor * spatial_factor * spatial_factor + subpixel;
  const int in_c = expanded_channel / repeats;
  if (in_c >= in_channels) return;
  const long long input_index =
      (((static_cast<long long>(batch) * in_channels + in_c) * frames +
        in_t) *
           height +
       in_h) *
          width +
      in_w;
  out[index] = x[input_index];
}

__global__ void pack_causal_cache3_nhwc_bf16_kernel(
    const __nv_bfloat16* __restrict__ previous,
    const __nv_bfloat16* __restrict__ current,
    __nv_bfloat16* __restrict__ out,
    int channels,
    int height,
    int width,
    long long total) {
  for (long long index =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int channel3 = index % (3LL * channels);
    long long rem = index / (3LL * channels);
    const int out_w = rem % width;
    rem /= width;
    const int out_h = rem % height;
    const int batch = rem / height;
    const int plane = channel3 / channels;
    const int channel = channel3 - plane * channels;
    const long long hw = static_cast<long long>(out_h) * width + out_w;
    if (plane < 2) {
      out[index] = previous[
          ((static_cast<long long>(batch) * channels + channel) * 2 +
           plane) *
              height * width +
          hw];
    } else {
      out[index] = current[
          (static_cast<long long>(batch) * channels + channel) *
              height * width +
          hw];
    }
  }
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

void patch_im2col_bf16(
    const void* input,
    void* output,
    int num_views,
    cudaStream_t stream) {
  const int total = num_views * 256 * 588;
  patch_im2col_bf16_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input),
      reinterpret_cast<__nv_bfloat16*>(output),
      num_views,
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

void avg_pool3d_channels_bf16(
    const void* x,
    void* out,
    int batch,
    int channels,
    int frames,
    int height,
    int width,
    int out_channels,
    int factor_t,
    int factor_s,
    int group_size,
    cudaStream_t stream) {
  const int pad_t = (factor_t - (frames % factor_t)) % factor_t;
  const int out_frames = (frames + pad_t) / factor_t;
  const int out_height = height / factor_s;
  const int out_width = width / factor_s;
  const long long total64 =
      static_cast<long long>(batch) * out_channels * out_frames * out_height *
      out_width;
  avg_pool3d_channels_bf16_kernel<<<
      static_cast<unsigned>((total64 + 255) / 256), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<__nv_bfloat16*>(out),
      static_cast<int>(total64),
      channels,
      frames,
      height,
      width,
      out_channels,
      out_frames,
      out_height,
      out_width,
      factor_t,
      factor_s,
      group_size,
      pad_t);
}

void channel_to_space3d_bf16(
    const void* x,
    void* out,
    int batch,
    int in_channels,
    int out_channels,
    int frames,
    int height,
    int width,
    int temporal_factor,
    int spatial_factor,
    int repeats,
    bool first_chunk,
    cudaStream_t stream) {
  const int out_frames =
      frames * temporal_factor - (first_chunk ? temporal_factor - 1 : 0);
  const long long total =
      static_cast<long long>(batch) * out_channels * out_frames *
      (height * spatial_factor) * (width * spatial_factor);
  channel_to_space3d_bf16_kernel<<<
      static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<__nv_bfloat16*>(out), in_channels, out_channels,
      frames, height, width, temporal_factor, spatial_factor, repeats,
      out_frames, total);
}

void pack_causal_cache3_nhwc_bf16(
    const void* previous,
    const void* current,
    void* out,
    int batch,
    int channels,
    int height,
    int width,
    cudaStream_t stream) {
  const long long total =
      static_cast<long long>(batch) * height * width * 3 * channels;
  int blocks = static_cast<int>((total + 255) / 256);
  if (blocks > 4096) blocks = 4096;
  pack_causal_cache3_nhwc_bf16_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(previous),
      reinterpret_cast<const __nv_bfloat16*>(current),
      reinterpret_cast<__nv_bfloat16*>(out), channels, height, width,
      total);
}

}  // namespace spatiotemporal_layout
}  // namespace flash_rt
