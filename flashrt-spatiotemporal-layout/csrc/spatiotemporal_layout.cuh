// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace spatiotemporal_layout {

void ncdhw_to_blc_bf16(
    const void* x,
    void* out,
    int batch,
    int channels,
    int frames,
    int height,
    int width,
    cudaStream_t stream);

void patch_im2col_bf16(
    const void* input,
    void* output,
    int num_views,
    cudaStream_t stream);

void time_unshuffle2_bf16(
    const void* x,
    void* out,
    int batch,
    int channels,
    int frames,
    int height,
    int width,
    cudaStream_t stream);

void add_bias_ncdhw_bf16(
    void* x,
    const void* bias,
    int batch,
    int channels,
    int frames,
    int height,
    int width,
    cudaStream_t stream);

void update_cache2_ncdhw_bf16(
    const void* cur,
    const void* prev,
    void* out,
    int batch,
    int channels,
    int frames,
    int height,
    int width,
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

void pack_causal_cache3_nhwc_bf16(
    const void* previous,
    const void* current,
    void* out,
    int batch,
    int channels,
    int height,
    int width,
    cudaStream_t stream);

}  // namespace spatiotemporal_layout
}  // namespace flash_rt
