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

}  // namespace spatiotemporal_layout
}  // namespace flash_rt
