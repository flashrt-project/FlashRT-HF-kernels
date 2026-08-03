// SPDX-License-Identifier: Apache-2.0
//
// Portable SIMT reference conv kernels (see portable_conv_simt.cu).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace conv {

int fp8_conv3d_v18_ncdhw_res_bf16out_simt(
    const void* cache_x, const void* new_x, const void* w, void* y,
    const void* bias, const void* residual,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream);

int fp8_conv3d_v17_ndhwc_bf16out_simt(
    const void* cache_x, const void* new_x, const void* w, void* y,
    const void* bias,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream);

int fp8_conv3d_v17_anyco_ndhwc_bf16out_simt(
    const void* cache_x, const void* new_x, const void* w, void* y,
    const void* bias,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream);

int fp8_conv2d_3x3_v2_nhwc_bf16out_simt(
    const void* x, const void* w, void* y, const void* bias,
    int N, int H, int W, int Ci, int Co, float alpha, cudaStream_t stream);

int fp8_conv2d_3x3_v2_nhwc_ncdhw_bf16out_simt(
    const void* x, const void* w, void* y, const void* bias,
    int B, int T, int H, int W, int Ci, int Co, float alpha,
    cudaStream_t stream);

int fp4_conv3d_ndhwc_bf16out_simt(
    const void* cache_x, const void* new_x, const void* w,
    const void* cache_sf, const void* new_sf, const void* w_sf,
    const void* outer_w, void* y, const void* bias,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream);

int fp4_conv3d_ncdhw_res_bf16out_simt(
    const void* cache_x, const void* new_x, const void* w,
    const void* cache_sf, const void* new_sf, const void* w_sf,
    const void* outer_w, void* y, const void* bias, const void* residual,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream);

}  // namespace conv
}  // namespace flash_rt
