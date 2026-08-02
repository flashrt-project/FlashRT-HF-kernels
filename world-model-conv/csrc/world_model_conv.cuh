// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace conv {

extern "C" int bf16_conv3d_v0_ndhwc_bf16out(
    const void* cache_x_bf16,
    const void* new_x_bf16,
    const void* w_bf16,
    void* y_bf16,
    const void* bias_bf16,
    int N,
    int T_cache,
    int T_new,
    int H,
    int W,
    int Ci,
    int Co,
    float alpha,
    cudaStream_t stream);

extern "C" int fp8_conv3d_v18_ncdhw_res_bf16out(
    const void* cache_x_fp8,
    const void* new_x_fp8,
    const void* w_fp8,
    void* y_bf16,
    const void* bias_bf16,
    const void* residual_bf16,
    int N,
    int T_cache,
    int T_new,
    int H,
    int W,
    int Ci,
    int Co,
    float alpha,
    cudaStream_t stream);

extern "C" int fp8_conv3d_v17_ndhwc_bf16out(
    const void* cache_x_fp8,
    const void* new_x_fp8,
    const void* w_fp8,
    void* y_bf16,
    const void* bias_bf16,
    int N,
    int T_cache,
    int T_new,
    int H,
    int W,
    int Ci,
    int Co,
    float alpha,
    cudaStream_t stream);

extern "C" int fp8_conv3d_v17_anyco_ndhwc_bf16out(
    const void* cache_x_fp8,
    const void* new_x_fp8,
    const void* w_fp8,
    void* y_bf16,
    const void* bias_bf16,
    int N,
    int T_cache,
    int T_new,
    int H,
    int W,
    int Ci,
    int Co,
    float alpha,
    cudaStream_t stream);

extern "C" int fp8_conv2d_3x3_v2_nhwc_bf16out(
    const void* x_fp8,
    const void* w_fp8,
    void* y_bf16,
    const void* bias_bf16,
    int N,
    int H,
    int W,
    int Ci,
    int Co,
    float alpha,
    cudaStream_t stream);

extern "C" int fp8_conv2d_3x3_v2_nhwc_ncdhw_bf16out(
    const void* x_fp8,
    const void* w_fp8,
    void* y_bf16,
    const void* bias_bf16,
    int B,
    int T,
    int H,
    int W,
    int Ci,
    int Co,
    float alpha,
    cudaStream_t stream);

extern "C" int motus_fp4_conv3d_v19sf_ndhwc_bf16out(
    const void*, const void*, const void*, const void*, const void*,
    const void*, void*, const void*, int, int, int, int, int, int, int,
    float, cudaStream_t);
extern "C" int motus_fp4_conv3d_v19sf_ndhwc_bf16out_v2(
    const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, void*, const void*, int, int, int, int, int,
    int, int, float, cudaStream_t);
extern "C" int motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out(
    const void*, const void*, const void*, const void*, const void*,
    const void*, void*, const void*, const void*, int, int, int, int, int,
    int, int, float, cudaStream_t);
extern "C" int motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out_v2(
    const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, void*, const void*, const void*, int, int,
    int, int, int, int, int, float, cudaStream_t);
extern "C" int motus_fp4_conv3d_v19sfbk128_ncdhw_res_bf16out(
    const void*, const void*, const void*, const void*, const void*,
    const void*, void*, const void*, const void*, int, int, int, int, int,
    int, int, float, cudaStream_t);

}  // namespace conv
}  // namespace flash_rt
