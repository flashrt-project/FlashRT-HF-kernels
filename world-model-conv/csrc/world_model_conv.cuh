// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace conv {

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

}  // namespace conv
}  // namespace flash_rt
