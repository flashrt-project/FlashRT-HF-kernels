// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace adaln_producers {

void layer_norm_no_affine_fp8_static_bf16(
    const void* x_bf16,
    void* out_fp8,
    const float* scale,
    int rows,
    int dim,
    float eps,
    cudaStream_t stream);

}  // namespace adaln_producers
}  // namespace flash_rt
