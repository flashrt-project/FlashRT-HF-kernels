// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt::kernels {

void relu2_quantize_fp8_static_bf16(
    const void* input,
    void* output,
    const float* scale,
    int numel,
    cudaStream_t stream);

}  // namespace flash_rt::kernels
