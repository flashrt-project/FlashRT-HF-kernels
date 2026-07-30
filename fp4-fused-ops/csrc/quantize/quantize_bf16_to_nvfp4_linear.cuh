// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::quantize {

void quantize_bf16_to_nvfp4_linear(
    const __nv_bfloat16* input,
    uint8_t* packed,
    uint8_t* scale_factors,
    int rows,
    int cols,
    cudaStream_t stream);

}  // namespace flash_rt::quantize
