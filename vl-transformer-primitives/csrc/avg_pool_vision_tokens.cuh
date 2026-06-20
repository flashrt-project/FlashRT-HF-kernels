// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace vl_transformer_primitives {

void avg_pool_vision_tokens_bf16(
    const void* x,
    void* out,
    int nv,
    int h,
    int w,
    int dim,
    int pool_factor,
    cudaStream_t stream);

}  // namespace vl_transformer_primitives
}  // namespace flash_rt
