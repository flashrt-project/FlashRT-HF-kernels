// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace adaln_producers {

void adaln_modulation6_bf16(
    const float* adaln_params,
    const float* layer_modulation,
    void* out0,
    void* out1,
    void* out2,
    void* out3,
    void* out4,
    void* out5,
    int batch,
    int sequence,
    int dim,
    cudaStream_t stream);

}  // namespace adaln_producers
}  // namespace flash_rt
