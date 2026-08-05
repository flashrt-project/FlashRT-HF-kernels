#pragma once

#include <cuda_runtime_api.h>

namespace flash_rt::adaln_producers::hub {

using AdaLayerNormFp4Dispatch = int (*)(
    const void* x, const void* scale, const void* shift,
    void* packed, void* sfa, int rows, int dim, float eps,
    cudaStream_t stream);

using LayerNormFp4Dispatch = int (*)(
    const void* x, void* packed, void* sfa, int rows, int dim,
    float eps, cudaStream_t stream);

extern AdaLayerNormFp4Dispatch ada_layer_norm_fp4_dispatch;
extern LayerNormFp4Dispatch layer_norm_fp4_dispatch;

}  // namespace flash_rt::adaln_producers::hub
