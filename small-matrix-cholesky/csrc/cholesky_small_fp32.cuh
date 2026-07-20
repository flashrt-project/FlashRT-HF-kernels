// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flashrt_hub {
namespace cholesky {

// Factor a contiguous batch of row-major FP32 SPD matrices. The output is a
// dense lower-triangular matrix with its upper triangle explicitly zeroed.
// Supported matrix orders are 32, 64, and 128.
cudaError_t cholesky_small_fp32(
    const float* input,
    float* output,
    int batch,
    int n,
    cudaStream_t stream);

}  // namespace cholesky
}  // namespace flashrt_hub
