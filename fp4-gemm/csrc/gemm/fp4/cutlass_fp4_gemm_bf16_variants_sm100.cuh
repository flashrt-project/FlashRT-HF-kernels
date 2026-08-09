// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt::fp4 {

// Native NVFP4 W4A4 GEMM variants with a BF16 output contract. Variant
// indices match cutlass_fp4_gemm_variant; the Hub package currently ships
// the PI0.5 production choices v7 and v10.
int cutlass_fp4_gemm_bf16_variant(
    int variant, const void* a_packed, const void* sfa,
    const void* b_packed, const void* sfb, void* out_bf16,
    int m, int n, int k, float alpha, float beta, cudaStream_t stream);

}  // namespace flash_rt::fp4
