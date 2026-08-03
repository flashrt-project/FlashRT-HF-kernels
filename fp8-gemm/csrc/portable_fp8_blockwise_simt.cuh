// SPDX-License-Identifier: Apache-2.0
//
// Portable SIMT reference for the block-128 FP8 GEMM.
//
// The sm_120a CUTLASS path (cutlass_sm120_block128_fp8_gemm.cu) requires
// SM120 tensor cores. This reference computes the same block-scaled
// FP8 x FP8 -> BF16 GEMM in pure SIMT FMA so the package is usable
// (slowly) on sm_110 Thor. sm_120 keeps the CUTLASS path.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// Block-128 FP8 GEMM, BF16 output, portable SIMT reference.
//
// Layout & shapes match cutlass_sm120_block128_fp8_gemm.cuh:
//   A_fp8      : (M, K)        e4m3 row-major
//   B_fp8      : (N, K)        e4m3 row-major
//   D_bf16     : (M, N)        bf16 row-major
//   act_scale  : (M, K/128)    fp32 row-major
//   w_scale    : (N/128, K/128) fp32 row-major
//
// Semantics (matches the CUTLASS kernel and test_fp8_gemm.py):
//   D[m, n] = sum_k  A[m, k] * act_scale[m, k/128]
//                  * B[n, k] * w_scale[n/128, k/128]
//
// Constraints: K and N must be multiples of 128. M is unrestricted.
void fp8_block128_gemm_simt_bf16out(
    const void* A_fp8,
    const void* B_fp8,
    void*       D_bf16,
    int M, int N, int K,
    const float* act_scale,
    const float* w_scale,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
