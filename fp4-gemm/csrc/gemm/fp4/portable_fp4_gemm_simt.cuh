// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

// Portable SIMT fallbacks for the FP4 block-scaled GEMM fused epilogues.
// The sm_120a CUTLASS kernels are unavailable on pre-sm120 devices; these
// reference kernels compute the same GEMM + epilogue in pure SIMT FMA so the
// ops stay usable (slowly) on sm_110 Thor. sm_120 keeps the CUTLASS path.

namespace flash_rt {
namespace gemm {

// out[M,N] = alpha * (A@B^T)
int nvfp4_gemm_linear_simt(
    const void* a_packed, const void* b_packed, const void* sfa,
    const void* sfb, void* out_bf16, int m, int n, int k, float alpha,
    cudaStream_t stream);

// out[M,N] = alpha * (A@B^T) + residual[M,N]
int nvfp4_gemm_residual_simt(
    const void* a_packed, const void* b_packed, const void* sfa,
    const void* sfb, const void* residual_bf16, void* out_bf16,
    int m, int n, int k, float alpha, cudaStream_t stream);

// out[M,N] = alpha * (A@B^T) + bias[N]
int nvfp4_gemm_bias_simt(
    const void* a_packed, const void* b_packed, const void* sfa,
    const void* sfb, const void* bias_bf16, void* out_bf16,
    int m, int n, int k, float alpha, cudaStream_t stream);

// out[M,N] = GELU_taylor(alpha * (A@B^T) + bias[N])
int nvfp4_gemm_bias_gelu_simt(
    const void* a_packed, const void* b_packed, const void* sfa,
    const void* sfb, const void* bias_bf16, void* out_bf16,
    int m, int n, int k, float alpha, cudaStream_t stream);

// out_packed[M,N/2] (e2m1 + SFA layout scale) = quant_fp4(GELU_taylor(alpha*(A@B^T)+bias[N]))
int nvfp4_gemm_bias_gelu_fp4_simt(
    const void* a_packed, const void* b_packed, const void* sfa,
    const void* sfb, const void* bias_bf16, void* out_packed, void* out_sfa,
    int m, int n, int k, float alpha, cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
