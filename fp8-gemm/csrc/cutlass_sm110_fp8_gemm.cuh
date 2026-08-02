// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

extern "C" {

int cutlass_fp8_sq(void* A, void* B, void* D, int M, int N, int K,
                   float alpha, float beta, cudaStream_t stream);
int cutlass_fp8_t1(void* A, void* B, void* D, int M, int N, int K,
                   float alpha, float beta, cudaStream_t stream);
int cutlass_fp8_wide(void* A, void* B, void* D, int M, int N, int K,
                     float alpha, float beta, cudaStream_t stream);
int cutlass_fp8_plain(void* A, void* B, void* D, int M, int N, int K,
                      float alpha, float beta, cudaStream_t stream);
int cutlass_fp8_gelu(void* A, void* B, void* D, int M, int N, int K,
                     float alpha, float beta, cudaStream_t stream);
int cutlass_fp8_sq_bf16out(void* A, void* B, void* D, int M, int N, int K,
                           float alpha, float beta, cudaStream_t stream);
int cutlass_fp8_wide_bf16out(void* A, void* B, void* D, int M, int N, int K,
                             float alpha, float beta, cudaStream_t stream);
int cutlass_fp8_t1_bf16out(void* A, void* B, void* D, int M, int N, int K,
                           float alpha, float beta, cudaStream_t stream);

}  // extern "C"
