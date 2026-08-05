#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" {

void attention_mha_fp16_masked(
    cublasHandle_t handle, const __half* q, const __half* k,
    const __half* v, __half* logits, __half* out, int sequence_q,
    int sequence_kv, int heads, int head_dim, float scale,
    cudaStream_t stream);
void attention_mha_bf16_masked(
    cublasHandle_t handle, const __nv_bfloat16* q,
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* logits, __nv_bfloat16* out, int sequence_q,
    int sequence_kv, int heads, int head_dim, float scale,
    int logits_kv_stride, int qkv_token_stride, cudaStream_t stream);

}
