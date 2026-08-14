#pragma once

#include <cuda_runtime.h>

namespace flashrt_hub::sage2 {

int padded_k64(int seqlen_k);
int q_scale_elems(int batch, int seqlen_q, int num_q_heads);
int k_scale_elems(int batch, int seqlen_k, int num_kv_heads);
int q_thread_scale_elems(int batch, int seqlen_q, int num_q_heads);
int k_thread_scale_elems(int batch, int seqlen_k, int num_kv_heads);
int v_scale_elems(int batch, int head_dim, int num_kv_heads);

void quant_per_warp_int8_bf16_d128(
    const void* x_bf16,
    void* out_i8,
    void* scale_f32,
    int batch,
    int seqlen,
    int heads,
    cudaStream_t stream);

void quant_per_block_int8_bf16_d128(
    const void* x_bf16,
    void* out_i8,
    void* scale_f32,
    int batch,
    int seqlen,
    int heads,
    cudaStream_t stream);

int quant_qk_per_thread_int8_bf16_d128(
    const void* q_bf16, const void* k_bf16, void* q_i8, void* k_i8,
    void* q_scale_f32, void* k_scale_f32, int batch, int seqlen_q,
    int seqlen_k, int q_heads, int k_heads, cudaStream_t stream);

void v_bf16_to_fp16_d128(
    const void* v_bf16,
    void* v_half,
    int batch,
    int seqlen,
    int heads,
    cudaStream_t stream);

void v_bf16_to_fp8_tpp_d128(
    const void* v_bf16,
    void* v_fp8,
    void* v_scale,
    int batch,
    int seqlen,
    int heads,
    cudaStream_t stream);

void v_bf16_to_fp8_tpp_native_d128(
    const void* v_bf16,
    void* v_tpp_bf16,
    void* v_fp8,
    void* v_scale,
    int batch,
    int seqlen,
    int heads,
    cudaStream_t stream);

int sage2_qk_int8_sv_f16_bf16_gqa_d128(
    const void* q_int8,
    const void* k_int8,
    const void* v_half,
    void* out_bf16,
    const void* q_scale,
    const void* k_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_q_heads,
    int num_kv_heads,
    float softmax_scale,
    bool causal,
    cudaStream_t stream);

int sage2_qk_int8_sv_f8_bf16_gqa_d128(
    const void* q_int8,
    const void* k_int8,
    const void* v_fp8,
    void* out_bf16,
    const void* q_scale,
    const void* k_scale,
    const void* v_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_q_heads,
    int num_kv_heads,
    float softmax_scale,
    bool causal,
    cudaStream_t stream);

int sage2_qk_int8_pt_sv_f16_bf16_gqa_d128(
    const void* q_int8, const void* k_int8, const void* v_half, void* out_bf16,
    const void* q_scale, const void* k_scale, int batch, int seqlen_q,
    int seqlen_k, int num_q_heads, int num_kv_heads, float softmax_scale,
    bool causal, cudaStream_t stream);

int sage2_qk_int8_pt_sv_f8_bf16_gqa_d128(
    const void* q_int8, const void* k_int8, const void* v_fp8, void* out_bf16,
    const void* q_scale, const void* k_scale, const void* v_scale, int batch,
    int seqlen_q, int seqlen_k, int num_q_heads, int num_kv_heads,
    float softmax_scale, bool causal, cudaStream_t stream);

}  // namespace flashrt_hub::sage2
