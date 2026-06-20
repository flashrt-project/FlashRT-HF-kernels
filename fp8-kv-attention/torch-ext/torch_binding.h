#pragma once

#include <torch/all.h>

void xqa_bf16_fp8kv(
    torch::Tensor const& q,
    torch::Tensor const& k_cache,
    torch::Tensor const& v_cache,
    torch::Tensor const& page_table,
    torch::Tensor const& seq_lens,
    torch::Tensor const& mask,
    torch::Tensor& out,
    torch::Tensor& semaphores,
    torch::Tensor& scratch,
    int64_t max_seq_len,
    double q_scale,
    double kv_scale,
    bool enable_pdl,
    int64_t sm_count,
    int64_t k_stride_page,
    int64_t k_stride_token,
    int64_t k_stride_head);
