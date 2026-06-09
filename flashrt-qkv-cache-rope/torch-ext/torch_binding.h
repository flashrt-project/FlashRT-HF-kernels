// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void qkv_split_rope_kvcache_bf16(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& rope,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t cache_offset,
    torch::Tensor& q_out,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache);

void qkv_split_bf16(
    torch::Tensor const& packed_qkv,
    int64_t heads,
    int64_t head_dim,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    torch::Tensor& v_out);

void qkv_split_norm_rope_bf16(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& norm_q_weight,
    torch::Tensor const& norm_k_weight,
    torch::Tensor const& freqs_re,
    torch::Tensor const& freqs_im,
    int64_t heads,
    int64_t head_dim,
    int64_t rope_seq_len,
    double eps,
    torch::Tensor& q_out,
    torch::Tensor& k_out);
