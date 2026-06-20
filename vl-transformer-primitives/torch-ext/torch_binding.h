// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void qwen3_q_norm_rope_qstage_bf16(
    torch::Tensor const& q_pre,
    torch::Tensor const& q_norm_weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    double eps,
    torch::Tensor& q_out);

void qwen3_k_norm_rope_kvwrite_bf16(
    torch::Tensor const& k_pre,
    torch::Tensor const& v_pre,
    torch::Tensor const& k_norm_weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    double eps,
    torch::Tensor& k_cache_dst,
    torch::Tensor& v_cache_dst);

void qwen3_k_norm_rope_kvwrite_devpos_bf16(
    torch::Tensor const& k_pre,
    torch::Tensor const& v_pre,
    torch::Tensor const& k_norm_weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    torch::Tensor const& cur_pos,
    double eps,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache);

void avg_pool_vision_tokens_bf16(
    torch::Tensor const& x,
    int64_t nv,
    int64_t h,
    int64_t w,
    int64_t pool_factor,
    torch::Tensor& out);
