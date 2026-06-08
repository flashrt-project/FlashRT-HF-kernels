// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

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
