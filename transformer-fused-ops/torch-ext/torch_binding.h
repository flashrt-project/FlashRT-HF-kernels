// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void rms_norm_gated_silu_bf16(torch::Tensor const& x, torch::Tensor const& gate,
                              torch::Tensor const& weight, double eps,
                              torch::Tensor& out);
void silu_mul_bf16(torch::Tensor const& gate, torch::Tensor const& up, torch::Tensor& out);
void sigmoid_mul_bf16(torch::Tensor const& gate, torch::Tensor const& x, torch::Tensor& out);
void embedding_lookup_bf16(torch::Tensor const& token_ids, torch::Tensor const& embed,
                           torch::Tensor& out);
void partial_rope_qk_bf16(torch::Tensor const& q_in, torch::Tensor const& k_in,
                          torch::Tensor const& cos, torch::Tensor const& sin,
                          torch::Tensor& q_out, torch::Tensor& k_out, int64_t rope_dim);
void argmax_bf16(torch::Tensor const& logits, torch::Tensor& argmax_out);
void spec_accept_greedy_bf16(torch::Tensor const& logits, torch::Tensor const& drafts,
                             torch::Tensor& argmax_out, torch::Tensor& accept_n,
                             int64_t spec_k);
void nexn2_lin_split_qkv_broadcast_bf16(torch::Tensor const& conv_out,
                                        torch::Tensor& q32, torch::Tensor& k32,
                                        torch::Tensor& v32);
void nexn2_split_q_gate_bf16(torch::Tensor const& q_proj,
                             torch::Tensor& q_pre, torch::Tensor& gate);
void nexn2_router_topk_bf16(torch::Tensor const& logits, torch::Tensor& out_idx,
                            torch::Tensor& out_val, int64_t k);
