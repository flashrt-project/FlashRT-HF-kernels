// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void msa_topk_from_scores(torch::Tensor const& score,
                          torch::Tensor const& seq_lens,
                          int64_t block_size,
                          int64_t topk,
                          torch::Tensor& topk_idx);

void msa_decode_sparse_attn(torch::Tensor const& q,
                            torch::Tensor const& kv_cache,
                            torch::Tensor const& seq_lens,
                            torch::Tensor const& slot_ids,
                            torch::Tensor const& topk_idx,
                            int64_t block_size,
                            double sm_scale,
                            torch::Tensor& out);

// Tensor-core (mma) fragment-resident variant; requires D=128, Hq/Hkv=16.
void msa_decode_sparse_attn_mma(torch::Tensor const& q,
                                torch::Tensor const& kv_cache,
                                torch::Tensor const& seq_lens,
                                torch::Tensor const& slot_ids,
                                torch::Tensor const& topk_idx,
                                int64_t block_size,
                                double sm_scale,
                                torch::Tensor& out);

// Paged tensor-core variant (separate k/v caches + req_to_token indirection).
void msa_decode_sparse_attn_mma_paged(torch::Tensor const& q,
                                      torch::Tensor const& k_cache,
                                      torch::Tensor const& v_cache,
                                      torch::Tensor const& req_to_token,
                                      torch::Tensor const& seq_lens,
                                      torch::Tensor const& slot_ids,
                                      torch::Tensor const& topk_idx,
                                      int64_t block_size,
                                      double sm_scale,
                                      torch::Tensor& out);
