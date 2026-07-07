// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void argmax_bf16(torch::Tensor const& logits, torch::Tensor& argmax_out);

void accept_greedy_bf16(torch::Tensor const& logits, torch::Tensor const& drafts,
                        torch::Tensor& argmax_out, torch::Tensor& accept_n,
                        int64_t spec_k);

void accept_partitioned_bf16(torch::Tensor const& logits, torch::Tensor const& drafts,
                             torch::Tensor& argmax_out, torch::Tensor& accept_n,
                             torch::Tensor& partial_vals, torch::Tensor& partial_idx,
                             int64_t spec_k, int64_t parts);
