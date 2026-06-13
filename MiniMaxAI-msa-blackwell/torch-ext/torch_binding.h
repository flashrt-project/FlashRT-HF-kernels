// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void msa_topk_from_scores(torch::Tensor const& score,
                          torch::Tensor const& seq_lens,
                          int64_t block_size,
                          int64_t topk,
                          torch::Tensor& topk_idx);
