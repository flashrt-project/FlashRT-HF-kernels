// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void ncdhw_to_blc_bf16(torch::Tensor const& x, torch::Tensor& out);
void time_unshuffle2_bf16(torch::Tensor const& x, torch::Tensor& out);
void add_bias_ncdhw_bf16(torch::Tensor& x, torch::Tensor const& bias);
void update_cache2_ncdhw_bf16(torch::Tensor const& cur, torch::Tensor const& prev, torch::Tensor& out);
