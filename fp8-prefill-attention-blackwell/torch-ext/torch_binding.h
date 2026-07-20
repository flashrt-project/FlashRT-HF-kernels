#pragma once

#include <torch/types.h>

void fp8_causal_gqa_attention_bf16_out(torch::Tensor const &query,
                                       torch::Tensor const &key,
                                       torch::Tensor const &value,
                                       double softmax_scale,
                                       torch::Tensor &output);
