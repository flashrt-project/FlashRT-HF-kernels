#pragma once

#include <torch/all.h>

void fp8_gqa_cross_attention_bf16_out(
    torch::Tensor const& query,
    torch::Tensor const& key,
    torch::Tensor const& value,
    double query_scale,
    double key_scale,
    double value_scale,
    torch::Tensor& output,
    torch::Tensor& lse,
    torch::Tensor& workspace);
