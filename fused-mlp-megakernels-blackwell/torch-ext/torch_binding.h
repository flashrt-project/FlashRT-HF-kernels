#pragma once

#include <torch/all.h>

void fp16_geglu_fused_out(
    torch::Tensor const& input,
    torch::Tensor const& gate_weight,
    torch::Tensor const& up_weight,
    torch::Tensor& gate_scratch,
    torch::Tensor& output);
