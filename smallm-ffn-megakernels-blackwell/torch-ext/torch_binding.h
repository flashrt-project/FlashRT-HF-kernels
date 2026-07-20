#pragma once
#include <torch/types.h>
void fp8_gelu_ffn_gated_residual_bf16_out(
    torch::Tensor const &, torch::Tensor const &, torch::Tensor const &,
    torch::Tensor const &, torch::Tensor const &, torch::Tensor const &,
    torch::Tensor const &, torch::Tensor const &, double, double, double,
    torch::Tensor &, torch::Tensor &);
void fp8_gelu_ffn_residual_bf16_out(
    torch::Tensor const &, torch::Tensor const &, torch::Tensor const &,
    torch::Tensor const &, torch::Tensor const &, torch::Tensor const &,
    torch::Tensor const &, torch::Tensor const &, double, double, double,
    double, bool, torch::Tensor &, torch::Tensor &, torch::Tensor &,
    torch::Tensor &);
