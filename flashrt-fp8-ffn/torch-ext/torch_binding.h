// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void fp8_gemm_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& input_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor& out);

void fp8_linear_bias_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    torch::Tensor const& input_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor& out);

void bf16_fp8_linear_bias_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    torch::Tensor const& input_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor& input_fp8,
    torch::Tensor& out);

void fp8_linear_bias_gelu_quant_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    torch::Tensor const& input_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor const& output_scale,
    torch::Tensor& hidden_bf16,
    torch::Tensor& out_fp8);

void fp8_gelu_mlp_bf16(
    torch::Tensor const& input,
    torch::Tensor const& up_weight,
    torch::Tensor const& up_bias,
    torch::Tensor const& down_weight,
    torch::Tensor const& down_bias,
    torch::Tensor const& input_scale,
    torch::Tensor const& up_weight_scale,
    torch::Tensor const& hidden_scale,
    torch::Tensor const& down_weight_scale,
    torch::Tensor& hidden_bf16,
    torch::Tensor& hidden_fp8,
    torch::Tensor& out);

void bf16_fp8_gelu_mlp_bf16(
    torch::Tensor const& input,
    torch::Tensor const& up_weight,
    torch::Tensor const& up_bias,
    torch::Tensor const& down_weight,
    torch::Tensor const& down_bias,
    torch::Tensor const& input_scale,
    torch::Tensor const& up_weight_scale,
    torch::Tensor const& hidden_scale,
    torch::Tensor const& down_weight_scale,
    torch::Tensor& input_fp8,
    torch::Tensor& hidden_bf16,
    torch::Tensor& hidden_fp8,
    torch::Tensor& out);
