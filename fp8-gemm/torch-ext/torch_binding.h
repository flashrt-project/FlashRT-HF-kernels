// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void fp8_linear_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    double alpha,
    int64_t variant,
    torch::Tensor& out);

void fp8_linear_residual_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    double alpha,
    int64_t variant,
    torch::Tensor& residual);

void fp8_linear_bias_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    double alpha,
    torch::Tensor& out);

void fp8_linear_bias_residual_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    double alpha,
    torch::Tensor& residual);

void fp8_linear_bias_gelu_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    double alpha,
    torch::Tensor& out);

void fp8_blockwise_linear_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& input_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor& out);

void fp8_blockwise_swiglu_quantize_fp8(
    torch::Tensor const& input,
    torch::Tensor const& gate_up_weight,
    torch::Tensor const& input_scale,
    torch::Tensor const& gate_up_weight_scale,
    torch::Tensor& output,
    torch::Tensor& output_scale);
