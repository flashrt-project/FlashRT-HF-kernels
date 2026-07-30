// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void quantize_fp8_block128_bf16(
    torch::Tensor const& input, torch::Tensor& output, torch::Tensor& scale);
void layer_norm_fp8_block128_bf16(
    torch::Tensor const& input, torch::Tensor const& weight,
    torch::Tensor const& bias, double eps,
    torch::Tensor& output, torch::Tensor& scale);
void rms_norm_fp8_block128_bf16(
    torch::Tensor const& input, torch::Tensor const& weight, double eps,
    torch::Tensor& output, torch::Tensor& scale);
void residual_add_rms_norm_fp8_block128_bf16(
    torch::Tensor const& residual, torch::Tensor const& input,
    torch::Tensor const& weight, double eps, torch::Tensor& residual_out,
    torch::Tensor& output, torch::Tensor& scale);
void gelu_tanh_fp8_block128_bf16(
    torch::Tensor const& input, torch::Tensor& output, torch::Tensor& scale);
void gelu_tanh_bias_fp8_block128_bf16(
    torch::Tensor const& input, torch::Tensor const& bias,
    torch::Tensor& output, torch::Tensor& scale);
void silu_mul_fp8_block128_bf16(
    torch::Tensor const& gate, torch::Tensor const& up,
    torch::Tensor& output, torch::Tensor& scale);
void silu_mul_merged_fp8_block128_bf16(
    torch::Tensor const& gate_up, torch::Tensor& output, torch::Tensor& scale);
