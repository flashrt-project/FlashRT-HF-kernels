// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void adaptive_rms_norm_quant_fp8_padded_bf16(
    torch::Tensor const& input, torch::Tensor const& weight,
    torch::Tensor const& gamma, torch::Tensor const& beta,
    torch::Tensor const& scale, double eps, torch::Tensor& output);
void residual_add_adaptive_rms_norm_quant_fp8_padded_bf16(
    torch::Tensor const& residual, torch::Tensor const& input,
    torch::Tensor const& weight, torch::Tensor const& gamma,
    torch::Tensor const& beta, torch::Tensor const& scale, double eps,
    torch::Tensor& residual_out, torch::Tensor& output);
void swiglu_quant_fp8_padded_bf16(
    torch::Tensor const& gate, torch::Tensor const& up,
    torch::Tensor const& scale, torch::Tensor& output);
void swiglu_merged_quant_fp8_padded_bf16(
    torch::Tensor const& gate_up, torch::Tensor const& scale,
    torch::Tensor& output);
void swiglu_merged_quant_fp8_padded_fp16(
    torch::Tensor const& gate_up, torch::Tensor const& scale,
    torch::Tensor& output);
