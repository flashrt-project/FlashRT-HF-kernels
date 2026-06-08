// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void fp8_gemm_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& input_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor& out);

void silu_mul_merged_quantize_fp8_static_bf16(
    torch::Tensor const& gate_up_bf16,
    torch::Tensor const& output_scale,
    torch::Tensor& out_fp8);

void fp8_swiglu_mlp_bf16(
    torch::Tensor const& input,
    torch::Tensor const& gate_up_weight,
    torch::Tensor const& down_weight,
    torch::Tensor const& input_scale,
    torch::Tensor const& gate_up_weight_scale,
    torch::Tensor const& hidden_scale,
    torch::Tensor const& down_weight_scale,
    torch::Tensor& gate_up_bf16,
    torch::Tensor& hidden_fp8,
    torch::Tensor& out);
