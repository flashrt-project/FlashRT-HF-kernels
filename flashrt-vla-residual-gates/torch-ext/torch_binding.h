// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void joint3_bias_gate_residual_bf16(
    torch::Tensor const& v_residual,
    torch::Tensor const& v_x,
    torch::Tensor const& v_bias,
    torch::Tensor const& v_gate,
    torch::Tensor& v_out,
    torch::Tensor const& a_residual,
    torch::Tensor const& a_x,
    torch::Tensor const& a_bias,
    torch::Tensor const& a_gate,
    torch::Tensor& a_out,
    torch::Tensor const& u_residual,
    torch::Tensor const& u_x,
    torch::Tensor& u_out);

void joint3_bias_gate_residual_action_nobias_bf16(
    torch::Tensor const& v_residual,
    torch::Tensor const& v_x,
    torch::Tensor const& v_bias,
    torch::Tensor const& v_gate,
    torch::Tensor& v_out,
    torch::Tensor const& a_residual,
    torch::Tensor const& a_x,
    torch::Tensor const& a_gate,
    torch::Tensor& a_out,
    torch::Tensor const& u_residual,
    torch::Tensor const& u_x,
    torch::Tensor& u_out);
