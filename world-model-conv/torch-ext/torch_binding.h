// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>

void fp8_conv3d_v18_ncdhw_res_bf16out(
    torch::Tensor const& cache_x,
    torch::Tensor const& new_x,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    torch::Tensor const& residual,
    double alpha,
    torch::Tensor& out);
void fp8_causal_conv3d_ndhwc_bf16(
    torch::Tensor const& cache_x,
    torch::Tensor const& new_x,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    double alpha,
    torch::Tensor& out);
void fp8_conv2d_3x3_nhwc_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    double alpha,
    torch::Tensor& out);
void fp8_conv2d_3x3_ncdhw_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    double alpha,
    torch::Tensor& out);
void nvfp4_causal_conv3d_ndhwc_bf16(
    torch::Tensor const& cache_packed, torch::Tensor const& new_packed,
    torch::Tensor const& weight_packed, torch::Tensor const& cache_sf,
    torch::Tensor const& new_sf, torch::Tensor const& weight_sf,
    torch::Tensor const& bias,
    c10::optional<torch::Tensor> const& outer_weight, double alpha,
    torch::Tensor& out);
void nvfp4_causal_conv3d_residual_ncdhw_bf16(
    torch::Tensor const& cache_packed, torch::Tensor const& new_packed,
    torch::Tensor const& weight_packed, torch::Tensor const& cache_sf,
    torch::Tensor const& new_sf, torch::Tensor const& weight_sf,
    torch::Tensor const& bias, torch::Tensor const& residual,
    c10::optional<torch::Tensor> const& outer_weight, double alpha,
    torch::Tensor& out);
