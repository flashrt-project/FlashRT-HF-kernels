#pragma once

#include <torch/all.h>

void bf16_linear_bf16(
    torch::Tensor const& x,
    torch::Tensor const& w,
    torch::Tensor& out);

void bf16_linear_bias_bf16(
    torch::Tensor const& x,
    torch::Tensor const& w,
    torch::Tensor const& bias,
    torch::Tensor& out);

void bf16_gemm_bias_gelu(
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& bias,
    torch::Tensor& out);

void bf16_gemm_bias(
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& bias,
    torch::Tensor& out);

void bias_gelu_quantize_fp8_static_bf16(
    torch::Tensor const& input,
    torch::Tensor const& bias,
    torch::Tensor const& scale,
    torch::Tensor& out);

void gelu_quantize_fp8_static_bf16(
    torch::Tensor const& input,
    torch::Tensor const& scale,
    torch::Tensor& out);

void channel_scale_quantize_fp8_static_bf16(
    torch::Tensor const& input,
    torch::Tensor const& channel_scale,
    torch::Tensor const& scale,
    torch::Tensor& out);
