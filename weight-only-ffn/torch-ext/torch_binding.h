#pragma once

#include <torch/all.h>

void quantize_w4_weight_bf16(torch::Tensor const& weight,
                             torch::Tensor& packed,
                             torch::Tensor& sfb);
void dequantize_w4_weight_bf16(torch::Tensor const& packed,
                               torch::Tensor const& sfb,
                               torch::Tensor& weight);
void w4a16_linear_bf16(torch::Tensor const& x,
                       torch::Tensor const& packed,
                       torch::Tensor const& sfb,
                       double alpha,
                       int64_t variant,
                       torch::Tensor& out);

void quantize_w8_weight_bf16(torch::Tensor const& weight,
                             torch::Tensor& quantized,
                             torch::Tensor& scales);
void dequantize_w8_weight_bf16(torch::Tensor const& quantized,
                               torch::Tensor const& scales,
                               torch::Tensor& weight);
void w8a16_linear_bf16(torch::Tensor const& x,
                       torch::Tensor const& quantized,
                       torch::Tensor const& scales,
                       int64_t variant,
                       torch::Tensor& out);
