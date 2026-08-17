// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void silu_mul_quantize_fp4_sfa_bf16(
    torch::Tensor const& merged, torch::Tensor& packed, torch::Tensor& sfa);
void rms_norm_quantize_fp4_sfa_bf16(
    torch::Tensor const& x, torch::Tensor const& weight, double eps,
    torch::Tensor& normed, torch::Tensor& packed, torch::Tensor& sfa);

int64_t sfa_size_bytes(int64_t rows, int64_t dim, bool is_sfb);
int64_t sfa_size_bytes_for(torch::Tensor const& anchor, int64_t rows, int64_t dim, bool is_sfb);
void rms_silu_nvfp4_ndhwc_bf16(
    torch::Tensor const& x, torch::Tensor const& gamma,
    c10::optional<torch::Tensor> const& awq_inv_scale, double eps,
    torch::Tensor& packed, torch::Tensor& scale_factors);
void quantize_bf16_to_nvfp4_linear(
    torch::Tensor const& input, torch::Tensor& packed,
    torch::Tensor& scale_factors);
void bf16_rms_silu_ncdhw(
    torch::Tensor const& x, torch::Tensor const& gamma,
    c10::optional<torch::Tensor> const& prev_cache, double eps,
    torch::Tensor& out, c10::optional<torch::Tensor> const& next_cache);
void bf16_rms_norm_ncdhw(
    torch::Tensor const& x, torch::Tensor const& gamma,
    c10::optional<torch::Tensor> const& bias, double eps,
    torch::Tensor& out);
