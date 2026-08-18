// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void fp4_w4a16_linear_bf16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor& out,
    double alpha,
    int64_t variant);

void quantize_fp4_sfa_fp16(
    torch::Tensor const& x,
    torch::Tensor& packed,
    torch::Tensor& sfa,
    bool is_sfb);

void quantize_fp4_sfa_bf16(
    torch::Tensor const& x,
    torch::Tensor& packed,
    torch::Tensor& sfa,
    bool is_sfb);

void quantize_fp4_sfa_bf16_pdl(
    torch::Tensor const& x,
    torch::Tensor& packed,
    torch::Tensor& sfa,
    bool is_sfb);

void fp4_w4a4_gemm_warpsplit_mrows_pdl_bf16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor& out,
    double alpha,
    int64_t warps,
    int64_t stages);

void quantize_fp4_sfa_mse_bf16(
    torch::Tensor const& x,
    torch::Tensor& packed,
    torch::Tensor& sfa,
    bool is_sfb);

void dequantize_fp4_sfa_fp16(
    torch::Tensor const& packed,
    torch::Tensor const& sfa,
    torch::Tensor& out,
    bool is_sfb);

void nvfp4_w4a16_marlin_bf16(
    torch::Tensor const& x,
    torch::Tensor const& weight_marlin,
    torch::Tensor const& weight_scale_marlin,
    torch::Tensor const& weight_global_scale,
    torch::Tensor& workspace,
    torch::Tensor& out);

void nvfp4_w4a16_marlin_repack(
    torch::Tensor const& qweight_kn,
    torch::Tensor& weight_marlin);
