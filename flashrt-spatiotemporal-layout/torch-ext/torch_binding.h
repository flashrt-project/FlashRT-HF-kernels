// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void ncdhw_to_blc_bf16(torch::Tensor const& x, torch::Tensor& out);
void patch_im2col_bf16(torch::Tensor const& x, torch::Tensor& out);
void time_unshuffle2_bf16(torch::Tensor const& x, torch::Tensor& out);
void add_bias_ncdhw_bf16(torch::Tensor& x, torch::Tensor const& bias);
void update_cache2_ncdhw_bf16(torch::Tensor const& cur, torch::Tensor const& prev, torch::Tensor& out);
void channel_to_space3d_bf16(
    torch::Tensor const& x, int64_t out_channels, int64_t temporal_factor,
    int64_t spatial_factor, int64_t repeats, bool first_chunk,
    torch::Tensor& out);
void pack_causal_cache3_nhwc_bf16(
    torch::Tensor const& previous, torch::Tensor const& current,
    torch::Tensor& out);
void ndhwc_to_ncdhw_bf16(torch::Tensor const& x, torch::Tensor& out);
void ndhwc_to_ncdhw_bias_bf16(
    torch::Tensor const& x, torch::Tensor const& bias, torch::Tensor& out);
void ndhwc_to_ncdhw_add_bf16(
    torch::Tensor const& x, torch::Tensor const& residual, torch::Tensor& out);
void ncdhw_quantize_fp8_static_ndhwc_bf16(
    torch::Tensor const& x, double scale, torch::Tensor& out);
void upsample2x_quantize_fp8_static_nhwc_bf16(
    torch::Tensor const& x, double scale, torch::Tensor& out);
