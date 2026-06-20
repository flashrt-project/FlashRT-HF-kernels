// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void unpack_packed_bf16(
    torch::Tensor const& k_idx_packed,
    torch::Tensor const& k_qjl_packed,
    torch::Tensor const& v_idx_packed,
    torch::Tensor const& cb_k_mse,
    torch::Tensor const& cb_v,
    int64_t b_k_mse,
    int64_t b_v,
    torch::Tensor& y_k,
    torch::Tensor& qjl_bf,
    torch::Tensor& y_v);

void unpack_packed_mixed(
    torch::Tensor const& k_idx_packed,
    torch::Tensor const& k_qjl_packed,
    torch::Tensor const& v_idx_packed,
    torch::Tensor const& cb_k_mse,
    torch::Tensor const& cb_v,
    int64_t b_k_mse,
    int64_t b_v,
    torch::Tensor& y_k,
    torch::Tensor& qjl_f,
    torch::Tensor& y_v);

void combine_kv_bf16(
    torch::Tensor const& k_mse,
    torch::Tensor const& k_qjl,
    torch::Tensor const& v_unit,
    torch::Tensor const& k_norm,
    torch::Tensor const& k_rnorm,
    torch::Tensor const& v_norm,
    double coef,
    torch::Tensor& k_out,
    torch::Tensor& v_out);
