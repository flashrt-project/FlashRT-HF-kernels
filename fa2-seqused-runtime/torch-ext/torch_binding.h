#pragma once

#include <torch/all.h>

void fa2_forward_static(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    torch::Tensor& out, torch::Tensor& softmax_lse,
    const c10::optional<torch::Tensor>& softmax_lse_accum,
    const c10::optional<torch::Tensor>& out_accum, double softmax_scale,
    bool causal, int64_t num_sms);

void fa2_forward_seqused_static(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& seqused_k, torch::Tensor& out,
    torch::Tensor& softmax_lse,
    const c10::optional<torch::Tensor>& softmax_lse_accum,
    const c10::optional<torch::Tensor>& out_accum, double softmax_scale,
    int64_t num_sms);
