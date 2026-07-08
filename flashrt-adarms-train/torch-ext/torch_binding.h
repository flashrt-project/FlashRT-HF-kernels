// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

torch::Tensor flashrt_training_package_marker(torch::Tensor x);

std::tuple<torch::Tensor, torch::Tensor> adarms_fwd(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& scale,
    const c10::optional<torch::Tensor>& shift,
    const c10::optional<torch::Tensor>& weight, double eps);

std::tuple<torch::Tensor, torch::Tensor> adarms_bwd(
    const torch::Tensor& dy, const torch::Tensor& x,
    const c10::optional<torch::Tensor>& scale,
    const c10::optional<torch::Tensor>& weight, const torch::Tensor& rstd);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> resgate_adarms_fwd(
    const torch::Tensor& x, const torch::Tensor& h,
    const c10::optional<torch::Tensor>& gate,
    const c10::optional<torch::Tensor>& scale,
    const c10::optional<torch::Tensor>& shift,
    const c10::optional<torch::Tensor>& weight, double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
resgate_adarms_bwd(const torch::Tensor& dy,
                   const c10::optional<torch::Tensor>& dyr,
                   const torch::Tensor& r, const torch::Tensor& h,
                   const c10::optional<torch::Tensor>& gate,
                   const c10::optional<torch::Tensor>& scale,
                   const c10::optional<torch::Tensor>& weight,
                   const torch::Tensor& rstd);
