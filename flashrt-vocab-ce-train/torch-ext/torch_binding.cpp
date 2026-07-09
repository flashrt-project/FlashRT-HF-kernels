// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>
#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"
#include "vocab_ce_train.cuh"

torch::Tensor flashrt_training_package_marker(torch::Tensor x) { return x; }

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
vocab_ce_fwd_stream(const torch::Tensor& hidden, const torch::Tensor& weight,
                    const torch::Tensor& labels) {
  namespace vc = flashrt_hub::vocab_ce_train;
  TORCH_CHECK(hidden.is_cuda() && hidden.is_contiguous() && hidden.dim() == 2 &&
                  hidden.scalar_type() == torch::kFloat32,
              "hidden must be contiguous CUDA fp32 (N, H)");
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.dim() == 2 &&
                  weight.scalar_type() == torch::kFloat32,
              "weight must be contiguous CUDA fp32 (V, H)");
  TORCH_CHECK(labels.is_cuda() && labels.is_contiguous() && labels.dim() == 1 &&
                  labels.scalar_type() == torch::kLong,
              "labels must be contiguous CUDA int64 (N,)");
  const long rows = hidden.size(0);
  const long h = hidden.size(1);
  const long v = weight.size(0);
  TORCH_CHECK(weight.size(1) == h, "weight/hidden H mismatch");
  TORCH_CHECK(labels.size(0) == rows, "labels/hidden rows mismatch");
  TORCH_CHECK(rows >= 1 && rows <= vc::kMaxRows, "rows must be in [1, ",
              vc::kMaxRows, "]");
  TORCH_CHECK(v % vc::kVTile == 0, "V must be a multiple of ", vc::kVTile);
  TORCH_CHECK(h % 128 == 0, "H must be a multiple of 128");

  const long tiles = v / vc::kVTile;
  auto logits = torch::empty({rows, v}, hidden.options());
  auto pmax = torch::empty({rows, tiles}, hidden.options());
  auto psum = torch::empty({rows, tiles}, hidden.options());
  auto label_logit = torch::zeros({rows}, hidden.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  vc::vocab_ce_fwd_launch(hidden.data_ptr<float>(), weight.data_ptr<float>(),
                          labels.data_ptr<long>(), logits.data_ptr<float>(),
                          pmax.data_ptr<float>(), psum.data_ptr<float>(),
                          label_logit.data_ptr<float>(), (int)rows, (int)v,
                          (int)h, stream.stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {logits, pmax, psum, label_logit};
}

std::tuple<torch::Tensor, torch::Tensor> vocab_ce_stats(
    const torch::Tensor& logits) {
  namespace vc = flashrt_hub::vocab_ce_train;
  TORCH_CHECK(logits.is_cuda() && logits.is_contiguous() && logits.dim() == 2 &&
                  logits.scalar_type() == torch::kFloat32,
              "logits must be contiguous CUDA fp32 (N, V)");
  const long rows = logits.size(0);
  const long v = logits.size(1);
  TORCH_CHECK(v % 32 == 0, "V must be a multiple of 32");
  auto pmax = torch::empty({rows, (long)vc::kStatsSplits}, logits.options());
  auto psum = torch::empty({rows, (long)vc::kStatsSplits}, logits.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  vc::vocab_ce_stats_launch(logits.data_ptr<float>(), pmax.data_ptr<float>(),
                            psum.data_ptr<float>(), (int)rows, (int)v,
                            stream.stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {pmax, psum};
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("_flashrt_training_package_marker(Tensor x) -> Tensor");
  ops.impl("_flashrt_training_package_marker",
           c10::DispatchKey::CompositeExplicitAutograd,
           &flashrt_training_package_marker);
  ops.def(
      "vocab_ce_fwd_stream(Tensor hidden, Tensor weight, Tensor labels) -> "
      "(Tensor, Tensor, Tensor, Tensor)");
  ops.impl("vocab_ce_fwd_stream", torch::kCUDA, &vocab_ce_fwd_stream);
  ops.def("vocab_ce_stats(Tensor logits) -> (Tensor, Tensor)");
  ops.impl("vocab_ce_stats", torch::kCUDA, &vocab_ce_stats);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
