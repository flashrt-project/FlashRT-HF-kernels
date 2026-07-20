// SPDX-License-Identifier: Apache-2.0
#include <torch/all.h>
#include <torch/library.h>
#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include "action_ffn_megakernel_v6t_sm120.cuh"
#include "registration.h"
#include "torch_binding.h"
#include "und_ffn_megakernel_v5split_stage3_sm120.cuh"
#include "und_ffn_megakernel_v5t_sm120.cuh"
namespace {
void ct(torch::Tensor const &t, c10::ScalarType d, const char *n) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), n, " must be contiguous CUDA");
  TORCH_CHECK(t.scalar_type() == d, n, " has incorrect dtype");
}
void same(torch::Tensor const &a, torch::Tensor const &b) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              "all tensors must share a device");
}
} // namespace
void fp8_gelu_ffn_gated_residual_bf16_out(
    torch::Tensor const &x, torch::Tensor const &uw, torch::Tensor const &ub,
    torch::Tensor const &dinv, torch::Tensor const &dw, torch::Tensor const &db,
    torch::Tensor const &gate, torch::Tensor const &res, double ua, double da,
    double hs, torch::Tensor &out, torch::Tensor &scratch) {
  ct(x, c10::ScalarType::Float8_e4m3fn, "input");
  ct(uw, c10::ScalarType::Float8_e4m3fn, "up_weight");
  ct(dw, c10::ScalarType::Float8_e4m3fn, "down_weight");
  ct(ub, torch::kBFloat16, "up_bias");
  ct(db, torch::kBFloat16, "down_bias");
  ct(gate, torch::kBFloat16, "gate");
  ct(res, torch::kBFloat16, "residual");
  ct(dinv, torch::kBFloat16, "down_inverse_scale");
  ct(out, torch::kBFloat16, "output");
  ct(scratch, c10::ScalarType::Float8_e4m3fn, "hidden_scratch");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(0) <= 32 &&
                  x.size(1) == 1024,
              "input must be [M,1024], 1<=M<=32");
  TORCH_CHECK(uw.sizes() == torch::IntArrayRef({4096, 1024}) &&
                  dw.sizes() == torch::IntArrayRef({1024, 4096}),
              "weight shapes must be [4096,1024] and [1024,4096]");
  TORCH_CHECK(ub.numel() == 4096 && db.numel() == 1024,
              "bias shapes are invalid");
  TORCH_CHECK(gate.sizes() == x.sizes() && res.sizes() == x.sizes() &&
                  out.sizes() == x.sizes(),
              "gate/residual/output shape mismatch");
  TORCH_CHECK(scratch.sizes() == torch::IntArrayRef({x.size(0), 4096}),
              "hidden_scratch must be [M,4096]");
  same(x, out);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard gd(x.device());
  auto s = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  int rc = flash_rt::megakernel::action_ffn_v6t_launch_sm120(
      x.data_ptr(), uw.data_ptr(), ub.data_ptr(), dinv.data_ptr(),
      dw.data_ptr(), db.data_ptr(), gate.data_ptr(), res.data_ptr(),
      out.data_ptr(), scratch.data_ptr(), x.size(0), 1024, 4096, 4096, 1024, ua,
      da, hs, s);
  TORCH_CHECK(rc == 0, "gated FFN megakernel failed rc=", rc);
#else
  TORCH_CHECK(false, "CUDA support was not built");
#endif
}
void fp8_gelu_ffn_residual_bf16_out(
    torch::Tensor const &x, torch::Tensor const &uinv, torch::Tensor const &uw,
    torch::Tensor const &ub, torch::Tensor const &dinv, torch::Tensor const &dw,
    torch::Tensor const &db, torch::Tensor const &res, double ua, double da,
    double us, double ds, bool split, torch::Tensor &out, torch::Tensor &xs,
    torch::Tensor &hs, torch::Tensor &barrier) {
  ct(x, torch::kBFloat16, "input");
  ct(uinv, torch::kBFloat16, "up_inverse_scale");
  ct(dinv, torch::kBFloat16, "down_inverse_scale");
  ct(uw, c10::ScalarType::Float8_e4m3fn, "up_weight");
  ct(dw, c10::ScalarType::Float8_e4m3fn, "down_weight");
  ct(ub, torch::kBFloat16, "up_bias");
  ct(db, torch::kBFloat16, "down_bias");
  ct(res, torch::kBFloat16, "residual");
  ct(out, torch::kBFloat16, "output");
  ct(xs, c10::ScalarType::Float8_e4m3fn, "input_scratch");
  ct(hs, c10::ScalarType::Float8_e4m3fn, "hidden_scratch");
  ct(barrier, torch::kUInt32, "barrier");
  int64_t cap = split ? 192 : 144;
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(0) <= cap &&
                  x.size(1) == 512,
              "input must be [M,512] in selected capacity");
  TORCH_CHECK(uw.sizes() == torch::IntArrayRef({2048, 512}) &&
                  dw.sizes() == torch::IntArrayRef({512, 2048}),
              "weight shapes must be [2048,512] and [512,2048]");
  TORCH_CHECK(res.sizes() == x.sizes() && out.sizes() == x.sizes() &&
                  xs.sizes() == x.sizes() &&
                  hs.sizes() == torch::IntArrayRef({x.size(0), 2048}),
              "buffer shape mismatch");
  TORCH_CHECK(barrier.numel() >= 2, "barrier needs two uint32 elements");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard gd(x.device());
  auto s = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  int rc =
      split ? flash_rt::megakernel::und_ffn_v5split_stage3_launch_sm120(
                  x.data_ptr(), uinv.data_ptr(), uw.data_ptr(), ub.data_ptr(),
                  dinv.data_ptr(), dw.data_ptr(), db.data_ptr(), res.data_ptr(),
                  out.data_ptr(), xs.data_ptr(), hs.data_ptr(), x.size(0), 512,
                  2048, 2048, 512, ua, da, us, ds, barrier.data_ptr(), s)
            : flash_rt::megakernel::und_ffn_v5t_launch_sm120(
                  x.data_ptr(), uinv.data_ptr(), uw.data_ptr(), ub.data_ptr(),
                  dinv.data_ptr(), dw.data_ptr(), db.data_ptr(), res.data_ptr(),
                  out.data_ptr(), xs.data_ptr(), hs.data_ptr(), x.size(0), 512,
                  2048, 2048, 512, ua, da, us, ds, barrier.data_ptr(), s);
  TORCH_CHECK(rc == 0, "residual FFN megakernel failed rc=", rc);
#else
  TORCH_CHECK(false, "CUDA support was not built");
#endif
}
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("fp8_gelu_ffn_gated_residual_bf16_out(Tensor x, Tensor uw, Tensor "
          "ub, Tensor dinv, Tensor dw, Tensor db, Tensor gate, Tensor "
          "residual, float up_alpha, float down_alpha, float hidden_scale, "
          "Tensor! output, Tensor! hidden_scratch) -> ()");
  ops.def("fp8_gelu_ffn_residual_bf16_out(Tensor x, Tensor uinv, Tensor uw, "
          "Tensor ub, Tensor dinv, Tensor dw, Tensor db, Tensor residual, "
          "float up_alpha, float down_alpha, float input_scale, float "
          "hidden_scale, bool split_stage, Tensor! output, Tensor! "
          "input_scratch, Tensor! hidden_scratch, Tensor! barrier) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("fp8_gelu_ffn_gated_residual_bf16_out", torch::kCUDA,
           &fp8_gelu_ffn_gated_residual_bf16_out);
  ops.impl("fp8_gelu_ffn_residual_bf16_out", torch::kCUDA,
           &fp8_gelu_ffn_residual_bf16_out);
#endif
}
REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
