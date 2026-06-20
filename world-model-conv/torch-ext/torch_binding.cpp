// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "registration.h"
#include "torch_binding.h"
#include "world_model_conv.cuh"

namespace {

void check_cuda_contiguous(torch::Tensor const& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_fp8(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kFloat8_e4m3fn, name, " must have dtype torch.float8_e4m3fn");
}

void check_bf16(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must have dtype torch.bfloat16");
}

}  // namespace

void fp8_conv3d_v18_ncdhw_res_bf16out(
    torch::Tensor const& cache_x,
    torch::Tensor const& new_x,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    torch::Tensor const& residual,
    double alpha,
    torch::Tensor& out) {
  check_fp8(cache_x, "cache_x");
  check_fp8(new_x, "new_x");
  check_fp8(weight, "weight");
  check_bf16(bias, "bias");
  check_bf16(residual, "residual");
  check_bf16(out, "out");
  TORCH_CHECK(cache_x.dim() == 5 && new_x.dim() == 5, "cache_x/new_x must be NDHWC");
  TORCH_CHECK(weight.dim() == 5, "weight must be (Co,3,3,3,Ci)");
  const int64_t n = new_x.size(0);
  const int64_t t_cache = cache_x.size(1);
  const int64_t t_new = new_x.size(1);
  const int64_t h = new_x.size(2);
  const int64_t w = new_x.size(3);
  const int64_t ci = new_x.size(4);
  const int64_t co = weight.size(0);
  TORCH_CHECK(cache_x.sizes() == torch::IntArrayRef({n, t_cache, h, w, ci}), "cache_x shape mismatch");
  TORCH_CHECK(t_cache == 2, "T_cache must be 2");
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({co, 3, 3, 3, ci}), "weight shape mismatch");
  TORCH_CHECK(ci % 32 == 0, "Ci must be a multiple of 32");
  TORCH_CHECK(co % 8 == 0, "Co must be a multiple of 8");
  TORCH_CHECK(bias.sizes() == torch::IntArrayRef({co}), "bias must have shape (Co,)");
  TORCH_CHECK(residual.sizes() == torch::IntArrayRef({n, co, t_new, h, w}), "residual must be NCDHW");
  TORCH_CHECK(out.sizes() == residual.sizes(), "out must be NCDHW");
  TORCH_CHECK(cache_x.get_device() == new_x.get_device() &&
              cache_x.get_device() == weight.get_device() &&
              cache_x.get_device() == out.get_device(),
              "all tensors must be on the same CUDA device");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(cache_x.device());
  auto stream = at::cuda::getCurrentCUDAStream(cache_x.get_device()).stream();
  int status = flash_rt::conv::fp8_conv3d_v18_ncdhw_res_bf16out(
      cache_x.data_ptr(), new_x.data_ptr(), weight.data_ptr(), out.data_ptr(),
      bias.data_ptr(), residual.data_ptr(), static_cast<int>(n), static_cast<int>(t_cache),
      static_cast<int>(t_new), static_cast<int>(h), static_cast<int>(w),
      static_cast<int>(ci), static_cast<int>(co), static_cast<float>(alpha), stream);
  TORCH_CHECK(status == 0, "fp8_conv3d_v18_ncdhw_res_bf16out failed with status ", status);
#else
  TORCH_CHECK(false, "world-model-conv was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("fp8_conv3d_v18_ncdhw_res_bf16out(Tensor cache_x, Tensor new_x, Tensor weight, Tensor bias, Tensor residual, float alpha, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("fp8_conv3d_v18_ncdhw_res_bf16out", torch::kCUDA, &fp8_conv3d_v18_ncdhw_res_bf16out);
#endif
}
