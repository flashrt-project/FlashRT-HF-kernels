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

void check_uint8(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kUInt8,
              name, " must have dtype torch.uint8");
}

struct Nvfp4ConvShape {
  int64_t n;
  int64_t tc;
  int64_t tn;
  int64_t h;
  int64_t w;
  int64_t ci;
  int64_t co;
};

Nvfp4ConvShape check_nvfp4_conv_inputs(
    torch::Tensor const& cache_packed,
    torch::Tensor const& new_packed,
    torch::Tensor const& weight_packed,
    torch::Tensor const& cache_sf,
    torch::Tensor const& new_sf,
    torch::Tensor const& weight_sf,
    torch::Tensor const& bias,
    c10::optional<torch::Tensor> const& outer_weight) {
  check_uint8(cache_packed, "cache_packed");
  check_uint8(new_packed, "new_packed");
  check_uint8(weight_packed, "weight_packed");
  check_uint8(cache_sf, "cache_sf");
  check_uint8(new_sf, "new_sf");
  check_uint8(weight_sf, "weight_sf");
  check_bf16(bias, "bias");
  TORCH_CHECK(cache_packed.dim() == 5 && new_packed.dim() == 5,
              "cache_packed/new_packed must be NDHWC packed tensors");
  TORCH_CHECK(weight_packed.dim() == 5,
              "weight_packed must have shape (Co,3,3,3,Ci/2)");
  const int64_t n = new_packed.size(0);
  const int64_t tc = cache_packed.size(1);
  const int64_t tn = new_packed.size(1);
  const int64_t h = new_packed.size(2);
  const int64_t w = new_packed.size(3);
  const int64_t ci = new_packed.size(4) * 2;
  const int64_t co = weight_packed.size(0);
  TORCH_CHECK(tc == 2, "T_cache must be 2");
  TORCH_CHECK(
      ci == 64 || ci % 128 == 0,
      "NVFP4 Conv3D accepts Ci=64 or multiples of 128; Ci=320 and "
      "other K64 multi-tile shapes must use the strong-library path");
  TORCH_CHECK(co % 8 == 0,
              "Co must be divisible by 8 for NVFP4 Conv3D");
  TORCH_CHECK(
      cache_packed.sizes() == torch::IntArrayRef({n, tc, h, w, ci / 2}),
      "cache_packed shape mismatch");
  TORCH_CHECK(
      weight_packed.sizes() ==
          torch::IntArrayRef({co, 3, 3, 3, ci / 2}),
      "weight_packed must have shape (Co,3,3,3,Ci/2)");
  TORCH_CHECK(
      cache_sf.sizes() == torch::IntArrayRef({n, tc, h, w, ci / 16}),
      "cache_sf must have shape (N,2,H,W,Ci/16)");
  TORCH_CHECK(
      new_sf.sizes() == torch::IntArrayRef({n, tn, h, w, ci / 16}),
      "new_sf must have shape (N,T,H,W,Ci/16)");
  TORCH_CHECK(
      weight_sf.sizes() ==
          torch::IntArrayRef({co, 3, 3, 3, ci / 16}),
      "weight_sf must have shape (Co,3,3,3,Ci/16)");
  TORCH_CHECK(bias.sizes() == torch::IntArrayRef({co}),
              "bias must have shape (Co,)");
  const int device = new_packed.get_device();
  for (auto const* tensor : {
           &cache_packed, &weight_packed, &cache_sf, &new_sf, &weight_sf,
           &bias}) {
    TORCH_CHECK(tensor->get_device() == device,
                "all NVFP4 Conv3D tensors must be on one CUDA device");
  }
  if (outer_weight.has_value()) {
    auto const& outer = outer_weight.value();
    check_cuda_contiguous(outer, "outer_weight");
    TORCH_CHECK(outer.scalar_type() == torch::kFloat32,
                "outer_weight must have dtype torch.float32");
    TORCH_CHECK(outer.sizes() == torch::IntArrayRef({co}),
                "outer_weight must have shape (Co,)");
    TORCH_CHECK(outer.get_device() == device,
                "outer_weight must be on the input CUDA device");
  }
  return {n, tc, tn, h, w, ci, co};
}

}  // namespace

void bf16_causal_conv3d_ndhwc_bf16(
    torch::Tensor const& cache_x,
    torch::Tensor const& new_x,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    double alpha,
    torch::Tensor& out) {
  check_bf16(cache_x, "cache_x");
  check_bf16(new_x, "new_x");
  check_bf16(weight, "weight");
  check_bf16(bias, "bias");
  check_bf16(out, "out");
  TORCH_CHECK(new_x.dim() == 5 && cache_x.dim() == 5,
              "cache_x/new_x must be NDHWC");
  const int64_t n = new_x.size(0), tn = new_x.size(1);
  const int64_t h = new_x.size(2), w = new_x.size(3), ci = new_x.size(4);
  TORCH_CHECK(weight.dim() == 5, "weight must have shape (Co,3,3,3,Ci)");
  const int64_t co = weight.size(0);
  TORCH_CHECK(cache_x.sizes() == torch::IntArrayRef({n, 2, h, w, ci}),
              "cache_x must have shape (N,2,H,W,Ci)");
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({co, 3, 3, 3, ci}),
              "weight must have shape (Co,3,3,3,Ci)");
  TORCH_CHECK(ci % 16 == 0 && co % 8 == 0,
              "Ci must be divisible by 16 and Co by 8");
  TORCH_CHECK(bias.sizes() == torch::IntArrayRef({co}),
              "bias must have shape (Co,)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({n, tn, h, w, co}),
              "out must have shape (N,T,H,W,Co)");
  TORCH_CHECK(cache_x.get_device() == new_x.get_device() &&
                  cache_x.get_device() == weight.get_device() &&
                  cache_x.get_device() == bias.get_device() &&
                  cache_x.get_device() == out.get_device(),
              "all tensors must be on the same CUDA device");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(new_x.device());
  const auto* props = at::cuda::getDeviceProperties(new_x.get_device());
  TORCH_CHECK(props->major == 11 && props->minor == 0,
              "bf16_causal_conv3d_ndhwc_bf16 requires SM110");
  auto stream = at::cuda::getCurrentCUDAStream(new_x.get_device()).stream();
  int status = flash_rt::conv::bf16_conv3d_v0_ndhwc_bf16out(
      cache_x.data_ptr(), new_x.data_ptr(), weight.data_ptr(), out.data_ptr(),
      bias.data_ptr(), n, 2, tn, h, w, ci, co,
      static_cast<float>(alpha), stream);
  TORCH_CHECK(status == 0,
              "bf16_causal_conv3d_ndhwc_bf16 failed with status ", status);
#else
  TORCH_CHECK(false, "world-model-conv was not built with CUDA support");
#endif
}

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
  TORCH_CHECK(ci <= 64,
              "FP8 Conv3D is accepted only for Ci=32/64; use the "
              "strong-library or NVFP4 path for larger channel counts");
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

void fp8_causal_conv3d_ndhwc_bf16(
    torch::Tensor const& cache_x,
    torch::Tensor const& new_x,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    double alpha,
    torch::Tensor& out) {
  check_fp8(cache_x, "cache_x");
  check_fp8(new_x, "new_x");
  check_fp8(weight, "weight");
  check_bf16(bias, "bias");
  check_bf16(out, "out");
  TORCH_CHECK(cache_x.dim() == 5 && new_x.dim() == 5,
              "cache_x/new_x must be NDHWC");
  const int64_t n = new_x.size(0);
  const int64_t tc = cache_x.size(1);
  const int64_t tn = new_x.size(1);
  const int64_t h = new_x.size(2);
  const int64_t w = new_x.size(3);
  const int64_t ci = new_x.size(4);
  TORCH_CHECK(weight.dim() == 5, "weight must be (Co,3,3,3,Ci)");
  const int64_t co = weight.size(0);
  TORCH_CHECK(cache_x.sizes() == torch::IntArrayRef({n, 2, h, w, ci}),
              "cache_x must have shape (N,2,H,W,Ci)");
  TORCH_CHECK(tc == 2, "T_cache must be 2");
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({co, 3, 3, 3, ci}),
              "weight must have shape (Co,3,3,3,Ci)");
  TORCH_CHECK(ci % 32 == 0, "Ci must be divisible by 32");
  TORCH_CHECK(ci <= 64,
              "FP8 Conv3D is accepted only for Ci=32/64; use the "
              "strong-library or NVFP4 path for larger channel counts");
  TORCH_CHECK(co % 2 == 0,
              "Co must be even because the native epilogue stores BF16 pairs");
  TORCH_CHECK(bias.sizes() == torch::IntArrayRef({co}),
              "bias must have shape (Co,)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({n, tn, h, w, co}),
              "out must have shape (N,T,H,W,Co)");
  TORCH_CHECK(cache_x.get_device() == new_x.get_device() &&
                  cache_x.get_device() == weight.get_device() &&
                  cache_x.get_device() == bias.get_device() &&
                  cache_x.get_device() == out.get_device(),
              "all tensors must be on the same CUDA device");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(cache_x.device());
  auto stream = at::cuda::getCurrentCUDAStream(cache_x.get_device()).stream();
  const int status =
      co % 8 == 0
          ? flash_rt::conv::fp8_conv3d_v17_ndhwc_bf16out(
                cache_x.data_ptr(), new_x.data_ptr(), weight.data_ptr(),
                out.data_ptr(), bias.data_ptr(), static_cast<int>(n),
                static_cast<int>(tc), static_cast<int>(tn),
                static_cast<int>(h), static_cast<int>(w),
                static_cast<int>(ci), static_cast<int>(co),
                static_cast<float>(alpha), stream)
          : flash_rt::conv::fp8_conv3d_v17_anyco_ndhwc_bf16out(
                cache_x.data_ptr(), new_x.data_ptr(), weight.data_ptr(),
                out.data_ptr(), bias.data_ptr(), static_cast<int>(n),
                static_cast<int>(tc), static_cast<int>(tn),
                static_cast<int>(h), static_cast<int>(w),
                static_cast<int>(ci), static_cast<int>(co),
                static_cast<float>(alpha), stream);
  TORCH_CHECK(status == 0,
              "fp8_causal_conv3d_ndhwc_bf16 failed with status ", status);
#else
  TORCH_CHECK(false, "world-model-conv was not built with CUDA support");
#endif
}

void fp8_conv2d_3x3_nhwc_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    double alpha,
    torch::Tensor& out) {
  check_fp8(input, "input");
  check_fp8(weight, "weight");
  check_bf16(bias, "bias");
  check_bf16(out, "out");
  TORCH_CHECK(input.dim() == 4, "input must have shape (N,H,W,Ci)");
  const int64_t n = input.size(0);
  const int64_t h = input.size(1);
  const int64_t w = input.size(2);
  const int64_t ci = input.size(3);
  TORCH_CHECK(weight.dim() == 4, "weight must have shape (Co,3,3,Ci)");
  const int64_t co = weight.size(0);
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({co, 3, 3, ci}),
              "weight must have shape (Co,3,3,Ci)");
  TORCH_CHECK(ci % 32 == 0 && co % 8 == 0,
              "Ci must be divisible by 32 and Co by 8");
  TORCH_CHECK(bias.sizes() == torch::IntArrayRef({co}),
              "bias must have shape (Co,)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({n, h, w, co}),
              "out must have shape (N,H,W,Co)");
  TORCH_CHECK(input.get_device() == weight.get_device() &&
                  input.get_device() == bias.get_device() &&
                  input.get_device() == out.get_device(),
              "all tensors must be on the same CUDA device");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  const int status = flash_rt::conv::fp8_conv2d_3x3_v2_nhwc_bf16out(
      input.data_ptr(), weight.data_ptr(), out.data_ptr(), bias.data_ptr(),
      static_cast<int>(n), static_cast<int>(h), static_cast<int>(w),
      static_cast<int>(ci), static_cast<int>(co), static_cast<float>(alpha),
      stream);
  TORCH_CHECK(status == 0,
              "fp8_conv2d_3x3_nhwc_bf16 failed with status ", status);
#else
  TORCH_CHECK(false, "world-model-conv was not built with CUDA support");
#endif
}

void fp8_conv2d_3x3_ncdhw_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    double alpha,
    torch::Tensor& out) {
  check_fp8(input, "input");
  check_fp8(weight, "weight");
  check_bf16(bias, "bias");
  check_bf16(out, "out");
  TORCH_CHECK(input.dim() == 5, "input must have shape (B,T,H,W,Ci)");
  const int64_t b = input.size(0);
  const int64_t t = input.size(1);
  const int64_t h = input.size(2);
  const int64_t w = input.size(3);
  const int64_t ci = input.size(4);
  TORCH_CHECK(weight.dim() == 4, "weight must have shape (Co,3,3,Ci)");
  const int64_t co = weight.size(0);
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({co, 3, 3, ci}),
              "weight must have shape (Co,3,3,Ci)");
  TORCH_CHECK(ci % 32 == 0 && co % 8 == 0,
              "Ci must be divisible by 32 and Co by 8");
  TORCH_CHECK(bias.sizes() == torch::IntArrayRef({co}),
              "bias must have shape (Co,)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({b, co, t, h, w}),
              "out must have shape (B,Co,T,H,W)");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  const int status =
      flash_rt::conv::fp8_conv2d_3x3_v2_nhwc_ncdhw_bf16out(
          input.data_ptr(), weight.data_ptr(), out.data_ptr(), bias.data_ptr(),
          static_cast<int>(b), static_cast<int>(t), static_cast<int>(h),
          static_cast<int>(w), static_cast<int>(ci), static_cast<int>(co),
          static_cast<float>(alpha), stream);
  TORCH_CHECK(status == 0,
              "fp8_conv2d_3x3_ncdhw_bf16 failed with status ", status);
#else
  TORCH_CHECK(false, "world-model-conv was not built with CUDA support");
#endif
}

void nvfp4_causal_conv3d_ndhwc_bf16(
    torch::Tensor const& cache_packed,
    torch::Tensor const& new_packed,
    torch::Tensor const& weight_packed,
    torch::Tensor const& cache_sf,
    torch::Tensor const& new_sf,
    torch::Tensor const& weight_sf,
    torch::Tensor const& bias,
    c10::optional<torch::Tensor> const& outer_weight,
    double alpha,
    torch::Tensor& out) {
  const auto shape = check_nvfp4_conv_inputs(
      cache_packed, new_packed, weight_packed, cache_sf, new_sf, weight_sf,
      bias, outer_weight);
  check_bf16(out, "out");
  TORCH_CHECK(
      out.sizes() ==
          torch::IntArrayRef(
              {shape.n, shape.tn, shape.h, shape.w, shape.co}),
      "out must have shape (N,T,H,W,Co)");
  TORCH_CHECK(out.get_device() == new_packed.get_device(),
              "out must be on the input CUDA device");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(new_packed.device());
  auto stream =
      at::cuda::getCurrentCUDAStream(new_packed.get_device()).stream();
  const int status = outer_weight.has_value()
      ? flash_rt::conv::motus_fp4_conv3d_v19sf_ndhwc_bf16out_v2(
            cache_packed.data_ptr(), new_packed.data_ptr(),
            weight_packed.data_ptr(), cache_sf.data_ptr(), new_sf.data_ptr(),
            weight_sf.data_ptr(), outer_weight->data_ptr(), out.data_ptr(),
            bias.data_ptr(), shape.n, shape.tc, shape.tn, shape.h, shape.w,
            shape.ci, shape.co, static_cast<float>(alpha), stream)
      : flash_rt::conv::motus_fp4_conv3d_v19sf_ndhwc_bf16out(
            cache_packed.data_ptr(), new_packed.data_ptr(),
            weight_packed.data_ptr(), cache_sf.data_ptr(), new_sf.data_ptr(),
            weight_sf.data_ptr(), out.data_ptr(), bias.data_ptr(), shape.n,
            shape.tc, shape.tn, shape.h, shape.w, shape.ci, shape.co,
            static_cast<float>(alpha), stream);
  TORCH_CHECK(status == 0,
              "nvfp4_causal_conv3d_ndhwc_bf16 failed with status ", status);
#endif
}

void nvfp4_causal_conv3d_residual_ncdhw_bf16(
    torch::Tensor const& cache_packed,
    torch::Tensor const& new_packed,
    torch::Tensor const& weight_packed,
    torch::Tensor const& cache_sf,
    torch::Tensor const& new_sf,
    torch::Tensor const& weight_sf,
    torch::Tensor const& bias,
    torch::Tensor const& residual,
    c10::optional<torch::Tensor> const& outer_weight,
    double alpha,
    torch::Tensor& out) {
  const auto shape = check_nvfp4_conv_inputs(
      cache_packed, new_packed, weight_packed, cache_sf, new_sf, weight_sf,
      bias, outer_weight);
  check_bf16(residual, "residual");
  check_bf16(out, "out");
  TORCH_CHECK(
      residual.sizes() ==
              torch::IntArrayRef(
                  {shape.n, shape.co, shape.tn, shape.h, shape.w}) &&
          out.sizes() ==
              torch::IntArrayRef(
                  {shape.n, shape.co, shape.tn, shape.h, shape.w}),
              "residual/out must have shape (N,Co,T,H,W)");
  TORCH_CHECK(
      residual.get_device() == new_packed.get_device() &&
          out.get_device() == new_packed.get_device(),
      "residual/out must be on the input CUDA device");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(new_packed.device());
  auto stream =
      at::cuda::getCurrentCUDAStream(new_packed.get_device()).stream();
  int status;
  if (outer_weight.has_value()) {
    status =
        flash_rt::conv::motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out_v2(
            cache_packed.data_ptr(), new_packed.data_ptr(),
            weight_packed.data_ptr(), cache_sf.data_ptr(), new_sf.data_ptr(),
            weight_sf.data_ptr(), outer_weight->data_ptr(), out.data_ptr(),
            bias.data_ptr(), residual.data_ptr(), shape.n, shape.tc, shape.tn,
            shape.h, shape.w, shape.ci, shape.co,
            static_cast<float>(alpha), stream);
  } else if (shape.ci % 128 == 0) {
    status =
        flash_rt::conv::motus_fp4_conv3d_v19sfbk128_ncdhw_res_bf16out(
            cache_packed.data_ptr(), new_packed.data_ptr(),
            weight_packed.data_ptr(), cache_sf.data_ptr(), new_sf.data_ptr(),
            weight_sf.data_ptr(), out.data_ptr(), bias.data_ptr(),
            residual.data_ptr(), shape.n, shape.tc, shape.tn, shape.h,
            shape.w, shape.ci, shape.co, static_cast<float>(alpha), stream);
  } else {
    status =
        flash_rt::conv::motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out(
            cache_packed.data_ptr(), new_packed.data_ptr(),
            weight_packed.data_ptr(), cache_sf.data_ptr(), new_sf.data_ptr(),
            weight_sf.data_ptr(), out.data_ptr(), bias.data_ptr(),
            residual.data_ptr(), shape.n, shape.tc, shape.tn, shape.h,
            shape.w, shape.ci, shape.co, static_cast<float>(alpha), stream);
  }
  TORCH_CHECK(
      status == 0,
      "nvfp4_causal_conv3d_residual_ncdhw_bf16 failed with status ", status);
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("bf16_causal_conv3d_ndhwc_bf16(Tensor cache_x, Tensor new_x, Tensor weight, Tensor bias, float alpha, Tensor! out) -> ()");
  ops.def("fp8_conv3d_v18_ncdhw_res_bf16out(Tensor cache_x, Tensor new_x, Tensor weight, Tensor bias, Tensor residual, float alpha, Tensor! out) -> ()");
  ops.def("fp8_causal_conv3d_ndhwc_bf16(Tensor cache_x, Tensor new_x, Tensor weight, Tensor bias, float alpha, Tensor! out) -> ()");
  ops.def("fp8_conv2d_3x3_nhwc_bf16(Tensor input, Tensor weight, Tensor bias, float alpha, Tensor! out) -> ()");
  ops.def("fp8_conv2d_3x3_ncdhw_bf16(Tensor input, Tensor weight, Tensor bias, float alpha, Tensor! out) -> ()");
  ops.def("nvfp4_causal_conv3d_ndhwc_bf16(Tensor cache_packed, Tensor new_packed, Tensor weight_packed, Tensor cache_sf, Tensor new_sf, Tensor weight_sf, Tensor bias, Tensor? outer_weight, float alpha, Tensor! out) -> ()");
  ops.def("nvfp4_causal_conv3d_residual_ncdhw_bf16(Tensor cache_packed, Tensor new_packed, Tensor weight_packed, Tensor cache_sf, Tensor new_sf, Tensor weight_sf, Tensor bias, Tensor residual, Tensor? outer_weight, float alpha, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("bf16_causal_conv3d_ndhwc_bf16", torch::kCUDA, &bf16_causal_conv3d_ndhwc_bf16);
  ops.impl("fp8_conv3d_v18_ncdhw_res_bf16out", torch::kCUDA, &fp8_conv3d_v18_ncdhw_res_bf16out);
  ops.impl("fp8_causal_conv3d_ndhwc_bf16", torch::kCUDA, &fp8_causal_conv3d_ndhwc_bf16);
  ops.impl("fp8_conv2d_3x3_nhwc_bf16", torch::kCUDA, &fp8_conv2d_3x3_nhwc_bf16);
  ops.impl("fp8_conv2d_3x3_ncdhw_bf16", torch::kCUDA, &fp8_conv2d_3x3_ncdhw_bf16);
  ops.impl("nvfp4_causal_conv3d_ndhwc_bf16", torch::kCUDA, &nvfp4_causal_conv3d_ndhwc_bf16);
  ops.impl("nvfp4_causal_conv3d_residual_ncdhw_bf16", torch::kCUDA, &nvfp4_causal_conv3d_residual_ncdhw_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
