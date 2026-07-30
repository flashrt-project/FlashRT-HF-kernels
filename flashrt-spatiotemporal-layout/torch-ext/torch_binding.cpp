// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "registration.h"
#include "bf16_ndhwc_to_ncdhw_transpose.cuh"
#include "bf16_quant_fp8_ncdhw_to_ndhwc.cuh"
#include "spatiotemporal_layout.cuh"
#include "torch_binding.h"

namespace {

void check_bf16(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
}

void check_ncdhw(torch::Tensor const& tensor, const char* name) {
  check_bf16(tensor, name);
  TORCH_CHECK(tensor.dim() == 5, name, " must have shape (B, C, T, H, W)");
  for (int i = 0; i < 5; ++i) {
    TORCH_CHECK(tensor.size(i) > 0, name, " dimensions must be positive");
  }
}

void check_ndhwc(torch::Tensor const& tensor, const char* name) {
  check_bf16(tensor, name);
  TORCH_CHECK(tensor.dim() == 5, name, " must have shape (B, T, H, W, C)");
  for (int i = 0; i < 5; ++i) {
    TORCH_CHECK(tensor.size(i) > 0, name, " dimensions must be positive");
  }
}

void check_fp8(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat8_e4m3fn,
              name, " must have dtype torch.float8_e4m3fn");
}

void check_same_device(torch::Tensor const& a,
                       torch::Tensor const& b,
                       const char* a_name,
                       const char* b_name) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              a_name, " and ", b_name, " must be on the same CUDA device");
}

}  // namespace

void ncdhw_to_blc_bf16(torch::Tensor const& x, torch::Tensor& out) {
  check_ncdhw(x, "x");
  check_bf16(out, "out");
  const int64_t b = x.size(0);
  const int64_t c = x.size(1);
  const int64_t t = x.size(2);
  const int64_t h = x.size(3);
  const int64_t w = x.size(4);
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({b, t * h * w, c}),
              "out must have shape (B, T * H * W, C)");
  check_same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::spatiotemporal_layout::ncdhw_to_blc_bf16(
      x.data_ptr(), out.data_ptr(),
      static_cast<int>(b), static_cast<int>(c), static_cast<int>(t),
      static_cast<int>(h), static_cast<int>(w), stream);
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

void patch_im2col_bf16(torch::Tensor const& x, torch::Tensor& out) {
  check_bf16(x, "x");
  check_bf16(out, "out");
  TORCH_CHECK(x.dim() == 4, "x must have shape (num_views, 224, 224, 3)");
  const int64_t nv = x.size(0);
  TORCH_CHECK(nv > 0, "num_views must be positive");
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({nv, 224, 224, 3}),
              "x must have shape (num_views, 224, 224, 3)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({nv * 256, 588}),
              "out must have shape (num_views * 256, 588)");
  check_same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::spatiotemporal_layout::patch_im2col_bf16(
      x.data_ptr(),
      out.data_ptr(),
      static_cast<int>(nv),
      stream);
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

void time_unshuffle2_bf16(torch::Tensor const& x, torch::Tensor& out) {
  check_ncdhw(x, "x");
  check_bf16(out, "out");
  const int64_t b = x.size(0);
  const int64_t c2 = x.size(1);
  const int64_t t = x.size(2);
  const int64_t h = x.size(3);
  const int64_t w = x.size(4);
  TORCH_CHECK((c2 % 2) == 0, "x.shape[1] must be even");
  const int64_t c = c2 / 2;
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({b, c, 2 * t, h, w}),
              "out must have shape (B, C / 2, 2 * T, H, W)");
  check_same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::spatiotemporal_layout::time_unshuffle2_bf16(
      x.data_ptr(), out.data_ptr(),
      static_cast<int>(b), static_cast<int>(c), static_cast<int>(t),
      static_cast<int>(h), static_cast<int>(w), stream);
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

void add_bias_ncdhw_bf16(torch::Tensor& x, torch::Tensor const& bias) {
  check_ncdhw(x, "x");
  check_bf16(bias, "bias");
  const int64_t b = x.size(0);
  const int64_t c = x.size(1);
  const int64_t t = x.size(2);
  const int64_t h = x.size(3);
  const int64_t w = x.size(4);
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c,
              "bias must have shape (C,)");
  check_same_device(x, bias, "x", "bias");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::spatiotemporal_layout::add_bias_ncdhw_bf16(
      x.data_ptr(), bias.data_ptr(),
      static_cast<int>(b), static_cast<int>(c), static_cast<int>(t),
      static_cast<int>(h), static_cast<int>(w), stream);
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

void update_cache2_ncdhw_bf16(torch::Tensor const& cur, torch::Tensor const& prev, torch::Tensor& out) {
  check_ncdhw(cur, "cur");
  check_bf16(prev, "prev");
  check_bf16(out, "out");
  const int64_t b = cur.size(0);
  const int64_t c = cur.size(1);
  const int64_t t = cur.size(2);
  const int64_t h = cur.size(3);
  const int64_t w = cur.size(4);
  TORCH_CHECK(prev.sizes() == torch::IntArrayRef({b, c, 2, h, w}),
              "prev must have shape (B, C, 2, H, W)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({b, c, 2, h, w}),
              "out must have shape (B, C, 2, H, W)");
  check_same_device(cur, prev, "cur", "prev");
  check_same_device(cur, out, "cur", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(cur.device());
  auto stream = at::cuda::getCurrentCUDAStream(cur.get_device()).stream();
  flash_rt::spatiotemporal_layout::update_cache2_ncdhw_bf16(
      cur.data_ptr(), prev.data_ptr(), out.data_ptr(),
      static_cast<int>(b), static_cast<int>(c), static_cast<int>(t),
      static_cast<int>(h), static_cast<int>(w), stream);
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

void channel_to_space3d_bf16(
    torch::Tensor const& x,
    int64_t out_channels,
    int64_t temporal_factor,
    int64_t spatial_factor,
    int64_t repeats,
    bool first_chunk,
    torch::Tensor& out) {
  check_ncdhw(x, "x");
  check_ncdhw(out, "out");
  const auto b = x.size(0);
  const auto in_channels = x.size(1);
  const auto t = x.size(2);
  const auto h = x.size(3);
  const auto w = x.size(4);
  TORCH_CHECK(
      out_channels > 0 && temporal_factor > 0 && spatial_factor > 0 &&
          repeats > 0,
      "channel-to-space parameters must be positive");
  TORCH_CHECK(
      in_channels * repeats >=
          out_channels * temporal_factor * spatial_factor * spatial_factor,
      "input channels times repeats must cover the channel-to-space output");
  const int64_t out_t =
      t * temporal_factor - (first_chunk ? temporal_factor - 1 : 0);
  TORCH_CHECK(
      out.sizes() ==
          torch::IntArrayRef(
              {b, out_channels, out_t, h * spatial_factor,
               w * spatial_factor}),
      "out has the wrong channel-to-space NCDHW shape");
  check_same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::spatiotemporal_layout::channel_to_space3d_bf16(
      x.data_ptr(), out.data_ptr(), static_cast<int>(b),
      static_cast<int>(in_channels), static_cast<int>(out_channels),
      static_cast<int>(t), static_cast<int>(h), static_cast<int>(w),
      static_cast<int>(temporal_factor), static_cast<int>(spatial_factor),
      static_cast<int>(repeats), first_chunk, stream);
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

void pack_causal_cache3_nhwc_bf16(
    torch::Tensor const& previous,
    torch::Tensor const& current,
    torch::Tensor& out) {
  check_ncdhw(previous, "previous");
  check_ncdhw(current, "current");
  check_bf16(out, "out");
  const auto b = current.size(0);
  const auto c = current.size(1);
  const auto h = current.size(3);
  const auto w = current.size(4);
  TORCH_CHECK(current.size(2) == 1, "current must have one frame");
  TORCH_CHECK(
      previous.sizes() == torch::IntArrayRef({b, c, 2, h, w}),
      "previous must have shape (B,C,2,H,W)");
  TORCH_CHECK(
      out.sizes() == torch::IntArrayRef({b, h, w, 3 * c}),
      "out must have shape (B,H,W,3C)");
  check_same_device(current, previous, "current", "previous");
  check_same_device(current, out, "current", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(current.device());
  auto stream =
      at::cuda::getCurrentCUDAStream(current.get_device()).stream();
  flash_rt::spatiotemporal_layout::pack_causal_cache3_nhwc_bf16(
      previous.data_ptr(), current.data_ptr(), out.data_ptr(),
      static_cast<int>(b), static_cast<int>(c), static_cast<int>(h),
      static_cast<int>(w), stream);
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

void avg_pool3d_channels_bf16(
    torch::Tensor const& x,
    int64_t out_channels,
    int64_t factor_t,
    int64_t factor_s,
    int64_t group_size,
    torch::Tensor& out) {
  check_ncdhw(x, "x");
  check_ncdhw(out, "out");
  const int64_t b = x.size(0);
  const int64_t c = x.size(1);
  const int64_t t = x.size(2);
  const int64_t h = x.size(3);
  const int64_t w = x.size(4);
  TORCH_CHECK(out_channels > 0 && factor_t > 0 && factor_s > 0 &&
                  group_size > 0 && group_size <= 8,
              "pooling parameters must be positive and group_size <= 8");
  TORCH_CHECK(h % factor_s == 0 && w % factor_s == 0,
              "H and W must be divisible by factor_s");
  TORCH_CHECK(c * factor_t * factor_s * factor_s ==
                  out_channels * group_size,
              "C * factor_t * factor_s^2 must equal out_channels * group_size");
  const int64_t out_t = (t + factor_t - 1) / factor_t;
  TORCH_CHECK(
      out.sizes() ==
          torch::IntArrayRef(
              {b, out_channels, out_t, h / factor_s, w / factor_s}),
      "out has the wrong pooled NCDHW shape");
  check_same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::spatiotemporal_layout::avg_pool3d_channels_bf16(
      x.data_ptr(),
      out.data_ptr(),
      static_cast<int>(b),
      static_cast<int>(c),
      static_cast<int>(t),
      static_cast<int>(h),
      static_cast<int>(w),
      static_cast<int>(out_channels),
      static_cast<int>(factor_t),
      static_cast<int>(factor_s),
      static_cast<int>(group_size),
      stream);
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

void ndhwc_to_ncdhw_bf16(torch::Tensor const& x, torch::Tensor& out) {
  check_ndhwc(x, "x");
  check_ncdhw(out, "out");
  const int64_t b = x.size(0);
  const int64_t t = x.size(1);
  const int64_t h = x.size(2);
  const int64_t w = x.size(3);
  const int64_t c = x.size(4);
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({b, c, t, h, w}),
              "out must have shape (B, C, T, H, W)");
  check_same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  TORCH_CHECK(
      flash_rt::quantize::bf16_ndhwc_to_ncdhw_transpose(
          x.data_ptr(), out.data_ptr(), static_cast<int>(b),
          static_cast<int>(c), static_cast<int>(t), static_cast<int>(h),
          static_cast<int>(w), stream) == 0,
      "ndhwc_to_ncdhw_bf16 launch failed");
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

void ndhwc_to_ncdhw_bias_bf16(
    torch::Tensor const& x, torch::Tensor const& bias, torch::Tensor& out) {
  check_ndhwc(x, "x");
  check_bf16(bias, "bias");
  check_ncdhw(out, "out");
  const int64_t b = x.size(0);
  const int64_t t = x.size(1);
  const int64_t h = x.size(2);
  const int64_t w = x.size(3);
  const int64_t c = x.size(4);
  TORCH_CHECK(bias.sizes() == torch::IntArrayRef({c}),
              "bias must have shape (C,)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({b, c, t, h, w}),
              "out must have shape (B, C, T, H, W)");
  check_same_device(x, bias, "x", "bias");
  check_same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  TORCH_CHECK(
      flash_rt::quantize::bf16_ndhwc_to_ncdhw_bias_bf16(
          x.data_ptr(), bias.data_ptr(), out.data_ptr(), static_cast<int>(b),
          static_cast<int>(c), static_cast<int>(t), static_cast<int>(h),
          static_cast<int>(w), stream) == 0,
      "ndhwc_to_ncdhw_bias_bf16 launch failed");
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

void ndhwc_to_ncdhw_add_bf16(
    torch::Tensor const& x, torch::Tensor const& residual, torch::Tensor& out) {
  check_ndhwc(x, "x");
  check_ncdhw(residual, "residual");
  check_ncdhw(out, "out");
  const int64_t b = x.size(0);
  const int64_t t = x.size(1);
  const int64_t h = x.size(2);
  const int64_t w = x.size(3);
  const int64_t c = x.size(4);
  TORCH_CHECK(residual.sizes() == torch::IntArrayRef({b, c, t, h, w}),
              "residual must have shape (B, C, T, H, W)");
  TORCH_CHECK(out.sizes() == residual.sizes(),
              "out must match residual shape");
  check_same_device(x, residual, "x", "residual");
  check_same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const auto rs = residual.strides();
  TORCH_CHECK(
      flash_rt::quantize::bf16_ndhwc_to_ncdhw_add_bf16(
          x.data_ptr(), residual.data_ptr(), out.data_ptr(),
          static_cast<int>(b), static_cast<int>(c), static_cast<int>(t),
          static_cast<int>(h), static_cast<int>(w), rs[0], rs[1], rs[2],
          rs[3], rs[4], stream) == 0,
      "ndhwc_to_ncdhw_add_bf16 launch failed");
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

void ncdhw_quantize_fp8_static_ndhwc_bf16(
    torch::Tensor const& x, double scale, torch::Tensor& out) {
  check_ncdhw(x, "x");
  check_fp8(out, "out");
  const int64_t b = x.size(0);
  const int64_t c = x.size(1);
  const int64_t t = x.size(2);
  const int64_t h = x.size(3);
  const int64_t w = x.size(4);
  TORCH_CHECK(c % 4 == 0, "C must be divisible by 4");
  TORCH_CHECK(scale > 0.0, "scale must be positive");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({b, t, h, w, c}),
              "out must have shape (B, T, H, W, C)");
  check_same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  TORCH_CHECK(
      flash_rt::quantize::bf16_quant_fp8_ncdhw_to_ndhwc(
          x.data_ptr(), out.data_ptr(), static_cast<int>(b),
          static_cast<int>(c), static_cast<int>(t), static_cast<int>(h),
          static_cast<int>(w), static_cast<float>(scale), stream) == 0,
      "ncdhw_quantize_fp8_static_ndhwc_bf16 launch failed");
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

void upsample2x_quantize_fp8_static_nhwc_bf16(
    torch::Tensor const& x, double scale, torch::Tensor& out) {
  check_bf16(x, "x");
  check_fp8(out, "out");
  TORCH_CHECK(x.dim() == 4, "x must have shape (N, C, H, W)");
  const int64_t n = x.size(0);
  const int64_t c = x.size(1);
  const int64_t h = x.size(2);
  const int64_t w = x.size(3);
  TORCH_CHECK(n > 0 && c > 0 && h > 0 && w > 0,
              "x dimensions must be positive");
  TORCH_CHECK(c % 4 == 0, "C must be divisible by 4");
  TORCH_CHECK(scale > 0.0, "scale must be positive");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({n, 2 * h, 2 * w, c}),
              "out must have shape (N, 2H, 2W, C)");
  check_same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  TORCH_CHECK(
      flash_rt::quantize::bf16_upsample2x_quant_fp8_nchw_to_nhwc(
          x.data_ptr(), out.data_ptr(), static_cast<int>(n),
          static_cast<int>(c), static_cast<int>(h), static_cast<int>(w),
          static_cast<float>(scale), stream) == 0,
      "upsample2x_quantize_fp8_static_nhwc_bf16 launch failed");
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("ncdhw_to_blc_bf16(Tensor x, Tensor! out) -> ()");
  ops.def("patch_im2col_bf16(Tensor x, Tensor! out) -> ()");
  ops.def("time_unshuffle2_bf16(Tensor x, Tensor! out) -> ()");
  ops.def("add_bias_ncdhw_bf16(Tensor! x, Tensor bias) -> ()");
  ops.def("update_cache2_ncdhw_bf16(Tensor cur, Tensor prev, Tensor! out) -> ()");
  ops.def("channel_to_space3d_bf16(Tensor x, int out_channels, int temporal_factor, int spatial_factor, int repeats, bool first_chunk, Tensor! out) -> ()");
  ops.def("pack_causal_cache3_nhwc_bf16(Tensor previous, Tensor current, Tensor! out) -> ()");
  ops.def("avg_pool3d_channels_bf16(Tensor x, int out_channels, int factor_t, int factor_s, int group_size, Tensor! out) -> ()");
  ops.def("ndhwc_to_ncdhw_bf16(Tensor x, Tensor! out) -> ()");
  ops.def("ndhwc_to_ncdhw_bias_bf16(Tensor x, Tensor bias, Tensor! out) -> ()");
  ops.def("ndhwc_to_ncdhw_add_bf16(Tensor x, Tensor residual, Tensor! out) -> ()");
  ops.def("ncdhw_quantize_fp8_static_ndhwc_bf16(Tensor x, float scale, Tensor! out) -> ()");
  ops.def("upsample2x_quantize_fp8_static_nhwc_bf16(Tensor x, float scale, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("ncdhw_to_blc_bf16", torch::kCUDA, &ncdhw_to_blc_bf16);
  ops.impl("patch_im2col_bf16", torch::kCUDA, &patch_im2col_bf16);
  ops.impl("time_unshuffle2_bf16", torch::kCUDA, &time_unshuffle2_bf16);
  ops.impl("add_bias_ncdhw_bf16", torch::kCUDA, &add_bias_ncdhw_bf16);
  ops.impl("update_cache2_ncdhw_bf16", torch::kCUDA, &update_cache2_ncdhw_bf16);
  ops.impl("channel_to_space3d_bf16", torch::kCUDA, &channel_to_space3d_bf16);
  ops.impl("pack_causal_cache3_nhwc_bf16", torch::kCUDA, &pack_causal_cache3_nhwc_bf16);
  ops.impl("avg_pool3d_channels_bf16", torch::kCUDA, &avg_pool3d_channels_bf16);
  ops.impl("ndhwc_to_ncdhw_bf16", torch::kCUDA, &ndhwc_to_ncdhw_bf16);
  ops.impl("ndhwc_to_ncdhw_bias_bf16", torch::kCUDA, &ndhwc_to_ncdhw_bias_bf16);
  ops.impl("ndhwc_to_ncdhw_add_bf16", torch::kCUDA, &ndhwc_to_ncdhw_add_bf16);
  ops.impl("ncdhw_quantize_fp8_static_ndhwc_bf16", torch::kCUDA, &ncdhw_quantize_fp8_static_ndhwc_bf16);
  ops.impl("upsample2x_quantize_fp8_static_nhwc_bf16", torch::kCUDA, &upsample2x_quantize_fp8_static_nhwc_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
