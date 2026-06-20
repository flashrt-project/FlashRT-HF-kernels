// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "diffusion_step_ops.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
}

void check_fp16(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype torch.float16");
}

void check_fp32(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              name, " must have dtype torch.float32");
}

void check_same_shape(torch::Tensor const& a,
                      torch::Tensor const& b,
                      const char* a_name,
                      const char* b_name) {
  TORCH_CHECK(a.sizes() == b.sizes(),
              a_name, " and ", b_name, " must have the same shape");
}

void check_same_device(torch::Tensor const& a,
                       torch::Tensor const& b,
                       const char* a_name,
                       const char* b_name) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              a_name, " and ", b_name, " must be on the same CUDA device");
}

void check_pair(torch::Tensor const& a,
                torch::Tensor const& b,
                torch::Tensor const& out,
                const char* a_name,
                const char* b_name,
                const char* out_name) {
  check_bf16(a, a_name);
  check_bf16(b, b_name);
  check_bf16(out, out_name);
  TORCH_CHECK(a.numel() > 0, a_name, " must be non-empty");
  check_same_shape(a, b, a_name, b_name);
  check_same_shape(a, out, a_name, out_name);
  check_same_device(a, b, a_name, b_name);
  check_same_device(a, out, a_name, out_name);
}

}  // namespace

void add_bf16_out(torch::Tensor const& a, torch::Tensor const& b, torch::Tensor& out) {
  check_pair(a, b, out, "a", "b", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
  flash_rt::diffusion_step_ops::add_bf16_out(
      a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel(), stream);
#else
  TORCH_CHECK(false, "diffusion-step-ops was not built with CUDA support");
#endif
}

void euler_step_bf16_out(
    torch::Tensor const& latent,
    torch::Tensor const& velocity,
    double dt,
    torch::Tensor& out) {
  check_pair(latent, velocity, out, "latent", "velocity", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(latent.device());
  auto stream = at::cuda::getCurrentCUDAStream(latent.get_device()).stream();
  flash_rt::diffusion_step_ops::euler_step_bf16_out(
      latent.data_ptr(), velocity.data_ptr(), out.data_ptr(),
      static_cast<float>(dt), latent.numel(), stream);
#else
  TORCH_CHECK(false, "diffusion-step-ops was not built with CUDA support");
#endif
}

void cfg_combine_into_residual_bf16(
    torch::Tensor& residual,
    torch::Tensor const& v_cond,
    torch::Tensor const& v_uncond,
    double beta) {
  check_pair(v_cond, v_uncond, residual, "v_cond", "v_uncond", "residual");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(residual.device());
  auto stream = at::cuda::getCurrentCUDAStream(residual.get_device()).stream();
  flash_rt::diffusion_step_ops::cfg_combine_into_residual_bf16(
      residual.data_ptr(), v_cond.data_ptr(), v_uncond.data_ptr(),
      static_cast<float>(beta), residual.numel(), stream);
#else
  TORCH_CHECK(false, "diffusion-step-ops was not built with CUDA support");
#endif
}

void cfg_combine_into_residual_fp16(
    torch::Tensor& residual,
    torch::Tensor const& v_cond,
    torch::Tensor const& v_uncond,
    double beta) {
  check_fp16(residual, "residual");
  check_fp16(v_cond, "v_cond");
  check_fp16(v_uncond, "v_uncond");
  TORCH_CHECK(residual.numel() > 0, "residual must be non-empty");
  check_same_shape(residual, v_cond, "residual", "v_cond");
  check_same_shape(residual, v_uncond, "residual", "v_uncond");
  check_same_device(residual, v_cond, "residual", "v_cond");
  check_same_device(residual, v_uncond, "residual", "v_uncond");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(residual.device());
  auto stream = at::cuda::getCurrentCUDAStream(residual.get_device()).stream();
  flash_rt::diffusion_step_ops::cfg_combine_into_residual_fp16(
      residual.data_ptr(), v_cond.data_ptr(), v_uncond.data_ptr(),
      static_cast<float>(beta), residual.numel(), stream);
#else
  TORCH_CHECK(false, "diffusion-step-ops was not built with CUDA support");
#endif
}

void teacher_force_first_frame_bf16(torch::Tensor& video_latent, torch::Tensor const& cond_latent) {
  check_bf16(video_latent, "video_latent");
  check_bf16(cond_latent, "cond_latent");
  TORCH_CHECK(video_latent.dim() == 5,
              "video_latent must have shape (B, C, T, H, W)");
  TORCH_CHECK(cond_latent.dim() == 4,
              "cond_latent must have shape (B, C, H, W)");
  const auto b = video_latent.size(0);
  const auto c = video_latent.size(1);
  const auto t = video_latent.size(2);
  const auto h = video_latent.size(3);
  const auto w = video_latent.size(4);
  TORCH_CHECK(t > 0, "video_latent T must be positive");
  TORCH_CHECK(cond_latent.sizes() == torch::IntArrayRef({b, c, h, w}),
              "cond_latent must have shape (B, C, H, W) matching video_latent");
  check_same_device(video_latent, cond_latent, "video_latent", "cond_latent");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(video_latent.device());
  auto stream = at::cuda::getCurrentCUDAStream(video_latent.get_device()).stream();
  flash_rt::diffusion_step_ops::teacher_force_first_frame_bf16(
      video_latent.data_ptr(), cond_latent.data_ptr(),
      static_cast<int>(b), static_cast<int>(c), static_cast<int>(t),
      static_cast<int>(h), static_cast<int>(w), stream);
#else
  TORCH_CHECK(false, "diffusion-step-ops was not built with CUDA support");
#endif
}

void motus_decode_postprocess_bf16_to_fp32(torch::Tensor const& decoded, torch::Tensor& out) {
  check_bf16(decoded, "decoded");
  check_fp32(out, "out");
  TORCH_CHECK(decoded.dim() == 5, "decoded must have shape (B, C, T_in, H, W)");
  const auto b = decoded.size(0);
  const auto c = decoded.size(1);
  const auto t = decoded.size(2);
  const auto h = decoded.size(3);
  const auto w = decoded.size(4);
  TORCH_CHECK(t >= 2, "decoded T_in must be >= 2");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({b, c, t - 1, h, w}),
              "out must have shape (B, C, T_in - 1, H, W)");
  check_same_device(decoded, out, "decoded", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(decoded.device());
  auto stream = at::cuda::getCurrentCUDAStream(decoded.get_device()).stream();
  flash_rt::diffusion_step_ops::motus_decode_postprocess_bf16_to_fp32(
      decoded.data_ptr(), out.data_ptr(),
      static_cast<int>(b), static_cast<int>(c), static_cast<int>(t),
      static_cast<int>(h), static_cast<int>(w), stream);
#else
  TORCH_CHECK(false, "diffusion-step-ops was not built with CUDA support");
#endif
}

void cast_bf16_to_fp32(torch::Tensor const& src, torch::Tensor& dst) {
  check_bf16(src, "src");
  check_fp32(dst, "dst");
  check_same_shape(src, dst, "src", "dst");
  check_same_device(src, dst, "src", "dst");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(src.device());
  auto stream = at::cuda::getCurrentCUDAStream(src.get_device()).stream();
  flash_rt::diffusion_step_ops::cast_bf16_to_fp32(src.data_ptr(), dst.data_ptr(), src.numel(), stream);
#else
  TORCH_CHECK(false, "diffusion-step-ops was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("add_bf16_out(Tensor a, Tensor b, Tensor! out) -> ()");
  ops.def("euler_step_bf16_out(Tensor latent, Tensor velocity, float dt, Tensor! out) -> ()");
  ops.def("cfg_combine_into_residual_bf16(Tensor! residual, Tensor v_cond, Tensor v_uncond, float beta) -> ()");
  ops.def("cfg_combine_into_residual_fp16(Tensor! residual, Tensor v_cond, Tensor v_uncond, float beta) -> ()");
  ops.def("teacher_force_first_frame_bf16(Tensor! video_latent, Tensor cond_latent) -> ()");
  ops.def("motus_decode_postprocess_bf16_to_fp32(Tensor decoded, Tensor! out) -> ()");
  ops.def("cast_bf16_to_fp32(Tensor src, Tensor! dst) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("add_bf16_out", torch::kCUDA, &add_bf16_out);
  ops.impl("euler_step_bf16_out", torch::kCUDA, &euler_step_bf16_out);
  ops.impl("cfg_combine_into_residual_bf16", torch::kCUDA, &cfg_combine_into_residual_bf16);
  ops.impl("cfg_combine_into_residual_fp16", torch::kCUDA, &cfg_combine_into_residual_fp16);
  ops.impl("teacher_force_first_frame_bf16", torch::kCUDA, &teacher_force_first_frame_bf16);
  ops.impl("motus_decode_postprocess_bf16_to_fp32", torch::kCUDA, &motus_decode_postprocess_bf16_to_fp32);
  ops.impl("cast_bf16_to_fp32", torch::kCUDA, &cast_bf16_to_fp32);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
