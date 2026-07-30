// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <array>

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

void pack_tail_bf16(
    torch::Tensor const& tail,
    int64_t flat_dim,
    torch::Tensor& out) {
  check_bf16(tail, "tail");
  check_bf16(out, "out");
  TORCH_CHECK(tail.dim() == 1, "tail must be one-dimensional");
  TORCH_CHECK(out.dim() == 1 && out.numel() == flat_dim,
              "out must be one-dimensional with flat_dim elements");
  TORCH_CHECK(flat_dim >= tail.numel() && tail.numel() > 0,
              "flat_dim must be at least tail.numel(), and tail must be non-empty");
  check_same_device(tail, out, "tail", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(tail.device());
  auto stream = at::cuda::getCurrentCUDAStream(tail.get_device()).stream();
  flash_rt::diffusion_step_ops::pack_tail_bf16(
      tail.data_ptr(), out.data_ptr(), flat_dim, tail.numel(), stream);
#else
  TORCH_CHECK(false, "diffusion-step-ops was not built with CUDA support");
#endif
}

void add_bias_zero_tail_bf16(
    torch::Tensor const& input,
    torch::Tensor const& bias,
    int64_t valid_cols,
    torch::Tensor& out) {
  check_bf16(input, "input");
  check_bf16(bias, "bias");
  check_bf16(out, "out");
  TORCH_CHECK(input.dim() == 2, "input must have shape (rows, cols)");
  TORCH_CHECK(bias.sizes() == torch::IntArrayRef({input.size(1)}),
              "bias must have shape (cols,)");
  TORCH_CHECK(out.sizes() == input.sizes(), "out must match input shape");
  TORCH_CHECK(valid_cols >= 0 && valid_cols <= input.size(1),
              "valid_cols must be in [0, cols]");
  check_same_device(input, bias, "input", "bias");
  check_same_device(input, out, "input", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  flash_rt::diffusion_step_ops::add_bias_zero_tail_bf16(
      input.data_ptr(), bias.data_ptr(), out.data_ptr(),
      input.size(0), input.size(1), valid_cols, stream);
#else
  TORCH_CHECK(false, "diffusion-step-ops was not built with CUDA support");
#endif
}

void extract_tail_f32_to_bf16(
    torch::Tensor const& flat,
    int64_t tail_numel,
    torch::Tensor& out) {
  check_fp32(flat, "flat");
  check_bf16(out, "out");
  TORCH_CHECK(flat.dim() == 1, "flat must be one-dimensional");
  TORCH_CHECK(tail_numel > 0 && tail_numel <= flat.numel(),
              "tail_numel must be in [1, flat.numel()]");
  TORCH_CHECK(out.dim() == 1 && out.numel() == tail_numel,
              "out must be one-dimensional with tail_numel elements");
  check_same_device(flat, out, "flat", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(flat.device());
  auto stream = at::cuda::getCurrentCUDAStream(flat.get_device()).stream();
  flash_rt::diffusion_step_ops::extract_tail_f32_to_bf16(
      flat.data_ptr(), out.data_ptr(), flat.numel(), tail_numel, stream);
#else
  TORCH_CHECK(false, "diffusion-step-ops was not built with CUDA support");
#endif
}

void add_bias_pair_bf16(
    torch::Tensor const& input,
    torch::Tensor const& bias_a,
    torch::Tensor const& bias_b,
    torch::Tensor& out) {
  check_bf16(input, "input");
  check_bf16(bias_a, "bias_a");
  check_bf16(bias_b, "bias_b");
  check_bf16(out, "out");
  TORCH_CHECK(input.dim() == 2, "input must have shape (rows, hidden)");
  const auto hidden = input.size(1);
  TORCH_CHECK(bias_a.sizes() == torch::IntArrayRef({hidden}),
              "bias_a must have shape (hidden,)");
  TORCH_CHECK(bias_b.sizes() == torch::IntArrayRef({hidden}),
              "bias_b must have shape (hidden,)");
  TORCH_CHECK(out.sizes() == input.sizes(), "out must match input shape");
  check_same_device(input, bias_a, "input", "bias_a");
  check_same_device(input, bias_b, "input", "bias_b");
  check_same_device(input, out, "input", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  flash_rt::diffusion_step_ops::add_bias_pair_bf16(
      input.data_ptr(), bias_a.data_ptr(), bias_b.data_ptr(), out.data_ptr(),
      input.size(0), hidden, stream);
#else
  TORCH_CHECK(false, "diffusion-step-ops was not built with CUDA support");
#endif
}

void unipc_step_f32_bf16(
    torch::Tensor const& sample,
    torch::Tensor const& velocity,
    torch::Tensor const& prev_m1,
    torch::Tensor const& prev_m2,
    torch::Tensor const& prev_last_sample,
    double sigma,
    int64_t corrector_order,
    int64_t predictor_order,
    double c_sample,
    double c_last,
    double c_prev_m1,
    double c_prev_m2,
    double c_curr_m,
    double p_sample,
    double p_curr_m,
    double p_prev_m1,
    torch::Tensor& next_sample,
    torch::Tensor& current_m,
    torch::Tensor& current_last_sample) {
  check_fp32(sample, "sample");
  check_bf16(velocity, "velocity");
  check_fp32(prev_m1, "prev_m1");
  check_fp32(prev_m2, "prev_m2");
  check_fp32(prev_last_sample, "prev_last_sample");
  check_fp32(next_sample, "next_sample");
  check_fp32(current_m, "current_m");
  check_fp32(current_last_sample, "current_last_sample");
  const std::array<torch::Tensor const*, 7> tensors = {
      &velocity,
      &prev_m1,
      &prev_m2,
      &prev_last_sample,
      &next_sample,
      &current_m,
      &current_last_sample,
  };
  for (auto const* tensor : tensors) {
    TORCH_CHECK(tensor->sizes() == sample.sizes(),
                "all UniPC tensors must have the same shape");
    TORCH_CHECK(tensor->get_device() == sample.get_device(),
                "all UniPC tensors must be on the same CUDA device");
  }
  TORCH_CHECK(corrector_order >= 0 && corrector_order <= 2,
              "corrector_order must be 0, 1, or 2");
  TORCH_CHECK(predictor_order >= 1 && predictor_order <= 2,
              "predictor_order must be 1 or 2");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(sample.device());
  auto stream = at::cuda::getCurrentCUDAStream(sample.get_device()).stream();
  flash_rt::diffusion_step_ops::unipc_step_f32_bf16(
      sample.data_ptr(),
      velocity.data_ptr(),
      prev_m1.data_ptr(),
      prev_m2.data_ptr(),
      prev_last_sample.data_ptr(),
      next_sample.data_ptr(),
      current_m.data_ptr(),
      current_last_sample.data_ptr(),
      sample.numel(),
      static_cast<float>(sigma),
      static_cast<int>(corrector_order),
      static_cast<int>(predictor_order),
      static_cast<float>(c_sample),
      static_cast<float>(c_last),
      static_cast<float>(c_prev_m1),
      static_cast<float>(c_prev_m2),
      static_cast<float>(c_curr_m),
      static_cast<float>(p_sample),
      static_cast<float>(p_curr_m),
      static_cast<float>(p_prev_m1),
      stream);
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
  ops.def("pack_tail_bf16(Tensor tail, int flat_dim, Tensor! out) -> ()");
  ops.def("add_bias_zero_tail_bf16(Tensor input, Tensor bias, int valid_cols, Tensor! out) -> ()");
  ops.def("extract_tail_f32_to_bf16(Tensor flat, int tail_numel, Tensor! out) -> ()");
  ops.def("add_bias_pair_bf16(Tensor input, Tensor bias_a, Tensor bias_b, Tensor! out) -> ()");
  ops.def("unipc_step_f32_bf16(Tensor sample, Tensor velocity, Tensor prev_m1, Tensor prev_m2, Tensor prev_last_sample, float sigma, int corrector_order, int predictor_order, float c_sample, float c_last, float c_prev_m1, float c_prev_m2, float c_curr_m, float p_sample, float p_curr_m, float p_prev_m1, Tensor! next_sample, Tensor! current_m, Tensor! current_last_sample) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("add_bf16_out", torch::kCUDA, &add_bf16_out);
  ops.impl("euler_step_bf16_out", torch::kCUDA, &euler_step_bf16_out);
  ops.impl("cfg_combine_into_residual_bf16", torch::kCUDA, &cfg_combine_into_residual_bf16);
  ops.impl("cfg_combine_into_residual_fp16", torch::kCUDA, &cfg_combine_into_residual_fp16);
  ops.impl("teacher_force_first_frame_bf16", torch::kCUDA, &teacher_force_first_frame_bf16);
  ops.impl("motus_decode_postprocess_bf16_to_fp32", torch::kCUDA, &motus_decode_postprocess_bf16_to_fp32);
  ops.impl("cast_bf16_to_fp32", torch::kCUDA, &cast_bf16_to_fp32);
  ops.impl("pack_tail_bf16", torch::kCUDA, &pack_tail_bf16);
  ops.impl("add_bias_zero_tail_bf16", torch::kCUDA, &add_bias_zero_tail_bf16);
  ops.impl("extract_tail_f32_to_bf16", torch::kCUDA, &extract_tail_f32_to_bf16);
  ops.impl("add_bias_pair_bf16", torch::kCUDA, &add_bias_pair_bf16);
  ops.impl("unipc_step_f32_bf16", torch::kCUDA, &unipc_step_f32_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
