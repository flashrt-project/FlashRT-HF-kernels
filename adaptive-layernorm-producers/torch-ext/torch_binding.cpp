// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "ada_layer_norm_fp8.cuh"
#include "adaln_modulation6.cuh"
#include "dit_layer_norm_fp8.cuh"
#include "registration.h"

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

void check_fp32(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              name, " must have dtype torch.float32");
}

void check_fp8(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat8_e4m3fn,
              name, " must have dtype torch.float8_e4m3fn");
}

void check_u8(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8,
              name, " must have dtype torch.uint8");
}

void check_same_device(torch::Tensor const& a,
                       torch::Tensor const& b,
                       const char* a_name,
                       const char* b_name) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              a_name, " and ", b_name, " must be on the same CUDA device");
}

void check_x(torch::Tensor const& x) {
  check_bf16(x, "x");
  TORCH_CHECK(x.dim() == 2, "x must have shape (rows, dim)");
  TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0,
              "x rows and dim must be positive");
  TORCH_CHECK((x.size(1) % 2) == 0, "x.shape[1] must be even");
}

void check_bf16_mod(torch::Tensor const& x,
                    torch::Tensor const& scale,
                    torch::Tensor const& shift) {
  check_bf16(scale, "scale");
  check_bf16(shift, "shift");
  TORCH_CHECK(scale.dim() == 1 && scale.size(0) == x.size(1),
              "scale must have shape (dim,)");
  TORCH_CHECK(shift.dim() == 1 && shift.size(0) == x.size(1),
              "shift must have shape (dim,)");
  check_same_device(x, scale, "x", "scale");
  check_same_device(x, shift, "x", "shift");
}

void check_fp8_mod(torch::Tensor const& x,
                   torch::Tensor const& scale,
                   torch::Tensor const& shift,
                   torch::Tensor const& scale_deq,
                   torch::Tensor const& shift_deq) {
  check_fp8(scale, "scale_fp8");
  check_fp8(shift, "shift_fp8");
  check_fp32(scale_deq, "scale_deq");
  check_fp32(shift_deq, "shift_deq");
  TORCH_CHECK(scale.dim() == 1 && scale.size(0) == x.size(1),
              "scale_fp8 must have shape (dim,)");
  TORCH_CHECK(shift.dim() == 1 && shift.size(0) == x.size(1),
              "shift_fp8 must have shape (dim,)");
  TORCH_CHECK(scale_deq.numel() == 1, "scale_deq must be a scalar tensor");
  TORCH_CHECK(shift_deq.numel() == 1, "shift_deq must be a scalar tensor");
  check_same_device(x, scale, "x", "scale_fp8");
  check_same_device(x, shift, "x", "shift_fp8");
  check_same_device(x, scale_deq, "x", "scale_deq");
  check_same_device(x, shift_deq, "x", "shift_deq");
}

void check_act_scale(torch::Tensor const& x, torch::Tensor const& act_scale) {
  check_fp32(act_scale, "act_scale");
  TORCH_CHECK(act_scale.numel() == 1, "act_scale must be a scalar tensor");
  check_same_device(x, act_scale, "x", "act_scale");
}

int64_t swizzled_sf_bytes(int64_t rows, int64_t dim) {
  TORCH_CHECK((dim % 16) == 0, "dim must be divisible by 16 for NVFP4 swizzled output");
  const int64_t blocks = dim / 16;
  const int64_t row_super = (rows + 127) / 128;
  const int64_t col_super = (blocks + 3) / 4;
  return row_super * col_super * 128 * 64;
}

void check_fp8_out(torch::Tensor const& x, torch::Tensor const& out) {
  check_fp8(out, "out");
  TORCH_CHECK(out.sizes() == x.sizes(), "out must have the same shape as x");
  check_same_device(x, out, "x", "out");
}

void check_nvfp4_out(torch::Tensor const& x,
                     torch::Tensor const& packed,
                     torch::Tensor const& sf_swizzled) {
  check_u8(packed, "packed");
  check_u8(sf_swizzled, "sf_swizzled");
  TORCH_CHECK((x.size(1) % 16) == 0, "x.shape[1] must be divisible by 16");
  TORCH_CHECK(packed.dim() == 2 && packed.size(0) == x.size(0) &&
                  packed.size(1) == x.size(1) / 2,
              "packed must have shape (rows, dim // 2)");
  TORCH_CHECK(sf_swizzled.numel() >= swizzled_sf_bytes(x.size(0), x.size(1)),
              "sf_swizzled is too small for the swizzled NVFP4 scale layout");
  check_same_device(x, packed, "x", "packed");
  check_same_device(x, sf_swizzled, "x", "sf_swizzled");
}

}  // namespace

void ada_layer_norm_quant_fp8_bf16(
    torch::Tensor const& x,
    torch::Tensor const& scale,
    torch::Tensor const& shift,
    torch::Tensor const& act_scale,
    double eps,
    torch::Tensor& out) {
  check_x(x);
  check_bf16_mod(x, scale, shift);
  check_act_scale(x, act_scale);
  check_fp8_out(x, out);

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::quantize::ada_layer_norm_fp8(
      x.data_ptr(),
      scale.data_ptr(),
      shift.data_ptr(),
      out.data_ptr(),
      static_cast<const float*>(act_scale.data_ptr()),
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1)),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "adaptive-layernorm-producers was not built with CUDA support");
#endif
}

void ada_layer_norm_quant_fp8_modfp8_bf16(
    torch::Tensor const& x,
    torch::Tensor const& scale_fp8,
    torch::Tensor const& shift_fp8,
    torch::Tensor const& scale_deq,
    torch::Tensor const& shift_deq,
    torch::Tensor const& act_scale,
    double eps,
    torch::Tensor& out) {
  check_x(x);
  check_fp8_mod(x, scale_fp8, shift_fp8, scale_deq, shift_deq);
  check_act_scale(x, act_scale);
  check_fp8_out(x, out);

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::quantize::ada_layer_norm_fp8_modfp8(
      x.data_ptr(),
      scale_fp8.data_ptr(),
      shift_fp8.data_ptr(),
      static_cast<const float*>(scale_deq.data_ptr()),
      static_cast<const float*>(shift_deq.data_ptr()),
      out.data_ptr(),
      static_cast<const float*>(act_scale.data_ptr()),
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1)),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "adaptive-layernorm-producers was not built with CUDA support");
#endif
}

void awq_ada_layer_norm_quant_fp8_bf16(
    torch::Tensor const& x,
    torch::Tensor const& scale,
    torch::Tensor const& shift,
    torch::Tensor const& inv_s,
    torch::Tensor const& act_scale,
    double eps,
    torch::Tensor& out) {
  check_x(x);
  check_bf16_mod(x, scale, shift);
  check_bf16(inv_s, "inv_s");
  TORCH_CHECK(inv_s.dim() == 1 && inv_s.size(0) == x.size(1),
              "inv_s must have shape (dim,)");
  check_same_device(x, inv_s, "x", "inv_s");
  check_act_scale(x, act_scale);
  check_fp8_out(x, out);

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::quantize::awq_ada_layer_norm_fp8(
      x.data_ptr(),
      scale.data_ptr(),
      shift.data_ptr(),
      inv_s.data_ptr(),
      out.data_ptr(),
      static_cast<const float*>(act_scale.data_ptr()),
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1)),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "adaptive-layernorm-producers was not built with CUDA support");
#endif
}

void ada_layer_norm_quant_nvfp4_swizzled_bf16(
    torch::Tensor const& x,
    torch::Tensor const& scale,
    torch::Tensor const& shift,
    double eps,
    torch::Tensor& packed,
    torch::Tensor& sf_swizzled) {
  check_x(x);
  check_bf16_mod(x, scale, shift);
  check_nvfp4_out(x, packed, sf_swizzled);

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::quantize::ada_layer_norm_nvfp4_swizzled(
      x.data_ptr(),
      scale.data_ptr(),
      shift.data_ptr(),
      packed.data_ptr(),
      sf_swizzled.data_ptr(),
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1)),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "adaptive-layernorm-producers was not built with CUDA support");
#endif
}

void ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16(
    torch::Tensor const& x,
    torch::Tensor const& scale_fp8,
    torch::Tensor const& shift_fp8,
    torch::Tensor const& scale_deq,
    torch::Tensor const& shift_deq,
    double eps,
    torch::Tensor& packed,
    torch::Tensor& sf_swizzled) {
  check_x(x);
  check_fp8_mod(x, scale_fp8, shift_fp8, scale_deq, shift_deq);
  check_nvfp4_out(x, packed, sf_swizzled);

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::quantize::ada_layer_norm_nvfp4_swizzled_modfp8(
      x.data_ptr(),
      scale_fp8.data_ptr(),
      shift_fp8.data_ptr(),
      static_cast<const float*>(scale_deq.data_ptr()),
      static_cast<const float*>(shift_deq.data_ptr()),
      packed.data_ptr(),
      sf_swizzled.data_ptr(),
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1)),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "adaptive-layernorm-producers was not built with CUDA support");
#endif
}

void layer_norm_no_affine_quant_fp8_static_bf16(
    torch::Tensor const& x,
    torch::Tensor const& act_scale,
    double eps,
    torch::Tensor& out) {
  check_x(x);
  check_act_scale(x, act_scale);
  check_fp8_out(x, out);

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::adaln_producers::layer_norm_no_affine_fp8_static_bf16(
      x.data_ptr(),
      out.data_ptr(),
      static_cast<const float*>(act_scale.data_ptr()),
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1)),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "adaptive-layernorm-producers was not built with CUDA support");
#endif
}

void adaln_modulation6_bf16(
    torch::Tensor const& adaln_params,
    torch::Tensor const& layer_modulation,
    torch::Tensor& out0,
    torch::Tensor& out1,
    torch::Tensor& out2,
    torch::Tensor& out3,
    torch::Tensor& out4,
    torch::Tensor& out5) {
  check_fp32(adaln_params, "adaln_params");
  check_fp32(layer_modulation, "layer_modulation");
  TORCH_CHECK(
      adaln_params.dim() == 4 && adaln_params.size(2) == 6,
      "adaln_params must have shape (batch, sequence, 6, dim)");
  const auto batch = adaln_params.size(0);
  const auto sequence = adaln_params.size(1);
  const auto dim = adaln_params.size(3);
  TORCH_CHECK(batch > 0 && sequence > 0 && dim > 0,
              "batch, sequence and dim must be positive");
  TORCH_CHECK(layer_modulation.dim() == 2 &&
                  layer_modulation.size(0) == 6 &&
                  layer_modulation.size(1) == dim,
              "layer_modulation must have shape (6, dim)");
  check_same_device(
      adaln_params, layer_modulation, "adaln_params", "layer_modulation");
  torch::Tensor* outputs[] = {&out0, &out1, &out2, &out3, &out4, &out5};
  for (int index = 0; index < 6; ++index) {
    const std::string name = "out" + std::to_string(index);
    check_bf16(*outputs[index], name.c_str());
    TORCH_CHECK(
        outputs[index]->dim() == 3 &&
            outputs[index]->size(0) == batch &&
            outputs[index]->size(1) == sequence &&
            outputs[index]->size(2) == dim,
        name, " must have shape (batch, sequence, dim)");
    check_same_device(
        adaln_params, *outputs[index], "adaln_params", name.c_str());
  }

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(adaln_params.device());
  auto stream =
      at::cuda::getCurrentCUDAStream(adaln_params.get_device()).stream();
  flash_rt::adaln_producers::adaln_modulation6_bf16(
      adaln_params.data_ptr<float>(),
      layer_modulation.data_ptr<float>(),
      out0.data_ptr(),
      out1.data_ptr(),
      out2.data_ptr(),
      out3.data_ptr(),
      out4.data_ptr(),
      out5.data_ptr(),
      static_cast<int>(batch),
      static_cast<int>(sequence),
      static_cast<int>(dim),
      stream);
#else
  TORCH_CHECK(false, "adaptive-layernorm-producers was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("ada_layer_norm_quant_fp8_bf16("
          "Tensor x, Tensor scale, Tensor shift, Tensor act_scale, float eps, Tensor! out) -> ()");
  ops.def("ada_layer_norm_quant_fp8_modfp8_bf16("
          "Tensor x, Tensor scale_fp8, Tensor shift_fp8, Tensor scale_deq, Tensor shift_deq, "
          "Tensor act_scale, float eps, Tensor! out) -> ()");
  ops.def("awq_ada_layer_norm_quant_fp8_bf16("
          "Tensor x, Tensor scale, Tensor shift, Tensor inv_s, Tensor act_scale, float eps, Tensor! out) -> ()");
  ops.def("ada_layer_norm_quant_nvfp4_swizzled_bf16("
          "Tensor x, Tensor scale, Tensor shift, float eps, Tensor! packed, Tensor! sf_swizzled) -> ()");
  ops.def("ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16("
          "Tensor x, Tensor scale_fp8, Tensor shift_fp8, Tensor scale_deq, Tensor shift_deq, "
          "float eps, Tensor! packed, Tensor! sf_swizzled) -> ()");
  ops.def("layer_norm_no_affine_quant_fp8_static_bf16("
          "Tensor x, Tensor act_scale, float eps, Tensor! out) -> ()");
  ops.def("adaln_modulation6_bf16("
          "Tensor adaln_params, Tensor layer_modulation, "
          "Tensor! out0, Tensor! out1, Tensor! out2, "
          "Tensor! out3, Tensor! out4, Tensor! out5) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("ada_layer_norm_quant_fp8_bf16",
           torch::kCUDA,
           &ada_layer_norm_quant_fp8_bf16);
  ops.impl("ada_layer_norm_quant_fp8_modfp8_bf16",
           torch::kCUDA,
           &ada_layer_norm_quant_fp8_modfp8_bf16);
  ops.impl("awq_ada_layer_norm_quant_fp8_bf16",
           torch::kCUDA,
           &awq_ada_layer_norm_quant_fp8_bf16);
  ops.impl("ada_layer_norm_quant_nvfp4_swizzled_bf16",
           torch::kCUDA,
           &ada_layer_norm_quant_nvfp4_swizzled_bf16);
  ops.impl("ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16",
           torch::kCUDA,
           &ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16);
  ops.impl("layer_norm_no_affine_quant_fp8_static_bf16",
           torch::kCUDA,
           &layer_norm_no_affine_quant_fp8_static_bf16);
  ops.impl("adaln_modulation6_bf16",
           torch::kCUDA,
           &adaln_modulation6_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
