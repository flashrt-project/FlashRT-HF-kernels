// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "registration.h"
#include "residual_gates.cuh"
#include "torch_binding.h"

namespace {

void check_bf16(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
}

void check_same_device(torch::Tensor const& a,
                       torch::Tensor const& b,
                       const char* a_name,
                       const char* b_name) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              a_name, " and ", b_name, " must be on the same CUDA device");
}

void check_matrix(torch::Tensor const& tensor, const char* name) {
  check_bf16(tensor, name);
  TORCH_CHECK(tensor.dim() == 2, name, " must have shape (rows, dim)");
  TORCH_CHECK(tensor.size(0) > 0 && tensor.size(1) > 0,
              name, " rows and dim must be positive");
  TORCH_CHECK((tensor.size(1) % 2) == 0, name, ".shape[1] must be even");
}

void check_like(torch::Tensor const& tensor,
                torch::Tensor const& reference,
                const char* name,
                const char* reference_name) {
  check_bf16(tensor, name);
  TORCH_CHECK(tensor.sizes() == reference.sizes(),
              name, " must have the same shape as ", reference_name);
}

void check_bias(torch::Tensor const& bias, int64_t dim, const char* name) {
  check_bf16(bias, name);
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == dim,
              name, " must have shape (dim,)");
}

}  // namespace

void gate_residual_bf16(
    torch::Tensor const& residual,
    torch::Tensor const& x,
    torch::Tensor const& gate,
    torch::Tensor& out) {
  check_matrix(residual, "residual");
  check_like(x, residual, "x", "residual");
  check_like(gate, residual, "gate", "residual");
  check_like(out, residual, "out", "residual");
  check_same_device(residual, x, "residual", "x");
  check_same_device(residual, gate, "residual", "gate");
  check_same_device(residual, out, "residual", "out");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(residual.device());
  auto stream = at::cuda::getCurrentCUDAStream(residual.get_device()).stream();
  flash_rt::vla_residual_gates::gate_residual_bf16(
      residual.data_ptr(),
      x.data_ptr(),
      gate.data_ptr(),
      out.data_ptr(),
      static_cast<int>(residual.numel()),
      stream);
#else
  TORCH_CHECK(false, "flashrt-vla-residual-gates was not built with CUDA support");
#endif
}

void bias_residual_bf16(
    torch::Tensor const& residual,
    torch::Tensor const& x,
    torch::Tensor const& bias,
    torch::Tensor& out) {
  check_matrix(residual, "residual");
  check_like(x, residual, "x", "residual");
  check_like(out, residual, "out", "residual");
  check_bias(bias, residual.size(1), "bias");
  check_same_device(residual, x, "residual", "x");
  check_same_device(residual, bias, "residual", "bias");
  check_same_device(residual, out, "residual", "out");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(residual.device());
  auto stream = at::cuda::getCurrentCUDAStream(residual.get_device()).stream();
  flash_rt::vla_residual_gates::bias_residual_bf16(
      residual.data_ptr(),
      x.data_ptr(),
      bias.data_ptr(),
      out.data_ptr(),
      static_cast<int>(residual.size(0)),
      static_cast<int>(residual.size(1)),
      stream);
#else
  TORCH_CHECK(false, "flashrt-vla-residual-gates was not built with CUDA support");
#endif
}

void joint3_bias_gate_residual_bf16(
    torch::Tensor const& v_residual,
    torch::Tensor const& v_x,
    torch::Tensor const& v_bias,
    torch::Tensor const& v_gate,
    torch::Tensor& v_out,
    torch::Tensor const& a_residual,
    torch::Tensor const& a_x,
    torch::Tensor const& a_bias,
    torch::Tensor const& a_gate,
    torch::Tensor& a_out,
    torch::Tensor const& u_residual,
    torch::Tensor const& u_x,
    torch::Tensor& u_out) {
  check_matrix(v_residual, "v_residual");
  check_like(v_x, v_residual, "v_x", "v_residual");
  check_like(v_gate, v_residual, "v_gate", "v_residual");
  check_like(v_out, v_residual, "v_out", "v_residual");
  check_bias(v_bias, v_residual.size(1), "v_bias");

  check_matrix(a_residual, "a_residual");
  check_like(a_x, a_residual, "a_x", "a_residual");
  check_like(a_gate, a_residual, "a_gate", "a_residual");
  check_like(a_out, a_residual, "a_out", "a_residual");
  check_bias(a_bias, a_residual.size(1), "a_bias");

  check_matrix(u_residual, "u_residual");
  check_like(u_x, u_residual, "u_x", "u_residual");
  check_like(u_out, u_residual, "u_out", "u_residual");

  check_same_device(v_residual, v_x, "v_residual", "v_x");
  check_same_device(v_residual, v_bias, "v_residual", "v_bias");
  check_same_device(v_residual, v_gate, "v_residual", "v_gate");
  check_same_device(v_residual, v_out, "v_residual", "v_out");
  check_same_device(v_residual, a_residual, "v_residual", "a_residual");
  check_same_device(v_residual, a_x, "v_residual", "a_x");
  check_same_device(v_residual, a_bias, "v_residual", "a_bias");
  check_same_device(v_residual, a_gate, "v_residual", "a_gate");
  check_same_device(v_residual, a_out, "v_residual", "a_out");
  check_same_device(v_residual, u_residual, "v_residual", "u_residual");
  check_same_device(v_residual, u_x, "v_residual", "u_x");
  check_same_device(v_residual, u_out, "v_residual", "u_out");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(v_residual.device());
  auto stream = at::cuda::getCurrentCUDAStream(v_residual.get_device()).stream();
  flash_rt::vla_residual_gates::joint3_bias_gate_residual_bf16(
      v_residual.data_ptr(),
      v_x.data_ptr(),
      v_bias.data_ptr(),
      v_gate.data_ptr(),
      v_out.data_ptr(),
      static_cast<int>(v_residual.numel()),
      static_cast<int>(v_residual.size(1)),
      a_residual.data_ptr(),
      a_x.data_ptr(),
      a_bias.data_ptr(),
      a_gate.data_ptr(),
      a_out.data_ptr(),
      static_cast<int>(a_residual.numel()),
      static_cast<int>(a_residual.size(1)),
      u_residual.data_ptr(),
      u_x.data_ptr(),
      u_out.data_ptr(),
      static_cast<int>(u_residual.numel()),
      static_cast<int>(u_residual.size(1)),
      stream);
#else
  TORCH_CHECK(false, "flashrt-vla-residual-gates was not built with CUDA support");
#endif
}

void joint3_bias_gate_residual_action_nobias_bf16(
    torch::Tensor const& v_residual,
    torch::Tensor const& v_x,
    torch::Tensor const& v_bias,
    torch::Tensor const& v_gate,
    torch::Tensor& v_out,
    torch::Tensor const& a_residual,
    torch::Tensor const& a_x,
    torch::Tensor const& a_gate,
    torch::Tensor& a_out,
    torch::Tensor const& u_residual,
    torch::Tensor const& u_x,
    torch::Tensor& u_out) {
  check_matrix(v_residual, "v_residual");
  check_like(v_x, v_residual, "v_x", "v_residual");
  check_like(v_gate, v_residual, "v_gate", "v_residual");
  check_like(v_out, v_residual, "v_out", "v_residual");
  check_bias(v_bias, v_residual.size(1), "v_bias");

  check_matrix(a_residual, "a_residual");
  check_like(a_x, a_residual, "a_x", "a_residual");
  check_like(a_gate, a_residual, "a_gate", "a_residual");
  check_like(a_out, a_residual, "a_out", "a_residual");

  check_matrix(u_residual, "u_residual");
  check_like(u_x, u_residual, "u_x", "u_residual");
  check_like(u_out, u_residual, "u_out", "u_residual");

  check_same_device(v_residual, v_x, "v_residual", "v_x");
  check_same_device(v_residual, v_bias, "v_residual", "v_bias");
  check_same_device(v_residual, v_gate, "v_residual", "v_gate");
  check_same_device(v_residual, v_out, "v_residual", "v_out");
  check_same_device(v_residual, a_residual, "v_residual", "a_residual");
  check_same_device(v_residual, a_x, "v_residual", "a_x");
  check_same_device(v_residual, a_gate, "v_residual", "a_gate");
  check_same_device(v_residual, a_out, "v_residual", "a_out");
  check_same_device(v_residual, u_residual, "v_residual", "u_residual");
  check_same_device(v_residual, u_x, "v_residual", "u_x");
  check_same_device(v_residual, u_out, "v_residual", "u_out");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(v_residual.device());
  auto stream = at::cuda::getCurrentCUDAStream(v_residual.get_device()).stream();
  flash_rt::vla_residual_gates::joint3_bias_gate_residual_action_nobias_bf16(
      v_residual.data_ptr(),
      v_x.data_ptr(),
      v_bias.data_ptr(),
      v_gate.data_ptr(),
      v_out.data_ptr(),
      static_cast<int>(v_residual.numel()),
      static_cast<int>(v_residual.size(1)),
      a_residual.data_ptr(),
      a_x.data_ptr(),
      a_gate.data_ptr(),
      a_out.data_ptr(),
      static_cast<int>(a_residual.numel()),
      static_cast<int>(a_residual.size(1)),
      u_residual.data_ptr(),
      u_x.data_ptr(),
      u_out.data_ptr(),
      static_cast<int>(u_residual.numel()),
      static_cast<int>(u_residual.size(1)),
      stream);
#else
  TORCH_CHECK(false, "flashrt-vla-residual-gates was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("gate_residual_bf16("
          "Tensor residual, Tensor x, Tensor gate, Tensor! out) -> ()");
  ops.def("bias_residual_bf16("
          "Tensor residual, Tensor x, Tensor bias, Tensor! out) -> ()");
  ops.def("joint3_bias_gate_residual_bf16("
          "Tensor v_residual, Tensor v_x, Tensor v_bias, Tensor v_gate, Tensor! v_out, "
          "Tensor a_residual, Tensor a_x, Tensor a_bias, Tensor a_gate, Tensor! a_out, "
          "Tensor u_residual, Tensor u_x, Tensor! u_out) -> ()");
  ops.def("joint3_bias_gate_residual_action_nobias_bf16("
          "Tensor v_residual, Tensor v_x, Tensor v_bias, Tensor v_gate, Tensor! v_out, "
          "Tensor a_residual, Tensor a_x, Tensor a_gate, Tensor! a_out, "
          "Tensor u_residual, Tensor u_x, Tensor! u_out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("gate_residual_bf16",
           torch::kCUDA,
           &gate_residual_bf16);
  ops.impl("bias_residual_bf16",
           torch::kCUDA,
           &bias_residual_bf16);
  ops.impl("joint3_bias_gate_residual_bf16",
           torch::kCUDA,
           &joint3_bias_gate_residual_bf16);
  ops.impl("joint3_bias_gate_residual_action_nobias_bf16",
           torch::kCUDA,
           &joint3_bias_gate_residual_action_nobias_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
