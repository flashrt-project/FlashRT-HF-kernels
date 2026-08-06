// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#ifndef FLASHRT_HAVE_COSMOS3_EDGE
#define FLASHRT_HAVE_COSMOS3_EDGE 1
#endif

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "fused_fp4/norm_silu_fp4_sfa.cuh"
#include "fused_fp4/layer_norm_fp4_sfa.cuh"
#include "fused_fp4/siglip_ln_vec.cuh"
#include "fused_fp4/dequantize_fp4_sfa.cuh"
#include "fused_fp4/adarms_fp8_static_fp16.cuh"
#include "fused_fp4/pi05_e0m3_act.cuh"
#include "fused_fp4/cosmos3_edge_fp4.cuh"
#include "fused_fp4/silu_mul_two_fp4_to_fp4.cuh"
#include "quantize/quantize_bf16_to_nvfp4_linear.cuh"
#include "quantize/bf16_rms_silu_ncdhw.cuh"
#include "quantize/reshape_scales_sfa.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace flash_rt::quantize {
extern "C" int motus_bf16_rms_silu_quant_nvfp4_to_ndhwc_v1(
    const void* x_bf16, const void* gamma_bf16,
    const void* awq_inv_scale_fp32, void* y_fp4, void* y_sf,
    int B, int C, int T, int H, int W, float eps, cudaStream_t stream);
}

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_fp16_matrix(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype torch.float16");
  TORCH_CHECK(tensor.dim() == 2, name, " must have shape (rows, cols)");
  TORCH_CHECK(tensor.size(0) > 0 && tensor.size(1) > 0,
              name, " dimensions must be positive");
  TORCH_CHECK(tensor.size(1) % 16 == 0, name, ".shape[1] must be divisible by 16");
}

void check_uint8(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8,
              name, " must have dtype torch.uint8");
}

void check_same_device(torch::Tensor const& a, torch::Tensor const& b,
                       const char* a_name, const char* b_name) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              a_name, " and ", b_name, " must be on the same CUDA device");
}

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in positive int");
  return static_cast<int>(value);
}

void check_packed_sfa(
    torch::Tensor const& packed,
    torch::Tensor const& sfa,
    torch::Tensor const& input,
    int64_t rows,
    int64_t dim) {
  check_uint8(packed, "packed");
  check_uint8(sfa, "sfa");
  TORCH_CHECK(packed.sizes() == torch::IntArrayRef({rows, dim / 2}),
              "packed must have shape (rows, dim / 2)");
  TORCH_CHECK(sfa.dim() == 1, "sfa must be a one-dimensional byte buffer");
  const int64_t required = sfa_size_bytes(rows, dim, false);
  TORCH_CHECK(required > 0, "sfa_size_bytes returned invalid size");
  TORCH_CHECK(sfa.numel() >= required,
              "sfa is too small for CUTLASS SFA layout");
  check_same_device(input, packed, "input", "packed");
  check_same_device(input, sfa, "input", "sfa");
}

struct TwoFp4Shape {
  int64_t rows;
  int64_t hidden;
};

TwoFp4Shape check_two_fp4_inputs(
    torch::Tensor const& gate_packed,
    torch::Tensor const& gate_sfa,
    torch::Tensor const& up_packed,
    torch::Tensor const& up_sfa,
    torch::Tensor const& out_packed,
    torch::Tensor const& out_sfa) {
  check_uint8(gate_packed, "gate_packed");
  check_uint8(gate_sfa, "gate_sfa");
  check_uint8(up_packed, "up_packed");
  check_uint8(up_sfa, "up_sfa");
  check_uint8(out_packed, "out_packed");
  check_uint8(out_sfa, "out_sfa");
  TORCH_CHECK(gate_packed.dim() == 2, "gate_packed must have shape (rows, hidden / 2)");
  TORCH_CHECK(gate_packed.size(0) > 0 && gate_packed.size(1) > 0,
              "gate_packed dimensions must be positive");
  TORCH_CHECK(up_packed.sizes() == gate_packed.sizes() &&
              out_packed.sizes() == gate_packed.sizes(),
              "up_packed and out_packed must match gate_packed shape");
  const int64_t rows = gate_packed.size(0);
  const int64_t hidden = gate_packed.size(1) * 2;
  const int64_t required = sfa_size_bytes(rows, hidden, false);
  TORCH_CHECK(gate_sfa.numel() >= required && up_sfa.numel() >= required &&
              out_sfa.numel() >= required,
              "SFA buffers are too small");
  check_same_device(gate_packed, gate_sfa, "gate_packed", "gate_sfa");
  check_same_device(gate_packed, up_packed, "gate_packed", "up_packed");
  check_same_device(gate_packed, up_sfa, "gate_packed", "up_sfa");
  check_same_device(gate_packed, out_packed, "gate_packed", "out_packed");
  check_same_device(gate_packed, out_sfa, "gate_packed", "out_sfa");
  return {rows, hidden};
}

}  // namespace

int64_t sfa_size_bytes(int64_t rows, int64_t dim, bool is_sfb) {
  TORCH_CHECK(rows > 0 && dim > 0, "rows and dim must be positive");
  TORCH_CHECK(dim % 16 == 0, "dim must be divisible by 16");
#if defined(CUDA_KERNEL)
  return flash_rt::fp4::sfa_size_bytes(
      checked_int(rows, "rows"), checked_int(dim, "dim"), is_sfb);
#else
  TORCH_CHECK(false, "fp4-fused-ops was not built with CUDA support");
#endif
}

int64_t sfa_size_bytes_for(
    torch::Tensor const& anchor,
    int64_t rows,
    int64_t dim,
    bool is_sfb) {
  check_cuda_contiguous(anchor, "anchor");
  return sfa_size_bytes(rows, dim, is_sfb);
}

void rms_norm_fp4_sfa_fp16(
    torch::Tensor const& x,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(x, "x");
  TORCH_CHECK(x.size(1) <= 2048,
              "rms_norm_fp4_sfa_fp16 supports dim <= 2048; use a v2 producer for larger dim");
  const int64_t rows = x.size(0);
  const int64_t dim = x.size(1);
  check_packed_sfa(packed, sfa, x, rows, dim);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::rms_norm_fp4_sfa_fp16(
      reinterpret_cast<const __half*>(x.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(dim, "dim"),
      stream);
#endif
}

void residual_add_rms_norm_fp4_sfa_fp16(
    torch::Tensor& residual,
    torch::Tensor const& x,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(residual, "residual");
  check_fp16_matrix(x, "x");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual and x must have the same shape");
  TORCH_CHECK(x.size(1) <= 2048,
              "residual_add_rms_norm_fp4_sfa_fp16 supports dim <= 2048; use residual_add_rms_norm_fp4_sfa_v2_fp16 for larger dim");
  check_same_device(residual, x, "residual", "x");
  const int64_t rows = x.size(0);
  const int64_t dim = x.size(1);
  check_packed_sfa(packed, sfa, x, rows, dim);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::residual_add_rms_norm_fp4_sfa_fp16(
      reinterpret_cast<__half*>(residual.data_ptr()),
      reinterpret_cast<const __half*>(x.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(dim, "dim"),
      stream);
#endif
}

void residual_add_rms_norm_fp4_sfa_v2_fp16(
    torch::Tensor& residual,
    torch::Tensor const& x,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(residual, "residual");
  check_fp16_matrix(x, "x");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual and x must have the same shape");
  TORCH_CHECK(x.size(1) <= 16384,
              "v2 residual path supports dim <= 16384");
  check_same_device(residual, x, "residual", "x");
  const int64_t rows = x.size(0);
  const int64_t dim = x.size(1);
  check_packed_sfa(packed, sfa, x, rows, dim);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::residual_add_rms_norm_fp4_sfa_v2_fp16(
      reinterpret_cast<__half*>(residual.data_ptr()),
      reinterpret_cast<const __half*>(x.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(dim, "dim"),
      stream);
#endif
}

void residual_add_rms_norm_mul_fp4_sfa_fp16(
    torch::Tensor& residual,
    torch::Tensor const& x,
    torch::Tensor const& inv_s,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(residual, "residual");
  check_fp16_matrix(x, "x");
  check_cuda_contiguous(inv_s, "inv_s");
  TORCH_CHECK(inv_s.scalar_type() == torch::kFloat16,
              "inv_s must have dtype torch.float16");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual and x must have the same shape");
  TORCH_CHECK(x.size(1) <= 2048,
              "residual_add_rms_norm_mul_fp4_sfa_fp16 supports dim <= 2048");
  TORCH_CHECK(inv_s.sizes() == torch::IntArrayRef({x.size(1)}),
              "inv_s must have shape (dim,)");
  check_same_device(residual, x, "residual", "x");
  check_same_device(residual, inv_s, "residual", "inv_s");
  const int64_t rows = x.size(0);
  const int64_t dim = x.size(1);
  check_packed_sfa(packed, sfa, x, rows, dim);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::residual_add_rms_norm_mul_fp4_sfa_fp16(
      reinterpret_cast<__half*>(residual.data_ptr()),
      reinterpret_cast<const __half*>(x.data_ptr()),
      reinterpret_cast<const __half*>(inv_s.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(dim, "dim"),
      stream);
#endif
}

void silu_mul_fp4_sfa_fp16(
    torch::Tensor const& merged,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(merged, "merged");
  TORCH_CHECK(merged.size(1) % 32 == 0,
              "merged.shape[1] must be divisible by 32 because hidden dim is half");
  const int64_t rows = merged.size(0);
  const int64_t hidden = merged.size(1) / 2;
  check_packed_sfa(packed, sfa, merged, rows, hidden);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(merged.device());
  auto stream = at::cuda::getCurrentCUDAStream(merged.get_device()).stream();
  flash_rt::fused_fp4::gate_silu_mul_fp4_sfa_fp16(
      reinterpret_cast<const __half*>(merged.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(hidden, "hidden"),
      stream);
#endif
}

void silu_mul_fp4_sfa_v2_fp16(
    torch::Tensor const& merged,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(merged, "merged");
  TORCH_CHECK(merged.size(1) % 32 == 0,
              "merged.shape[1] must be divisible by 32 because hidden dim is half");
  const int64_t rows = merged.size(0);
  const int64_t hidden = merged.size(1) / 2;
  check_packed_sfa(packed, sfa, merged, rows, hidden);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(merged.device());
  auto stream = at::cuda::getCurrentCUDAStream(merged.get_device()).stream();
  flash_rt::fused_fp4::gate_silu_mul_fp4_sfa_v2_fp16(
      reinterpret_cast<const __half*>(merged.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(hidden, "hidden"),
      stream);
#endif
}

void silu_mul_mul_fp4_sfa_v2_fp16(
    torch::Tensor const& merged,
    torch::Tensor const& inv_s,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(merged, "merged");
  check_cuda_contiguous(inv_s, "inv_s");
  TORCH_CHECK(inv_s.scalar_type() == torch::kFloat16,
              "inv_s must have dtype torch.float16");
  TORCH_CHECK(merged.size(1) % 32 == 0,
              "merged.shape[1] must be divisible by 32 because hidden dim is half");
  const int64_t rows = merged.size(0);
  const int64_t hidden = merged.size(1) / 2;
  TORCH_CHECK(inv_s.sizes() == torch::IntArrayRef({hidden}),
              "inv_s must have shape (hidden,)");
  check_same_device(merged, inv_s, "merged", "inv_s");
  check_packed_sfa(packed, sfa, merged, rows, hidden);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(merged.device());
  auto stream = at::cuda::getCurrentCUDAStream(merged.get_device()).stream();
  flash_rt::fused_fp4::gate_silu_mul_mul_fp4_sfa_v2_fp16(
      reinterpret_cast<const __half*>(merged.data_ptr()),
      reinterpret_cast<const __half*>(inv_s.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(hidden, "hidden"),
      stream);
#endif
}

void silu_mul_two_fp4_to_fp4(
    torch::Tensor const& gate_packed,
    torch::Tensor const& gate_sfa,
    torch::Tensor const& up_packed,
    torch::Tensor const& up_sfa,
    torch::Tensor& out_packed,
    torch::Tensor& out_sfa) {
  auto shape = check_two_fp4_inputs(
      gate_packed, gate_sfa, up_packed, up_sfa, out_packed, out_sfa);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(gate_packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(gate_packed.get_device()).stream();
  flash_rt::fused_fp4::silu_mul_two_fp4_to_fp4(
      reinterpret_cast<const uint8_t*>(gate_packed.data_ptr()),
      reinterpret_cast<const uint8_t*>(gate_sfa.data_ptr()),
      reinterpret_cast<const uint8_t*>(up_packed.data_ptr()),
      reinterpret_cast<const uint8_t*>(up_sfa.data_ptr()),
      reinterpret_cast<uint8_t*>(out_packed.data_ptr()),
      reinterpret_cast<uint8_t*>(out_sfa.data_ptr()),
      checked_int(shape.rows, "rows"),
      checked_int(shape.hidden, "hidden"),
      stream);
#endif
}

void silu_mul_two_mul_fp4_to_fp4(
    torch::Tensor const& gate_packed,
    torch::Tensor const& gate_sfa,
    torch::Tensor const& up_packed,
    torch::Tensor const& up_sfa,
    torch::Tensor const& inv_s,
    torch::Tensor& out_packed,
    torch::Tensor& out_sfa) {
  check_cuda_contiguous(inv_s, "inv_s");
  TORCH_CHECK(inv_s.scalar_type() == torch::kFloat16,
              "inv_s must have dtype torch.float16");
  auto shape = check_two_fp4_inputs(
      gate_packed, gate_sfa, up_packed, up_sfa, out_packed, out_sfa);
  TORCH_CHECK(inv_s.dim() == 1 && inv_s.size(0) == shape.hidden,
              "inv_s must have shape (hidden,)");
  check_same_device(gate_packed, inv_s, "gate_packed", "inv_s");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(gate_packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(gate_packed.get_device()).stream();
  flash_rt::fused_fp4::silu_mul_two_mul_fp4_to_fp4(
      reinterpret_cast<const uint8_t*>(gate_packed.data_ptr()),
      reinterpret_cast<const uint8_t*>(gate_sfa.data_ptr()),
      reinterpret_cast<const uint8_t*>(up_packed.data_ptr()),
      reinterpret_cast<const uint8_t*>(up_sfa.data_ptr()),
      reinterpret_cast<const __half*>(inv_s.data_ptr()),
      reinterpret_cast<uint8_t*>(out_packed.data_ptr()),
      reinterpret_cast<uint8_t*>(out_sfa.data_ptr()),
      checked_int(shape.rows, "rows"),
      checked_int(shape.hidden, "hidden"),
      stream);
#endif
}

void adaptive_rms_norm_nvfp4_fp16(
    torch::Tensor const& x,
    torch::Tensor const& style,
    torch::Tensor& packed,
    torch::Tensor& sfa,
    torch::Tensor& gate) {
  check_fp16_matrix(x, "x");
  check_fp16_matrix(style, "style");
  check_fp16_matrix(gate, "gate");
  const int64_t rows = x.size(0);
  const int64_t dim = x.size(1);
  TORCH_CHECK(dim == 1024,
              "adaptive_rms_norm_nvfp4_fp16 currently supports dim=1024");
  TORCH_CHECK(style.sizes() == torch::IntArrayRef({rows, 3 * dim}),
              "style must have shape (rows, 3 * dim)");
  TORCH_CHECK(gate.sizes() == x.sizes(), "gate must match x shape");
  check_packed_sfa(packed, sfa, x, rows, dim);
  check_same_device(x, style, "x", "style");
  check_same_device(x, gate, "x", "gate");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::pi05_adarms_fp4_sfa_native_fp16(
      reinterpret_cast<const __half*>(x.data_ptr()),
      reinterpret_cast<const __half*>(style.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      reinterpret_cast<__half*>(gate.data_ptr()), checked_int(rows, "rows"),
      checked_int(dim, "dim"), stream);
#endif
}

void gated_residual_adaptive_rms_norm_nvfp4_fp16(
    torch::Tensor const& x,
    torch::Tensor const& previous_gate,
    torch::Tensor& residual,
    torch::Tensor const& style,
    torch::Tensor& packed,
    torch::Tensor& sfa,
    torch::Tensor& gate) {
  check_fp16_matrix(x, "x");
  check_fp16_matrix(previous_gate, "previous_gate");
  check_fp16_matrix(residual, "residual");
  check_fp16_matrix(style, "style");
  check_fp16_matrix(gate, "gate");
  const int64_t rows = x.size(0);
  const int64_t dim = x.size(1);
  TORCH_CHECK(dim == 1024,
              "gated_residual_adaptive_rms_norm_nvfp4_fp16 currently supports dim=1024");
  TORCH_CHECK(previous_gate.sizes() == x.sizes() &&
                  residual.sizes() == x.sizes() && gate.sizes() == x.sizes(),
              "previous_gate, residual, and gate must match x shape");
  TORCH_CHECK(style.sizes() == torch::IntArrayRef({rows, 3 * dim}),
              "style must have shape (rows, 3 * dim)");
  check_packed_sfa(packed, sfa, x, rows, dim);
  check_same_device(x, previous_gate, "x", "previous_gate");
  check_same_device(x, residual, "x", "residual");
  check_same_device(x, style, "x", "style");
  check_same_device(x, gate, "x", "gate");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::pi05_gate_res_adarms_fp4_sfa_native_fp16(
      reinterpret_cast<const __half*>(x.data_ptr()),
      reinterpret_cast<const __half*>(previous_gate.data_ptr()),
      reinterpret_cast<__half*>(residual.data_ptr()),
      reinterpret_cast<const __half*>(style.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      reinterpret_cast<__half*>(gate.data_ptr()), checked_int(rows, "rows"),
      checked_int(dim, "dim"), stream);
#endif
}

void adaptive_rms_norm_fp8_static_fp16(
    torch::Tensor const& x,
    torch::Tensor const& style,
    torch::Tensor const& scale,
    torch::Tensor& out,
    torch::Tensor& gate) {
  check_fp16_matrix(x, "x");
  check_fp16_matrix(style, "style");
  check_fp16_matrix(gate, "gate");
  TORCH_CHECK(style.sizes() == torch::IntArrayRef({x.size(0), 3 * x.size(1)}),
              "style must have shape (rows, 3 * dim)");
  TORCH_CHECK(gate.sizes() == x.sizes(), "gate must match x shape");
  check_cuda_contiguous(scale, "scale");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32 && scale.numel() == 1,
              "scale must be a float32 scalar tensor");
  check_cuda_contiguous(out, "out");
  TORCH_CHECK(out.scalar_type() == c10::ScalarType::Float8_e4m3fn &&
                  out.sizes() == x.sizes(),
              "out must be float8_e4m3fn with the same shape as x");
  check_same_device(x, style, "x", "style");
  check_same_device(x, scale, "x", "scale");
  check_same_device(x, out, "x", "out");
  check_same_device(x, gate, "x", "gate");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::adaptive_rms_norm_fp8_static_fp16(
      x.data_ptr(), style.data_ptr(), out.data_ptr(), gate.data_ptr(),
      checked_int(x.size(0), "rows"), checked_int(x.size(1), "dim"),
      static_cast<const float*>(scale.data_ptr()), stream);
#endif
}

void gated_residual_adaptive_rms_norm_fp8_static_fp16(
    torch::Tensor const& x,
    torch::Tensor const& previous_gate,
    torch::Tensor& residual,
    torch::Tensor const& style,
    torch::Tensor const& scale,
    torch::Tensor& out,
    torch::Tensor& gate) {
  check_fp16_matrix(x, "x");
  check_fp16_matrix(previous_gate, "previous_gate");
  check_fp16_matrix(residual, "residual");
  TORCH_CHECK(previous_gate.sizes() == x.sizes() && residual.sizes() == x.sizes(),
              "previous_gate and residual must match x shape");
  check_fp16_matrix(style, "style");
  check_fp16_matrix(gate, "gate");
  TORCH_CHECK(style.sizes() == torch::IntArrayRef({x.size(0), 3 * x.size(1)}),
              "style must have shape (rows, 3 * dim)");
  TORCH_CHECK(gate.sizes() == x.sizes(), "gate must match x shape");
  check_cuda_contiguous(scale, "scale");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32 && scale.numel() == 1,
              "scale must be a float32 scalar tensor");
  check_cuda_contiguous(out, "out");
  TORCH_CHECK(out.scalar_type() == c10::ScalarType::Float8_e4m3fn &&
                  out.sizes() == x.sizes(),
              "out must be float8_e4m3fn with the same shape as x");
  check_same_device(x, previous_gate, "x", "previous_gate");
  check_same_device(x, residual, "x", "residual");
  check_same_device(x, style, "x", "style");
  check_same_device(x, scale, "x", "scale");
  check_same_device(x, out, "x", "out");
  check_same_device(x, gate, "x", "gate");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::gated_residual_adaptive_rms_norm_fp8_static_fp16(
      x.data_ptr(), previous_gate.data_ptr(), residual.data_ptr(),
      style.data_ptr(), out.data_ptr(), gate.data_ptr(),
      checked_int(x.size(0), "rows"), checked_int(x.size(1), "dim"),
      static_cast<const float*>(scale.data_ptr()), stream);
#endif
}

void adaptive_rms_norm_e0m3_fp16(
    torch::Tensor const& x, torch::Tensor const& style, bool use_rht,
    torch::Tensor& packed, torch::Tensor& sfa, torch::Tensor& gate) {
  check_fp16_matrix(x, "x");
  check_fp16_matrix(style, "style");
  check_fp16_matrix(gate, "gate");
  TORCH_CHECK(x.size(1) == 1024,
              "adaptive_rms_norm_e0m3_fp16 currently supports dim=1024");
  TORCH_CHECK(style.sizes() == torch::IntArrayRef({x.size(0), 3 * x.size(1)}),
              "style must have shape (rows, 3 * dim)");
  TORCH_CHECK(gate.sizes() == x.sizes(), "gate must match x shape");
  check_packed_sfa(packed, sfa, x, x.size(0), x.size(1));
  check_same_device(x, style, "x", "style");
  check_same_device(x, gate, "x", "gate");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::pi05_adarms_e0m3_sfa_fp16(
      static_cast<const __half*>(x.data_ptr()),
      static_cast<const __half*>(style.data_ptr()),
      static_cast<uint8_t*>(packed.data_ptr()),
      static_cast<uint8_t*>(sfa.data_ptr()),
      static_cast<__half*>(gate.data_ptr()), checked_int(x.size(0), "rows"),
      checked_int(x.size(1), "dim"), use_rht ? 1 : 0, stream);
#endif
}

void gated_residual_adaptive_rms_norm_e0m3_fp16(
    torch::Tensor const& x, torch::Tensor const& previous_gate,
    torch::Tensor& residual, torch::Tensor const& style, bool use_rht,
    torch::Tensor& packed, torch::Tensor& sfa, torch::Tensor& gate) {
  check_fp16_matrix(x, "x");
  check_fp16_matrix(previous_gate, "previous_gate");
  check_fp16_matrix(residual, "residual");
  check_fp16_matrix(style, "style");
  check_fp16_matrix(gate, "gate");
  TORCH_CHECK(x.size(1) == 1024,
              "gated_residual_adaptive_rms_norm_e0m3_fp16 currently supports dim=1024");
  TORCH_CHECK(previous_gate.sizes() == x.sizes() &&
                  residual.sizes() == x.sizes() && gate.sizes() == x.sizes(),
              "previous_gate, residual, and gate must match x shape");
  TORCH_CHECK(style.sizes() == torch::IntArrayRef({x.size(0), 3 * x.size(1)}),
              "style must have shape (rows, 3 * dim)");
  check_packed_sfa(packed, sfa, x, x.size(0), x.size(1));
  check_same_device(x, previous_gate, "x", "previous_gate");
  check_same_device(x, residual, "x", "residual");
  check_same_device(x, style, "x", "style");
  check_same_device(x, gate, "x", "gate");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::pi05_gate_res_adarms_e0m3_sfa_fp16(
      static_cast<const __half*>(x.data_ptr()),
      static_cast<const __half*>(previous_gate.data_ptr()),
      static_cast<__half*>(residual.data_ptr()),
      static_cast<const __half*>(style.data_ptr()),
      static_cast<uint8_t*>(packed.data_ptr()),
      static_cast<uint8_t*>(sfa.data_ptr()),
      static_cast<__half*>(gate.data_ptr()), checked_int(x.size(0), "rows"),
      checked_int(x.size(1), "dim"), use_rht ? 1 : 0, stream);
#endif
}

void gelu_mul_e0m3_fp16(
    torch::Tensor const& merged, bool use_rht, torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(merged, "merged");
  TORCH_CHECK(merged.size(1) % 32 == 0,
              "merged.shape[1] must be divisible by 32");
  const int64_t hidden = merged.size(1) / 2;
  check_packed_sfa(packed, sfa, merged, merged.size(0), hidden);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(merged.device());
  auto stream = at::cuda::getCurrentCUDAStream(merged.get_device()).stream();
  const int rc = flash_rt::fused_fp4::gate_geglu_e0m3_sfa_vec_fp16(
      static_cast<const __half*>(merged.data_ptr()),
      static_cast<uint8_t*>(packed.data_ptr()),
      static_cast<uint8_t*>(sfa.data_ptr()), checked_int(merged.size(0), "rows"),
      checked_int(hidden, "hidden"), use_rht ? 1 : 0, stream);
  TORCH_CHECK(rc == 0, "gelu_mul_e0m3_fp16 failed with rc=", rc);
#endif
}

void residual_add_rms_norm_quant_nvfp4_swizzled_bf16(
    torch::Tensor& residual, torch::Tensor const& x,
    torch::Tensor const& weight, double eps, torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_cuda_contiguous(residual, "residual");
  check_cuda_contiguous(x, "x");
  check_cuda_contiguous(weight, "weight");
  TORCH_CHECK(residual.scalar_type() == torch::kBFloat16 &&
                  x.scalar_type() == torch::kBFloat16 &&
                  weight.scalar_type() == torch::kBFloat16,
              "residual, x, and weight must be torch.bfloat16");
  TORCH_CHECK(residual.dim() == 2 && x.sizes() == residual.sizes(),
              "residual and x must have the same 2D shape");
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({x.size(1)}),
              "weight must have shape (dim,)");
  TORCH_CHECK(x.size(1) <= 16384 && x.size(1) % 16 == 0,
              "dim must be a multiple of 16 and at most 16384");
  check_packed_sfa(packed, sfa, x, x.size(0), x.size(1));
  check_same_device(x, residual, "x", "residual");
  check_same_device(x, weight, "x", "weight");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::cosmos3_edge_res_rms_fp4_sfa_bf16(
      static_cast<__nv_bfloat16*>(residual.data_ptr()),
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<const __nv_bfloat16*>(weight.data_ptr()),
      static_cast<uint8_t*>(packed.data_ptr()),
      static_cast<uint8_t*>(sfa.data_ptr()), checked_int(x.size(0), "rows"),
      checked_int(x.size(1), "dim"), static_cast<float>(eps), stream);
#endif
}

void relu2_quant_nvfp4_swizzled_fp16(
    torch::Tensor const& x, torch::Tensor& packed, torch::Tensor& sfa) {
  check_fp16_matrix(x, "x");
  check_packed_sfa(packed, sfa, x, x.size(0), x.size(1));
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::cosmos3_edge_relu2_fp4_sfa_fp16(
      static_cast<const __half*>(x.data_ptr()),
      static_cast<uint8_t*>(packed.data_ptr()),
      static_cast<uint8_t*>(sfa.data_ptr()), checked_int(x.size(0), "rows"),
      checked_int(x.size(1), "dim"), stream);
#endif
}

void layer_norm_fp8_fp16(
    torch::Tensor const& x,
    torch::Tensor const& gamma,
    torch::Tensor const& beta,
    double eps,
    torch::Tensor& out) {
  check_fp16_matrix(x, "x");
  check_cuda_contiguous(gamma, "gamma");
  check_cuda_contiguous(beta, "beta");
  check_cuda_contiguous(out, "out");
  TORCH_CHECK(gamma.scalar_type() == torch::kFloat16 &&
                  beta.scalar_type() == torch::kFloat16,
              "gamma and beta must have dtype torch.float16");
  TORCH_CHECK(out.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              "out must have dtype torch.float8_e4m3fn");
  TORCH_CHECK(gamma.sizes() == torch::IntArrayRef({x.size(1)}) &&
                  beta.sizes() == gamma.sizes(),
              "gamma and beta must have shape (dim,)");
  TORCH_CHECK(out.sizes() == x.sizes(), "out must match x shape");
  check_same_device(x, gamma, "x", "gamma");
  check_same_device(x, beta, "x", "beta");
  check_same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int rc = flash_rt::fused_fp4::layer_norm_fp8_vec_fp16(
      reinterpret_cast<const __half*>(x.data_ptr()),
      reinterpret_cast<const __half*>(gamma.data_ptr()),
      reinterpret_cast<const __half*>(beta.data_ptr()), out.data_ptr(),
      checked_int(x.size(0), "rows"), checked_int(x.size(1), "dim"),
      static_cast<float>(eps), stream);
  TORCH_CHECK(rc == 0, "layer_norm_fp8_fp16 unsupported or failed with rc=", rc);
#endif
}

void layer_norm_nvfp4_fp16(
    torch::Tensor const& x,
    torch::Tensor const& gamma,
    torch::Tensor const& beta,
    c10::optional<torch::Tensor> const& inv_s,
    double eps,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(x, "x");
  check_cuda_contiguous(gamma, "gamma");
  check_cuda_contiguous(beta, "beta");
  TORCH_CHECK(gamma.scalar_type() == torch::kFloat16 &&
                  beta.scalar_type() == torch::kFloat16,
              "gamma and beta must have dtype torch.float16");
  TORCH_CHECK(gamma.sizes() == torch::IntArrayRef({x.size(1)}) &&
                  beta.sizes() == gamma.sizes(),
              "gamma and beta must have shape (dim,)");
  check_packed_sfa(packed, sfa, x, x.size(0), x.size(1));
  check_same_device(x, gamma, "x", "gamma");
  check_same_device(x, beta, "x", "beta");
  const __half* inv_ptr = nullptr;
  if (inv_s.has_value()) {
    auto const& scale = inv_s.value();
    check_cuda_contiguous(scale, "inv_s");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat16 &&
                    scale.sizes() == gamma.sizes(),
                "inv_s must be float16 with shape (dim,)");
    check_same_device(x, scale, "x", "inv_s");
    inv_ptr = reinterpret_cast<const __half*>(scale.data_ptr());
  }
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  int rc = flash_rt::fused_fp4::layer_norm_mul_fp4_sfa_vec_fp16(
      reinterpret_cast<const __half*>(x.data_ptr()),
      reinterpret_cast<const __half*>(gamma.data_ptr()),
      reinterpret_cast<const __half*>(beta.data_ptr()), inv_ptr,
      packed.data_ptr(), sfa.data_ptr(), checked_int(x.size(0), "rows"),
      checked_int(x.size(1), "dim"), static_cast<float>(eps), stream);
  if (rc != 0) {
    rc = flash_rt::fused_fp4::layer_norm_mul_fp4_sfa_fp16(
        reinterpret_cast<const __half*>(x.data_ptr()),
        reinterpret_cast<const __half*>(gamma.data_ptr()),
        reinterpret_cast<const __half*>(beta.data_ptr()), inv_ptr,
        packed.data_ptr(), sfa.data_ptr(), checked_int(x.size(0), "rows"),
        checked_int(x.size(1), "dim"), static_cast<float>(eps), stream);
  }
  TORCH_CHECK(rc == 0, "layer_norm_nvfp4_fp16 failed with rc=", rc);
#endif
}

void gelu_mul_nvfp4_fp16(
    torch::Tensor const& merged,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(merged, "merged");
  TORCH_CHECK(merged.size(1) % 32 == 0,
              "merged.shape[1] must be divisible by 32");
  const int64_t rows = merged.size(0);
  const int64_t hidden = merged.size(1) / 2;
  check_packed_sfa(packed, sfa, merged, rows, hidden);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(merged.device());
  auto stream = at::cuda::getCurrentCUDAStream(merged.get_device()).stream();
  int rc = flash_rt::fused_fp4::gate_silu_mul_fp4_sfa_vec_fp16(
      reinterpret_cast<const __half*>(merged.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()), checked_int(rows, "rows"),
      checked_int(hidden, "hidden"), stream);
  if (rc != 0) {
    flash_rt::fused_fp4::gate_silu_mul_fp4_sfa_v2_fp16(
        reinterpret_cast<const __half*>(merged.data_ptr()),
        reinterpret_cast<uint8_t*>(packed.data_ptr()),
        reinterpret_cast<uint8_t*>(sfa.data_ptr()), checked_int(rows, "rows"),
        checked_int(hidden, "hidden"), stream);
  }
#endif
}

void dequantize_fp4_sfa_fp16(
    torch::Tensor const& packed,
    torch::Tensor const& sfa,
    torch::Tensor& out) {
  check_uint8(packed, "packed");
  check_uint8(sfa, "sfa");
  check_cuda_contiguous(out, "out");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "out must have dtype torch.float16");
  TORCH_CHECK(out.dim() == 2, "out must have shape (rows, dim)");
  const int64_t rows = out.size(0);
  const int64_t dim = out.size(1);
  TORCH_CHECK(dim % 16 == 0, "out.shape[1] must be divisible by 16");
  TORCH_CHECK(packed.sizes() == torch::IntArrayRef({rows, dim / 2}),
              "packed must have shape (rows, dim / 2)");
  const int64_t required = sfa_size_bytes(rows, dim, false);
  TORCH_CHECK(sfa.numel() >= required,
              "sfa is too small for CUTLASS SFA layout");
  check_same_device(packed, sfa, "packed", "sfa");
  check_same_device(packed, out, "packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed.get_device()).stream();
  flash_rt::fused_fp4::dequantize_fp4_sfa_fp16(
      reinterpret_cast<const uint8_t*>(packed.data_ptr()),
      reinterpret_cast<const uint8_t*>(sfa.data_ptr()),
      reinterpret_cast<__half*>(out.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(dim, "dim"),
      stream);
#endif
}

void rms_silu_nvfp4_ndhwc_bf16(
    torch::Tensor const& x,
    torch::Tensor const& gamma,
    c10::optional<torch::Tensor> const& awq_inv_scale,
    double eps,
    torch::Tensor& packed,
    torch::Tensor& scale_factors) {
  check_cuda_contiguous(x, "x");
  check_cuda_contiguous(gamma, "gamma");
  check_uint8(packed, "packed");
  check_uint8(scale_factors, "scale_factors");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16,
              "x must have dtype torch.bfloat16");
  TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16,
              "gamma must have dtype torch.bfloat16");
  TORCH_CHECK(x.dim() == 5, "x must have shape (B,C,T,H,W)");
  const auto b = x.size(0);
  const auto c = x.size(1);
  const auto t = x.size(2);
  const auto h = x.size(3);
  const auto w = x.size(4);
  TORCH_CHECK(c % 128 == 0 && c <= 1024,
              "C must be divisible by 128 and at most 1024");
  TORCH_CHECK(gamma.sizes() == torch::IntArrayRef({c}),
              "gamma must have shape (C,)");
  TORCH_CHECK(
      packed.sizes() == torch::IntArrayRef({b, t, h, w, c / 2}),
      "packed must have shape (B,T,H,W,C/2)");
  TORCH_CHECK(
      scale_factors.sizes() == torch::IntArrayRef({b, t, h, w, c / 16}),
      "scale_factors must have shape (B,T,H,W,C/16)");
  check_same_device(x, gamma, "x", "gamma");
  check_same_device(x, packed, "x", "packed");
  check_same_device(x, scale_factors, "x", "scale_factors");
  const void* awq_ptr = nullptr;
  if (awq_inv_scale.has_value()) {
    auto const& scale = awq_inv_scale.value();
    check_cuda_contiguous(scale, "awq_inv_scale");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32,
                "awq_inv_scale must have dtype torch.float32");
    TORCH_CHECK(scale.sizes() == torch::IntArrayRef({c}),
                "awq_inv_scale must have shape (C,)");
    check_same_device(x, scale, "x", "awq_inv_scale");
    awq_ptr = scale.data_ptr();
  }
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int status =
      flash_rt::quantize::motus_bf16_rms_silu_quant_nvfp4_to_ndhwc_v1(
          x.data_ptr(), gamma.data_ptr(), awq_ptr, packed.data_ptr(),
          scale_factors.data_ptr(), checked_int(b, "B"),
          checked_int(c, "C"), checked_int(t, "T"), checked_int(h, "H"),
          checked_int(w, "W"), static_cast<float>(eps), stream);
  TORCH_CHECK(status == 0,
              "rms_silu_nvfp4_ndhwc_bf16 failed with status ", status);
#endif
}

void quantize_bf16_to_nvfp4_linear(
    torch::Tensor const& input,
    torch::Tensor& packed,
    torch::Tensor& scale_factors) {
  check_cuda_contiguous(input, "input");
  check_uint8(packed, "packed");
  check_uint8(scale_factors, "scale_factors");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
              "input must have dtype torch.bfloat16");
  TORCH_CHECK(input.dim() == 2, "input must have shape (rows, cols)");
  const auto rows = input.size(0);
  const auto cols = input.size(1);
  TORCH_CHECK(rows > 0, "rows must be positive");
  TORCH_CHECK(cols > 0 && cols % 16 == 0,
              "cols must be positive and divisible by 16");
  TORCH_CHECK(
      packed.sizes() == torch::IntArrayRef({rows, cols / 2}),
      "packed must have shape (rows, cols/2)");
  TORCH_CHECK(
      scale_factors.sizes() == torch::IntArrayRef({rows, cols / 16}),
      "scale_factors must have shape (rows, cols/16)");
  check_same_device(input, packed, "input", "packed");
  check_same_device(input, scale_factors, "input", "scale_factors");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(input.device());
  auto stream =
      at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  flash_rt::quantize::quantize_bf16_to_nvfp4_linear(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(scale_factors.data_ptr()),
      checked_int(rows, "rows"), checked_int(cols, "cols"), stream);
#endif
}

void bf16_rms_silu_ncdhw(
    torch::Tensor const& x,
    torch::Tensor const& gamma,
    c10::optional<torch::Tensor> const& prev_cache,
    double eps,
    torch::Tensor& out,
    c10::optional<torch::Tensor> const& next_cache) {
  check_cuda_contiguous(x, "x");
  check_cuda_contiguous(gamma, "gamma");
  check_cuda_contiguous(out, "out");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16,
              "x must have dtype torch.bfloat16");
  TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16,
              "gamma must have dtype torch.bfloat16");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16,
              "out must have dtype torch.bfloat16");
  TORCH_CHECK(x.dim() == 5, "x must have shape (B,C,T,H,W)");
  const auto b = x.size(0);
  const auto c = x.size(1);
  const auto t = x.size(2);
  const auto h = x.size(3);
  const auto w = x.size(4);
  TORCH_CHECK(c > 0 && (c % 2) == 0 && c <= 1024,
              "C must be even and at most 1024");
  TORCH_CHECK(gamma.sizes() == torch::IntArrayRef({c}),
              "gamma must have shape (C,)");
  TORCH_CHECK(out.sizes() == x.sizes(), "out must have the same shape as x");
  check_same_device(x, gamma, "x", "gamma");
  check_same_device(x, out, "x", "out");
  const void* prev_ptr = nullptr;
  void* next_ptr = nullptr;
  if (prev_cache.has_value()) {
    auto const& cache = prev_cache.value();
    check_cuda_contiguous(cache, "prev_cache");
    TORCH_CHECK(cache.scalar_type() == torch::kBFloat16,
                "prev_cache must have dtype torch.bfloat16");
    TORCH_CHECK(
        cache.sizes() == torch::IntArrayRef({b, c, 2, h, w}),
        "prev_cache must have shape (B,C,2,H,W)");
    check_same_device(x, cache, "x", "prev_cache");
    prev_ptr = cache.data_ptr();
  }
  if (next_cache.has_value()) {
    auto const& cache = next_cache.value();
    check_cuda_contiguous(cache, "next_cache");
    TORCH_CHECK(cache.scalar_type() == torch::kBFloat16,
                "next_cache must have dtype torch.bfloat16");
    TORCH_CHECK(
        cache.sizes() == torch::IntArrayRef({b, c, 2, h, w}),
        "next_cache must have shape (B,C,2,H,W)");
    check_same_device(x, cache, "x", "next_cache");
    next_ptr = cache.data_ptr();
  }
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int status = flash_rt::quantize::bf16_rms_silu_ncdhw(
      x.data_ptr(), gamma.data_ptr(), out.data_ptr(), prev_ptr, next_ptr,
      checked_int(b, "B"), checked_int(c, "C"), checked_int(t, "T"),
      checked_int(h, "H"), checked_int(w, "W"), static_cast<float>(eps),
      stream);
  TORCH_CHECK(status == 0, "bf16_rms_silu_ncdhw failed with status ", status);
#endif
}

void bf16_rms_norm_ncdhw(
    torch::Tensor const& x,
    torch::Tensor const& gamma,
    c10::optional<torch::Tensor> const& bias,
    double eps,
    torch::Tensor& out) {
  check_cuda_contiguous(x, "x");
  check_cuda_contiguous(gamma, "gamma");
  check_cuda_contiguous(out, "out");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16,
              "x must have dtype torch.bfloat16");
  TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16,
              "gamma must have dtype torch.bfloat16");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16,
              "out must have dtype torch.bfloat16");
  TORCH_CHECK(x.dim() == 5, "x must have shape (B,C,T,H,W)");
  const auto b = x.size(0);
  const auto c = x.size(1);
  const auto t = x.size(2);
  const auto h = x.size(3);
  const auto w = x.size(4);
  TORCH_CHECK(c > 0 && (c % 2) == 0 && c <= 1024,
              "C must be even and at most 1024");
  TORCH_CHECK(gamma.sizes() == torch::IntArrayRef({c}),
              "gamma must have shape (C,)");
  TORCH_CHECK(out.sizes() == x.sizes(), "out must have the same shape as x");
  check_same_device(x, gamma, "x", "gamma");
  check_same_device(x, out, "x", "out");
  const void* bias_ptr = nullptr;
  if (bias.has_value()) {
    auto const& bias_tensor = bias.value();
    check_cuda_contiguous(bias_tensor, "bias");
    TORCH_CHECK(bias_tensor.scalar_type() == torch::kBFloat16,
                "bias must have dtype torch.bfloat16");
    TORCH_CHECK(bias_tensor.sizes() == torch::IntArrayRef({c}),
                "bias must have shape (C,)");
    check_same_device(x, bias_tensor, "x", "bias");
    bias_ptr = bias_tensor.data_ptr();
  }
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int status = flash_rt::quantize::bf16_rms_norm_ncdhw(
      x.data_ptr(), gamma.data_ptr(), bias_ptr, out.data_ptr(),
      checked_int(b, "B"), checked_int(c, "C"), checked_int(t, "T"),
      checked_int(h, "H"), checked_int(w, "W"), static_cast<float>(eps),
      stream);
  TORCH_CHECK(status == 0, "bf16_rms_norm_ncdhw failed with status ", status);
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("sfa_size_bytes_for(Tensor anchor, int rows, int dim, bool is_sfb=False) -> int");
  ops.def("rms_norm_fp4_sfa_fp16(Tensor x, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("residual_add_rms_norm_fp4_sfa_fp16(Tensor! residual, Tensor x, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("residual_add_rms_norm_fp4_sfa_v2_fp16(Tensor! residual, Tensor x, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("residual_add_rms_norm_mul_fp4_sfa_fp16(Tensor! residual, Tensor x, Tensor inv_s, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("silu_mul_fp4_sfa_fp16(Tensor merged, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("silu_mul_fp4_sfa_v2_fp16(Tensor merged, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("silu_mul_mul_fp4_sfa_v2_fp16(Tensor merged, Tensor inv_s, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("silu_mul_two_fp4_to_fp4(Tensor gate_packed, Tensor gate_sfa, Tensor up_packed, Tensor up_sfa, Tensor! out_packed, Tensor! out_sfa) -> ()");
  ops.def("silu_mul_two_mul_fp4_to_fp4(Tensor gate_packed, Tensor gate_sfa, Tensor up_packed, Tensor up_sfa, Tensor inv_s, Tensor! out_packed, Tensor! out_sfa) -> ()");
  ops.def("adaptive_rms_norm_nvfp4_fp16(Tensor x, Tensor style, Tensor! packed, Tensor! sfa, Tensor! gate) -> ()");
  ops.def("gated_residual_adaptive_rms_norm_nvfp4_fp16(Tensor x, Tensor previous_gate, Tensor! residual, Tensor style, Tensor! packed, Tensor! sfa, Tensor! gate) -> ()");
  ops.def("adaptive_rms_norm_fp8_static_fp16(Tensor x, Tensor style, Tensor scale, Tensor! out, Tensor! gate) -> ()");
  ops.def("gated_residual_adaptive_rms_norm_fp8_static_fp16(Tensor x, Tensor previous_gate, Tensor! residual, Tensor style, Tensor scale, Tensor! out, Tensor! gate) -> ()");
  ops.def("adaptive_rms_norm_e0m3_fp16(Tensor x, Tensor style, bool use_rht, Tensor! packed, Tensor! sfa, Tensor! gate) -> ()");
  ops.def("gated_residual_adaptive_rms_norm_e0m3_fp16(Tensor x, Tensor previous_gate, Tensor! residual, Tensor style, bool use_rht, Tensor! packed, Tensor! sfa, Tensor! gate) -> ()");
  ops.def("gelu_mul_e0m3_fp16(Tensor merged, bool use_rht, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("residual_add_rms_norm_quant_nvfp4_swizzled_bf16(Tensor! residual, Tensor x, Tensor weight, float eps, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("relu2_quant_nvfp4_swizzled_fp16(Tensor x, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("layer_norm_fp8_fp16(Tensor x, Tensor gamma, Tensor beta, float eps, Tensor! out) -> ()");
  ops.def("layer_norm_nvfp4_fp16(Tensor x, Tensor gamma, Tensor beta, Tensor? inv_s, float eps, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("gelu_mul_nvfp4_fp16(Tensor merged, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("dequantize_fp4_sfa_fp16(Tensor packed, Tensor sfa, Tensor! out) -> ()");
  ops.def("rms_silu_nvfp4_ndhwc_bf16(Tensor x, Tensor gamma, Tensor? awq_inv_scale, float eps, Tensor! packed, Tensor! scale_factors) -> ()");
  ops.def("quantize_bf16_to_nvfp4_linear(Tensor input, Tensor! packed, Tensor! scale_factors) -> ()");
  ops.def("bf16_rms_silu_ncdhw(Tensor x, Tensor gamma, Tensor? prev_cache, float eps, Tensor! out, Tensor? next_cache) -> ()");
  ops.def("bf16_rms_norm_ncdhw(Tensor x, Tensor gamma, Tensor? bias, float eps, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("sfa_size_bytes_for", torch::kCUDA, &sfa_size_bytes_for);
  ops.impl("rms_norm_fp4_sfa_fp16", torch::kCUDA, &rms_norm_fp4_sfa_fp16);
  ops.impl("residual_add_rms_norm_fp4_sfa_fp16", torch::kCUDA, &residual_add_rms_norm_fp4_sfa_fp16);
  ops.impl("residual_add_rms_norm_fp4_sfa_v2_fp16", torch::kCUDA, &residual_add_rms_norm_fp4_sfa_v2_fp16);
  ops.impl("residual_add_rms_norm_mul_fp4_sfa_fp16", torch::kCUDA, &residual_add_rms_norm_mul_fp4_sfa_fp16);
  ops.impl("silu_mul_fp4_sfa_fp16", torch::kCUDA, &silu_mul_fp4_sfa_fp16);
  ops.impl("silu_mul_fp4_sfa_v2_fp16", torch::kCUDA, &silu_mul_fp4_sfa_v2_fp16);
  ops.impl("silu_mul_mul_fp4_sfa_v2_fp16", torch::kCUDA, &silu_mul_mul_fp4_sfa_v2_fp16);
  ops.impl("silu_mul_two_fp4_to_fp4", torch::kCUDA, &silu_mul_two_fp4_to_fp4);
  ops.impl("silu_mul_two_mul_fp4_to_fp4", torch::kCUDA, &silu_mul_two_mul_fp4_to_fp4);
  ops.impl("adaptive_rms_norm_nvfp4_fp16", torch::kCUDA, &adaptive_rms_norm_nvfp4_fp16);
  ops.impl("gated_residual_adaptive_rms_norm_nvfp4_fp16", torch::kCUDA, &gated_residual_adaptive_rms_norm_nvfp4_fp16);
  ops.impl("adaptive_rms_norm_fp8_static_fp16", torch::kCUDA, &adaptive_rms_norm_fp8_static_fp16);
  ops.impl("gated_residual_adaptive_rms_norm_fp8_static_fp16", torch::kCUDA, &gated_residual_adaptive_rms_norm_fp8_static_fp16);
  ops.impl("adaptive_rms_norm_e0m3_fp16", torch::kCUDA, &adaptive_rms_norm_e0m3_fp16);
  ops.impl("gated_residual_adaptive_rms_norm_e0m3_fp16", torch::kCUDA, &gated_residual_adaptive_rms_norm_e0m3_fp16);
  ops.impl("gelu_mul_e0m3_fp16", torch::kCUDA, &gelu_mul_e0m3_fp16);
  ops.impl("residual_add_rms_norm_quant_nvfp4_swizzled_bf16", torch::kCUDA, &residual_add_rms_norm_quant_nvfp4_swizzled_bf16);
  ops.impl("relu2_quant_nvfp4_swizzled_fp16", torch::kCUDA, &relu2_quant_nvfp4_swizzled_fp16);
  ops.impl("layer_norm_fp8_fp16", torch::kCUDA, &layer_norm_fp8_fp16);
  ops.impl("layer_norm_nvfp4_fp16", torch::kCUDA, &layer_norm_nvfp4_fp16);
  ops.impl("gelu_mul_nvfp4_fp16", torch::kCUDA, &gelu_mul_nvfp4_fp16);
  ops.impl("dequantize_fp4_sfa_fp16", torch::kCUDA, &dequantize_fp4_sfa_fp16);
  ops.impl("rms_silu_nvfp4_ndhwc_bf16", torch::kCUDA, &rms_silu_nvfp4_ndhwc_bf16);
  ops.impl("quantize_bf16_to_nvfp4_linear", torch::kCUDA, &quantize_bf16_to_nvfp4_linear);
  ops.impl("bf16_rms_silu_ncdhw", torch::kCUDA, &bf16_rms_silu_ncdhw);
  ops.impl("bf16_rms_norm_ncdhw", torch::kCUDA, &bf16_rms_norm_ncdhw);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
