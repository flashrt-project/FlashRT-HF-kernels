#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "registration.h"
#include "torch_binding.h"
#include "weight_only_ffn.cuh"

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must be torch.bfloat16");
}

void check_u8(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8,
              name, " must be torch.uint8");
}

void check_i8(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kInt8,
              name, " must be torch.int8");
}

void check_f32(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              name, " must be torch.float32");
}

int as_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in a positive int");
  return static_cast<int>(value);
}

int64_t sfb_bytes(int64_t rows, int64_t cols) {
  const int64_t row_super = (rows + 127) / 128;
  const int64_t col_super = ((cols / 16) + 3) / 4;
  return row_super * col_super * 512;
}

void check_same_device(torch::Tensor const& reference,
                       torch::Tensor const& other,
                       const char* name) {
  TORCH_CHECK(reference.device() == other.device(),
              name, " must be on ", reference.device());
}

const void* optional_bf16_ptr(c10::optional<torch::Tensor> const& tensor,
                              torch::Tensor const& reference,
                              int64_t expected_numel,
                              const char* name) {
  if (!tensor.has_value()) return nullptr;
  check_bf16(*tensor, name);
  check_same_device(reference, *tensor, name);
  TORCH_CHECK(tensor->numel() == expected_numel,
              name, " must contain ", expected_numel, " elements");
  return tensor->data_ptr();
}

struct MatrixShape {
  int64_t m;
  int64_t n;
  int64_t k;
};

void check_variant(int64_t variant, int64_t m) {
  TORCH_CHECK(variant >= 0 && variant <= 3,
              "variant must be 0(auto), 1(dense), 2(4-warp small-M), or 3(8-warp small-M)");
  TORCH_CHECK(variant != 0 || m <= 4,
              "weight-only auto dispatch supports M in [1,4]; got M=", m);
}

bool w4_auto_linear_supported(MatrixShape const& shape) {
  const int64_t elements = shape.n * shape.k;
  if (shape.m == 1) return shape.n >= 4096;
  return shape.m == 2 && shape.n >= 2048 && shape.k >= 2048 &&
         shape.k <= 8192 && elements >= 8 * 1024 * 1024;
}

bool w8_auto_linear_supported(MatrixShape const& shape) {
  if (shape.k <= 1024) return true;
  if (shape.k <= 4096) return shape.n >= 1024;
  return shape.n >= (shape.m <= 2 ? 1024 : 2048);
}

void check_w4_auto_linear(MatrixShape const& shape, int64_t variant) {
  TORCH_CHECK(variant != 0 || w4_auto_linear_supported(shape),
              "W4A16 auto dispatch has no qualified fast path for M=", shape.m,
              ", N=", shape.n, ", K=", shape.k,
              "; use W8A16 or an explicit diagnostic variant");
}

void check_w8_auto_linear(MatrixShape const& shape, int64_t variant) {
  TORCH_CHECK(variant != 0 || w8_auto_linear_supported(shape),
              "W8A16 auto dispatch has no qualified fast path for M=", shape.m,
              ", N=", shape.n, ", K=", shape.k,
              "; use an explicit diagnostic variant for analysis");
}

MatrixShape check_w4_linear(torch::Tensor const& x,
                            torch::Tensor const& packed,
                            torch::Tensor const& sfb,
                            torch::Tensor const& out) {
  check_bf16(x, "x");
  check_u8(packed, "packed");
  check_u8(sfb, "sfb");
  check_bf16(out, "out");
  TORCH_CHECK(x.dim() == 2, "x must have shape (M,K)");
  TORCH_CHECK(packed.dim() == 2, "packed must have shape (N,K/2)");
  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t n = packed.size(0);
  TORCH_CHECK(k % 64 == 0, "K must be divisible by 64");
  TORCH_CHECK(packed.size(1) == k / 2, "packed must have shape (N,K/2)");
  TORCH_CHECK(sfb.numel() >= sfb_bytes(n, k), "sfb is too small");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({m, n}),
              "out must have shape (M,N)");
  check_same_device(x, packed, "packed");
  check_same_device(x, sfb, "sfb");
  check_same_device(x, out, "out");
  return {m, n, k};
}

MatrixShape check_w8_linear(torch::Tensor const& x,
                            torch::Tensor const& quantized,
                            torch::Tensor const& scales,
                            torch::Tensor const& out) {
  check_bf16(x, "x");
  check_i8(quantized, "quantized");
  check_f32(scales, "scales");
  check_bf16(out, "out");
  TORCH_CHECK(x.dim() == 2, "x must have shape (M,K)");
  TORCH_CHECK(quantized.dim() == 2, "quantized must have shape (N,K)");
  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t n = quantized.size(0);
  TORCH_CHECK(k % 64 == 0, "K must be divisible by 64");
  TORCH_CHECK(quantized.size(1) == k, "quantized must have shape (N,K)");
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({n}), "scales must have shape (N,)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({m, n}), "out must have shape (M,N)");
  check_same_device(x, quantized, "quantized");
  check_same_device(x, scales, "scales");
  check_same_device(x, out, "out");
  return {m, n, k};
}

#if defined(CUDA_KERNEL)
cudaStream_t current_stream(torch::Tensor const& tensor) {
  return at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}
#endif

}  // namespace

void quantize_w4_weight_bf16(torch::Tensor const& weight,
                             torch::Tensor& packed,
                             torch::Tensor& sfb) {
  check_bf16(weight, "weight");
  check_u8(packed, "packed");
  check_u8(sfb, "sfb");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape (N,K)");
  const int64_t n = weight.size(0);
  const int64_t k = weight.size(1);
  TORCH_CHECK(k % 64 == 0, "K must be divisible by 64");
  TORCH_CHECK(packed.sizes() == torch::IntArrayRef({n, k / 2}),
              "packed must have shape (N,K/2)");
  TORCH_CHECK(sfb.numel() >= sfb_bytes(n, k), "sfb is too small");
  check_same_device(weight, packed, "packed");
  check_same_device(weight, sfb, "sfb");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(weight.device());
  const int rc = flashrt_weight_only::quantize_w4_weight_bf16(
      weight.data_ptr(), packed.data_ptr(), sfb.data_ptr(),
      as_int(n, "N"), as_int(k, "K"), current_stream(weight));
  TORCH_CHECK(rc == 0, "quantize_w4_weight_bf16 failed with rc=", rc);
#endif
}

void dequantize_w4_weight_bf16(torch::Tensor const& packed,
                               torch::Tensor const& sfb,
                               torch::Tensor& weight) {
  check_u8(packed, "packed");
  check_u8(sfb, "sfb");
  check_bf16(weight, "weight");
  TORCH_CHECK(weight.dim() == 2 && packed.dim() == 2,
              "weight and packed must be matrices");
  const int64_t n = weight.size(0);
  const int64_t k = weight.size(1);
  TORCH_CHECK(packed.sizes() == torch::IntArrayRef({n, k / 2}), "packed shape mismatch");
  TORCH_CHECK(sfb.numel() >= sfb_bytes(n, k), "sfb is too small");
  check_same_device(weight, packed, "packed");
  check_same_device(weight, sfb, "sfb");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(weight.device());
  const int rc = flashrt_weight_only::dequantize_w4_weight_bf16(
      packed.data_ptr(), sfb.data_ptr(), weight.data_ptr(),
      as_int(n, "N"), as_int(k, "K"), current_stream(weight));
  TORCH_CHECK(rc == 0, "dequantize_w4_weight_bf16 failed with rc=", rc);
#endif
}

void w4a16_linear_bf16(torch::Tensor const& x,
                       torch::Tensor const& packed,
                       torch::Tensor const& sfb,
                       double alpha,
                       int64_t variant,
                       torch::Tensor& out) {
  const auto shape = check_w4_linear(x, packed, sfb, out);
  check_variant(variant, shape.m);
  check_w4_auto_linear(shape, variant);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  const int rc = flashrt_weight_only::w4a16_linear_bf16(
      x.data_ptr(), packed.data_ptr(), sfb.data_ptr(), out.data_ptr(),
      as_int(shape.m, "M"), as_int(shape.n, "N"), as_int(shape.k, "K"),
      static_cast<float>(alpha), static_cast<int>(variant), current_stream(x));
  TORCH_CHECK(rc == 0, "w4a16_linear_bf16 failed with rc=", rc);
#endif
}

void quantize_w8_weight_bf16(torch::Tensor const& weight,
                             torch::Tensor& quantized,
                             torch::Tensor& scales) {
  check_bf16(weight, "weight");
  check_i8(quantized, "quantized");
  check_f32(scales, "scales");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape (N,K)");
  const int64_t n = weight.size(0);
  const int64_t k = weight.size(1);
  TORCH_CHECK(k % 64 == 0, "K must be divisible by 64");
  TORCH_CHECK(quantized.sizes() == weight.sizes(), "quantized shape mismatch");
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({n}), "scales must have shape (N,)");
  check_same_device(weight, quantized, "quantized");
  check_same_device(weight, scales, "scales");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(weight.device());
  const int rc = flashrt_weight_only::quantize_w8_weight_bf16(
      weight.data_ptr(), quantized.data_ptr(), scales.data_ptr(),
      as_int(n, "N"), as_int(k, "K"), current_stream(weight));
  TORCH_CHECK(rc == 0, "quantize_w8_weight_bf16 failed with rc=", rc);
#endif
}

void dequantize_w8_weight_bf16(torch::Tensor const& quantized,
                               torch::Tensor const& scales,
                               torch::Tensor& weight) {
  check_i8(quantized, "quantized");
  check_f32(scales, "scales");
  check_bf16(weight, "weight");
  TORCH_CHECK(quantized.dim() == 2 && weight.sizes() == quantized.sizes(),
              "weight must match quantized shape");
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({quantized.size(0)}),
              "scales must have shape (N,)");
  check_same_device(weight, quantized, "quantized");
  check_same_device(weight, scales, "scales");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(weight.device());
  const int rc = flashrt_weight_only::dequantize_w8_weight_bf16(
      quantized.data_ptr(), scales.data_ptr(), weight.data_ptr(),
      as_int(weight.size(0), "N"), as_int(weight.size(1), "K"), current_stream(weight));
  TORCH_CHECK(rc == 0, "dequantize_w8_weight_bf16 failed with rc=", rc);
#endif
}

void w8a16_linear_bf16(torch::Tensor const& x,
                       torch::Tensor const& quantized,
                       torch::Tensor const& scales,
                       int64_t variant,
                       torch::Tensor& out) {
  const auto shape = check_w8_linear(x, quantized, scales, out);
  check_variant(variant, shape.m);
  check_w8_auto_linear(shape, variant);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  const int rc = flashrt_weight_only::w8a16_linear_bf16(
      x.data_ptr(), quantized.data_ptr(), scales.data_ptr(), out.data_ptr(),
      as_int(shape.m, "M"), as_int(shape.n, "N"), as_int(shape.k, "K"),
      static_cast<int>(variant), current_stream(x));
  TORCH_CHECK(rc == 0, "w8a16_linear_bf16 failed with rc=", rc);
#endif
}

namespace {

template <bool W4>
void gated_ffn(torch::Tensor const& x,
               torch::Tensor const& gate_up_weight,
               torch::Tensor const& gate_up_scale,
               torch::Tensor const& down_weight,
               torch::Tensor const& down_scale,
               c10::optional<torch::Tensor> const& gate_up_bias,
               c10::optional<torch::Tensor> const& down_bias,
               bool gelu,
               double gate_up_alpha,
               double down_alpha,
               int64_t variant,
               torch::Tensor& gate_up,
               torch::Tensor& hidden,
               torch::Tensor& out) {
  const auto first = W4
      ? check_w4_linear(x, gate_up_weight, gate_up_scale, gate_up)
      : check_w8_linear(x, gate_up_weight, gate_up_scale, gate_up);
  check_variant(variant, first.m);
  TORCH_CHECK(first.n % 2 == 0, "gate_up output width must be even");
  const int64_t h = first.n / 2;
  TORCH_CHECK(hidden.sizes() == torch::IntArrayRef({first.m, h}),
              "hidden must have shape (M,H)");
  check_bf16(hidden, "hidden");
  check_same_device(x, hidden, "hidden");
  const void* gu_bias = optional_bf16_ptr(gate_up_bias, x, 2 * h, "gate_up_bias");
  const int64_t output_n = W4 ? down_weight.size(0) : down_weight.size(0);
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({first.m, output_n}), "out shape mismatch");
  const auto second = W4
      ? check_w4_linear(hidden, down_weight, down_scale, out)
      : check_w8_linear(hidden, down_weight, down_scale, out);
  TORCH_CHECK(second.k == h, "down weight K must equal hidden size");
  const void* dn_bias = optional_bf16_ptr(down_bias, x, second.n, "down_bias");
  const int64_t weight_elements = first.n * first.k + second.n * second.k;
  if constexpr (W4) {
    const int64_t minimum_elements =
        first.m == 1 ? 12 * 1024 * 1024
                     : (first.m == 2 ? 32 * 1024 * 1024 : 64 * 1024 * 1024);
    TORCH_CHECK(variant != 0 ||
                    (first.m <= 3 && weight_elements >= minimum_elements),
                "W4A16 gated FFN auto dispatch is not qualified for this shape; "
                "use W8A16 or an explicit diagnostic variant");
  } else {
    const int64_t minimum_elements =
        first.m == 4 ? 16 * 1024 * 1024 : 8 * 1024 * 1024;
    TORCH_CHECK(variant != 0 || weight_elements >= minimum_elements,
                "W8A16 gated FFN auto dispatch is not qualified for this shape; "
                "use an explicit diagnostic variant for analysis");
  }
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  const auto stream = current_stream(x);
  int rc;
  if constexpr (W4) {
    rc = flashrt_weight_only::w4a16_linear_bf16(
        x.data_ptr(), gate_up_weight.data_ptr(), gate_up_scale.data_ptr(), gate_up.data_ptr(),
        as_int(first.m, "M"), as_int(first.n, "2H"), as_int(first.k, "K"),
        static_cast<float>(gate_up_alpha), static_cast<int>(variant), stream);
  } else {
    rc = flashrt_weight_only::w8a16_linear_bf16(
        x.data_ptr(), gate_up_weight.data_ptr(), gate_up_scale.data_ptr(), gate_up.data_ptr(),
        as_int(first.m, "M"), as_int(first.n, "2H"), as_int(first.k, "K"),
        static_cast<int>(variant), stream);
  }
  TORCH_CHECK(rc == 0, "gate/up projection failed with rc=", rc);
  flashrt_weight_only::gated_activation_bf16(
      gate_up.data_ptr(), gu_bias, hidden.data_ptr(),
      as_int(first.m, "M"), as_int(h, "H"), gelu, stream);
  if constexpr (W4) {
    rc = flashrt_weight_only::w4a16_linear_bf16(
        hidden.data_ptr(), down_weight.data_ptr(), down_scale.data_ptr(), out.data_ptr(),
        as_int(second.m, "M"), as_int(second.n, "N"), as_int(second.k, "H"),
        static_cast<float>(down_alpha), static_cast<int>(variant), stream);
  } else {
    rc = flashrt_weight_only::w8a16_linear_bf16(
        hidden.data_ptr(), down_weight.data_ptr(), down_scale.data_ptr(), out.data_ptr(),
        as_int(second.m, "M"), as_int(second.n, "N"), as_int(second.k, "H"),
        static_cast<int>(variant), stream);
  }
  TORCH_CHECK(rc == 0, "down projection failed with rc=", rc);
  flashrt_weight_only::add_bias_bf16(out.data_ptr(), dn_bias,
      as_int(second.m, "M"), as_int(second.n, "N"), stream);
  const auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "gated FFN launch failed: ", cudaGetErrorString(error));
#endif
}

template <bool W4>
void gelu_ffn(torch::Tensor const& x,
              torch::Tensor const& up_weight,
              torch::Tensor const& up_scale,
              torch::Tensor const& down_weight,
              torch::Tensor const& down_scale,
              c10::optional<torch::Tensor> const& up_bias,
              c10::optional<torch::Tensor> const& down_bias,
              double up_alpha,
              double down_alpha,
              int64_t variant,
              torch::Tensor& up,
              torch::Tensor& hidden,
              torch::Tensor& out) {
  const auto first = W4
      ? check_w4_linear(x, up_weight, up_scale, up)
      : check_w8_linear(x, up_weight, up_scale, up);
  check_variant(variant, first.m);
  TORCH_CHECK(hidden.sizes() == up.sizes(), "hidden must match up shape");
  check_bf16(hidden, "hidden");
  check_same_device(x, hidden, "hidden");
  const void* up_bias_ptr = optional_bf16_ptr(up_bias, x, first.n, "up_bias");
  const int64_t output_n = down_weight.size(0);
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({first.m, output_n}), "out shape mismatch");
  const auto second = W4
      ? check_w4_linear(hidden, down_weight, down_scale, out)
      : check_w8_linear(hidden, down_weight, down_scale, out);
  TORCH_CHECK(second.k == first.n, "down weight K must equal hidden size");
  const void* down_bias_ptr = optional_bf16_ptr(down_bias, x, second.n, "down_bias");
  const int64_t weight_elements = first.n * first.k + second.n * second.k;
  if constexpr (W4) {
    TORCH_CHECK(variant != 0 || (first.m <= 3 && weight_elements >= 64 * 1024 * 1024),
                "W4A16 GELU FFN auto dispatch is not qualified for this shape; "
                "use W8A16 or an explicit diagnostic variant");
  } else {
    const int64_t minimum_elements =
        first.m == 1 ? 8 * 1024 * 1024 : 16 * 1024 * 1024;
    TORCH_CHECK(variant != 0 || weight_elements >= minimum_elements,
                "W8A16 GELU FFN auto dispatch is not qualified for this shape; "
                "use an explicit diagnostic variant for analysis");
  }
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  const auto stream = current_stream(x);
  int rc;
  if constexpr (W4) {
    rc = flashrt_weight_only::w4a16_linear_bf16(
        x.data_ptr(), up_weight.data_ptr(), up_scale.data_ptr(), up.data_ptr(),
        as_int(first.m, "M"), as_int(first.n, "H"), as_int(first.k, "K"),
        static_cast<float>(up_alpha), static_cast<int>(variant), stream);
  } else {
    rc = flashrt_weight_only::w8a16_linear_bf16(
        x.data_ptr(), up_weight.data_ptr(), up_scale.data_ptr(), up.data_ptr(),
        as_int(first.m, "M"), as_int(first.n, "H"), as_int(first.k, "K"),
        static_cast<int>(variant), stream);
  }
  TORCH_CHECK(rc == 0, "up projection failed with rc=", rc);
  flashrt_weight_only::gelu_activation_bf16(
      up.data_ptr(), up_bias_ptr, hidden.data_ptr(),
      as_int(first.m, "M"), as_int(first.n, "H"), stream);
  if constexpr (W4) {
    rc = flashrt_weight_only::w4a16_linear_bf16(
        hidden.data_ptr(), down_weight.data_ptr(), down_scale.data_ptr(), out.data_ptr(),
        as_int(second.m, "M"), as_int(second.n, "N"), as_int(second.k, "H"),
        static_cast<float>(down_alpha), static_cast<int>(variant), stream);
  } else {
    rc = flashrt_weight_only::w8a16_linear_bf16(
        hidden.data_ptr(), down_weight.data_ptr(), down_scale.data_ptr(), out.data_ptr(),
        as_int(second.m, "M"), as_int(second.n, "N"), as_int(second.k, "H"),
        static_cast<int>(variant), stream);
  }
  TORCH_CHECK(rc == 0, "down projection failed with rc=", rc);
  flashrt_weight_only::add_bias_bf16(out.data_ptr(), down_bias_ptr,
      as_int(second.m, "M"), as_int(second.n, "N"), stream);
  const auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "GELU FFN launch failed: ", cudaGetErrorString(error));
#endif
}

}  // namespace

void w4a16_gated_ffn_bf16(torch::Tensor const& x, torch::Tensor const& gu_w,
    torch::Tensor const& gu_s, torch::Tensor const& dn_w, torch::Tensor const& dn_s,
    c10::optional<torch::Tensor> const& gu_b, c10::optional<torch::Tensor> const& dn_b,
    bool gelu, double gu_alpha, double dn_alpha, int64_t variant,
    torch::Tensor& gu, torch::Tensor& hidden, torch::Tensor& out) {
  gated_ffn<true>(x, gu_w, gu_s, dn_w, dn_s, gu_b, dn_b, gelu,
                   gu_alpha, dn_alpha, variant, gu, hidden, out);
}

void w8a16_gated_ffn_bf16(torch::Tensor const& x, torch::Tensor const& gu_w,
    torch::Tensor const& gu_s, torch::Tensor const& dn_w, torch::Tensor const& dn_s,
    c10::optional<torch::Tensor> const& gu_b, c10::optional<torch::Tensor> const& dn_b,
    bool gelu, int64_t variant, torch::Tensor& gu, torch::Tensor& hidden,
    torch::Tensor& out) {
  gated_ffn<false>(x, gu_w, gu_s, dn_w, dn_s, gu_b, dn_b, gelu,
                    1.0, 1.0, variant, gu, hidden, out);
}

void w4a16_gelu_ffn_bf16(torch::Tensor const& x, torch::Tensor const& up_w,
    torch::Tensor const& up_s, torch::Tensor const& dn_w, torch::Tensor const& dn_s,
    c10::optional<torch::Tensor> const& up_b, c10::optional<torch::Tensor> const& dn_b,
    double up_alpha, double dn_alpha, int64_t variant,
    torch::Tensor& up, torch::Tensor& hidden, torch::Tensor& out) {
  gelu_ffn<true>(x, up_w, up_s, dn_w, dn_s, up_b, dn_b,
                  up_alpha, dn_alpha, variant, up, hidden, out);
}

void w8a16_gelu_ffn_bf16(torch::Tensor const& x, torch::Tensor const& up_w,
    torch::Tensor const& up_s, torch::Tensor const& dn_w, torch::Tensor const& dn_s,
    c10::optional<torch::Tensor> const& up_b, c10::optional<torch::Tensor> const& dn_b,
    int64_t variant, torch::Tensor& up, torch::Tensor& hidden, torch::Tensor& out) {
  gelu_ffn<false>(x, up_w, up_s, dn_w, dn_s, up_b, dn_b,
                   1.0, 1.0, variant, up, hidden, out);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("quantize_w4_weight_bf16(Tensor weight, Tensor! packed, Tensor! sfb) -> ()");
  ops.def("dequantize_w4_weight_bf16(Tensor packed, Tensor sfb, Tensor! weight) -> ()");
  ops.def("w4a16_linear_bf16(Tensor x, Tensor packed, Tensor sfb, float alpha, int variant, Tensor! out) -> ()");
  ops.def("quantize_w8_weight_bf16(Tensor weight, Tensor! quantized, Tensor! scales) -> ()");
  ops.def("dequantize_w8_weight_bf16(Tensor quantized, Tensor scales, Tensor! weight) -> ()");
  ops.def("w8a16_linear_bf16(Tensor x, Tensor quantized, Tensor scales, int variant, Tensor! out) -> ()");
  ops.def("w4a16_gated_ffn_bf16(Tensor x, Tensor gu_w, Tensor gu_s, Tensor dn_w, Tensor dn_s, Tensor? gu_b, Tensor? dn_b, bool gelu, float gu_alpha, float dn_alpha, int variant, Tensor! gu, Tensor! hidden, Tensor! out) -> ()");
  ops.def("w8a16_gated_ffn_bf16(Tensor x, Tensor gu_w, Tensor gu_s, Tensor dn_w, Tensor dn_s, Tensor? gu_b, Tensor? dn_b, bool gelu, int variant, Tensor! gu, Tensor! hidden, Tensor! out) -> ()");
  ops.def("w4a16_gelu_ffn_bf16(Tensor x, Tensor up_w, Tensor up_s, Tensor dn_w, Tensor dn_s, Tensor? up_b, Tensor? dn_b, float up_alpha, float dn_alpha, int variant, Tensor! up, Tensor! hidden, Tensor! out) -> ()");
  ops.def("w8a16_gelu_ffn_bf16(Tensor x, Tensor up_w, Tensor up_s, Tensor dn_w, Tensor dn_s, Tensor? up_b, Tensor? dn_b, int variant, Tensor! up, Tensor! hidden, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("quantize_w4_weight_bf16", torch::kCUDA, &quantize_w4_weight_bf16);
  ops.impl("dequantize_w4_weight_bf16", torch::kCUDA, &dequantize_w4_weight_bf16);
  ops.impl("w4a16_linear_bf16", torch::kCUDA, &w4a16_linear_bf16);
  ops.impl("quantize_w8_weight_bf16", torch::kCUDA, &quantize_w8_weight_bf16);
  ops.impl("dequantize_w8_weight_bf16", torch::kCUDA, &dequantize_w8_weight_bf16);
  ops.impl("w8a16_linear_bf16", torch::kCUDA, &w8a16_linear_bf16);
  ops.impl("w4a16_gated_ffn_bf16", torch::kCUDA, &w4a16_gated_ffn_bf16);
  ops.impl("w8a16_gated_ffn_bf16", torch::kCUDA, &w8a16_gated_ffn_bf16);
  ops.impl("w4a16_gelu_ffn_bf16", torch::kCUDA, &w4a16_gelu_ffn_bf16);
  ops.impl("w8a16_gelu_ffn_bf16", torch::kCUDA, &w8a16_gelu_ffn_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
