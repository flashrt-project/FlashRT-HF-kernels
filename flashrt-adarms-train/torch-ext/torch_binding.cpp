// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/library.h>

#include "adarms_train.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

using flashrt_hub::adarms_train::ModView;

void check_io(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.dim() == 3, name, " must be (B, T, H)");
  TORCH_CHECK(
      t.scalar_type() == torch::kBFloat16 || t.scalar_type() == torch::kFloat32,
      name, " must be bf16 or fp32");
}

void check_h(const torch::Tensor& x) {
  const long h = x.size(-1);
  const long lanes = x.scalar_type() == torch::kBFloat16 ? 8 : 4;
  TORCH_CHECK(h % lanes == 0, "H must be a multiple of ", lanes);
  TORCH_CHECK(h <= 256 * lanes * 4, "H too large for the register-cached kernel");
}

ModView mod_view(const c10::optional<torch::Tensor>& m, const torch::Tensor& x,
                 const char* name) {
  if (!m.has_value()) return ModView{nullptr, 0, 0};
  const torch::Tensor& t = m.value();
  TORCH_CHECK(t.is_cuda() && t.dim() == 3, name, " must be CUDA (B, 1|T, H)");
  TORCH_CHECK(t.scalar_type() == x.scalar_type() ||
                  t.scalar_type() == torch::kFloat32,
              name, " dtype must match x or be fp32");
  TORCH_CHECK(t.stride(2) == 1, name, " last dim must be contiguous");
  TORCH_CHECK(t.size(0) == x.size(0) && t.size(2) == x.size(2), name,
              " shape mismatch");
  TORCH_CHECK(t.size(1) == 1 || t.size(1) == x.size(1), name,
              " token dim must be 1 or T");
  const long token_stride = t.size(1) == 1 ? 0 : t.stride(1);
  return ModView{t.data_ptr(), t.stride(0), token_stride};
}

const void* weight_ptr(const c10::optional<torch::Tensor>& w,
                       const torch::Tensor& x) {
  if (!w.has_value()) return nullptr;
  const torch::Tensor& t = w.value();
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dim() == 1 &&
                  t.size(0) == x.size(-1) &&
                  (t.scalar_type() == x.scalar_type() ||
                   t.scalar_type() == torch::kFloat32),
              "weight must be contiguous CUDA (H,) in x dtype or fp32");
  return t.data_ptr();
}

void check_mode(const c10::optional<torch::Tensor>& scale,
                const c10::optional<torch::Tensor>& weight) {
  TORCH_CHECK(scale.has_value() != weight.has_value(),
              "exactly one of scale/shift or weight is required");
}

torch::ScalarType mod_dtype(const c10::optional<torch::Tensor>& scale,
                            const c10::optional<torch::Tensor>& weight) {
  return scale.has_value() ? scale.value().scalar_type()
                           : weight.value().scalar_type();
}

bool mod_is_fp32_mixed(const torch::Tensor& x,
                       const c10::optional<torch::Tensor>& scale,
                       const c10::optional<torch::Tensor>& weight) {
  return x.scalar_type() == torch::kBFloat16 &&
         mod_dtype(scale, weight) == torch::kFloat32;
}

}  // namespace

torch::Tensor flashrt_training_package_marker(torch::Tensor x) { return x; }

std::tuple<torch::Tensor, torch::Tensor> adarms_fwd(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& scale,
    const c10::optional<torch::Tensor>& shift,
    const c10::optional<torch::Tensor>& weight, double eps) {
  check_io(x, "x");
  check_h(x);
  TORCH_CHECK(scale.has_value() == shift.has_value(),
              "scale and shift must be provided together");
  check_mode(scale, weight);
  auto y = torch::empty_like(x);
  auto rstd =
      torch::empty({x.size(0) * x.size(1)}, x.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  flashrt_hub::adarms_train::adarms_fwd_launch(
      x.data_ptr(), mod_view(scale, x, "scale"), mod_view(shift, x, "shift"),
      weight_ptr(weight, x), y.data_ptr(), rstd.data_ptr<float>(),
      (int)(x.size(0) * x.size(1)), (int)x.size(1), (int)x.size(2), (float)eps,
      x.scalar_type() == torch::kBFloat16, mod_is_fp32_mixed(x, scale, weight),
      stream.stream());
  return {y, rstd};
}

std::tuple<torch::Tensor, torch::Tensor> adarms_bwd(
    const torch::Tensor& dy, const torch::Tensor& x,
    const c10::optional<torch::Tensor>& scale,
    const c10::optional<torch::Tensor>& weight, const torch::Tensor& rstd) {
  check_io(dy, "dy");
  check_io(x, "x");
  check_mode(scale, weight);
  auto dx = torch::empty_like(x);
  const int rows = (int)(x.size(0) * x.size(1));
  // Adaptive: per-row dscale elements. Weight mode: per-CTA fp32 partial
  // weight-grad rows, summed below.
  const int partial_rows = flashrt_hub::adarms_train::bwd_weight_grid(rows);
  auto dmod_elem =
      scale.has_value()
          ? torch::empty_like(x, x.options().dtype(mod_dtype(scale, weight)))
          : torch::empty({partial_rows, x.size(-1)},
                         x.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  flashrt_hub::adarms_train::adarms_bwd_launch(
      dy.data_ptr(), x.data_ptr(), mod_view(scale, x, "scale"),
      weight_ptr(weight, x), rstd.data_ptr<float>(), dx.data_ptr(),
      dmod_elem.data_ptr(), rows, (int)x.size(1), (int)x.size(2),
      x.scalar_type() == torch::kBFloat16, mod_is_fp32_mixed(x, scale, weight),
      stream.stream());
  torch::Tensor dmod =
      scale.has_value() ? dmod_elem : dmod_elem.sum(0);
  return {dx, dmod};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> resgate_adarms_fwd(
    const torch::Tensor& x, const torch::Tensor& h,
    const c10::optional<torch::Tensor>& gate,
    const c10::optional<torch::Tensor>& scale,
    const c10::optional<torch::Tensor>& shift,
    const c10::optional<torch::Tensor>& weight, double eps) {
  check_io(x, "x");
  check_io(h, "h");
  check_h(x);
  TORCH_CHECK(scale.has_value() == shift.has_value(),
              "scale and shift must be provided together");
  check_mode(scale, weight);
  if (gate.has_value()) check_io(gate.value(), "gate");
  auto r = torch::empty_like(x);
  auto y = torch::empty_like(x);
  auto rstd =
      torch::empty({x.size(0) * x.size(1)}, x.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  flashrt_hub::adarms_train::resgate_adarms_fwd_launch(
      x.data_ptr(), h.data_ptr(),
      gate.has_value() ? gate.value().data_ptr() : nullptr,
      mod_view(scale, x, "scale"), mod_view(shift, x, "shift"),
      weight_ptr(weight, x), r.data_ptr(), y.data_ptr(), rstd.data_ptr<float>(),
      (int)(x.size(0) * x.size(1)), (int)x.size(1), (int)x.size(2), (float)eps,
      x.scalar_type() == torch::kBFloat16, mod_is_fp32_mixed(x, scale, weight),
      stream.stream());
  return {r, y, rstd};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
resgate_adarms_bwd(const torch::Tensor& dy,
                   const c10::optional<torch::Tensor>& dyr,
                   const torch::Tensor& r, const torch::Tensor& h,
                   const c10::optional<torch::Tensor>& gate,
                   const c10::optional<torch::Tensor>& scale,
                   const c10::optional<torch::Tensor>& weight,
                   const torch::Tensor& rstd) {
  check_io(dy, "dy");
  check_io(r, "r");
  check_io(h, "h");
  check_mode(scale, weight);
  if (dyr.has_value()) check_io(dyr.value(), "dyr");
  if (gate.has_value()) check_io(gate.value(), "gate");
  auto dr = torch::empty_like(r);
  auto dh = torch::empty_like(r);
  auto dg = gate.has_value() ? torch::empty_like(r)
                             : torch::empty({0}, r.options());
  const int rows = (int)(r.size(0) * r.size(1));
  const int partial_rows = flashrt_hub::adarms_train::bwd_weight_grid(rows);
  auto dmod_elem =
      scale.has_value()
          ? torch::empty_like(r, r.options().dtype(mod_dtype(scale, weight)))
          : torch::empty({partial_rows, r.size(-1)},
                         r.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  flashrt_hub::adarms_train::resgate_adarms_bwd_launch(
      dy.data_ptr(), dyr.has_value() ? dyr.value().data_ptr() : nullptr,
      r.data_ptr(), h.data_ptr(),
      gate.has_value() ? gate.value().data_ptr() : nullptr,
      mod_view(scale, r, "scale"), weight_ptr(weight, r),
      rstd.data_ptr<float>(), dr.data_ptr(), dh.data_ptr(),
      gate.has_value() ? dg.data_ptr() : nullptr, dmod_elem.data_ptr(), rows,
      (int)r.size(1), (int)r.size(2), r.scalar_type() == torch::kBFloat16,
      mod_is_fp32_mixed(r, scale, weight), stream.stream());
  torch::Tensor dmod =
      scale.has_value() ? dmod_elem : dmod_elem.sum(0);
  return {dr, dh, dg, dmod};
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("_flashrt_training_package_marker(Tensor x) -> Tensor");
  ops.impl("_flashrt_training_package_marker",
           c10::DispatchKey::CompositeExplicitAutograd,
           &flashrt_training_package_marker);
  ops.def(
      "adarms_fwd(Tensor x, Tensor? scale, Tensor? shift, Tensor? weight, "
      "float eps) -> (Tensor, Tensor)");
  ops.impl("adarms_fwd", torch::kCUDA, &adarms_fwd);
  ops.def(
      "adarms_bwd(Tensor dy, Tensor x, Tensor? scale, Tensor? weight, "
      "Tensor rstd) -> (Tensor, Tensor)");
  ops.impl("adarms_bwd", torch::kCUDA, &adarms_bwd);
  ops.def(
      "resgate_adarms_fwd(Tensor x, Tensor h, Tensor? gate, Tensor? scale, "
      "Tensor? shift, Tensor? weight, float eps) -> (Tensor, Tensor, Tensor)");
  ops.impl("resgate_adarms_fwd", torch::kCUDA, &resgate_adarms_fwd);
  ops.def(
      "resgate_adarms_bwd(Tensor dy, Tensor? dyr, Tensor r, Tensor h, "
      "Tensor? gate, Tensor? scale, Tensor? weight, Tensor rstd) -> "
      "(Tensor, Tensor, Tensor, Tensor)");
  ops.impl("resgate_adarms_bwd", torch::kCUDA, &resgate_adarms_bwd);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
