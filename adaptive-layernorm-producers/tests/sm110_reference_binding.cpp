#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "quantize_fp4_sfa_bf16.cuh"

void ada_layer_norm_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale,
    const __nv_bfloat16* shift, __nv_bfloat16* out,
    int seq_len, int dim, float eps, cudaStream_t stream);
void layer_norm_no_affine_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out, int seq_len, int dim, float eps,
    cudaStream_t stream);

namespace {

void reference_adaln_fp4(
    const torch::Tensor& x, const torch::Tensor& scale,
    const torch::Tensor& shift, double eps, torch::Tensor& packed,
    torch::Tensor& sf) {
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  auto norm = torch::empty_like(x);
  ada_layer_norm_bf16(
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<const __nv_bfloat16*>(scale.data_ptr()),
      static_cast<const __nv_bfloat16*>(shift.data_ptr()),
      static_cast<__nv_bfloat16*>(norm.data_ptr()),
      static_cast<int>(x.size(0)), static_cast<int>(x.size(1)),
      static_cast<float>(eps), stream);
  const int rc = flash_rt::fp4::quantize_fp4_dynamic_sfa_bf16_vec(
      norm.data_ptr(), packed.data_ptr(), sf.data_ptr(),
      static_cast<int>(x.size(0)), static_cast<int>(x.size(1)), false, stream);
  TORCH_CHECK(rc == 0, "SM110 staged AdaLN-FP4 reference failed: rc=", rc);
}

void reference_ln_fp4(
    const torch::Tensor& x, double eps, torch::Tensor& packed,
    torch::Tensor& sf) {
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  auto norm = torch::empty_like(x);
  layer_norm_no_affine_bf16(
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<__nv_bfloat16*>(norm.data_ptr()),
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1)), static_cast<float>(eps), stream);
  const int rc = flash_rt::fp4::quantize_fp4_dynamic_sfa_bf16_vec(
      norm.data_ptr(), packed.data_ptr(), sf.data_ptr(),
      static_cast<int>(x.size(0)), static_cast<int>(x.size(1)), false, stream);
  TORCH_CHECK(rc == 0, "SM110 staged LayerNorm-FP4 reference failed: rc=", rc);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(adaptive_layernorm_producers_test, ops) {
  ops.def("_reference_adaln_fp4(Tensor x, Tensor scale, Tensor shift, float eps, Tensor! packed, Tensor! sf) -> ()");
  ops.def("_reference_ln_fp4(Tensor x, float eps, Tensor! packed, Tensor! sf) -> ()");
  ops.impl("_reference_adaln_fp4", torch::kCUDA, &reference_adaln_fp4);
  ops.impl("_reference_ln_fp4", torch::kCUDA, &reference_ln_fp4);
}
