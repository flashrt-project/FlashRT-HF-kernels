#include "gemm/fp4/sm110_dispatch.cuh"

#include "gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm100.cuh"

namespace flash_rt::hub {
namespace {

void launch_sm110(
    const void* a,
    const void* b,
    void* out,
    int m,
    int n,
    int k,
    const void* sfa,
    const void* sfb,
    float alpha,
    int variant,
    cudaStream_t stream) {
  if (variant == 1) {
    gemm::fp4_w4a16_gemm_sm100_bf16out_widen(
        a, b, out, m, n, k, sfa, sfb, alpha, stream);
  } else if (variant == 2) {
    gemm::fp4_w4a16_gemm_sm100_bf16out_pingpong(
        a, b, out, m, n, k, sfa, sfb, alpha, stream);
  } else {
    gemm::fp4_w4a16_gemm_sm100_bf16out(
        a, b, out, m, n, k, sfa, sfb, alpha, stream);
  }
}

struct Sm110DispatchRegistration {
  Sm110DispatchRegistration() { sm110_gemm_dispatch = &launch_sm110; }
};

Sm110DispatchRegistration registration;

}  // namespace
}  // namespace flash_rt::hub
