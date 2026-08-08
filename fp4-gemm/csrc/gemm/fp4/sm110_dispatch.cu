#include "gemm/fp4/sm110_dispatch.cuh"

#include "gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm100.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_bias_bf16_sm100.cuh"
#include "gemm/fp4/cutlass_fp4_gemm.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_bf16_variants_sm100.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_fp4out.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_geglu_il_sm100.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_siglip_ffn_sm100.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_e0m3w_sm100.cuh"
#include "gemm/fp4/cosmos3_edge_fp4_gemm_relu2_fp4out.cuh"
#include "quantize/quantize_fp4_sfa_bf16.cuh"

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

int launch_geglu_fp4_sm110(
    const void* a, const void* sfa, const void* b, const void* sfb,
    void* scratch, void* out_packed, void* out_sfa, int m, int n_twice,
    int k, bool skinny, cudaStream_t stream) {
  return skinny
      ? fp4::cutlass_fp4_gemm_geglu_il_hw_v10(
            a, sfa, b, sfb, scratch, out_packed, out_sfa, m, n_twice, k,
            stream)
      : fp4::cutlass_fp4_gemm_geglu_il_hw(
            a, sfa, b, sfb, scratch, out_packed, out_sfa, m, n_twice, k,
            stream);
}

struct Sm110DispatchRegistration {
  Sm110DispatchRegistration() {
    sm110_gemm_dispatch = &launch_sm110;
    sm110_gemm_bias_dispatch = &fp4::cutlass_fp4_gemm_bias_bf16;
    sm110_gemm_bias_residual_dispatch =
        &fp4::cutlass_fp4_gemm_bias_res_bf16;
    sm110_gemm_bias_gelu_fp4_dispatch =
        &fp4::cutlass_fp4_gemm_bias_gelu_fp4out_bf16;
    sm110_quantize_bf16_dispatch =
        &fp4::quantize_fp4_dynamic_sfa_bf16_vec;
    sm110_gemm_fp16_dispatch = &fp4::cutlass_fp4_gemm_variant;
    sm110_gemm_bf16_variant_dispatch =
        &fp4::cutlass_fp4_gemm_bf16_variant;
    sm110_gemm_fp4out_dispatch = &fp4::cutlass_fp4_gemm_fp4out;
    sm110_gemm_geglu_fp4_dispatch = &launch_geglu_fp4_sm110;
    sm110_gemm_bias_gelu_fp4_fp16_dispatch =
        &fp4::cutlass_fp4_gemm_bias_gelu_fp4out;
    sm110_gemm_bias_residual_fp16_dispatch =
        &fp4::cutlass_fp4_gemm_bias_res_fp16;
    sm110_e0m3_gemm_dispatch = &fp4::cutlass_fp4_gemm_e0m3w;
    sm110_gemm_relu2_fp4_dispatch =
        &fp4::cosmos3_edge_fp4_gemm_relu2_fp4out;
  }
};

Sm110DispatchRegistration registration;

}  // namespace
}  // namespace flash_rt::hub
