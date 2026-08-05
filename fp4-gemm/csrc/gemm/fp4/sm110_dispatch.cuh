#pragma once

#include <cuda_runtime_api.h>

namespace flash_rt::hub {

using Sm110GemmDispatch = void (*)(
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
    cudaStream_t stream);

using Sm110GemmBiasDispatch = int (*)(
    const void* a, const void* sfa, const void* b, const void* sfb,
    const void* bias, void* out, int m, int n, int k,
    cudaStream_t stream);

using Sm110GemmBiasResidualDispatch = int (*)(
    const void* a, const void* sfa, const void* b, const void* sfb,
    const void* bias, const void* residual, void* out,
    int m, int n, int k, cudaStream_t stream);

using Sm110GemmBiasGeluFp4Dispatch = int (*)(
    const void* a, const void* sfa, const void* b, const void* sfb,
    const void* bias, void* out_packed, void* out_sfa,
    int m, int n, int k, cudaStream_t stream);

using Sm110QuantizeBf16Dispatch = int (*)(
    const void* x, void* packed, void* sfa, int rows, int dim,
    bool is_sfb, cudaStream_t stream);

using Sm110GemmFp16Dispatch = int (*)(
    int variant, const void* a, const void* sfa, const void* b,
    const void* sfb, void* out, int m, int n, int k, float alpha,
    float beta, cudaStream_t stream);

using Sm110GemmGegluFp4Dispatch = int (*)(
    const void* a, const void* sfa, const void* b, const void* sfb,
    void* scratch, void* out_packed, void* out_sfa, int m, int n_twice,
    int k, bool skinny, cudaStream_t stream);

using Sm110GemmBiasGeluFp4Fp16Dispatch = int (*)(
    const void* a, const void* sfa, const void* b, const void* sfb,
    const void* bias, void* out_packed, void* out_sfa, int m, int n,
    int k, cudaStream_t stream);

using Sm110GemmBiasResidualFp16Dispatch = int (*)(
    const void* a, const void* sfa, const void* b, const void* sfb,
    const void* bias, const void* residual, void* out, int m, int n,
    int k, cudaStream_t stream);

extern Sm110GemmDispatch sm110_gemm_dispatch;
extern Sm110GemmBiasDispatch sm110_gemm_bias_dispatch;
extern Sm110GemmBiasResidualDispatch sm110_gemm_bias_residual_dispatch;
extern Sm110GemmBiasGeluFp4Dispatch sm110_gemm_bias_gelu_fp4_dispatch;
extern Sm110QuantizeBf16Dispatch sm110_quantize_bf16_dispatch;
extern Sm110GemmFp16Dispatch sm110_gemm_fp16_dispatch;
extern Sm110GemmGegluFp4Dispatch sm110_gemm_geglu_fp4_dispatch;
extern Sm110GemmBiasGeluFp4Fp16Dispatch
    sm110_gemm_bias_gelu_fp4_fp16_dispatch;
extern Sm110GemmBiasResidualFp16Dispatch
    sm110_gemm_bias_residual_fp16_dispatch;

}  // namespace flash_rt::hub
