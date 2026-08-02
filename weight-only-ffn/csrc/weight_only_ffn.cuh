#pragma once

#include <cuda_runtime.h>

namespace flashrt_weight_only {

#if defined(FLASHRT_WEIGHT_ONLY_OPTIONAL_SYMBOLS)
#define FLASHRT_WEIGHT_ONLY_OPTIONAL __attribute__((weak))
#else
#define FLASHRT_WEIGHT_ONLY_OPTIONAL
#endif

FLASHRT_WEIGHT_ONLY_OPTIONAL int quantize_w4_weight_bf16(
    const void* weight, void* packed, void* sfb, int rows, int cols,
    cudaStream_t stream);
FLASHRT_WEIGHT_ONLY_OPTIONAL int dequantize_w4_weight_bf16(
    const void* packed, const void* sfb, void* weight, int rows, int cols,
    cudaStream_t stream);
FLASHRT_WEIGHT_ONLY_OPTIONAL int w4a16_linear_bf16(
    const void* x, const void* packed, const void* sfb, void* out, int m,
    int n, int k, float alpha, int variant, cudaStream_t stream);

FLASHRT_WEIGHT_ONLY_OPTIONAL int quantize_w8_weight_bf16(
    const void* weight, void* quantized, void* scales, int rows, int cols,
    cudaStream_t stream);
FLASHRT_WEIGHT_ONLY_OPTIONAL int dequantize_w8_weight_bf16(
    const void* quantized, const void* scales, void* weight, int rows, int cols,
    cudaStream_t stream);
FLASHRT_WEIGHT_ONLY_OPTIONAL int w8a16_linear_bf16(
    const void* x, const void* quantized, const void* scales, void* out, int m,
    int n, int k, int variant, cudaStream_t stream);

FLASHRT_WEIGHT_ONLY_OPTIONAL void gated_activation_bf16(
    const void* merged, const void* bias, void* hidden, int rows,
    int hidden_size, bool gelu, cudaStream_t stream);
FLASHRT_WEIGHT_ONLY_OPTIONAL void gelu_activation_bf16(
    const void* input, const void* bias, void* output, int rows,
    int hidden_size, cudaStream_t stream);
FLASHRT_WEIGHT_ONLY_OPTIONAL void add_bias_bf16(
    void* output, const void* bias, int rows, int cols, cudaStream_t stream);

#undef FLASHRT_WEIGHT_ONLY_OPTIONAL

}  // namespace flashrt_weight_only
