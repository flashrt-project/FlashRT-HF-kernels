// CUTLASS FP8 GEMM implementations for SM110a. The package dispatcher uses
// BF16-output Sq/T1/Wide variants. Weight B is column-major in the CUTLASS
// problem and is supplied by the public API as a contiguous [N, K] tensor.

#include "gemm_types_sm110.h"
#include "cutlass_sm110_fp8_gemm.cuh"
#include "cutlass/util/device_memory.h"
#include <cuda_runtime.h>
#include <cstdio>

// Generic runner: initialize and run on the caller's stream.
template <typename GemmOp>
static int cutlass_run_impl(void* A, void* B, void* D,
                             int M, int N, int K,
                             float alpha, float beta,
                             cudaStream_t stream) {
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementD = typename GemmOp::ElementD;

    // CUTLASS stride computation
    auto stride_A = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideD{}, {M, N, 1});

    typename GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},  // problem size
        {(ElementA*)A, stride_A, (ElementB*)B, stride_B},
        {{alpha, beta}, (ElementD*)D, stride_D, (ElementD*)D, stride_D}
    };

    GemmOp gemm;
    size_t ws_size = GemmOp::get_workspace_size(args);
    static cutlass::device_memory::allocation<uint8_t> workspace(0);
    if (ws_size > workspace.size()) {
        workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
    }

    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] cannot implement: M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }

    status = gemm.initialize(args, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] init failed: M=%d N=%d K=%d\n", M, N, K);
        return -2;
    }

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] run failed: M=%d N=%d K=%d\n", M, N, K);
        return -3;
    }
    return 0;
}

template <typename GemmOp>
static int cutlass_run_bias_impl(
    void* A, void* B, void* bias, void* D, int M, int N, int K,
    float alpha, float beta, cudaStream_t stream) {
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementC = typename GemmOp::ElementC;
    using ElementD = typename GemmOp::ElementD;
    using ElementBias = cutlass::bfloat16_t;

    auto stride_A = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideD{}, {M, N, 1});

    typename GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<ElementA*>(A), stride_A,
         reinterpret_cast<ElementB*>(B), stride_B},
        {{alpha, beta}, reinterpret_cast<ElementC*>(D), stride_C,
         reinterpret_cast<ElementD*>(D), stride_D}
    };
    args.epilogue.thread.bias_ptr =
        reinterpret_cast<ElementBias const*>(bias);

    GemmOp gemm;
    const size_t ws_size = GemmOp::get_workspace_size(args);
    static cutlass::device_memory::allocation<uint8_t> workspace(0);
    if (ws_size > workspace.size()) {
        workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
    }
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -11;
    status = gemm.initialize(args, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) return -12;
    status = gemm.run(stream);
    return status == cutlass::Status::kSuccess ? 0 : -13;
}

// Exported C functions.
extern "C" {

int cutlass_fp8_sq(void* A, void* B, void* D, int M, int N, int K,
                    float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_sq::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_t1(void* A, void* B, void* D, int M, int N, int K,
                    float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_t1::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_wide(void* A, void* B, void* D, int M, int N, int K,
                      float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_wide::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_plain(void* A, void* B, void* D, int M, int N, int K,
                       float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_plain::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_gelu(void* A, void* B, void* D, int M, int N, int K,
                      float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_gelu::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

// FP32 output variants for models with activations exceeding FP16 range.
int cutlass_fp8_sq_f32out(void* A, void* B, void* D, int M, int N, int K,
                           float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_sq_f32out::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_wide_f32out(void* A, void* B, void* D, int M, int N, int K,
                             float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_wide_f32out::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

// BF16 output variants
int cutlass_fp8_sq_bf16out(void* A, void* B, void* D, int M, int N, int K,
                            float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_sq_bf16out::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_wide_bf16out(void* A, void* B, void* D, int M, int N, int K,
                              float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_wide_bf16out::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_t1_bf16out(void* A, void* B, void* D, int M, int N, int K,
                            float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_t1_bf16out::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_wide_bias_bf16out(
    void* A, void* B, void* bias, void* D, int M, int N, int K,
    float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_bias_impl<sm100_wide_bias_bf16out::Gemm>(
        A, B, bias, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_wide_bias_gelu_bf16out(
    void* A, void* B, void* bias, void* D, int M, int N, int K,
    float alpha, cudaStream_t stream) {
    return cutlass_run_bias_impl<sm100_wide_bias_gelu_bf16out::Gemm>(
        A, B, bias, D, M, N, K, alpha, 0.0f, stream);
}

}  // extern "C"
