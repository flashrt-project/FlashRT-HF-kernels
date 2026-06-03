// SPDX-License-Identifier: Apache-2.0
//
// Tensor-facing FP8 FFN building blocks extracted from FlashRT serving paths.
// The GEMM descriptor convention mirrors official/FlashRT
// csrc/kernels/decoder_fused.cu::fp8_gemm_descale_bf16out:
//
//   out[M,N] = (input_fp8[M,K] * input_scale)
//            @ (weight_fp8[N,K] * weight_scale).T
//
// Both scales are CUDA float32 scalar tensors.

#include "fp8_ffn.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace flash_rt {
namespace fp8_ffn {
namespace {

constexpr float kFp8Max = 448.0f;

cublasLtHandle_t g_fp8_lt = nullptr;
void* g_fp8_ws = nullptr;
size_t g_fp8_ws_sz = 32 * 1024 * 1024;

struct LtGemmKey {
  int M;
  int N;
  int K;
  bool operator==(const LtGemmKey& o) const {
    return M == o.M && N == o.N && K == o.K;
  }
};

struct LtGemmKeyHash {
  size_t operator()(const LtGemmKey& k) const {
    size_t h = std::hash<int>()(k.M);
    h ^= std::hash<int>()(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct CachedLtGemm {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr;
  cublasLtMatrixLayout_t Bdesc = nullptr;
  cublasLtMatrixLayout_t Ddesc = nullptr;
  cublasLtMatmulAlgo_t algo{};
};

std::unordered_map<LtGemmKey, CachedLtGemm, LtGemmKeyHash> g_lt_cache;

std::string shape_string(const char* name, int M, int N, int K) {
  return std::string(name) + " [" + std::to_string(M) + "," +
         std::to_string(N) + "," + std::to_string(K) + "]";
}

void check_cublaslt(
    cublasStatus_t status,
    const char* name,
    int M,
    int N,
    int K,
    const char* op) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        shape_string(name, M, N, K) + ": " + op +
        " failed with cuBLAS status " +
        std::to_string(static_cast<int>(status)));
  }
}

void check_cuda(
    cudaError_t status,
    const char* name,
    int M,
    int N,
    int K,
    const char* op) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        shape_string(name, M, N, K) + ": " + op +
        " failed with CUDA error " + cudaGetErrorString(status));
  }
}

void ensure_fp8_lt(const char* name, int M, int N, int K) {
  if (!g_fp8_lt) {
    check_cublaslt(cublasLtCreate(&g_fp8_lt), name, M, N, K, "cublasLtCreate");
    check_cuda(cudaMalloc(&g_fp8_ws, g_fp8_ws_sz), name, M, N, K,
               "cudaMalloc workspace");
  }
}

void check_heuristic(
    cublasStatus_t status,
    int returned_results,
    const char* name,
    int M,
    int N,
    int K) {
  check_cublaslt(status, name, M, N, K, "cublasLtMatmulAlgoGetHeuristic");
  if (returned_results == 0) {
    throw std::runtime_error(
        shape_string(name, M, N, K) +
        ": cuBLASLt returned no FP8 GEMM algorithm");
  }
}

__device__ __forceinline__ float gelu_tanh(float x) {
  return 0.5f * x *
         (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void bias_gelu_quantize_fp8_static_bf16_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ bias,
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ scale,
    long long tiles_per_row,
    int N,
    int has_bias) {
  const long long tile = static_cast<long long>(blockIdx.x);
  const long long row = tile / tiles_per_row;
  const long long col_tile = tile - row * tiles_per_row;
  const int col = static_cast<int>(col_tile * blockDim.x + threadIdx.x);
  if (col >= N) {
    return;
  }

  const long long idx = row * static_cast<long long>(N) + col;
  float v = __bfloat162float(in[idx]);
  if (has_bias) {
    v += __bfloat162float(bias[col]);
  }
  float q = gelu_tanh(v) * (1.0f / *scale);
  q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
  out[idx] = __nv_fp8_e4m3(q);
}

__global__ void add_bias_bf16_kernel(
    __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ bias,
    long long total,
    int N) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) +
                        threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int col = static_cast<int>(idx % N);
  const float v = __bfloat162float(input[idx]) + __bfloat162float(bias[col]);
  input[idx] = __float2bfloat16(v);
}

int quant_block_size(long long M, int N, bool has_bias) {
  const char* value = std::getenv("FLASHRT_FP8_FFN_QUANT_BLOCK_SIZE");
  if (value != nullptr) {
    const int block_size = std::atoi(value);
    if (block_size == 128 || block_size == 256 || block_size == 512 ||
        block_size == 1024) {
      return block_size;
    }
  }
  if (N >= 12288) {
    return (has_bias && M <= 32) ? 512 : 256;
  }
  if (M == 1) {
    return has_bias ? 512 : 256;
  }
  if (M <= 2) {
    return has_bias ? 512 : 1024;
  }
  if (M <= 32) {
    return 1024;
  }
  return 256;
}

}  // namespace

void fp8_gemm_descale_bf16out(
    const void* input_fp8,
    const void* weight_fp8,
    void* out_bf16,
    int M,
    int N,
    int K,
    const float* input_scale,
    const float* weight_scale,
    cudaStream_t stream) {
  const char* name = "fp8_gemm_descale_bf16out";
  ensure_fp8_lt(name, M, N, K);

  LtGemmKey key{M, N, K};
  auto it = g_lt_cache.find(key);
  if (it == g_lt_cache.end()) {
    CachedLtGemm cg{};
    check_cublaslt(
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
        name, M, N, K, "cublasLtMatmulDescCreate");
    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasOperation_t opT = CUBLAS_OP_T;
    check_cublaslt(
        cublasLtMatmulDescSetAttribute(
            cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)),
        name, M, N, K, "set TRANSA");
    check_cublaslt(
        cublasLtMatmulDescSetAttribute(
            cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)),
        name, M, N, K, "set TRANSB");
    check_cublaslt(
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, M, K, K),
        name, M, N, K, "create A layout");
    check_cublaslt(
        cublasLtMatrixLayoutSetAttribute(
            cg.Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
            sizeof(row_order)),
        name, M, N, K, "set A row-major order");
    check_cublaslt(
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, N, K, K),
        name, M, N, K, "create B layout");
    check_cublaslt(
        cublasLtMatrixLayoutSetAttribute(
            cg.Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
            sizeof(row_order)),
        name, M, N, K, "set B row-major order");
    check_cublaslt(
        cublasLtMatrixLayoutCreate(&cg.Ddesc, CUDA_R_16BF, M, N, N),
        name, M, N, K, "create D layout");
    check_cublaslt(
        cublasLtMatrixLayoutSetAttribute(
            cg.Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
            sizeof(row_order)),
        name, M, N, K, "set D row-major order");

    cublasLtMatmulPreference_t pref;
    check_cublaslt(cublasLtMatmulPreferenceCreate(&pref), name, M, N, K,
                   "cublasLtMatmulPreferenceCreate");
    check_cublaslt(
        cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_fp8_ws_sz,
            sizeof(g_fp8_ws_sz)),
        name, M, N, K, "set workspace preference");
    cublasLtMatmulHeuristicResult_t result;
    int ret = 0;
    cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
        g_fp8_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Ddesc, cg.Ddesc, pref, 1,
        &result, &ret);
    cublasLtMatmulPreferenceDestroy(pref);
    check_heuristic(heuristic_status, ret, name, M, N, K);
    cg.algo = result.algo;
    g_lt_cache[key] = cg;
    it = g_lt_cache.find(key);
  }

  auto& cg = it->second;
  check_cublaslt(
      cublasLtMatmulDescSetAttribute(
          cg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &input_scale,
          sizeof(input_scale)),
      name, M, N, K, "set A scale pointer");
  check_cublaslt(
      cublasLtMatmulDescSetAttribute(
          cg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &weight_scale,
          sizeof(weight_scale)),
      name, M, N, K, "set B scale pointer");
  float alpha = 1.0f;
  float beta = 0.0f;
  check_cublaslt(
      cublasLtMatmul(
          g_fp8_lt, cg.desc, &alpha, input_fp8, cg.Adesc, weight_fp8, cg.Bdesc,
          &beta, out_bf16, cg.Ddesc, out_bf16, cg.Ddesc, &cg.algo, g_fp8_ws,
          g_fp8_ws_sz, stream),
      name, M, N, K, "cublasLtMatmul");
}

void bias_gelu_quantize_fp8_static_bf16(
    const void* input_bf16,
    const void* bias_bf16,
    void* out_fp8,
    const float* scale,
    long long M,
    int N,
    cudaStream_t stream) {
  const long long total = M * static_cast<long long>(N);
  if (total <= 0) {
    return;
  }

  const int has_bias = bias_bf16 != nullptr ? 1 : 0;
  const int block_sz = quant_block_size(M, N, has_bias != 0);
  const long long tiles_per_row =
      (static_cast<long long>(N) + block_sz - 1) / block_sz;
  const unsigned grid = static_cast<unsigned>(M * tiles_per_row);

  bias_gelu_quantize_fp8_static_bf16_kernel<<<grid, block_sz, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
      scale,
      tiles_per_row,
      N,
      has_bias);
}

void add_bias_bf16(
    void* input_bf16,
    const void* bias_bf16,
    long long M,
    int N,
    cudaStream_t stream) {
  const long long total = M * static_cast<long long>(N);
  if (total <= 0) {
    return;
  }
  constexpr int block = 256;
  const int grid = static_cast<int>((total + block - 1) / block);
  add_bias_bf16_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(input_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      total,
      N);
}

}  // namespace fp8_ffn
}  // namespace flash_rt
