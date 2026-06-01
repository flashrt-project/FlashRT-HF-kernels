#include "bf16_gemm_bias_gelu.cuh"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace flash_rt::gemm {
namespace {

constexpr size_t kWorkspaceSize = 32 * 1024 * 1024;

std::string cublas_error(cublasStatus_t status, const char* expr) {
  return std::string("cuBLASLt error ") + std::to_string(static_cast<int>(status)) +
         " from " + expr;
}

std::string cuda_error(cudaError_t status, const char* expr) {
  return std::string("CUDA error ") + cudaGetErrorString(status) + " from " + expr;
}

#define FLASHRT_CUBLAS_CHECK(expr)                                      \
  do {                                                                  \
    const cublasStatus_t status = (expr);                               \
    if (status != CUBLAS_STATUS_SUCCESS) {                              \
      throw std::runtime_error(cublas_error(status, #expr));            \
    }                                                                   \
  } while (0)

#define FLASHRT_CUDA_CHECK(expr)                                        \
  do {                                                                  \
    const cudaError_t status = (expr);                                  \
    if (status != cudaSuccess) {                                        \
      throw std::runtime_error(cuda_error(status, #expr));              \
    }                                                                   \
  } while (0)

struct DeviceRuntime {
  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  size_t workspace_size = kWorkspaceSize;
  int device = -1;

  explicit DeviceRuntime(int device_id) : device(device_id) {
    FLASHRT_CUDA_CHECK(cudaSetDevice(device));
    FLASHRT_CUBLAS_CHECK(cublasLtCreate(&handle));
    FLASHRT_CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  }

  ~DeviceRuntime() {
    const int previous_device = [] {
      int current = -1;
      cudaGetDevice(&current);
      return current;
    }();
    cudaSetDevice(device);
    if (workspace != nullptr) {
      cudaFree(workspace);
    }
    if (handle != nullptr) {
      cublasLtDestroy(handle);
    }
    if (previous_device >= 0) {
      cudaSetDevice(previous_device);
    }
  }

  DeviceRuntime(const DeviceRuntime&) = delete;
  DeviceRuntime& operator=(const DeviceRuntime&) = delete;
};

DeviceRuntime& runtime_for_current_device() {
  int device = -1;
  FLASHRT_CUDA_CHECK(cudaGetDevice(&device));

  static std::mutex mutex;
  static std::map<int, std::unique_ptr<DeviceRuntime>> runtimes;

  std::lock_guard<std::mutex> lock(mutex);
  auto it = runtimes.find(device);
  if (it == runtimes.end()) {
    it = runtimes.emplace(device, std::make_unique<DeviceRuntime>(device)).first;
  }
  return *it->second;
}

struct MatmulDesc {
  cublasLtMatmulDesc_t value = nullptr;
  ~MatmulDesc() {
    if (value != nullptr) {
      cublasLtMatmulDescDestroy(value);
    }
  }
};

struct MatrixLayout {
  cublasLtMatrixLayout_t value = nullptr;
  ~MatrixLayout() {
    if (value != nullptr) {
      cublasLtMatrixLayoutDestroy(value);
    }
  }
};

struct MatmulPreference {
  cublasLtMatmulPreference_t value = nullptr;
  ~MatmulPreference() {
    if (value != nullptr) {
      cublasLtMatmulPreferenceDestroy(value);
    }
  }
};

}  // namespace

void bf16_gemm_bias(
    const void* A,
    const void* B,
    const void* bias,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
  auto& runtime = runtime_for_current_device();

  MatmulDesc matmul_desc;
  MatrixLayout A_desc;
  MatrixLayout B_desc;
  MatrixLayout D_desc;
  MatmulPreference preference;

  FLASHRT_CUBLAS_CHECK(
      cublasLtMatmulDescCreate(&matmul_desc.value, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  cublasOperation_t op_n = CUBLAS_OP_N;
  FLASHRT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc.value, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
  FLASHRT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc.value, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  FLASHRT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc.value, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  FLASHRT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc.value, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  cudaDataType_t bias_type = CUDA_R_16BF;
  FLASHRT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc.value, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));

  FLASHRT_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&A_desc.value, CUDA_R_16BF, N, K, N));
  FLASHRT_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&B_desc.value, CUDA_R_16BF, K, M, K));
  FLASHRT_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&D_desc.value, CUDA_R_16BF, N, M, N));

  FLASHRT_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference.value));
  FLASHRT_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference.value,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &runtime.workspace_size,
      sizeof(runtime.workspace_size)));

  int returned_results = 0;
  cublasLtMatmulHeuristicResult_t heuristic;
  FLASHRT_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      runtime.handle,
      matmul_desc.value,
      A_desc.value,
      B_desc.value,
      D_desc.value,
      D_desc.value,
      preference.value,
      1,
      &heuristic,
      &returned_results));
  if (returned_results == 0) {
    throw std::runtime_error("cuBLASLt bf16_gemm_bias: no algorithm found");
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  FLASHRT_CUBLAS_CHECK(cublasLtMatmul(
      runtime.handle,
      matmul_desc.value,
      &alpha,
      B,
      A_desc.value,
      A,
      B_desc.value,
      &beta,
      D,
      D_desc.value,
      D,
      D_desc.value,
      &heuristic.algo,
      runtime.workspace,
      runtime.workspace_size,
      stream));
}

void bf16_gemm_bias_gelu(
    const void* A,
    const void* B,
    const void* bias,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
  auto& runtime = runtime_for_current_device();

  MatmulDesc matmul_desc;
  MatrixLayout A_desc;
  MatrixLayout B_desc;
  MatrixLayout D_desc;
  MatmulPreference preference;

  FLASHRT_CUBLAS_CHECK(
      cublasLtMatmulDescCreate(&matmul_desc.value, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  cublasOperation_t op_n = CUBLAS_OP_N;
  FLASHRT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc.value, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
  FLASHRT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc.value, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
  FLASHRT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc.value, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  FLASHRT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc.value, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  cudaDataType_t bias_type = CUDA_R_16BF;
  FLASHRT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc.value, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));

  // Expose row-major tensors at the public API, but use cuBLASLt's native
  // column-major view internally: D_row(M,N) is D_col^T(N,M), so compute
  // D_col = B_col^T(N,K) @ A_col^T(K,M). This matches the layout style used
  // by FlashRT's fp8 cuBLASLt epilogue path and enables GELU_BIAS heuristics.
  FLASHRT_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&A_desc.value, CUDA_R_16BF, N, K, N));
  FLASHRT_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&B_desc.value, CUDA_R_16BF, K, M, K));
  FLASHRT_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&D_desc.value, CUDA_R_16BF, N, M, N));

  FLASHRT_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference.value));
  FLASHRT_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference.value,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &runtime.workspace_size,
      sizeof(runtime.workspace_size)));

  int returned_results = 0;
  cublasLtMatmulHeuristicResult_t heuristic;
  FLASHRT_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      runtime.handle,
      matmul_desc.value,
      A_desc.value,
      B_desc.value,
      D_desc.value,
      D_desc.value,
      preference.value,
      1,
      &heuristic,
      &returned_results));
  if (returned_results == 0) {
    throw std::runtime_error("cuBLASLt bf16_gemm_bias_gelu: no algorithm found");
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  FLASHRT_CUBLAS_CHECK(cublasLtMatmul(
      runtime.handle,
      matmul_desc.value,
      &alpha,
      B,
      A_desc.value,
      A,
      B_desc.value,
      &beta,
      D,
      D_desc.value,
      D,
      D_desc.value,
      &heuristic.algo,
      runtime.workspace,
      runtime.workspace_size,
      stream));
}

}  // namespace flash_rt::gemm
