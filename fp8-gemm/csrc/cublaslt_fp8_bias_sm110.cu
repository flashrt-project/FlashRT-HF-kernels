#include "cublaslt_fp8_bias_sm110.cuh"

#include <cublasLt.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace {

constexpr size_t kWorkspaceBytes = 32 * 1024 * 1024;

struct Key {
  int M;
  int N;
  int K;
  int epilogue;

  bool operator==(const Key& other) const {
    return M == other.M && N == other.N && K == other.K &&
           epilogue == other.epilogue;
  }
};

struct KeyHash {
  size_t operator()(const Key& key) const {
    size_t value = std::hash<int>{}(key.M);
    value ^= std::hash<int>{}(key.N) + 0x9e3779b9 + (value << 6) +
             (value >> 2);
    value ^= std::hash<int>{}(key.K) + 0x9e3779b9 + (value << 6) +
             (value >> 2);
    value ^= std::hash<int>{}(key.epilogue) + 0x9e3779b9 + (value << 6) +
             (value >> 2);
    return value;
  }
};

struct Entry {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t weight = nullptr;
  cublasLtMatrixLayout_t input = nullptr;
  cublasLtMatrixLayout_t output = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
};

cublasLtHandle_t handle = nullptr;
void* workspace = nullptr;
std::unordered_map<Key, Entry, KeyHash> cache;
std::mutex cache_mutex;

int status_code(cublasStatus_t status) {
  return status == CUBLAS_STATUS_SUCCESS ? 0 : -1000 - static_cast<int>(status);
}

void destroy_entry(Entry& entry) {
  if (entry.operation) cublasLtMatmulDescDestroy(entry.operation);
  if (entry.weight) cublasLtMatrixLayoutDestroy(entry.weight);
  if (entry.input) cublasLtMatrixLayoutDestroy(entry.input);
  if (entry.output) cublasLtMatrixLayoutDestroy(entry.output);
  entry = Entry{};
}

int ensure_runtime() {
  if (handle) return 0;
  cublasStatus_t status = cublasLtCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) return status_code(status);
  const cudaError_t cuda_status = cudaMalloc(&workspace, kWorkspaceBytes);
  return cuda_status == cudaSuccess ? 0 : -2000 - static_cast<int>(cuda_status);
}

int create_entry(const Key& key, Entry* entry) {
  cublasStatus_t status = cublasLtMatmulDescCreate(
      &entry->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t transpose = CUBLAS_OP_T;
  cublasOperation_t no_transpose = CUBLAS_OP_N;
  const bool has_bias =
      key.epilogue != static_cast<int>(FlashRtFp8BiasEpilogue::kNone);
  cublasLtEpilogue_t epilogue =
      key.epilogue == static_cast<int>(FlashRtFp8BiasEpilogue::kBiasGelu)
          ? CUBLASLT_EPILOGUE_GELU_BIAS
          : CUBLASLT_EPILOGUE_BIAS;
  cudaDataType_t bias_type = CUDA_R_16BF;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(
        entry->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
        sizeof(transpose));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(
        entry->operation, CUBLASLT_MATMUL_DESC_TRANSB, &no_transpose,
        sizeof(no_transpose));
  }
  if (status == CUBLAS_STATUS_SUCCESS && has_bias) {
    status = cublasLtMatmulDescSetAttribute(
        entry->operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
        sizeof(epilogue));
  }
  if (status == CUBLAS_STATUS_SUCCESS && has_bias) {
    status = cublasLtMatmulDescSetAttribute(
        entry->operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
        sizeof(bias_type));
  }

  // Row-major weight [N,K] is column-major [K,N]. Row-major input [M,K]
  // is column-major [K,M]. The logical result [N,M] has row-major [M,N]
  // storage, so no layout conversion or transpose kernel is required.
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(
        &entry->weight, CUDA_R_8F_E4M3, key.K, key.N, key.K);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(
        &entry->input, CUDA_R_8F_E4M3, key.K, key.M, key.K);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(
        &entry->output, CUDA_R_16BF, key.N, key.M, key.N);
  }

  cublasLtMatmulPreference_t preference = nullptr;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulPreferenceCreate(&preference);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    const size_t workspace_bytes = kWorkspaceBytes;
    status = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes, sizeof(workspace_bytes));
  }
  cublasLtMatmulHeuristicResult_t results[16]{};
  int returned = 0;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulAlgoGetHeuristic(
        handle, entry->operation, entry->weight, entry->input, entry->output,
        entry->output, preference, 16, results, &returned);
  }
  if (preference) cublasLtMatmulPreferenceDestroy(preference);
  if (status == CUBLAS_STATUS_SUCCESS && returned > 0) {
    entry->algorithm = results[0].algo;
    return 0;
  }
  destroy_entry(*entry);
  return status == CUBLAS_STATUS_SUCCESS ? -1100 : status_code(status);
}

}  // namespace

int fp8_linear_cublaslt_bf16(
    const void* input_fp8,
    const void* weight_fp8,
    const void* bias_bf16,
    void* out_bf16,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    FlashRtFp8BiasEpilogue epilogue,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(cache_mutex);
  int rc = ensure_runtime();
  if (rc != 0) return rc;
  const Key key{M, N, K, static_cast<int>(epilogue)};
  auto iterator = cache.find(key);
  if (iterator == cache.end()) {
    Entry entry;
    rc = create_entry(key, &entry);
    if (rc != 0) return rc;
    iterator = cache.emplace(key, entry).first;
  }
  Entry& entry = iterator->second;
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  if (epilogue != FlashRtFp8BiasEpilogue::kNone) {
    if (bias_bf16 == nullptr) return -1200;
    status = cublasLtMatmulDescSetAttribute(
        entry.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_bf16,
        sizeof(bias_bf16));
    if (status != CUBLAS_STATUS_SUCCESS) return status_code(status);
  }
  status = cublasLtMatmul(
      handle, entry.operation, &alpha, weight_fp8, entry.weight, input_fp8,
      entry.input, &beta, out_bf16, entry.output, out_bf16, entry.output,
      &entry.algorithm, workspace, kWorkspaceBytes, stream);
  return status_code(status);
}

int fp8_linear_bias_sm110_bf16(
    const void* input_fp8,
    const void* weight_fp8,
    const void* bias_bf16,
    void* out_bf16,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    FlashRtFp8BiasEpilogue epilogue,
    cudaStream_t stream) {
  return fp8_linear_cublaslt_bf16(
      input_fp8, weight_fp8, bias_bf16, out_bf16, M, N, K, alpha, beta,
      epilogue, stream);
}
