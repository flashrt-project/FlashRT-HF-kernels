#include "rocm/hipblaslt_simple.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <hipblaslt/hipblaslt.h>

namespace flash_rt {
namespace rocm {
namespace {

void check_hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::ostringstream oss;
    oss << what << " failed: " << hipGetErrorString(status);
    throw std::runtime_error(oss.str());
  }
}

void check_hipblas(hipblasStatus_t status, const char* what) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    std::ostringstream oss;
    oss << what << " failed with hipBLASLt status "
        << static_cast<int>(status);
    throw std::runtime_error(oss.str());
  }
}

struct LtContext {
  hipblasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 32ull * 1024ull * 1024ull;

  LtContext() {
    check_hipblas(hipblasLtCreate(&handle), "hipblasLtCreate");
    check_hip(hipMalloc(&workspace, workspace_bytes), "hipMalloc(workspace)");
  }

  ~LtContext() {
    if (workspace != nullptr) {
      hipFree(workspace);
    }
    if (handle != nullptr) {
      hipblasLtDestroy(handle);
    }
  }
};

LtContext& context() {
  static LtContext ctx;
  return ctx;
}

struct MatmulDesc {
  hipblasLtMatmulDesc_t value = nullptr;
  explicit MatmulDesc(hipblasComputeType_t compute) {
    check_hipblas(hipblasLtMatmulDescCreate(&value, compute, HIP_R_32F),
                  "hipblasLtMatmulDescCreate");
  }
  ~MatmulDesc() {
    if (value != nullptr) {
      hipblasLtMatmulDescDestroy(value);
    }
  }
};

struct MatrixLayout {
  hipblasLtMatrixLayout_t value = nullptr;
  MatrixLayout(hipDataType type, uint64_t rows, uint64_t cols, int64_t ld) {
    check_hipblas(hipblasLtMatrixLayoutCreate(&value, type, rows, cols, ld),
                  "hipblasLtMatrixLayoutCreate");
  }
  ~MatrixLayout() {
    if (value != nullptr) {
      hipblasLtMatrixLayoutDestroy(value);
    }
  }
};

struct Preference {
  hipblasLtMatmulPreference_t value = nullptr;
  explicit Preference(uint64_t workspace_bytes) {
    check_hipblas(hipblasLtMatmulPreferenceCreate(&value),
                  "hipblasLtMatmulPreferenceCreate");
    check_hipblas(
        hipblasLtMatmulPreferenceSetAttribute(
            value, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes, sizeof(workspace_bytes)),
        "hipblasLtMatmulPreferenceSetAttribute(workspace)");
  }
  ~Preference() {
    if (value != nullptr) {
      hipblasLtMatmulPreferenceDestroy(value);
    }
  }
};

struct CachedAlgo {
  hipblasLtMatmulAlgo_t algo{};
  size_t workspace_size = 0;
};

std::mutex& cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, CachedAlgo>& algo_cache() {
  static std::unordered_map<std::string, CachedAlgo> cache;
  return cache;
}

CachedAlgo select_algo(
    const std::string& key,
    hipblasLtMatmulDesc_t op,
    hipblasLtMatrixLayout_t a,
    hipblasLtMatrixLayout_t b,
    hipblasLtMatrixLayout_t d) {
  {
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto it = algo_cache().find(key);
    if (it != algo_cache().end()) {
      return it->second;
    }
  }

  LtContext& ctx = context();
  Preference pref(static_cast<uint64_t>(ctx.workspace_bytes));
  hipblasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  check_hipblas(
      hipblasLtMatmulAlgoGetHeuristic(
          ctx.handle, op, a, b, d, d, pref.value, 1, &heuristic, &returned),
      "hipblasLtMatmulAlgoGetHeuristic");
  if (returned <= 0 || heuristic.state != HIPBLAS_STATUS_SUCCESS ||
      heuristic.workspaceSize > ctx.workspace_bytes) {
    throw std::runtime_error("hipBLASLt returned no usable algorithm");
  }

  CachedAlgo selected{heuristic.algo, heuristic.workspaceSize};
  {
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto [it, _] = algo_cache().emplace(key, selected);
    return it->second;
  }
}

void set_scalar_fp8_scale(
    hipblasLtMatmulDesc_t desc,
    const float* a_scale,
    const float* b_scale) {
  const void* a_scale_ptr = a_scale;
  const void* b_scale_ptr = b_scale;
  const hipblasLtMatmulMatrixScale_t scale_mode =
      HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
  check_hipblas(
      hipblasLtMatmulDescSetAttribute(
          desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
          &a_scale_ptr, sizeof(a_scale_ptr)),
      "set A scale");
  check_hipblas(
      hipblasLtMatmulDescSetAttribute(
          desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
          &b_scale_ptr, sizeof(b_scale_ptr)),
      "set B scale");
  check_hipblas(
      hipblasLtMatmulDescSetAttribute(
          desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
          &scale_mode, sizeof(scale_mode)),
      "set A scale mode");
  check_hipblas(
      hipblasLtMatmulDescSetAttribute(
          desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
          &scale_mode, sizeof(scale_mode)),
      "set B scale mode");
}

}  // namespace

void hipblaslt_matmul_fp8_e4m3fnuz_bf16(
    const void* a,
    const void* b,
    const float* a_scale,
    const float* b_scale,
    void* out,
    int64_t m,
    int64_t n,
    int64_t k,
    hipStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) {
    throw std::invalid_argument("FP8 matmul dimensions must be positive");
  }

  LtContext& ctx = context();
  MatmulDesc op(HIPBLAS_COMPUTE_32F_FAST_8F_FNUZ);
  const hipblasOperation_t trans_a = HIPBLAS_OP_T;
  check_hipblas(
      hipblasLtMatmulDescSetAttribute(
          op.value, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)),
      "set transA");
  set_scalar_fp8_scale(op.value, b_scale, a_scale);

  // Row-major (N, K) FP8 weight memory is column-major (K, N). Transpose it
  // so the logical lhs is (N, K), then compute out^T = weight @ input^T.
  MatrixLayout b_as_lhs(HIP_R_8F_E4M3_FNUZ, static_cast<uint64_t>(k),
                        static_cast<uint64_t>(n), k);
  MatrixLayout a_as_rhs(HIP_R_8F_E4M3_FNUZ, static_cast<uint64_t>(k),
                        static_cast<uint64_t>(m), k);
  MatrixLayout out_t(HIP_R_16BF, static_cast<uint64_t>(n),
                     static_cast<uint64_t>(m), n);

  std::ostringstream key;
  key << "fp8:" << m << "x" << n << "x" << k;
  CachedAlgo algo = select_algo(key.str(), op.value, b_as_lhs.value,
                                a_as_rhs.value, out_t.value);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_hipblas(
      hipblasLtMatmul(ctx.handle, op.value, &alpha,
                      b, b_as_lhs.value,
                      a, a_as_rhs.value,
                      &beta,
                      out, out_t.value,
                      out, out_t.value,
                      &algo.algo,
                      ctx.workspace, algo.workspace_size,
                      stream),
      "hipblasLtMatmul(fp8)");
}

void hipblaslt_linear_bf16(
    const void* x,
    const void* weight,
    const void* bias,
    void* out,
    int64_t m,
    int64_t n,
    int64_t k,
    hipStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) {
    throw std::invalid_argument("BF16 linear dimensions must be positive");
  }

  LtContext& ctx = context();
  MatmulDesc op(HIPBLAS_COMPUTE_32F);

  if (bias != nullptr) {
    const hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
    check_hipblas(
        hipblasLtMatmulDescSetAttribute(
            op.value, HIPBLASLT_MATMUL_DESC_EPILOGUE,
            &epilogue, sizeof(epilogue)),
        "set bias epilogue");
    check_hipblas(
        hipblasLtMatmulDescSetAttribute(
            op.value, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias, sizeof(bias)),
        "set bias pointer");
    const hipDataType bias_type = HIP_R_16BF;
    check_hipblas(
        hipblasLtMatmulDescSetAttribute(
            op.value, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
            &bias_type, sizeof(bias_type)),
        "set bias dtype");
  }

  // Row-major (K, N) weight memory is column-major (N, K). Compute
  // out^T = weight^T @ x^T directly into row-major out storage.
  MatrixLayout weight_layout(HIP_R_16BF, static_cast<uint64_t>(n),
                             static_cast<uint64_t>(k), n);
  MatrixLayout x_rhs(HIP_R_16BF, static_cast<uint64_t>(k),
                     static_cast<uint64_t>(m), k);
  MatrixLayout out_t(HIP_R_16BF, static_cast<uint64_t>(n),
                     static_cast<uint64_t>(m), n);

  std::ostringstream key;
  key << "bf16_linear:" << m << "x" << n << "x" << k
      << ":bias=" << (bias != nullptr);
  CachedAlgo algo = select_algo(key.str(), op.value, weight_layout.value,
                                x_rhs.value, out_t.value);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_hipblas(
      hipblasLtMatmul(ctx.handle, op.value, &alpha,
                      weight, weight_layout.value,
                      x, x_rhs.value,
                      &beta,
                      out, out_t.value,
                      out, out_t.value,
                      &algo.algo,
                      ctx.workspace, algo.workspace_size,
                      stream),
      "hipblasLtMatmul(bf16 linear)");
}

}  // namespace rocm
}  // namespace flash_rt
