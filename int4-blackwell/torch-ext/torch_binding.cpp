// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <torch/all.h>
#include <torch/library.h>

#include <limits>
#include <mutex>
#include <unordered_map>

#include "registration.h"
#include "torch_binding.h"

namespace {

#define CU_CHECK(expr)                                                        \
  do {                                                                        \
    CUresult status = (expr);                                                  \
    if (status != CUDA_SUCCESS) {                                              \
      const char* message = nullptr;                                           \
      cuGetErrorString(status, &message);                                      \
      TORCH_CHECK(false, #expr, " failed: ", message ? message : "unknown"); \
    }                                                                         \
  } while (false)

void check_environment(int64_t device) {
  TORCH_CHECK(device >= 0 && device < at::cuda::device_count(),
              "device must identify an available CUDA device");
  cudaDeviceProp prop{};
  TORCH_CHECK(cudaGetDeviceProperties(&prop, static_cast<int>(device)) == cudaSuccess,
              "cudaGetDeviceProperties failed");
  TORCH_CHECK(prop.major == 12 && prop.minor == 0,
              "int4-blackwell requires SM120; got SM", prop.major, prop.minor);
  int driver = 0;
  TORCH_CHECK(cudaDriverGetVersion(&driver) == cudaSuccess,
              "cudaDriverGetVersion failed");
  TORCH_CHECK(driver >= 13000, "int4-blackwell requires a CUDA 13.0+ driver");
  TORCH_CHECK(cudaSetDevice(static_cast<int>(device)) == cudaSuccess,
              "cudaSetDevice failed");
  // Ensure the runtime has retained the primary context before using the
  // Driver API. A cold get_kernel process may not have allocated a tensor yet.
  TORCH_CHECK(cudaFree(nullptr) == cudaSuccess,
              "failed to initialize the CUDA primary context");
}

void check_cubin(torch::Tensor const& cubin) {
  TORCH_CHECK(cubin.device().is_cpu(), "cubin must be a CPU tensor");
  TORCH_CHECK(cubin.scalar_type() == torch::kUInt8, "cubin must be uint8");
  TORCH_CHECK(cubin.is_contiguous(), "cubin must be contiguous");
  TORCH_CHECK(cubin.numel() > 0, "cubin must not be empty");
}

CUmodule load_module(torch::Tensor const& cubin, int64_t device) {
  check_cubin(cubin);
  static std::mutex mutex;
  static std::unordered_map<uint64_t, CUmodule> modules;
  const auto key = static_cast<uint64_t>(device) << 56 |
      (reinterpret_cast<uintptr_t>(cubin.const_data_ptr<uint8_t>()) &
       0x00ffffffffffffffULL);
  std::lock_guard<std::mutex> lock(mutex);
  if (auto it = modules.find(key); it != modules.end()) {
    return it->second;
  }
  CUmodule module = nullptr;
  CU_CHECK(cuModuleLoadData(&module, cubin.const_data_ptr<uint8_t>()));
  modules.emplace(key, module);
  return module;
}

}  // namespace

torch::Tensor run_codebook_probe(torch::Tensor const& cubin, int64_t device) {
  check_environment(device);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  CUmodule module = load_module(cubin, device);
  CUfunction function = nullptr;
  CU_CHECK(cuModuleGetFunction(&function, module, "codebook_probe"));

  auto output = torch::empty({16, 128}, torch::TensorOptions()
      .device(torch::kCUDA, device).dtype(torch::kFloat32));
  uint32_t scale_one = 0x38383838u;
  for (uint32_t value = 0; value < 16; ++value) {
    uint32_t aword = value * 0x11111111u;
    uint32_t bword = 0x11111111u;
    float* row = output[value].data_ptr<float>();
    void* args[] = {&aword, &bword, &scale_one, &scale_one, &row};
    auto stream = reinterpret_cast<CUstream>(
        at::cuda::getCurrentCUDAStream(static_cast<int>(device)).stream());
    CU_CHECK(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, stream, args, nullptr));
  }
  return output;
}

torch::Tensor run_mma_probe(torch::Tensor const& cubin, int64_t iterations,
                            int64_t blocks, int64_t device) {
  check_environment(device);
  TORCH_CHECK(iterations > 0 && iterations <= std::numeric_limits<int>::max(),
              "iterations must fit in a positive int");
  TORCH_CHECK(blocks > 0 && blocks <= std::numeric_limits<unsigned>::max(),
              "blocks must fit in a positive unsigned int");
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  CUmodule module = load_module(cubin, device);
  CUfunction function = nullptr;
  CU_CHECK(cuModuleGetFunction(&function, module, "perf_mma"));

  constexpr int threads = 256;
  auto output = torch::empty({blocks, threads}, torch::TensorOptions()
      .device(torch::kCUDA, device).dtype(torch::kFloat32));
  int iters = static_cast<int>(iterations);
  uint32_t aword = 0x25142514u;
  uint32_t bword = 0x13521352u;
  uint32_t scale_one = 0x38383838u;
  float* out = output.data_ptr<float>();
  void* args[] = {&iters, &aword, &bword, &scale_one, &scale_one, &out};
  auto stream = reinterpret_cast<CUstream>(
      at::cuda::getCurrentCUDAStream(static_cast<int>(device)).stream());
  CU_CHECK(cuLaunchKernel(function, static_cast<unsigned>(blocks), 1, 1,
                          threads, 1, 1, 0, stream, args, nullptr));
  return output;
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("run_codebook_probe(Tensor cubin, int device) -> Tensor");
  ops.def("run_mma_probe(Tensor cubin, int iterations, int blocks, int device) -> Tensor");
  ops.impl("run_codebook_probe", torch::kCPU, &run_codebook_probe);
  ops.impl("run_mma_probe", torch::kCPU, &run_mma_probe);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
