// SPDX-License-Identifier: Apache-2.0

#include "cholesky_small_fp32.cuh"

#include <cuda_runtime.h>

#include <mutex>

namespace flashrt_hub {
namespace cholesky {
namespace {

template <int N, int MATRICES_PER_BLOCK>
__global__ void cholesky_small_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch) {
  constexpr int kLd = N + 1;
  constexpr int kTileElements = N * kLd;
  constexpr int kMatrixElements = N * N;
  extern __shared__ float shared[];

  const int group = threadIdx.x / N;
  const int row = threadIdx.x - group * N;
  const int matrix = blockIdx.x * MATRICES_PER_BLOCK + group;
  const bool valid = matrix < batch;
  float* tile = shared + group * kTileElements;

  for (int linear = threadIdx.x;
       linear < MATRICES_PER_BLOCK * kMatrixElements;
       linear += blockDim.x) {
    const int load_group = linear / kMatrixElements;
    const int element = linear - load_group * kMatrixElements;
    const int load_matrix = blockIdx.x * MATRICES_PER_BLOCK + load_group;
    const int load_row = element / N;
    const int load_col = element - load_row * N;
    shared[load_group * kTileElements + load_row * kLd + load_col] =
        load_matrix < batch && load_row >= load_col
            ? input[load_matrix * kMatrixElements + element]
            : 0.0f;
  }
  __syncthreads();

#pragma unroll 1
  for (int k = 0; k < N; ++k) {
    if (valid && row == k) {
      float value = tile[k * kLd + k];
#pragma unroll 4
      for (int j = 0; j < k; ++j) {
        const float item = tile[k * kLd + j];
        value = fmaf(-item, item, value);
      }
      tile[k * kLd + k] = sqrtf(fmaxf(value, 0.0f));
    }
    if constexpr (N == 32) {
      __syncwarp();
    } else {
      __syncthreads();
    }

    if (valid && row > k) {
      float value = tile[row * kLd + k];
#pragma unroll 4
      for (int j = 0; j < k; ++j) {
        value = fmaf(
            -tile[row * kLd + j], tile[k * kLd + j], value);
      }
      tile[row * kLd + k] = value / tile[k * kLd + k];
    }
    if constexpr (N == 32) {
      __syncwarp();
    } else {
      __syncthreads();
    }
  }
  __syncthreads();

  for (int linear = threadIdx.x;
       linear < MATRICES_PER_BLOCK * kMatrixElements;
       linear += blockDim.x) {
    const int store_group = linear / kMatrixElements;
    const int element = linear - store_group * kMatrixElements;
    const int store_matrix = blockIdx.x * MATRICES_PER_BLOCK + store_group;
    if (store_matrix < batch) {
      const int store_row = element / N;
      const int store_col = element - store_row * N;
      output[store_matrix * kMatrixElements + element] =
          store_row >= store_col
              ? shared[store_group * kTileElements + store_row * kLd + store_col]
              : 0.0f;
    }
  }
}

__global__ void cholesky_128_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch) {
  constexpr int kN = 128;
  constexpr int kLd = kN + 1;
  constexpr int kPanel = 32;
  constexpr int kMatrixElements = kN * kN;
  extern __shared__ float tile[];

  const int matrix = blockIdx.x;
  if (matrix >= batch) {
    return;
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  for (int element = threadIdx.x; element < kMatrixElements;
       element += blockDim.x) {
    const int row = element / kN;
    const int col = element - row * kN;
    tile[row * kLd + col] =
        row >= col ? input[matrix * kMatrixElements + element] : 0.0f;
  }
  __syncthreads();

#pragma unroll 1
  for (int start = 0; start < kN; start += kPanel) {
    if (warp == 0) {
#pragma unroll
      for (int local_col = 0; local_col < kPanel; ++local_col) {
        if (lane == local_col) {
          const int diagonal = start + local_col;
          float value = tile[diagonal * kLd + diagonal];
#pragma unroll
          for (int previous = 0; previous < local_col; ++previous) {
            const float item = tile[diagonal * kLd + start + previous];
            value = fmaf(-item, item, value);
          }
          tile[diagonal * kLd + diagonal] = sqrtf(fmaxf(value, 0.0f));
        }
        __syncwarp();
        if (lane > local_col) {
          const int row = start + lane;
          const int col = start + local_col;
          float value = tile[row * kLd + col];
#pragma unroll
          for (int previous = 0; previous < local_col; ++previous) {
            value = fmaf(
                -tile[row * kLd + start + previous],
                tile[col * kLd + start + previous], value);
          }
          tile[row * kLd + col] = value / tile[col * kLd + col];
        }
        __syncwarp();
      }
    }
    __syncthreads();

    const int panel_end = start + kPanel;
    for (int row = panel_end + threadIdx.x; row < kN; row += blockDim.x) {
#pragma unroll
      for (int local_col = 0; local_col < kPanel; ++local_col) {
        const int col = start + local_col;
        float value = tile[row * kLd + col];
#pragma unroll
        for (int previous = 0; previous < local_col; ++previous) {
          value = fmaf(
              -tile[row * kLd + start + previous],
              tile[col * kLd + start + previous], value);
        }
        tile[row * kLd + col] = value / tile[col * kLd + col];
      }
    }
    __syncthreads();

    const int trailing = kN - panel_end;
    const int trailing_elements = trailing * trailing;
    for (int local = threadIdx.x; local < trailing_elements;
         local += blockDim.x) {
      const int local_row = local / trailing;
      const int local_col = local - local_row * trailing;
      if (local_row >= local_col) {
        const int row = panel_end + local_row;
        const int col = panel_end + local_col;
        float value = tile[row * kLd + col];
#pragma unroll
        for (int previous = 0; previous < kPanel; ++previous) {
          value = fmaf(
              -tile[row * kLd + start + previous],
              tile[col * kLd + start + previous], value);
        }
        tile[row * kLd + col] = value;
      }
    }
    __syncthreads();
  }

  for (int element = threadIdx.x; element < kMatrixElements;
       element += blockDim.x) {
    const int row = element / kN;
    const int col = element - row * kN;
    output[matrix * kMatrixElements + element] =
        row >= col ? tile[row * kLd + col] : 0.0f;
  }
}

template <int N, int MATRICES_PER_BLOCK>
cudaError_t launch_small(
    const float* input,
    float* output,
    int batch,
    cudaStream_t stream) {
  constexpr int kThreads = N * MATRICES_PER_BLOCK;
  constexpr int kSharedBytes =
      MATRICES_PER_BLOCK * N * (N + 1) * sizeof(float);
  const int blocks = (batch + MATRICES_PER_BLOCK - 1) / MATRICES_PER_BLOCK;
  cholesky_small_kernel<N, MATRICES_PER_BLOCK>
      <<<blocks, kThreads, kSharedBytes, stream>>>(input, output, batch);
  return cudaGetLastError();
}

cudaError_t configure_128_for_current_device() {
  constexpr int kMaxDevices = 64;
  static std::once_flag flags[kMaxDevices];
  static cudaError_t results[kMaxDevices] = {};

  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    return error;
  }
  if (device < 0 || device >= kMaxDevices) {
    return cudaErrorInvalidDevice;
  }
  std::call_once(flags[device], [device]() {
    constexpr int kSharedBytes = 128 * 129 * sizeof(float);
    (void)device;
    results[device] = cudaFuncSetAttribute(
        cholesky_128_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedBytes);
  });
  return results[device];
}

cudaError_t launch_128(
    const float* input,
    float* output,
    int batch,
    cudaStream_t stream) {
  cudaError_t error = configure_128_for_current_device();
  if (error != cudaSuccess) {
    return error;
  }
  constexpr int kSharedBytes = 128 * 129 * sizeof(float);
  cholesky_128_kernel<<<batch, 256, kSharedBytes, stream>>>(
      input, output, batch);
  return cudaGetLastError();
}

}  // namespace

cudaError_t cholesky_small_fp32(
    const float* input,
    float* output,
    int batch,
    int n,
    cudaStream_t stream) {
  switch (n) {
    case 32:
      return launch_small<32, 4>(input, output, batch, stream);
    case 64:
      return launch_small<64, 2>(input, output, batch, stream);
    case 128:
      return launch_128(input, output, batch, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace cholesky
}  // namespace flashrt_hub
