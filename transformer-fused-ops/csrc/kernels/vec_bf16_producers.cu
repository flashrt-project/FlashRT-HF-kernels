#include "kernels/vec_bf16_producers.cuh"

#include <cstdint>

namespace {

constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_sum(float value, float* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < ((blockDim.x + 31) >> 5) ? shared[lane] : 0.0f;
    value = warp_sum(value);
    if (lane == 0) shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ __nv_fp8_e4m3 to_fp8(float value, float inv_scale) {
  const float scaled = fminf(fmaxf(value * inv_scale, -kFp8Max), kFp8Max);
  return __nv_fp8_e4m3(scaled);
}

__global__ void quantize_fp8_static_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    const float* __restrict__ scale,
    int numel) {
  const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (base >= numel) return;
  const float inv_scale = 1.0f / fmaxf(*scale, 1e-12f);
  if (base + 8 <= numel &&
      !(reinterpret_cast<uintptr_t>(input + base) & 15) &&
      !(reinterpret_cast<uintptr_t>(output + base) & 7)) {
    const int4 packed = *reinterpret_cast<const int4*>(input + base);
    const __nv_bfloat16* values =
        reinterpret_cast<const __nv_bfloat16*>(&packed);
    uint2 result;
    __nv_fp8_e4m3* quantized = reinterpret_cast<__nv_fp8_e4m3*>(&result);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      quantized[i] = to_fp8(__bfloat162float(values[i]), inv_scale);
    }
    *reinterpret_cast<uint2*>(output + base) = result;
    return;
  }
  for (int i = base; i < min(base + 8, numel); ++i) {
    output[i] = to_fp8(__bfloat162float(input[i]), inv_scale);
  }
}

__global__ void layer_norm_fp8_static_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ scale,
    int dim,
    float eps) {
  const int row = blockIdx.x;
  const int vecs = dim / 8;
  const int4* x_vec =
      reinterpret_cast<const int4*>(x + static_cast<long long>(row) * dim);
  const int4* weight_vec = reinterpret_cast<const int4*>(weight);
  const int4* bias_vec = reinterpret_cast<const int4*>(bias);
  uint2* out_vec =
      reinterpret_cast<uint2*>(out + static_cast<long long>(row) * dim);
  __shared__ float shared[32];

  float sum = 0.0f;
  for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
    const int4 packed = x_vec[i];
    const __nv_bfloat16* values =
        reinterpret_cast<const __nv_bfloat16*>(&packed);
    #pragma unroll
    for (int j = 0; j < 8; ++j) sum += __bfloat162float(values[j]);
  }
  const float mean = block_sum(sum, shared) / static_cast<float>(dim);

  float variance = 0.0f;
  for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
    const int4 packed = x_vec[i];
    const __nv_bfloat16* values =
        reinterpret_cast<const __nv_bfloat16*>(&packed);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float centered = __bfloat162float(values[j]) - mean;
      variance += centered * centered;
    }
  }
  const float inv_std =
      rsqrtf(block_sum(variance, shared) / static_cast<float>(dim) + eps);
  const float inv_scale = 1.0f / fmaxf(*scale, 1e-12f);

  for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
    const int4 x_packed = x_vec[i];
    const int4 w_packed = weight_vec[i];
    const int4 b_packed = bias_vec[i];
    const __nv_bfloat16* xv =
        reinterpret_cast<const __nv_bfloat16*>(&x_packed);
    const __nv_bfloat16* wv =
        reinterpret_cast<const __nv_bfloat16*>(&w_packed);
    const __nv_bfloat16* bv =
        reinterpret_cast<const __nv_bfloat16*>(&b_packed);
    uint2 result;
    __nv_fp8_e4m3* quantized = reinterpret_cast<__nv_fp8_e4m3*>(&result);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float normalized =
          (__bfloat162float(xv[j]) - mean) * inv_std *
              __bfloat162float(wv[j]) +
          __bfloat162float(bv[j]);
      const float rounded = __bfloat162float(__float2bfloat16(normalized));
      quantized[j] = to_fp8(rounded, inv_scale);
    }
    out_vec[i] = result;
  }
}

__global__ void gate_geglu_merged_fp8_static_bf16_kernel(
    const __nv_bfloat16* __restrict__ merged,
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ scale,
    int rows,
    int hidden) {
  const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const int total = rows * hidden;
  if (base >= total) return;
  const float inv_scale = 1.0f / fmaxf(*scale, 1e-12f);
  #pragma unroll
  for (int offset = 0; offset < 4; ++offset) {
    const int index = base + offset;
    if (index >= total) break;
    const int row = index / hidden;
    const int column = index - row * hidden;
    const long long row_base = static_cast<long long>(row) * 2 * hidden;
    const float gate = __bfloat162float(merged[row_base + column]);
    const float up = __bfloat162float(merged[row_base + hidden + column]);
    const float gelu = gate /
        (1.0f + expf(-1.5957691216057308f * gate *
                     (1.0f + 0.044715f * gate * gate)));
    out[index] = to_fp8(gelu * up, inv_scale);
  }
}

int norm_threads(int dim) {
  int threads = ((dim / 8 + 31) / 32) * 32;
  if (threads < 32) threads = 32;
  if (threads > 256) threads = 256;
  return threads;
}

}  // namespace

extern "C" {

int quantize_fp8_static_bf16_vec(const __nv_bfloat16* input,
                                 __nv_fp8_e4m3* output,
                                 const float* scale,
                                 int numel,
                                 cudaStream_t stream) {
  const int threads = 256;
  const int work_items = (numel + 7) / 8;
  quantize_fp8_static_bf16_kernel<<<
      (work_items + threads - 1) / threads, threads, 0, stream>>>(
      input, output, scale, numel);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? 0 : -static_cast<int>(error);
}

int layer_norm_fp8_static_bf16_vec(const __nv_bfloat16* x,
                                   const __nv_bfloat16* weight,
                                   const __nv_bfloat16* bias,
                                   __nv_fp8_e4m3* out,
                                   const float* scale,
                                   int rows,
                                   int dim,
                                   float eps,
                                   cudaStream_t stream) {
  if (dim % 8 != 0 ||
      (reinterpret_cast<uintptr_t>(x) & 15) ||
      (reinterpret_cast<uintptr_t>(weight) & 15) ||
      (reinterpret_cast<uintptr_t>(bias) & 15) ||
      (reinterpret_cast<uintptr_t>(out) & 7)) {
    return -1;
  }
  layer_norm_fp8_static_bf16_kernel<<<
      rows, norm_threads(dim), 0, stream>>>(x, weight, bias, out, scale, dim, eps);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? 0 : -static_cast<int>(error);
}

int gate_geglu_merged_fp8_static_bf16_vec(const __nv_bfloat16* merged,
                                          __nv_fp8_e4m3* out,
                                          const float* scale,
                                          int rows,
                                          int hidden,
                                          cudaStream_t stream) {
  const int threads = 256;
  const int work_items = (rows * hidden + 3) / 4;
  gate_geglu_merged_fp8_static_bf16_kernel<<<
      (work_items + threads - 1) / threads, threads, 0, stream>>>(
      merged, out, scale, rows, hidden);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? 0 : -static_cast<int>(error);
}

}
