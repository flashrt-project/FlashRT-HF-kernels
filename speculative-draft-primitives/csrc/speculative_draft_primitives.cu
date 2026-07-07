// SPDX-License-Identifier: Apache-2.0

#include "speculative_draft_primitives.cuh"

#include <limits>

namespace flashrt_hub::speculative {
namespace {

__global__ void argmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    int64_t* __restrict__ argmax_out,
    int rows,
    int vocab)
{
  extern __shared__ unsigned char smem[];
  float* s_val = reinterpret_cast<float*>(smem);
  int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

  const int row = blockIdx.x;
  if (row >= rows) return;
  const __nv_bfloat16* row_logits =
      logits + static_cast<size_t>(row) * vocab;

  float best_val = -std::numeric_limits<float>::infinity();
  int best_idx = 0;
  for (int col = threadIdx.x; col < vocab; col += blockDim.x) {
    const float v = static_cast<float>(row_logits[col]);
    if (v > best_val || (v == best_val && col < best_idx)) {
      best_val = v;
      best_idx = col;
    }
  }
  s_val[threadIdx.x] = best_val;
  s_idx[threadIdx.x] = best_idx;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other_val = s_val[threadIdx.x + stride];
      const int other_idx = s_idx[threadIdx.x + stride];
      const float cur_val = s_val[threadIdx.x];
      const int cur_idx = s_idx[threadIdx.x];
      if (other_val > cur_val
          || (other_val == cur_val && other_idx < cur_idx)) {
        s_val[threadIdx.x] = other_val;
        s_idx[threadIdx.x] = other_idx;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    argmax_out[row] = static_cast<int64_t>(s_idx[0]);
  }
}

__global__ void argmax_bf16_partition_kernel(
    const __nv_bfloat16* __restrict__ logits,
    float* __restrict__ partial_vals,
    int* __restrict__ partial_idx,
    int rows,
    int vocab,
    int parts)
{
  extern __shared__ unsigned char smem[];
  float* s_val = reinterpret_cast<float*>(smem);
  int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

  const int row = blockIdx.x;
  const int part = blockIdx.y;
  if (row >= rows || part >= parts) return;

  const int cols_per_part = (vocab + parts - 1) / parts;
  const int begin = part * cols_per_part;
  const int end = min(vocab, begin + cols_per_part);
  const __nv_bfloat16* row_logits =
      logits + static_cast<size_t>(row) * vocab;

  float best_val = -std::numeric_limits<float>::infinity();
  int best_idx = 0;
  for (int col = begin + threadIdx.x; col < end; col += blockDim.x) {
    const float v = static_cast<float>(row_logits[col]);
    if (v > best_val || (v == best_val && col < best_idx)) {
      best_val = v;
      best_idx = col;
    }
  }
  s_val[threadIdx.x] = best_val;
  s_idx[threadIdx.x] = best_idx;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other_val = s_val[threadIdx.x + stride];
      const int other_idx = s_idx[threadIdx.x + stride];
      const float cur_val = s_val[threadIdx.x];
      const int cur_idx = s_idx[threadIdx.x];
      if (other_val > cur_val
          || (other_val == cur_val && other_idx < cur_idx)) {
        s_val[threadIdx.x] = other_val;
        s_idx[threadIdx.x] = other_idx;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const int out = row * parts + part;
    partial_vals[out] = s_val[0];
    partial_idx[out] = s_idx[0];
  }
}

__global__ void reduce_accept_partition_kernel(
    const float* __restrict__ partial_vals,
    const int* __restrict__ partial_idx,
    const int64_t* __restrict__ drafts,
    int64_t* __restrict__ argmax_out,
    int* __restrict__ accept_n,
    int rows,
    int parts,
    int spec_k)
{
  for (int row = threadIdx.x; row < rows; row += blockDim.x) {
    float best_val = -std::numeric_limits<float>::infinity();
    int best_idx = 0;
    for (int part = 0; part < parts; ++part) {
      const int off = row * parts + part;
      const float v = partial_vals[off];
      const int idx = partial_idx[off];
      if (v > best_val || (v == best_val && idx < best_idx)) {
        best_val = v;
        best_idx = idx;
      }
    }
    argmax_out[row] = static_cast<int64_t>(best_idx);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int n = 0;
    for (; n < spec_k; ++n) {
      if (argmax_out[n] != drafts[n]) break;
    }
    accept_n[0] = n;
  }
}

__global__ void accept_kernel(
    const int64_t* __restrict__ argmax_out,
    const int64_t* __restrict__ drafts,
    int* __restrict__ accept_n,
    int spec_k)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int n = 0;
  for (; n < spec_k; ++n) {
    if (argmax_out[n] != drafts[n]) break;
  }
  accept_n[0] = n;
}

}  // namespace

void argmax_bf16(
    const __nv_bfloat16* logits,
    int64_t* argmax_out,
    int rows,
    int vocab,
    cudaStream_t stream)
{
  if (rows <= 0 || vocab <= 0) return;
  const int threads = 1024;
  const size_t smem = threads * (sizeof(float) + sizeof(int));
  argmax_bf16_kernel<<<rows, threads, smem, stream>>>(
      logits, argmax_out, rows, vocab);
}

void accept_greedy_bf16(
    const __nv_bfloat16* logits,
    const int64_t* drafts,
    int64_t* argmax_out,
    int* accept_n,
    int rows,
    int vocab,
    int spec_k,
    cudaStream_t stream)
{
  if (rows <= 0 || vocab <= 0 || spec_k <= 0) return;
  const int threads = 1024;
  const size_t smem = threads * (sizeof(float) + sizeof(int));
  argmax_bf16_kernel<<<rows, threads, smem, stream>>>(
      logits, argmax_out, rows, vocab);
  accept_kernel<<<1, 1, 0, stream>>>(argmax_out, drafts, accept_n, spec_k);
}

void accept_partitioned_bf16(
    const __nv_bfloat16* logits,
    const int64_t* drafts,
    int64_t* argmax_out,
    int* accept_n,
    float* partial_vals,
    int* partial_idx,
    int rows,
    int vocab,
    int spec_k,
    int parts,
    cudaStream_t stream)
{
  if (rows <= 0 || vocab <= 0 || spec_k <= 0 || parts <= 0) return;
  parts = min(parts, 128);
  const int threads = 256;
  const size_t smem = threads * (sizeof(float) + sizeof(int));
  const dim3 grid(rows, parts);
  argmax_bf16_partition_kernel<<<grid, threads, smem, stream>>>(
      logits, partial_vals, partial_idx, rows, vocab, parts);
  reduce_accept_partition_kernel<<<1, 128, 0, stream>>>(
      partial_vals, partial_idx, drafts, argmax_out, accept_n,
      rows, parts, spec_k);
}

}  // namespace flashrt_hub::speculative
