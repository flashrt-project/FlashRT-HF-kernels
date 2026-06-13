// SPDX-License-Identifier: Apache-2.0
//
// Native CUDA block-max QK scoring for the MSA lightning indexer. Replaces the
// O(batch*blocks*queries*heads) Python reference loop in
// fp4_indexer_block_scores: one warp per (query, q_head) walks that query's
// causal kv pages, and for each page reduces the raw dot product q . k over the
// block's visible keys to a single max. The fp4 -> bf16 dequant is done by the
// caller (linear cost); this kernel handles the cubic block-max QK work.

#include "msa_indexer_block_scores.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

namespace flashrt_minimax_msa {
namespace {

constexpr int WARP = 32;

__global__ void indexer_block_scores_kernel(
    const __nv_bfloat16* __restrict__ q,        // [total_q, Hq, D]
    const __nv_bfloat16* __restrict__ k_pages,   // [num_pages, Hkv, 128, D]
    const int* __restrict__ batch_of_q, const int* __restrict__ cu_q,
    const int* __restrict__ cu_k, const int* __restrict__ cu_pages,
    const int* __restrict__ kv_indices, float* __restrict__ scores,
    int total_q, int Hq, int Hkv, int D, int num_pages, int max_blocks,
    int page_size, int causal) {
  const int qi = blockIdx.x;
  const int qh = blockIdx.y;
  if (qi >= total_q) return;
  const int lane = threadIdx.x & (WARP - 1);
  const int gqa = Hq / Hkv;
  const int kh = qh / gqa;
  const int DPL = D / WARP;                       // dims per lane

  const int b = batch_of_q[qi];
  const int q_start = cu_q[b];
  const int local_q = qi - q_start;
  const int k_len = cu_k[b + 1] - cu_k[b];
  const int page_start = cu_pages[b];
  const int npages = cu_pages[b + 1] - page_start;

  // load this lane's slice of the query vector
  float qr[8];
  const __nv_bfloat16* qrow = q + ((long)qi * Hq + qh) * D + lane * DPL;
  #pragma unroll
  for (int i = 0; i < DPL; ++i) qr[i] = __bfloat162float(qrow[i]);

  for (int lblk = 0; lblk < npages; ++lblk) {
    const long out_idx =
        ((long)qh * max_blocks + lblk) * total_q + qi;
    if (lblk >= max_blocks) break;
    const int pp = kv_indices[page_start + lblk];
    int visible = 0;
    if (pp >= 0 && pp < num_pages) {
      const int valid = min(page_size, k_len - lblk * page_size);
      if (valid > 0) {
        if (causal) {
          if (lblk * page_size > local_q) {
            visible = 0;
          } else {
            visible = min(valid, local_q - lblk * page_size + 1);
          }
        } else {
          visible = valid;
        }
      }
    }
    if (visible <= 0) {
      if (lane == 0) scores[out_idx] = -CUDART_INF_F;
      continue;
    }
    const __nv_bfloat16* kbase =
        k_pages + (((long)pp * Hkv + kh) * page_size) * D + lane * DPL;
    float bmax = -CUDART_INF_F;
    for (int key = 0; key < visible; ++key) {
      const __nv_bfloat16* kr = kbase + (long)key * D;
      float part = 0.f;
      #pragma unroll
      for (int i = 0; i < DPL; ++i) part += qr[i] * __bfloat162float(kr[i]);
      #pragma unroll
      for (int o = WARP / 2; o > 0; o >>= 1)
        part += __shfl_xor_sync(0xffffffffu, part, o);
      bmax = fmaxf(bmax, part);
    }
    if (lane == 0) scores[out_idx] = bmax;
  }
}

}  // namespace

void msa_indexer_block_scores_cuda(
    const void* q, const void* k_pages, const int* batch_of_q, const int* cu_q,
    const int* cu_k, const int* cu_pages, const int* kv_indices, float* scores,
    int total_q, int Hq, int Hkv, int D, int num_pages, int max_blocks,
    int page_size, bool causal, cudaStream_t stream) {
  if (total_q == 0 || Hq == 0) return;
  dim3 grid(total_q, Hq);
  indexer_block_scores_kernel<<<grid, WARP, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k_pages), batch_of_q, cu_q, cu_k,
      cu_pages, kv_indices, scores, total_q, Hq, Hkv, D, num_pages, max_blocks,
      page_size, causal ? 1 : 0);
}

}  // namespace flashrt_minimax_msa
