// SPDX-License-Identifier: Apache-2.0
//
// Native CUDA block-sparse GQA decode attention for Blackwell (sm_120/121).
// Replaces the Triton flash_decode_with_gqa_share_sparse for the M3 MSA
// decode path.
//
// Split-K flash decode:
//   Phase 1: grid (B, Hkv, num_splits). Each CTA stages its selected K/V
//            blocks into shared memory ONCE (shared by all q-heads in the
//            GQA group -> kills the 16x redundant KV reads of the naive
//            warp-per-head version) and computes a PARTIAL online softmax
//            (m, l, acc) for every q-head over its block subset.
//   Phase 2: grid (B, Hq). Combine the num_splits partials per (b, qh) via
//            the standard log-sum-exp merge.
// head_dim split across the 32 lanes (DPL = D/32). BF16 in, FP32 accum.

#include "msa_decode_attn.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

namespace flashrt_minimax_msa {
namespace {

constexpr int WARP = 32;
constexpr int WARPS = 16;                // 512 threads -> 1 q-head/warp
constexpr int THREADS = WARP * WARPS;
constexpr int CHUNK = 128;               // K/V smem sub-tile (= block_size:
                                         // full-block staging was fastest;
                                         // ncu showed MIO-stall-bound, not
                                         // occupancy-bound, so smaller CHUNK
                                         // (more occupancy) did not help)

// Phase 1: partial softmax over this split's blocks for all q-heads.
//   m_part [B, Hkv, gqa, nsplit]   l_part [B, Hkv, gqa, nsplit]
//   o_part [B, Hkv, gqa, nsplit, D]
__global__ void msa_decode_p1_kernel(
    const __nv_bfloat16* __restrict__ q,         // [B, Hq, D]
    const __nv_bfloat16* __restrict__ kv,        // [max_slots,2,max_len,Hkv,D]
    const int* __restrict__ seq_lens,            // [B]
    const int64_t* __restrict__ slot_ids,        // [B]
    const int* __restrict__ topk_idx,            // [Hkv, B, topk]
    float* __restrict__ m_part, float* __restrict__ l_part,
    float* __restrict__ o_part,
    int B, int Hq, int Hkv, int D, int max_slots, int max_len,
    int block_size, int topk, int nsplit, int bps, float sm_scale, int DPL) {
  extern __shared__ __nv_bfloat16 smem[];      // K[CHUNK*D] + V[CHUNK*D]
  __nv_bfloat16* sK = smem;
  __nv_bfloat16* sV = smem + CHUNK * D;

  const int b = blockIdx.x;
  const int kh = blockIdx.y;
  const int split = blockIdx.z;
  const int gqa = Hq / Hkv;
  const int tid = threadIdx.x;
  const int warp = tid / WARP;
  const int lane = tid & (WARP - 1);

  const int seq_len = seq_lens[b] > 0 ? seq_lens[b] : 0;
  long sid = ((slot_ids[b] % max_slots) + max_slots) % max_slots;
  const long kv_k_base = (((sid * 2 + 0) * max_len) * Hkv + kh) * D;
  const long kv_v_base = (((sid * 2 + 1) * max_len) * Hkv + kh) * D;
  const long pos_stride = (long)Hkv * D;
  const int* idx_row = topk_idx + ((long)kh * B + b) * topk;

  // per-q-head partial state (this warp owns q-heads warp, warp+WARPS, ...)
  // gqa(16) q-heads over WARPS(8) warps -> 2 q-heads per warp.
  const int qpw = (gqa + WARPS - 1) / WARPS;   // q-heads per warp (=2)
  float m[2], l[2], acc[2][8];
  #pragma unroll
  for (int j = 0; j < qpw; ++j) {
    m[j] = -CUDART_INF_F; l[j] = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[j][i] = 0.f;
  }
  const int d0 = lane * DPL;
  float qreg[2][8];
  #pragma unroll
  for (int j = 0; j < qpw; ++j) {
    const int g = warp + j * WARPS;
    if (g < gqa) {
      const __nv_bfloat16* qr = q + ((long)b * Hq + (kh * gqa + g)) * D + d0;
      #pragma unroll
      for (int i = 0; i < DPL; ++i) qreg[j][i] = __bfloat162float(qr[i]);
    }
  }

  // iterate this split's blocks
  for (int t = split * bps; t < (split + 1) * bps && t < topk; ++t) {
    const int blk = idx_row[t];
    if (blk < 0) continue;
    const int start = blk * block_size;
    if (start >= seq_len) continue;
    const int n = min(block_size, seq_len - start);
    const int DU = D / 8;                         // uint4 chunks per row

    // Process the block in CHUNK-key sub-tiles so smem holds only CHUNK rows
    // of K+V (not block_size). This halves smem -> 2 blocks/SM (ncu showed
    // occupancy was shared-mem-limited to 1 block/SM at 33%), doubling warp
    // occupancy to hide the MIO (smem+shuffle) stalls. Online softmax state
    // (m/l/acc) persists across chunks.
    for (int cs = 0; cs < n; cs += CHUNK) {
      const int ce = min(CHUNK, n - cs);
      for (int c = tid; c < ce * DU; c += THREADS) {
        const int pos = c / DU, du = (c - pos * DU) * 8;
        const long gp = (long)(start + cs + pos) * pos_stride + du;
        *reinterpret_cast<uint4*>(sK + pos * D + du) =
            *reinterpret_cast<const uint4*>(kv + kv_k_base + gp);
        *reinterpret_cast<uint4*>(sV + pos * D + du) =
            *reinterpret_cast<const uint4*>(kv + kv_v_base + gp);
      }
      __syncthreads();

      #pragma unroll
      for (int j = 0; j < qpw; ++j) {
        const int g = warp + j * WARPS;
        if (g >= gqa) continue;
        for (int pos = 0; pos < ce; ++pos) {
          const __nv_bfloat16* kr = sK + pos * D + d0;
          float part = 0.f;
          #pragma unroll
          for (int i = 0; i < DPL; ++i)
            part += qreg[j][i] * __bfloat162float(kr[i]);
          #pragma unroll
          for (int o = WARP / 2; o > 0; o >>= 1)
            part += __shfl_xor_sync(0xffffffffu, part, o);
          const float score = part * sm_scale;
          const float mn = fmaxf(m[j], score);
          const float corr = __expf(m[j] - mn);
          const float p = __expf(score - mn);
          l[j] = l[j] * corr + p;
          const __nv_bfloat16* vr = sV + pos * D + d0;
          #pragma unroll
          for (int i = 0; i < DPL; ++i)
            acc[j][i] = acc[j][i] * corr + p * __bfloat162float(vr[i]);
          m[j] = mn;
        }
      }
      __syncthreads();
    }
  }

  // write partials
  #pragma unroll
  for (int j = 0; j < qpw; ++j) {
    const int g = warp + j * WARPS;
    if (g >= gqa) continue;
    const long pidx = (((long)b * Hkv + kh) * gqa + g) * nsplit + split;
    if (lane == 0) { m_part[pidx] = m[j]; l_part[pidx] = l[j]; }
    float* op = o_part + pidx * D + d0;
    #pragma unroll
    for (int i = 0; i < DPL; ++i) op[i] = acc[j][i];
  }
}

// Phase 2: combine nsplit partials per (b, qh). grid (B, Hq), block D.
__global__ void msa_decode_combine_kernel(
    const float* __restrict__ m_part, const float* __restrict__ l_part,
    const float* __restrict__ o_part, __nv_bfloat16* __restrict__ out,
    int B, int Hq, int Hkv, int D, int nsplit) {
  const int b = blockIdx.x, qh = blockIdx.y, d = threadIdx.x;
  const int gqa = Hq / Hkv, kh = qh / gqa, g = qh - kh * gqa;
  const long base = (((long)b * Hkv + kh) * gqa + g) * nsplit;
  float M = -CUDART_INF_F;
  for (int s = 0; s < nsplit; ++s) M = fmaxf(M, m_part[base + s]);
  float denom = 0.f, num = 0.f;
  for (int s = 0; s < nsplit; ++s) {
    const float ms = m_part[base + s];
    if (ms == -CUDART_INF_F) continue;
    const float w = __expf(ms - M);
    denom += w * l_part[base + s];
    num += w * o_part[(base + s) * D + d];
  }
  out[((long)b * Hq + qh) * D + d] =
      __float2bfloat16(denom > 0.f ? num / denom : 0.f);
}

}  // namespace

void msa_decode_sparse_attn_cuda(const void* q, const void* kv_cache,
                                 const int* seq_lens, const int64_t* slot_ids,
                                 const int* topk_idx, void* out,
                                 int B, int Hq, int Hkv, int D,
                                 int max_slots, int max_len,
                                 int block_size, int topk,
                                 float sm_scale, cudaStream_t stream) {
  const int gqa = Hq / Hkv;
  const int DPL = D / WARP;
  // 1 block per split (max occupancy for small B); cap CTAs sanely.
  const int nsplit = topk;
  const int bps = 1;

  // Cached workspace (grows as needed) — avoids per-call cudaMalloc/Free,
  // which dominate at decode's tiny ~15us roofline.
  static float* ws = nullptr;
  static long ws_floats = 0;
  const long n_pl = (long)B * Hkv * gqa * nsplit;
  const long need = n_pl * 2 + n_pl * D;       // m + l + o
  if (need > ws_floats) {
    if (ws) cudaFree(ws);
    cudaMalloc(&ws, need * sizeof(float));
    ws_floats = need;
  }
  float* m_part = ws;
  float* l_part = ws + n_pl;
  float* o_part = ws + n_pl * 2;

  dim3 g1(B, Hkv, nsplit);
  const size_t smem = (size_t)2 * CHUNK * D * sizeof(__nv_bfloat16);
  static int set_smem = -1;
  if ((int)smem != set_smem) {
    cudaFuncSetAttribute(msa_decode_p1_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    set_smem = (int)smem;
  }
  msa_decode_p1_kernel<<<g1, THREADS, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(kv_cache),
      seq_lens, slot_ids, topk_idx, m_part, l_part, o_part,
      B, Hq, Hkv, D, max_slots, max_len, block_size, topk, nsplit, bps,
      sm_scale, DPL);

  dim3 g2(B, Hq);
  msa_decode_combine_kernel<<<g2, D, 0, stream>>>(
      m_part, l_part, o_part,
      reinterpret_cast<__nv_bfloat16*>(out), B, Hq, Hkv, D, nsplit);
}

}  // namespace flashrt_minimax_msa
