// SPDX-License-Identifier: Apache-2.0
//
// Head-dimension 256, GQA group-size 8 XQA instantiation for the
// 16Q/2KV profile used by 35B-A3B transformer full-attention layers.

#define NDEBUG 1
#define BEAM_WIDTH 1
#define USE_INPUT_KV 0
#define USE_CUSTOM_BARRIER 1
#define MLA_WRAPPER 0
#define USE_SM90_MHA 0
#define INPUT_FP16 0
#define DTYPE __nv_bfloat16
#define CACHE_ELEM_ENUM 2
#ifndef TOKENS_PER_PAGE
#define TOKENS_PER_PAGE 32
#endif
#define HEAD_ELEMS 256
#define HEAD_GRP_SIZE 8
#define SLIDING_WINDOW 0
#define LOW_PREC_OUTPUT 0
#define SPEC_DEC 1

// Keep this instantiation's generated symbols distinct from the D256/G6 and
// D128 images linked into the same extension.
#define kernelType kernelType_d256_g8
#define smemSize smemSize_d256_g8
#define kernel_mha kernel_mha_d256_g8
#define launchMHA launchMHA_d256_g8
#define launchMHAFlashInfer launchMHAFlashInfer_d256_g8

#include "flashinfer_xqa_mha_include.cuh"
#include "xqa_bf16_fp8kv.cuh"

void flashrt_xqa_bf16_fp8kv_d256_g8(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* page_table,
    const uint32_t* seq_lens,
    const uint32_t* mask,
    void* out,
    uint32_t* semaphores,
    void* scratch,
    int max_seq_len,
    int q_seq_len,
    int num_kv_heads,
    int sm_count,
    float q_scale,
    float kv_scale,
    bool enable_pdl,
    int64_t k_stride_page,
    int64_t k_stride_token,
    int64_t k_stride_head,
    cudaStream_t stream) {
  launchMHAFlashInfer_d256_g8(
      static_cast<uint32_t>(sm_count),
      static_cast<uint32_t>(num_kv_heads),
      8,
      0,
      q_scale,
      nullptr,
      reinterpret_cast<OutputHead*>(out),
      reinterpret_cast<InputHead const*>(q),
      nullptr,
      reinterpret_cast<GMemCacheHead*>(const_cast<void*>(k_cache)),
      reinterpret_cast<GMemCacheHead*>(const_cast<void*>(v_cache)),
      reinterpret_cast<KVCachePageIndex const*>(page_table),
      static_cast<uint32_t>(max_seq_len),
      seq_lens,
      1,
      kv_scale,
      nullptr,
      static_cast<uint32_t>(q_seq_len),
      nullptr,
      reinterpret_cast<MaskType const*>(mask),
      semaphores,
      scratch,
      enable_pdl,
      static_cast<uint64_t>(k_stride_page),
      static_cast<uint64_t>(k_stride_token),
      static_cast<uint64_t>(k_stride_head),
      stream);
}
