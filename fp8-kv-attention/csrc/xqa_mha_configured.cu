// SPDX-License-Identifier: Apache-2.0
//
// Fixed public XQA instantiation used by FlashRT Qwen3.6 production paths:
// BF16 Q/O, FP8 E4M3 paged K/V, 24 Q heads, 4 KV heads, head_dim=256,
// page size selected by TOKENS_PER_PAGE, speculative/decode q_seq enabled.

#include "xqa_config.cuh"

#include "flashinfer_xqa_mha_include.cuh"
