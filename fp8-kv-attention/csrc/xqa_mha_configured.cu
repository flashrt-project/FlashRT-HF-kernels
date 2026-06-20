// SPDX-License-Identifier: Apache-2.0
//
// Fixed public XQA instantiation used by FlashRT Qwen3.6 production paths:
// BF16 Q/O, FP8 E4M3 paged K/V, 24 Q heads, 4 KV heads, head_dim=256,
// page size 128, speculative/decode q_seq support enabled.

#include "xqa_config.cuh"

#include "attention/flashinfer_xqa_src/mha.cu"
