# SPDX-License-Identifier: Apache-2.0
"""FlashRT Blackwell extension of MiniMax MSA decode sparse attention.

This package targets NVIDIA Blackwell-family CUDA 12.x GPUs and is validated on
DGX Spark / GB10 / SM121. It exposes the
MiniMax-M3 decode-sparse attention path used by FlashRT's MiniMax Spark runtime.

Implementation status:
  * native CUDA helper: score -> top-k sparse block ids;
  * native CUDA tensor-core sparse decode for the MiniMax-M3 Blackwell shape;
  * native CUDA FP4 block-score indexer;
  * native CUDA helper: swizzled NVFP4 -> BF16 dequant for the W4A16 quality path;
  * Blackwell-validated sparse prefill path;
  * MiniMaxAI/msa compatibility wrappers for CSR prefill, decode, NVFP4 helpers,
    and FP4 block-score helpers.

The upstream MiniMaxAI/msa package is SM100-only; this package is the Blackwell
extension path. The public API is Tensor-oriented and independent from
FlashRT's serving runtime.

Public API
----------
Lightning indexer (q.k blockmax score -> top-k block select), paged KV:
    flash_decode_with_topk_idx      (single-token decode, split-K)

Block-sparse GQA attention (consumes block_indices from the indexer):
    flash_decode_with_gqa_share_sparse    (decode, split-K over top-k blocks)
    sparse_atten_func                     (official CSR prefill wrapper)
    sparse_decode_atten_func              (official paged decode wrapper)

PyTorch naive references (for correctness checks):
    naive_flash_decode_with_gqa_share_sparse   (attention given topk_idx)
    naive_flash_decode_with_topk_idx           (joint indexer + attention)

All Triton entrypoints expect a paged KV cache:
    k_cache / v_cache : [max_slots, num_kv_heads, head_dim]
    req_to_token      : [max_reqs, max_kv_len]  (logical pos -> physical slot)
    slot_ids          : [batch]                 (per-request row in req_to_token)
    topk_idx          : [num_kv_heads, n, topk] int32, valid ids left-packed,
                        -1 right-padding (the M3 indexer contract).
See VENDOR_NOTES.md for full shape/dtype/layout contracts.
"""

from .common.utils import get_cu_seqblocks, robust_allocator
from .decode.flash_with_topk_idx import flash_decode_with_topk_idx
from .decode.topk_sparse import flash_decode_with_gqa_share_sparse
from .interface import (
    SparseDecodePagedAttentionWrapper,
    fp4_indexer_block_scores,
    sparse_atten_func,
    sparse_atten_nvfp4_kv_func,
    sparse_decode_atten_func,
)
from ._native import (
    has_native_ops,
    native_nvfp4_dequant_swizzled_to_bf16,
    native_topk_from_scores,
)
from .naive.flash_with_topk_idx import naive_flash_decode_with_topk_idx
from .naive.topk_sparse import naive_flash_decode_with_gqa_share_sparse
from .quantize import (
    Nvfp4QuantizedTensor,
    dequantize_nvfp4_128x4_to_bf16,
    nvfp4_global_scale_from_amax,
    quantize_bf16_to_nvfp4_128x4,
    quantize_kv_bf16_to_nvfp4_128x4,
    swizzle_nvfp4_scale_to_128x4,
)
from .sparse_index_utils import SparseK2qCsrBuilderSm100, build_k2q_csr
from .api_status import (
    OFFICIAL_API_STATUS,
    OFFICIAL_MINIMAX_MSA_FUNCTIONS,
    V1_AVAILABLE_FUNCTIONS,
    available_functions,
    official_api_status,
    official_minimax_msa_functions,
    unsupported_official_functions,
)

__all__ = [
    "sparse_atten_func",
    "sparse_atten_nvfp4_kv_func",
    "sparse_decode_atten_func",
    "SparseDecodePagedAttentionWrapper",
    "fp4_indexer_block_scores",
    "build_k2q_csr",
    "SparseK2qCsrBuilderSm100",
    "Nvfp4QuantizedTensor",
    "quantize_bf16_to_nvfp4_128x4",
    "quantize_kv_bf16_to_nvfp4_128x4",
    "dequantize_nvfp4_128x4_to_bf16",
    "swizzle_nvfp4_scale_to_128x4",
    "nvfp4_global_scale_from_amax",
    "flash_decode_with_topk_idx",
    "flash_decode_with_gqa_share_sparse",
    "has_native_ops",
    "native_topk_from_scores",
    "native_nvfp4_dequant_swizzled_to_bf16",
    "naive_flash_decode_with_gqa_share_sparse",
    "naive_flash_decode_with_topk_idx",
    "get_cu_seqblocks",
    "robust_allocator",
    "OFFICIAL_API_STATUS",
    "OFFICIAL_MINIMAX_MSA_FUNCTIONS",
    "V1_AVAILABLE_FUNCTIONS",
    "available_functions",
    "official_api_status",
    "official_minimax_msa_functions",
    "unsupported_official_functions",
]
