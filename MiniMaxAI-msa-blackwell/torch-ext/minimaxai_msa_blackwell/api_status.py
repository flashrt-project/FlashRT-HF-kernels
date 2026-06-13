# SPDX-License-Identifier: Apache-2.0
"""Public API status for the FlashRT MiniMax MSA Blackwell package."""

from __future__ import annotations

from types import MappingProxyType


V1_AVAILABLE_FUNCTIONS = (
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
    "has_native_ops",
    "native_topk_from_scores",
    "native_nvfp4_dequant_swizzled_to_bf16",
    "flash_decode_with_topk_idx",
    "flash_decode_with_gqa_share_sparse",
    "naive_flash_decode_with_topk_idx",
    "naive_flash_decode_with_gqa_share_sparse",
    "get_cu_seqblocks",
    "robust_allocator",
)

OFFICIAL_MINIMAX_MSA_FUNCTIONS = (
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
)

_OFFICIAL_API_STATUS = {
    "sparse_atten_func": {
        "status": "available",
        "target": "blackwell-prefill",
        "reason": "Official CSR sparse prefill API maps to the Blackwell Triton BF16/FP16 prefill wrapper.",
    },
    "sparse_atten_nvfp4_kv_func": {
        "status": "available",
        "target": "blackwell-prefill-nvfp4",
        "reason": "Official NVFP4 KV prefill API is available via native swizzled NVFP4 dequantization followed by the Blackwell prefill wrapper.",
    },
    "sparse_decode_atten_func": {
        "status": "available",
        "target": "v1",
        "reason": "Official decode API name maps to the Blackwell validated BF16/FP16 paged sparse decode wrapper.",
    },
    "SparseDecodePagedAttentionWrapper": {
        "status": "available",
        "target": "v1",
        "reason": "Official wrapper name is available for the Blackwell single-token paged sparse decode path.",
    },
    "fp4_indexer_block_scores": {
        "status": "available",
        "target": "blackwell-fp4-indexer",
        "reason": "Official FP4 block-score API uses the native Blackwell block-score kernel in built artifacts, with a reference fallback in source-tree mode.",
    },
    "build_k2q_csr": {
        "status": "available",
        "target": "v1",
        "reason": "CSR construction helper is available for the official prefill API.",
    },
    "SparseK2qCsrBuilderSm100": {
        "status": "available",
        "target": "v1",
        "reason": "Compatibility class name is exported and delegates build() to build_k2q_csr.",
    },
    "Nvfp4QuantizedTensor": {
        "status": "available",
        "target": "v1",
        "reason": "NVFP4 metadata dataclass is available.",
    },
    "quantize_bf16_to_nvfp4_128x4": {
        "status": "available_optional_te",
        "target": "v1",
        "reason": "Function is available and uses Transformer Engine when installed.",
    },
    "quantize_kv_bf16_to_nvfp4_128x4": {
        "status": "available_optional_te",
        "target": "v1",
        "reason": "Function is available and uses Transformer Engine when installed.",
    },
    "dequantize_nvfp4_128x4_to_bf16": {
        "status": "available",
        "target": "v1",
        "reason": "Reference dequant helper is available.",
    },
    "swizzle_nvfp4_scale_to_128x4": {
        "status": "available",
        "target": "v1",
        "reason": "Scale swizzle helper is available.",
    },
    "nvfp4_global_scale_from_amax": {
        "status": "available",
        "target": "v1",
        "reason": "Global scale helper is available.",
    },
}

OFFICIAL_API_STATUS = MappingProxyType(_OFFICIAL_API_STATUS)


def available_functions() -> tuple[str, ...]:
    """Return the functions/classes intentionally available in v1."""

    return V1_AVAILABLE_FUNCTIONS


def official_minimax_msa_functions() -> tuple[str, ...]:
    """Return the upstream MiniMaxAI/msa public function/class names tracked here."""

    return OFFICIAL_MINIMAX_MSA_FUNCTIONS


def official_api_status() -> dict[str, dict[str, str]]:
    """Return a copy of the official API compatibility status table."""

    return {name: dict(status) for name, status in OFFICIAL_API_STATUS.items()}


def unsupported_official_functions() -> tuple[str, ...]:
    """Return official MiniMaxAI/msa names that are not exported by this v1 package."""

    return tuple(
        name
        for name in OFFICIAL_MINIMAX_MSA_FUNCTIONS
        if OFFICIAL_API_STATUS[name]["status"] not in {"available", "available_optional_te"}
    )
