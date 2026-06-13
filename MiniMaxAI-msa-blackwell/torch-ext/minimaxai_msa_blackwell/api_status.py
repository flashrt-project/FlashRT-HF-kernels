# SPDX-License-Identifier: Apache-2.0
"""Public API status for the FlashRT MiniMax MSA Blackwell package."""

from __future__ import annotations

from types import MappingProxyType


V1_AVAILABLE_FUNCTIONS = (
    "has_native_ops",
    "native_topk_from_scores",
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
        "status": "planned",
        "target": "v2+",
        "reason": "SM100 prefill CSR/CUTE API; Blackwell implementation must be validated separately.",
    },
    "sparse_atten_nvfp4_kv_func": {
        "status": "planned",
        "target": "v2+",
        "reason": "SM100 NVFP4 KV sparse prefill API; requires Blackwell NVFP4 attention validation.",
    },
    "sparse_decode_atten_func": {
        "status": "planned",
        "target": "v2",
        "reason": "Official FP8 paged decode contract differs from the current validated BF16 decode-sparse API.",
    },
    "SparseDecodePagedAttentionWrapper": {
        "status": "planned",
        "target": "v2",
        "reason": "Wrapper should preserve official plan/run semantics; current v1 exposes direct Tensor functions.",
    },
    "fp4_indexer_block_scores": {
        "status": "planned",
        "target": "v2+",
        "reason": "Official FP4 indexer is CUTE/SM100-specific; Blackwell score path must be validated.",
    },
    "build_k2q_csr": {
        "status": "planned",
        "target": "v2+",
        "reason": "Official CSR builder uses a dedicated CUDA op; Blackwell CSR builder is not packaged in v1.",
    },
    "SparseK2qCsrBuilderSm100": {
        "status": "not_applicable_name",
        "target": "v2+",
        "reason": "The public class name is SM100-specific; Blackwell should use a new name plus optional compatibility alias.",
    },
    "Nvfp4QuantizedTensor": {
        "status": "planned",
        "target": "v2+",
        "reason": "NVFP4 helper API should be added with quant/dequant correctness tests.",
    },
    "quantize_bf16_to_nvfp4_128x4": {
        "status": "planned",
        "target": "v2+",
        "reason": "Depends on Transformer Engine NVFP4 semantics and 128x4 scale layout validation.",
    },
    "quantize_kv_bf16_to_nvfp4_128x4": {
        "status": "planned",
        "target": "v2+",
        "reason": "Depends on Transformer Engine NVFP4 semantics and 128x4 scale layout validation.",
    },
    "dequantize_nvfp4_128x4_to_bf16": {
        "status": "planned",
        "target": "v2+",
        "reason": "Reference dequant helper must be paired with quantization layout tests.",
    },
    "swizzle_nvfp4_scale_to_128x4": {
        "status": "planned",
        "target": "v2+",
        "reason": "Scale swizzle helper must be validated against the official 128x4 layout.",
    },
    "nvfp4_global_scale_from_amax": {
        "status": "planned",
        "target": "v2+",
        "reason": "Scale helper is straightforward but should ship together with NVFP4 validation.",
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
        if OFFICIAL_API_STATUS[name]["status"] != "available"
    )
