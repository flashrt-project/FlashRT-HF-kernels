# SPDX-License-Identifier: Apache-2.0
"""API-surface checks for the MiniMax MSA Blackwell Hub package."""

from __future__ import annotations

import minimaxai_msa_blackwell as msa


OFFICIAL_NAMES = {
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
}


def test_v1_available_functions_are_exported() -> None:
    available = set(msa.available_functions())
    assert available == set(msa.V1_AVAILABLE_FUNCTIONS)
    for name in available:
        assert hasattr(msa, name), f"{name} is listed as available but not exported"


def test_official_api_status_is_complete() -> None:
    tracked = set(msa.official_minimax_msa_functions())
    assert tracked == OFFICIAL_NAMES
    status = msa.official_api_status()
    assert set(status) == OFFICIAL_NAMES
    for name, item in status.items():
        assert item["status"] in {"available", "planned", "not_applicable_name"}
        assert item["target"]
        assert item["reason"]


def test_v1_does_not_export_unvalidated_official_names() -> None:
    unsupported = set(msa.unsupported_official_functions())
    assert unsupported == OFFICIAL_NAMES
    for name in unsupported:
        assert not hasattr(msa, name), f"{name} must not be exported until validated"
