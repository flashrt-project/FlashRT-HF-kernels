# SPDX-License-Identifier: Apache-2.0
"""API-surface checks for the MiniMax MSA Blackwell Hub package."""

from __future__ import annotations

import minimaxai_msa_blackwell as msa
import pytest


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
        assert item["status"] in {
            "available",
            "available_optional_te",
            "unsupported_callable",
        }
        assert item["target"]
        assert item["reason"]


def test_official_names_are_exported_at_root() -> None:
    for name in OFFICIAL_NAMES:
        assert hasattr(msa, name), f"{name} must be exported for API compatibility"


def test_unsupported_blackwell_paths_fail_explicitly() -> None:
    for name in {
        "sparse_atten_func",
        "sparse_atten_nvfp4_kv_func",
        "fp4_indexer_block_scores",
    }:
        with pytest.raises(NotImplementedError, match=name):
            getattr(msa, name)()


def test_pure_python_compat_helpers() -> None:
    import torch

    scale = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    swizzled = msa.swizzle_nvfp4_scale_to_128x4(scale, rows=2, cols=4)
    assert swizzled.shape == (128, 4)
    assert msa.nvfp4_global_scale_from_amax(torch.tensor([2688.0])).item() == 1.0

    q2k = torch.tensor([[[0, 1], [1, -1]]], dtype=torch.int32)
    cu_q = torch.tensor([0, 2], dtype=torch.int32)
    cu_k = torch.tensor([0, 256], dtype=torch.int32)
    row_ptr, q_idx = msa.build_k2q_csr(q2k, cu_q, cu_k, 128, total_k=256)
    assert row_ptr.dtype == torch.int32
    assert q_idx.dtype == torch.int32
    assert row_ptr.shape == (1, 3)
