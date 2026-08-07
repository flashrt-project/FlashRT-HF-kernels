# SPDX-License-Identifier: Apache-2.0
"""API-surface checks for the MiniMax MSA Blackwell Hub package."""
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

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


def _load_module(artifact):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("minimaxai_msa_blackwell")
    finally:
        if artifact:
            sys.path.remove(artifact)


def _check_v1_available_functions_are_exported(msa) -> None:
    available = set(msa.available_functions())
    if available != set(msa.V1_AVAILABLE_FUNCTIONS):
        raise AssertionError("available_functions() must equal V1_AVAILABLE_FUNCTIONS")
    for name in available:
        if not hasattr(msa, name):
            raise AssertionError(f"{name} is listed as available but not exported")
    print("PASS v1 available functions are exported")


def _check_official_api_status_is_complete(msa) -> None:
    tracked = set(msa.official_minimax_msa_functions())
    if tracked != OFFICIAL_NAMES:
        raise AssertionError(f"tracked {tracked} != official {OFFICIAL_NAMES}")
    status = msa.official_api_status()
    if set(status) != OFFICIAL_NAMES:
        raise AssertionError("official_api_status() keys must match OFFICIAL_NAMES")
    for name, item in status.items():
        if item["status"] not in {"available", "available_optional_te"}:
            raise AssertionError(f"{name}: unexpected status {item['status']}")
        if not item["target"]:
            raise AssertionError(f"{name}: missing target")
        if not item["reason"]:
            raise AssertionError(f"{name}: missing reason")
    print("PASS official API status is complete")


def _check_official_names_are_exported_at_root(msa) -> None:
    for name in OFFICIAL_NAMES:
        if not hasattr(msa, name):
            raise AssertionError(f"{name} must be exported for API compatibility")
    print("PASS official names are exported at root")


def _check_pure_python_compat_helpers(msa) -> None:
    import torch

    scale = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    swizzled = msa.swizzle_nvfp4_scale_to_128x4(scale, rows=2, cols=4)
    if swizzled.shape != (128, 4):
        raise AssertionError("swizzle_nvfp4_scale_to_128x4 shape mismatch")
    if msa.nvfp4_global_scale_from_amax(torch.tensor([2688.0])).item() != 1.0:
        raise AssertionError("nvfp4_global_scale_from_amax mismatch")

    q2k = torch.tensor([[[0, 1], [1, -1]]], dtype=torch.int32)
    cu_q = torch.tensor([0, 2], dtype=torch.int32)
    cu_k = torch.tensor([0, 256], dtype=torch.int32)
    row_ptr, q_idx = msa.build_k2q_csr(q2k, cu_q, cu_k, 128, total_k=256)
    if row_ptr.dtype != torch.int32 or q_idx.dtype != torch.int32:
        raise AssertionError("build_k2q_csr must return int32 tensors")
    if row_ptr.shape != (1, 3):
        raise AssertionError(f"build_k2q_csr row_ptr shape {row_ptr.shape}")
    print("PASS pure-python compat helpers")


def _check_fp4_indexer_block_scores_is_callable(msa) -> None:
    import torch

    total_q, hq, hkv, pages, packed_d = 2, 4, 1, 1, 64
    q_fp4 = torch.zeros(total_q, hq, packed_d, dtype=torch.uint8)
    k_fp4 = torch.zeros(pages, hkv, 128, packed_d, dtype=torch.uint8)
    q_scale = torch.ones(total_q, hq, 8, dtype=torch.float32)
    k_scale = torch.ones(pages, hkv, 128, 8, dtype=torch.float32)
    cu_q = torch.tensor([0, total_q], dtype=torch.int32)
    cu_k = torch.tensor([0, 128], dtype=torch.int32)
    cu_pages = torch.tensor([0, 1], dtype=torch.int32)
    kv_indices = torch.tensor([0], dtype=torch.int32)

    scores = msa.fp4_indexer_block_scores(
        q_fp4, k_fp4, q_scale, k_scale, cu_q, cu_k, cu_pages,
        max_seqlen_q=total_q, max_seqlen_k=128, kv_indices=kv_indices,
        fp4_format="nvfp4", causal=True, scale_layout="public",
    )
    if scores.shape != (hq, 1, total_q):
        raise AssertionError(f"fp4_indexer_block_scores shape {scores.shape}")
    if not torch.isfinite(scores).all():
        raise AssertionError("fp4_indexer_block_scores must be finite")
    print("PASS fp4_indexer_block_scores is callable")


def run(args) -> None:
    msa = _load_module(args.artifact)
    _check_v1_available_functions_are_exported(msa)
    _check_official_api_status_is_complete(msa)
    _check_official_names_are_exported_at_root(msa)
    _check_pure_python_compat_helpers(msa)
    _check_fp4_indexer_block_scores_is_callable(msa)
    print("PASS MiniMaxAI-msa-blackwell API surface: 5 checks")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source",
                        help="the Hub package is the supported path; source falls back to sys.path")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps({"passed": 1, "total": 1, "backend": args.backend}) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
