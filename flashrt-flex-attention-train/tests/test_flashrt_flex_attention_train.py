#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys

import torch
import torch.nn.functional as F


def load_ops(backend: str, artifact: str | None):
    if backend == "source":
        sys.path.insert(0, "flashrt-flex-attention-train/torch-ext")
        try:
            return importlib.import_module("flashrt_flex_attention_train")
        finally:
            sys.path.remove("flashrt-flex-attention-train/torch-ext")
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("flashrt_flex_attention_train")
    finally:
        if artifact:
            sys.path.remove(artifact)


def _shape(mode: str):
    if mode == "full" and torch.cuda.is_available():
        return "cuda", torch.bfloat16, 2, 8, 13, 6, 256
    return "cpu", torch.float32, 2, 2, 5, 4, 16


def test_matches_explicit_sdpa(flex_ops, mode: str) -> None:
    device, dtype, bsz, heads, prefix, action, dim = _shape(mode)
    torch.manual_seed(11)
    q = torch.randn(bsz, heads, prefix + action, dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    prefix_valid = torch.ones(bsz, prefix, device=device, dtype=torch.bool)
    prefix_valid[0, -1] = False
    prefix_att = torch.zeros(bsz, prefix, device=device, dtype=torch.bool)
    prefix_att[:, prefix // 2 :] = True
    action_valid = torch.ones(bsz, action, device=device, dtype=torch.bool)
    action_valid[1, -1] = False

    out = flex_ops.flex_attention(
        q,
        k,
        v,
        prefix_len=prefix,
        action_block_size=2,
        prefix_valid=prefix_valid,
        prefix_att=prefix_att,
        action_valid=action_valid,
        non_fast_prefix_len=prefix - 1,
    )

    pm, am = flex_ops.build_block_sparse_bool_masks(
        prefix_valid,
        prefix_att,
        batch=bsz,
        prefix_len=prefix,
        action_len=action,
        action_block_size=2,
        non_fast_prefix_len=prefix - 1,
        action_valid=action_valid,
        device=q.device,
    )
    pm = torch.where(
        pm[:, None],
        torch.zeros((), device=q.device, dtype=q.dtype),
        torch.full((), flex_ops.MASK_VALUE_F32, device=q.device, dtype=q.dtype),
    )
    am = torch.where(
        am[:, None],
        torch.zeros((), device=q.device, dtype=q.dtype),
        torch.full((), flex_ops.MASK_VALUE_F32, device=q.device, dtype=q.dtype),
    )
    expected_p = F.scaled_dot_product_attention(q[:, :, :prefix], k, v, attn_mask=pm, scale=dim**-0.5)
    kd = torch.cat([k[:, :, :prefix].detach(), k[:, :, prefix:]], dim=2)
    vd = torch.cat([v[:, :, :prefix].detach(), v[:, :, prefix:]], dim=2)
    expected_a = F.scaled_dot_product_attention(q[:, :, prefix:], kd, vd, attn_mask=am, scale=dim**-0.5)
    expected = torch.cat([expected_p, expected_a], dim=2)
    tol = 2e-3 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)


def test_detached_prefix_semantics(flex_ops, mode: str) -> None:
    device, dtype, bsz, heads, prefix, action, dim = _shape(mode)
    torch.manual_seed(17)
    q = torch.randn(bsz, heads, prefix + action, dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    out = flex_ops.flex_attention(q, k, v, prefix_len=prefix, action_block_size=2)
    out[:, :, prefix:].float().square().mean().backward()
    assert torch.count_nonzero(k.grad[:, :, :prefix]) == 0
    assert torch.count_nonzero(v.grad[:, :, :prefix]) == 0
    assert torch.count_nonzero(k.grad[:, :, prefix:]) > 0
    assert torch.count_nonzero(v.grad[:, :, prefix:]) > 0


def test_dense_attention_mask_path(flex_ops, mode: str) -> None:
    device, dtype, bsz, heads, prefix, action, dim = _shape(mode)
    torch.manual_seed(23)
    q = torch.randn(bsz, heads, prefix + action, dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    pm, am = flex_ops.build_block_sparse_bool_masks(
        None,
        None,
        batch=bsz,
        prefix_len=prefix,
        action_len=action,
        action_block_size=2,
        device=q.device,
    )
    full = torch.cat([pm, am], dim=1)
    mask = torch.where(
        full[:, None],
        torch.zeros((), device=q.device, dtype=q.dtype),
        torch.full((), flex_ops.MASK_VALUE_F32, device=q.device, dtype=q.dtype),
    )
    out_from_dense = flex_ops.flex_attention(q, k, v, prefix_len=prefix, action_block_size=2, attention_mask=mask)
    out_from_parts = flex_ops.flex_attention(q, k, v, prefix_len=prefix, action_block_size=2)
    tol = 2e-3 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(out_from_dense, out_from_parts, atol=tol, rtol=tol)


def run(flex_ops, mode: str) -> None:
    test_matches_explicit_sdpa(flex_ops, mode)
    test_detached_prefix_semantics(flex_ops, mode)
    test_dense_attention_mask_path(flex_ops, mode)
    x = torch.ones(1)
    torch.testing.assert_close(flex_ops.backend_marker(x), x)
    print(f"flashrt-flex-attention-train {mode}: passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    run(load_ops(args.backend, args.artifact), args.mode)
