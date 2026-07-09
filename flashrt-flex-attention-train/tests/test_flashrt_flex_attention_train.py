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


def test_manual_matches_reference(flex_ops, mode: str) -> None:
    device, dtype, bsz, heads, prefix, action, dim = _shape(mode)
    kv_heads = 1 if mode == "full" else heads  # GQA on the real-shape run
    torch.manual_seed(29)
    total = prefix + action
    q1 = torch.randn(bsz, heads, total, dim, device=device, dtype=dtype, requires_grad=True)
    k1 = torch.randn(bsz, kv_heads, total, dim, device=device, dtype=dtype, requires_grad=True)
    v1 = torch.randn_like(k1, requires_grad=True)
    q2 = q1.detach().clone().requires_grad_(True)
    k2 = k1.detach().clone().requires_grad_(True)
    v2 = v1.detach().clone().requires_grad_(True)
    prefix_att = torch.zeros(bsz, prefix, device=device, dtype=torch.bool)
    prefix_att[:, prefix // 2 :] = True
    kwargs = dict(prefix_len=prefix, action_block_size=2, prefix_att=prefix_att)

    ref = flex_ops.flex_attention(q1, k1, v1, **kwargs)
    got = flex_ops.manual_attention(q2, k2, v2, compile_part=device != "cpu", **kwargs)
    # bf16-logits class: the manual path stores logits in the io dtype
    # between the GEMM and the fp32 softmax.
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(got, ref, atol=tol, rtol=tol)

    ref.float().square().mean().backward()
    got.float().square().mean().backward()
    for a, b in ((q1, q2), (k1, k2), (v1, v2)):
        denom = torch.linalg.vector_norm(a.grad.float()).clamp_min(1e-12)
        rel = torch.linalg.vector_norm((a.grad - b.grad).float()) / denom
        assert float(rel) <= 2e-2, f"grad rel diff {float(rel)}"

    # impl dispatch reaches the same path
    via_impl = flex_ops.flex_attention(
        q2.detach(), k2.detach(), v2.detach(), impl="manual", **kwargs
    )
    torch.testing.assert_close(via_impl, got.detach(), atol=tol, rtol=tol)


def run(flex_ops, mode: str) -> None:
    test_matches_explicit_sdpa(flex_ops, mode)
    test_detached_prefix_semantics(flex_ops, mode)
    test_dense_attention_mask_path(flex_ops, mode)
    test_manual_matches_reference(flex_ops, mode)
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
