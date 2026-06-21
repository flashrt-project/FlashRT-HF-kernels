#!/usr/bin/env python3
"""Correctness tests for transformer-fused-ops."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "transformer-fused-ops"
REGISTRATION_INCLUDE = ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject" / "templates" / "torch"


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self.ops = getattr(torch.ops, namespace)

    def rms_norm_gated_silu_bf16(self, x, gate, weight, eps=1e-6):
        out = torch.empty_like(x)
        self.ops.rms_norm_gated_silu_bf16(x, gate, weight, float(eps), out)
        return out

    def silu_mul_bf16(self, gate, up):
        out = torch.empty_like(gate)
        self.ops.silu_mul_bf16(gate, up, out)
        return out

    def sigmoid_mul_bf16(self, gate, x):
        out = torch.empty_like(gate)
        self.ops.sigmoid_mul_bf16(gate, x, out)
        return out

    def embedding_lookup_bf16(self, token_ids, embed):
        out = torch.empty((token_ids.shape[0], embed.shape[1]), device=embed.device, dtype=torch.bfloat16)
        self.ops.embedding_lookup_bf16(token_ids, embed, out)
        return out

    def partial_rope_qk_bf16(self, q, k, cos, sin, rope_dim):
        qo = torch.empty_like(q)
        ko = torch.empty_like(k)
        self.ops.partial_rope_qk_bf16(q, k, cos, sin, qo, ko, int(rope_dim))
        return qo, ko

    def argmax_bf16(self, logits):
        out = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
        self.ops.argmax_bf16(logits, out)
        return out

    def spec_accept_greedy_bf16(self, logits, drafts, spec_k):
        argmax = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
        accept_n = torch.empty((1,), device=logits.device, dtype=torch.int32)
        self.ops.spec_accept_greedy_bf16(logits, drafts, argmax, accept_n, int(spec_k))
        return argmax, accept_n

    def nexn2_lin_split_qkv_broadcast_bf16(self, conv_out):
        q = torch.empty((conv_out.shape[0], 32, 128), device=conv_out.device, dtype=torch.bfloat16)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        self.ops.nexn2_lin_split_qkv_broadcast_bf16(conv_out, q, k, v)
        return q, k, v

    def nexn2_split_q_gate_bf16(self, q_proj):
        q_pre = torch.empty((q_proj.shape[0], 16, 256), device=q_proj.device, dtype=torch.bfloat16)
        gate = torch.empty((q_proj.shape[0], 16 * 256), device=q_proj.device, dtype=torch.bfloat16)
        self.ops.nexn2_split_q_gate_bf16(q_proj, q_pre, gate)
        return q_pre, gate

    def nexn2_router_topk_bf16(self, logits, k=8):
        idx = torch.empty((k,), device=logits.device, dtype=torch.int32)
        val = torch.empty((k,), device=logits.device, dtype=torch.float32)
        self.ops.nexn2_router_topk_bf16(logits, idx, val, int(k))
        return idx, val


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return "12.0a" if major >= 12 else f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "transformer_fused_ops_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "kernels" / "rms_norm_gated_silu_qwen36.cu"),
            str(PACKAGE / "csrc" / "kernels" / "silu_mul_qwen36.cu"),
            str(PACKAGE / "csrc" / "kernels" / "qwen36_misc.cu"),
            str(PACKAGE / "csrc" / "kernels" / "nexn2_misc.cu"),
            str(PACKAGE / "csrc" / "kernels" / "nexn2_router_topk.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"],
        is_python_module=False,
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("transformer_fused_ops")
    finally:
        if artifact:
            sys.path.remove(artifact)


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float = 0.00390625) -> None:
    diff = (got.float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    cos = float(torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item())
    if max_abs > atol or cos < 0.999:
        raise AssertionError(f"{name}: max_abs={max_abs:.8f} cos={cos:.8f}")


def rope_ref(x, cos, sin, rope_dim):
    out = x.clone()
    half = rope_dim // 2
    left = x[:, :, :half].float()
    right = x[:, :, half:rope_dim].float()
    out[:, :, :half] = ((-right * sin[:, None, :half].float()).to(torch.bfloat16).float() + left * cos[:, None, :half].float()).to(torch.bfloat16)
    out[:, :, half:rope_dim] = ((left * sin[:, None, half:rope_dim].float()).to(torch.bfloat16).float() + right * cos[:, None, half:rope_dim].float()).to(torch.bfloat16)
    return out


def run(ops, mode: str) -> int:
    rows = [1, 8] if mode == "smoke" else [1, 8, 64, 257]
    count = 0
    for m in rows:
        x = (torch.randn((m, 128), device="cuda") * 0.1).to(torch.bfloat16)
        gate = (torch.randn_like(x.float()) * 0.1).to(torch.bfloat16)
        w = torch.randn((128,), device="cuda").to(torch.bfloat16)
        got = ops.rms_norm_gated_silu_bf16(x, gate, w)
        norm = x.float() * torch.rsqrt((x.float() * x.float()).mean(dim=1, keepdim=True) + 1e-6)
        weighted = (w.float() * norm.to(torch.bfloat16).float()).to(torch.bfloat16)
        ref = (weighted.float() * torch.nn.functional.silu(gate.float())).to(torch.bfloat16)
        assert_close(f"rms_norm_gated_silu rows={m}", got, ref, 0.00390625)
        count += 1

    gate = (torch.randn((4, 1024), device="cuda") * 0.2).to(torch.bfloat16)
    up = (torch.randn_like(gate.float()) * 0.2).to(torch.bfloat16)
    assert_close("silu_mul", ops.silu_mul_bf16(gate, up), (torch.nn.functional.silu(gate.float()).to(torch.bfloat16).float() * up.float()).to(torch.bfloat16))
    assert_close("sigmoid_mul", ops.sigmoid_mul_bf16(gate, up), (torch.sigmoid(gate.float()).to(torch.bfloat16).float() * up.float()).to(torch.bfloat16))
    count += 2

    token_ids = torch.tensor([0, 3, 7, 11], device="cuda", dtype=torch.int64)
    embed = torch.arange(16 * 32, device="cuda", dtype=torch.float32).reshape(16, 32).to(torch.bfloat16)
    got = ops.embedding_lookup_bf16(token_ids, embed)
    if not torch.equal(got.cpu(), embed[token_ids].cpu()):
        raise AssertionError("embedding lookup mismatch")
    count += 1

    q = torch.randn((8, 4, 128), device="cuda").to(torch.bfloat16)
    k = torch.randn((8, 2, 128), device="cuda").to(torch.bfloat16)
    cos = torch.randn((8, 64), device="cuda").to(torch.bfloat16)
    sin = torch.randn((8, 64), device="cuda").to(torch.bfloat16)
    qg, kg = ops.partial_rope_qk_bf16(q, k, cos, sin, 64)
    assert_close("partial_rope_q", qg, rope_ref(q, cos, sin, 64))
    assert_close("partial_rope_k", kg, rope_ref(k, cos, sin, 64))
    count += 2

    logits = torch.randn((5, 1024), device="cuda").to(torch.bfloat16)
    got = ops.argmax_bf16(logits)
    if not torch.equal(got.cpu(), torch.argmax(logits.float(), dim=1).cpu()):
        raise AssertionError("argmax mismatch")
    drafts = got.clone()
    drafts[3:] += 1
    _, accept_n = ops.spec_accept_greedy_bf16(logits, drafts, 5)
    torch.cuda.synchronize()
    if int(accept_n.cpu()[0]) != 3:
        raise AssertionError("spec accept mismatch")
    count += 2

    conv = torch.randn((3, 8192), device="cuda").to(torch.bfloat16)
    q32, k32, v32 = ops.nexn2_lin_split_qkv_broadcast_bf16(conv)
    ref_q = conv[:, :2048].reshape(3, 16, 128)[:, torch.arange(32, device="cuda") // 2]
    ref_k = conv[:, 2048:4096].reshape(3, 16, 128)[:, torch.arange(32, device="cuda") // 2]
    ref_v = conv[:, 4096:].reshape(3, 32, 128)
    assert_close("nexn2_lin_q", q32, ref_q)
    assert_close("nexn2_lin_k", k32, ref_k)
    assert_close("nexn2_lin_v", v32, ref_v)
    count += 3

    q_proj = torch.randn((3, 16, 512), device="cuda").to(torch.bfloat16)
    q_pre, q_gate = ops.nexn2_split_q_gate_bf16(q_proj)
    assert_close("nexn2_q_pre", q_pre, q_proj[:, :, :256].contiguous())
    assert_close("nexn2_gate", q_gate, q_proj[:, :, 256:].reshape(3, 16 * 256).contiguous())
    count += 2

    router = torch.linspace(-1.0, 1.0, 256, device="cuda").to(torch.bfloat16)
    idx, val = ops.nexn2_router_topk_bf16(router, 8)
    ref_val, ref_idx = torch.topk(router.float(), 8)
    if not torch.equal(idx.cpu(), ref_idx.to(torch.int32).cpu()) or not torch.allclose(val.cpu(), ref_val.cpu()):
        raise AssertionError("router topk mismatch")
    count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    count = run(ops, args.mode)
    print(f"transformer-fused-ops {args.backend} {args.mode}: passed {count}/{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
