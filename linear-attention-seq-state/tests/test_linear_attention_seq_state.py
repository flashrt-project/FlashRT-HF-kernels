#!/usr/bin/env python3
"""Correctness tests for linear-attention-seq-state."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "linear-attention-seq-state"
REGISTRATION_INCLUDE = ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject" / "templates" / "torch"


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self.ops = getattr(torch.ops, namespace)

    def gated_delta_recurrent_seq_bf16(self, q, k, v, g, beta, state, use_qk_l2norm=False):
        out = torch.empty_like(q)
        self.ops.gated_delta_recurrent_seq_bf16(q, k, v, g, beta, state, out, bool(use_qk_l2norm))
        return out, state


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return "12.0a" if major >= 12 else f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "linear_attention_seq_state_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "kernels" / "nexn2_gdn_seq.cu"),
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
        return importlib.import_module("linear_attention_seq_state")
    finally:
        if artifact:
            sys.path.remove(artifact)


def ref_scan(q, k, v, g, beta, state, use_qk_l2norm):
    qf, kf, vf = q.float(), k.float(), v.float()
    gf, bf = g.float(), beta.float()
    st = state.float().clone()
    out = torch.empty_like(qf)
    inv_sqrt = 1.0 / (q.shape[-1] ** 0.5)
    for s in range(q.shape[0]):
        qs = qf[s].clone()
        ks = kf[s].clone()
        if use_qk_l2norm:
            qs = qs / torch.sqrt((qs * qs).sum(dim=-1, keepdim=True) + 1e-6)
            ks = ks / torch.sqrt((ks * ks).sum(dim=-1, keepdim=True) + 1e-6)
        qs = qs * inv_sqrt
        for h in range(q.shape[1]):
            st[h] = st[h] * torch.exp(gf[s, h])
            kv_mem = torch.mv(st[h].t(), ks[h])
            delta = (vf[s, h] - kv_mem) * bf[s, h]
            st[h] = st[h] + torch.outer(ks[h], delta)
            out[s, h] = torch.mv(st[h].t(), qs[h])
    return out.to(torch.bfloat16), st.to(torch.bfloat16)


def assert_close(name: str, got, ref) -> None:
    diff = (got.float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    cos = float(torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item())
    if max_abs > 0.0625 or cos < 0.999:
        raise AssertionError(f"{name}: max_abs={max_abs:.8f} cos={cos:.8f}")


def run(ops, mode: str) -> int:
    shapes = [(2, 2), (4, 4)] if mode == "smoke" else [(2, 2), (4, 4), (8, 8)]
    count = 0
    for s, h in shapes:
        gen = torch.Generator(device="cuda").manual_seed(9100 + s + h)
        q = (torch.randn((s, h, 128), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
        k = (torch.randn((s, h, 128), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
        v = (torch.randn((s, h, 128), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
        g = (torch.randn((s, h), device="cuda", generator=gen) * 0.01).to(torch.bfloat16)
        beta = torch.sigmoid(torch.randn((s, h), device="cuda", generator=gen)).to(torch.bfloat16)
        state0 = (torch.randn((h, 128, 128), device="cuda", generator=gen) * 0.01).to(torch.bfloat16)
        state = state0.clone()
        got_out, got_state = ops.gated_delta_recurrent_seq_bf16(
            q, k, v, g, beta, state, use_qk_l2norm=False
        )
        torch.cuda.synchronize()
        ref_out, ref_state = ref_scan(q, k, v, g, beta, state0, False)
        assert_close(f"gdn_seq_out S={s} H={h}", got_out, ref_out)
        assert_close(f"gdn_seq_state S={s} H={h}", got_state, ref_state)
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
    print(f"linear-attention-seq-state {args.backend} {args.mode}: passed {count}/{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
