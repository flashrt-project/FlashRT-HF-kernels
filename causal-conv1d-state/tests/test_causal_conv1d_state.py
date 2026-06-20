#!/usr/bin/env python3
"""Correctness tests for causal-conv1d-state."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "causal-conv1d-state"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)

SHAPES = {
    "fwd_s8_c1024": ("fwd", 1, 8, 1024, 4),
    "decode_c1024": ("decode", 1, 1, 1024, 4),
    "inout_c1024": ("inout", 1, 1, 1024, 4),
    "chunk_s8_c1024": ("chunk", 1, 8, 1024, 4),
    "parallel_s16_c1024": ("parallel", 1, 16, 1024, 4),
    "gqa_s8_c10240": ("gqa", 1, 8, 10240, 4),
}
MODES = {
    "smoke": ["decode_c1024"],
    "headline": ["decode_c1024", "parallel_s16_c1024", "gqa_s8_c10240"],
    "full": list(SHAPES.keys()),
}


@dataclass
class Row:
    name: str
    kind: str
    B: int
    S: int
    C: int
    K: int
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    passed: bool


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    @staticmethod
    def empty_bias(x):
        return torch.empty((0,), device=x.device, dtype=torch.bfloat16)

    def fwd(self, x, w, bias, apply_silu=True):
        out = torch.empty_like(x)
        self._ops.causal_conv1d_bf16(x, w, bias, out, True, apply_silu)
        return out

    def update(self, x_new, w, state, bias, apply_silu=True):
        out = torch.empty_like(x_new)
        self._ops.causal_conv1d_update_bf16(x_new, w, bias, state, out, True, apply_silu)
        return out

    def update_inout(self, x_new, w, state_in, bias, apply_silu=True):
        out = torch.empty_like(x_new)
        state_out = torch.empty_like(state_in)
        self._ops.causal_conv1d_update_inout_bf16(
            x_new, w, bias, state_in, state_out, out, True, apply_silu
        )
        return out, state_out

    def chunk(self, x, w, state, bias, apply_silu=True):
        out = torch.empty_like(x)
        self._ops.causal_conv1d_update_chunk_bf16(x, w, bias, state, out, True, apply_silu)
        return out

    def parallel(self, x, w, state, bias, apply_silu=True):
        out = torch.empty_like(x)
        self._ops.causal_conv1d_update_chunk_parallel_bf16(x, w, bias, state, out, True, apply_silu)
        return out

    def gqa(self, x, w, state, bias, apply_silu=True):
        b, s, _ = x.shape
        q = torch.empty((b, s, 16, 128), device=x.device, dtype=torch.bfloat16)
        k = torch.empty_like(q)
        v = torch.empty((b, s, 48, 128), device=x.device, dtype=torch.bfloat16)
        self._ops.causal_conv1d_update_chunk_parallel_gqa_bf16(
            x, w, bias, state, q, k, v, True, apply_silu
        )
        return q, k, v


class InstalledOps:
    def __init__(self, mod) -> None:
        self._mod = mod

    def fwd(self, x, w, bias, apply_silu=True):
        return self._mod.causal_conv1d_bf16(x, w, bias, apply_silu=apply_silu)

    def update(self, x_new, w, state, bias, apply_silu=True):
        return self._mod.causal_conv1d_update_bf16(x_new, w, state, bias, apply_silu=apply_silu)

    def update_inout(self, x_new, w, state_in, bias, apply_silu=True):
        return self._mod.causal_conv1d_update_inout_bf16(
            x_new, w, state_in, bias, apply_silu=apply_silu
        )

    def chunk(self, x, w, state, bias, apply_silu=True):
        return self._mod.causal_conv1d_update_chunk_bf16(x, w, state, bias, apply_silu=apply_silu)

    def parallel(self, x, w, state, bias, apply_silu=True):
        return self._mod.causal_conv1d_update_chunk_parallel_bf16(
            x, w, state, bias, apply_silu=apply_silu
        )

    def gqa(self, x, w, state, bias, apply_silu=True):
        return self._mod.causal_conv1d_update_chunk_parallel_gqa_bf16(
            x, w, state, bias, apply_silu=apply_silu
        )


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if major == 12 and minor == 1:
        return "12.1"
    if major >= 12:
        return "12.0a"
    return f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "causal_conv1d_state_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "causal_conv1d_state.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=[
            "-O3",
            "-DCUDA_KERNEL",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return InstalledOps(importlib.import_module("causal_conv1d_state"))
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_inputs(B: int, S: int, C: int, K: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = (torch.randn((B, S, C), device="cuda", generator=gen) * 0.25).to(torch.bfloat16)
    w = (torch.randn((C, K), device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    bias = (torch.randn((C,), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    state = (torch.randn((B, C, K - 1), device="cuda", generator=gen) * 0.2).to(torch.bfloat16)
    return x, w, bias, state


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.exp(-x))


def ref_fwd(x, w, bias, apply_silu=True):
    B, S, C = x.shape
    K = w.shape[1]
    rows = []
    for b in range(B):
        brow = []
        for s in range(S):
            acc = bias.float().clone()
            for i in range(K):
                t = s + i - (K - 1)
                if t >= 0:
                    acc = acc + x[b, t].float() * w[:, i].float()
            if apply_silu:
                acc = _silu(acc)
            brow.append(acc.to(torch.bfloat16))
        rows.append(torch.stack(brow, dim=0))
    return torch.stack(rows, dim=0)


def ref_chunk(x, w, bias, state, apply_silu=True):
    B, S, C = x.shape
    K = w.shape[1]
    sk = K - 1
    state_next = state.clone()
    outs = []
    for s in range(S):
        x_new = x[:, s]
        acc = bias.float().view(1, C).expand(B, C).clone()
        for i in range(sk):
            acc = acc + state_next[:, :, i].float() * w[:, i].float().view(1, C)
        acc = acc + x_new.float() * w[:, sk].float().view(1, C)
        if apply_silu:
            acc = _silu(acc)
        outs.append(acc.to(torch.bfloat16))
        if sk > 1:
            state_next[:, :, :-1] = state_next[:, :, 1:].clone()
        state_next[:, :, sk - 1] = x_new
    return torch.stack(outs, dim=1), state_next


def metrics(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float, float]:
    diff = (got.float() - ref.float()).abs()
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        float(torch.quantile(diff.flatten(), 0.99).item()),
        float(torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()),
    )


def run_case(ops, name: str) -> Row:
    kind, B, S, C, K = SHAPES[name]
    x, w, bias, state = make_inputs(B, S, C, K, seed=8000 + S * 13 + C)
    if kind == "fwd":
        got = ops.fwd(x, w, bias)
        ref = ref_fwd(x, w, bias)
    elif kind == "decode":
        state_work = state.clone()
        got = ops.update(x[:, 0], w, state_work, bias)
        ref, ref_state = ref_chunk(x[:, :1], w, bias, state)
        torch.cuda.synchronize()
        state_err = metrics(state_work, ref_state)[0]
        assert state_err == 0.0, f"decode state mismatch {state_err}"
        ref = ref[:, 0]
    elif kind == "inout":
        got, got_state = ops.update_inout(x[:, 0], w, state, bias)
        ref, ref_state = ref_chunk(x[:, :1], w, bias, state)
        torch.cuda.synchronize()
        state_err = metrics(got_state, ref_state)[0]
        assert state_err == 0.0, f"inout state mismatch {state_err}"
        ref = ref[:, 0]
    elif kind == "chunk":
        state_work = state.clone()
        got = ops.chunk(x, w, state_work, bias)
        ref, ref_state = ref_chunk(x, w, bias, state)
        torch.cuda.synchronize()
        state_err = metrics(state_work, ref_state)[0]
        assert state_err == 0.0, f"chunk state mismatch {state_err}"
    elif kind == "parallel":
        state_work = state.clone()
        got = ops.parallel(x, w, state_work, bias)
        ref, ref_state = ref_chunk(x, w, bias, state)
        torch.cuda.synchronize()
        state_err = metrics(state_work, ref_state)[0]
        assert state_err == 0.0, f"parallel state mismatch {state_err}"
    elif kind == "gqa":
        state_work = state.clone()
        q, k, v = ops.gqa(x, w, state_work, bias)
        ref, ref_state = ref_chunk(x, w, bias, state)
        torch.cuda.synchronize()
        state_err = metrics(state_work, ref_state)[0]
        assert state_err == 0.0, f"gqa state mismatch {state_err}"
        got = torch.cat([q.reshape(B, S, 2048), k.reshape(B, S, 2048), v.reshape(B, S, 6144)], dim=2)
    else:
        raise AssertionError(kind)
    torch.cuda.synchronize()
    max_abs, mean_abs, p99_abs, cos = metrics(got, ref)
    passed = max_abs <= 0.00390625 and mean_abs <= 0.0008 and cos >= 0.999
    return Row(name, kind, B, S, C, K, max_abs, mean_abs, p99_abs, cos, passed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=sorted(MODES), default="smoke")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    rows = []
    for name in MODES[args.mode]:
        row = run_case(ops, name)
        rows.append(row)
        print(
            f"{row.name}: max_abs={row.max_abs:.6f} mean_abs={row.mean_abs:.6f} "
            f"p99_abs={row.p99_abs:.6f} cosine={row.cosine:.8f} passed={row.passed}"
        )
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps([asdict(r) for r in rows], indent=2) + "\n")
    if not all(r.passed for r in rows):
        raise AssertionError("causal-conv1d-state correctness failed")
    print(f"PASS causal-conv1d-state {args.backend} mode={args.mode}: {len(rows)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
