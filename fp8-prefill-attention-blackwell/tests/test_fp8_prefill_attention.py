#!/usr/bin/env python3
"""Strict source or installed-artifact validation."""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _source_loader import load_source_ops  # noqa: E402


def reference(q, k, v):
    qf = q.float().transpose(0, 1).unsqueeze(0)
    kf = k.float().repeat_interleave(4, dim=1).transpose(0, 1).unsqueeze(0)
    vf = v.float().repeat_interleave(4, dim=1).transpose(0, 1).unsqueeze(0)
    return (
        torch.nn.functional.scaled_dot_product_attention(
            qf, kf, vf, is_causal=True, scale=1.0 / math.sqrt(128)
        )
        .squeeze(0)
        .transpose(0, 1)
        .contiguous()
    )


def metrics(actual, expected):
    delta = (actual.float() - expected.float()).abs().flatten()
    p99_index = max(1, math.ceil(0.99 * delta.numel()))
    return {
        "max": delta.max().item(),
        "p99": delta.kthvalue(p99_index).values.item(),
        "mean": delta.mean().item(),
        "cos": torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        ).item(),
    }


def run(ops, full):
    torch.manual_seed(1701)
    lengths = [256, 512] + ([1024, 2048, 4096, 8192] if full else [])
    count = 0
    for s in lengths:
        q = (torch.randn(s, 32, 128, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        k = (torch.randn(s, 8, 128, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        v = (torch.randn(s, 8, 128, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        out = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
        got = ops.fp8_causal_gqa_attention_bf16(
            q, k, v, softmax_scale=1.0 / math.sqrt(128), out=out
        )
        torch.cuda.synchronize()
        stat = metrics(got, reference(q, k, v))
        print(
            f"S={s} dtype={got.dtype} max={stat['max']:.7f} p99={stat['p99']:.7f} mean={stat['mean']:.7f} cos={stat['cos']:.8f}"
        )
        assert got.data_ptr() == out.data_ptr() and got.dtype == torch.bfloat16
        assert stat["cos"] >= 0.999
        count += 1

    q = torch.zeros(256, 32, 128, device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.zeros(256, 8, 128, device="cuda", dtype=torch.float8_e4m3fn)
    for bad in (
        lambda: ops.fp8_causal_gqa_attention_bf16(
            q[:128], k[:128], k[:128], softmax_scale=0.1
        ),
        lambda: ops.fp8_causal_gqa_attention_bf16(q.float(), k, k, softmax_scale=0.1),
    ):
        try:
            bad()
        except RuntimeError:
            count += 1
        else:
            raise AssertionError("unsupported input did not raise RuntimeError")

    compile_q = (torch.randn(256, 32, 128, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    compile_k = (torch.randn(256, 8, 128, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    compiled = torch.compile(
        lambda a, b, c: ops.fp8_causal_gqa_attention_bf16(
            a, b, c, softmax_scale=1.0 / math.sqrt(128)
        ),
        fullgraph=True,
    )
    out = torch.empty(compile_q.shape, device="cuda", dtype=torch.bfloat16)
    compiled_out = compiled(compile_q, compile_k, compile_k)
    torch.cuda.synchronize()
    compile_stat = metrics(compiled_out, reference(compile_q, compile_k, compile_k))
    print(f"compile cos={compile_stat['cos']:.8f}")
    assert compile_stat["cos"] >= 0.999
    count += 1

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.fp8_causal_gqa_attention_bf16(
            compile_q, compile_k, compile_k, softmax_scale=1.0 / math.sqrt(128), out=out
        )
    graph.replay()
    torch.cuda.synchronize()
    assert metrics(out, reference(compile_q, compile_k, compile_k))["cos"] >= 0.999
    count += 1
    print(f"PASS checks={count}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("source", "installed"), default="source")
    p.add_argument("--artifact")
    p.add_argument("--mode", choices=("smoke", "full"), default="full")
    p.add_argument("--registration-include")
    a = p.parse_args()
    if a.backend == "source":
        module = load_source_ops(a.registration_include)
    else:
        if a.artifact:
            sys.path.insert(0, a.artifact)
        module = importlib.import_module("fp8_prefill_attention_blackwell")
    run(module, a.mode == "full")
