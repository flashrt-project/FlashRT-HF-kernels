#!/usr/bin/env python3
"""Correctness and source-extension tests for fp8-kv-attention."""

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
PACKAGE = ROOT / "fp8-kv-attention"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)

PAGE = 128
CONFIGS = {
    "qwen36": (24, 4, 256),
    "qwen3_vl": (32, 8, 128),
    "gqa32_kv16": (32, 16, 128),
    "cosmos_edge": (16, 8, 128),
}

SHAPES = {
    "qwen36_decode_128": ("qwen36", 1, 128),
    "qwen36_verify8_4096": ("qwen36", 8, 4096),
    "qwen3_vl_decode_1024": ("qwen3_vl", 1, 1024),
    "qwen3_vl_verify4_4096": ("qwen3_vl", 4, 4096),
    "qwen3_vl_verify8_32768": ("qwen3_vl", 8, 32768),
    "cosmos_edge_decode_1024": ("cosmos_edge", 1, 1024),
    "cosmos_edge_verify4_4096": ("cosmos_edge", 4, 4096),
    "cosmos_edge_verify8_32768": ("cosmos_edge", 8, 32768),
    "gqa32_kv16_decode_128": ("gqa32_kv16", 1, 128),
    "gqa32_kv16_decode_1024": ("gqa32_kv16", 1, 1024),
    "gqa32_kv16_verify4_4096": ("gqa32_kv16", 4, 4096),
    "gqa32_kv16_verify8_32768": ("gqa32_kv16", 8, 32768),
}
MODES = {
    "smoke": ["qwen36_decode_128", "qwen3_vl_decode_1024", "cosmos_edge_decode_1024", "gqa32_kv16_decode_128"],
    "headline": [
        "qwen36_verify8_4096",
        "qwen3_vl_verify4_4096",
        "cosmos_edge_verify4_4096",
        "gqa32_kv16_verify4_4096",
    ],
    "full": list(SHAPES.keys()),
}


@dataclass
class Metrics:
    shape: str
    config: str
    q_heads: int
    kv_heads: int
    head_dim: int
    q_seq: int
    kv_seq: int
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    dtype: str
    tolerance: str
    passed: bool


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    @staticmethod
    def causal_spec_mask(q_seq: int, device="cuda"):
        words = (q_seq + 31) // 32
        rows = torch.zeros((q_seq, words), dtype=torch.int32)
        for i in range(q_seq):
            upto = i + 1
            full = upto // 32
            rem = upto % 32
            if full:
                rows[i, :full] = -1
            if rem:
                rows[i, full] = (1 << rem) - 1
        return rows.to(device=device)

    @staticmethod
    def allocate_workspace(q_seq: int, q_heads: int, kv_heads: int, device="cuda"):
        sem_count = kv_heads * (((q_seq * (q_heads // kv_heads)) + 31) // 32)
        sem = torch.zeros(max(256, sem_count), device=device, dtype=torch.int32)
        scratch = torch.empty(256 << 20, device=device, dtype=torch.uint8)
        return sem, scratch

    def xqa_bf16_fp8kv(
        self,
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        mask,
        *,
        out,
        semaphores,
        scratch,
        max_seq_len,
        q_scale=1.0,
        kv_scale=1.0,
        enable_pdl=True,
        sm_count=0,
        k_stride_page=0,
        k_stride_token=0,
        k_stride_head=0,
    ):
        self._ops.xqa_bf16_fp8kv(
            q,
            k_cache,
            v_cache,
            page_table,
            seq_lens,
            mask,
            out,
            semaphores,
            scratch,
            int(max_seq_len),
            float(q_scale),
            float(kv_scale),
            bool(enable_pdl),
            int(sm_count),
            int(k_stride_page),
            int(k_stride_token),
            int(k_stride_head),
        )
        return out


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if major == 12 and minor == 1:
        return "12.1"
    if major >= 12:
        return "12.0"
    return f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "fp8_kv_attention_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "xqa_mha_configured.cu"),
            str(PACKAGE / "csrc" / "xqa_bf16_fp8kv.cu"),
            str(PACKAGE / "csrc" / "xqa_mha_d128.cu"),
        ],
        extra_include_paths=[
            str(PACKAGE / "csrc"),
            str(PACKAGE / "csrc" / "attention" / "flashinfer_xqa_src"),
            str(REGISTRATION_INCLUDE),
        ],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "--ftz=true",
            "--prec-div=false",
            "--prec-sqrt=false",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-DCUDA_KERNEL",
        ],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("fp8_kv_attention")
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_inputs(config: str, q_seq: int, kv_seq: int, seed: int):
    q_heads, kv_heads, head_dim = CONFIGS[config]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    pages = (kv_seq + PAGE - 1) // PAGE
    q = (torch.randn((q_seq, q_heads, head_dim), device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    k_bf16 = (
        torch.randn((pages, PAGE, kv_heads, head_dim), device="cuda", generator=gen) * 0.1
    ).to(torch.bfloat16)
    v_bf16 = (
        torch.randn((pages, PAGE, kv_heads, head_dim), device="cuda", generator=gen) * 0.1
    ).to(torch.bfloat16)
    return q, k_bf16.to(torch.float8_e4m3fn), v_bf16.to(torch.float8_e4m3fn)


def reference(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, kv_seq: int) -> torch.Tensor:
    q_seq = q.shape[0]
    q_heads, head_dim = q.shape[1:]
    kv_heads = k_cache.shape[2]
    group = q_heads // kv_heads
    k = k_cache.reshape(-1, kv_heads, head_dim)[:kv_seq].float()
    v = v_cache.reshape(-1, kv_heads, head_dim)[:kv_seq].float()
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * (head_dim**-0.5)
    positions = torch.arange(kv_seq, device=q.device)
    valid = kv_seq - q_seq + torch.arange(1, q_seq + 1, device=q.device)
    scores = scores.masked_fill(positions.view(1, 1, -1) >= valid.view(1, -1, 1), -torch.inf)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v).to(torch.bfloat16)


def metrics(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float, float]:
    diff = (got.float() - ref.float()).abs()
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        float(torch.quantile(diff.flatten(), 0.99).item()),
        float(torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()),
    )


def call_public_xqa(ops, q, k_cache, v_cache, kv_seq):
    q_seq = q.shape[0]
    q_heads, head_dim = q.shape[1:]
    kv_heads = k_cache.shape[2]
    pages = k_cache.shape[0]
    page_table = torch.arange(
        pages, device=q.device, dtype=torch.int32
    ).view(1, pages)
    seq_lens = torch.tensor(
        [[kv_seq]], device=q.device, dtype=torch.int32
    )
    mask = SourceOps.causal_spec_mask(q_seq, q.device)
    out = torch.empty_like(q)
    semaphores, scratch = SourceOps.allocate_workspace(
        q_seq, q_heads, kv_heads, q.device
    )
    return ops.xqa_bf16_fp8kv(
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        mask,
        out=out,
        semaphores=semaphores,
        scratch=scratch,
        max_seq_len=pages * PAGE,
        k_stride_page=PAGE * kv_heads * head_dim,
        k_stride_token=kv_heads * head_dim,
        k_stride_head=head_dim,
    )


def run_shape(ops, name: str, config: str, q_seq: int, kv_seq: int) -> Metrics:
    q_heads, kv_heads, head_dim = CONFIGS[config]
    q, k, v = make_inputs(config, q_seq, kv_seq, seed=1000 + q_seq * 17 + kv_seq)
    got = call_public_xqa(ops, q, k, v, kv_seq)
    torch.cuda.synchronize()
    ref = reference(q, k, v, kv_seq)
    max_abs, mean_abs, p99_abs, cos = metrics(got, ref)
    passed = max_abs <= 0.02 and mean_abs <= 0.0025 and cos >= 0.999
    return Metrics(
        shape=name,
        config=config,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        q_seq=q_seq,
        kv_seq=kv_seq,
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        cosine=cos,
        dtype="q/out=bf16, kv=float8_e4m3fn",
        tolerance="max_abs<=0.02, mean_abs<=0.0025, cosine>=0.999",
        passed=passed,
    )


def run_h32_kv16_graph(ops) -> None:
    q, k, v = make_inputs("gqa32_kv16", 4, 4096, seed=424242)
    q_seq, q_heads, head_dim = q.shape
    kv_heads, pages = k.shape[2], k.shape[0]
    page_table = torch.arange(pages, device="cuda", dtype=torch.int32).view(1, pages)
    seq_lens = torch.tensor([[4096]], device="cuda", dtype=torch.int32)
    mask = SourceOps.causal_spec_mask(q_seq, q.device)
    semaphores, scratch = SourceOps.allocate_workspace(q_seq, q_heads, kv_heads, q.device)
    out = torch.empty_like(q)

    def launch():
        return ops.xqa_bf16_fp8kv(
            q, k, v, page_table, seq_lens, mask, out=out,
            semaphores=semaphores, scratch=scratch,
            max_seq_len=pages * PAGE,
            k_stride_page=PAGE * kv_heads * head_dim,
            k_stride_token=kv_heads * head_dim,
            k_stride_head=head_dim,
        )

    expected = launch().clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def run_h32_kv16_compile(ops) -> None:
    q, k, v = make_inputs("gqa32_kv16", 4, 4096, seed=424243)
    q_seq, q_heads, head_dim = q.shape
    kv_heads, pages = k.shape[2], k.shape[0]
    page_table = torch.arange(pages, device="cuda", dtype=torch.int32).view(1, pages)
    seq_lens = torch.tensor([[4096]], device="cuda", dtype=torch.int32)
    mask = SourceOps.causal_spec_mask(q_seq, q.device)
    semaphores, scratch = SourceOps.allocate_workspace(q_seq, q_heads, kv_heads, q.device)
    eager_out = torch.empty_like(q)
    compiled_out = torch.empty_like(q)

    def launch(q_, k_, v_, page_table_, seq_lens_, mask_, out_, semaphores_, scratch_):
        return ops.xqa_bf16_fp8kv(
            q_, k_, v_, page_table_, seq_lens_, mask_, out=out_,
            semaphores=semaphores_, scratch=scratch_,
            max_seq_len=pages * PAGE,
            k_stride_page=PAGE * kv_heads * head_dim,
            k_stride_token=kv_heads * head_dim,
            k_stride_head=head_dim,
        )

    semaphores.zero_()
    expected = launch(q, k, v, page_table, seq_lens, mask, eager_out, semaphores, scratch).clone()
    semaphores.zero_()
    compiled = torch.compile(launch, fullgraph=True)
    got = compiled(q, k, v, page_table, seq_lens, mask, compiled_out, semaphores, scratch)
    torch.cuda.synchronize()
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=sorted(MODES), default="smoke")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability(0)
    if capability not in {(10, 0), (10, 3), (11, 0), (12, 0), (12, 1)}:
        raise RuntimeError(
            "fp8-kv-attention requires a supported Blackwell capability "
            f"(SM100, SM103, SM110, SM120, or SM121); got SM{capability[0]}{capability[1]}"
        )

    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    rows = []
    for name in MODES[args.mode]:
        config, q_seq, kv_seq = SHAPES[name]
        row = run_shape(ops, name, config, q_seq, kv_seq)
        rows.append(row)
        print(
            f"{row.shape}: max_abs={row.max_abs:.6f} mean_abs={row.mean_abs:.6f} "
            f"p99_abs={row.p99_abs:.6f} cosine={row.cosine:.8f} passed={row.passed}"
        )
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps([asdict(r) for r in rows], indent=2) + "\n")
    if not all(r.passed for r in rows):
        raise AssertionError("fp8-kv-attention correctness failed")
    if args.mode == "full":
        run_h32_kv16_graph(ops)
        if args.backend == "installed":
            run_h32_kv16_compile(ops)
    print(f"PASS fp8-kv-attention {args.backend} mode={args.mode}: {len(rows)} checks" +
          (" + H32/KV16 CUDA Graph" +
           (" + torch.compile(fullgraph=True)" if args.backend == "installed" else "")
           if args.mode == "full" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
