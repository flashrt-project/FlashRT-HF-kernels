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
QH = 24
KVH = 4
HD = 256

SHAPES = {
    "decode_128": (1, 128),
    "decode_1024": (1, 1024),
    "verify4_1024": (4, 1024),
    "verify8_4096": (8, 4096),
}
MODES = {
    "smoke": ["decode_128"],
    "headline": ["decode_1024", "verify4_1024"],
    "full": list(SHAPES.keys()),
}


@dataclass
class Metrics:
    shape: str
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
    def allocate_workspace(q_seq: int, device="cuda"):
        sem_count = KVH * (((q_seq * (QH // KVH)) + 31) // 32)
        sem = torch.zeros(max(256, sem_count), device=device, dtype=torch.int32)
        scratch = torch.empty(256 << 20, device=device, dtype=torch.uint8)
        return sem, scratch

    def xqa_bf16_fp8kv(self, q, k_cache, v_cache, kv_seq):
        q_seq = q.shape[0]
        pages = k_cache.shape[0]
        page_table = torch.arange(pages, device=q.device, dtype=torch.int32).view(1, pages)
        seq_lens = torch.tensor([[kv_seq]], device=q.device, dtype=torch.int32)
        mask = self.causal_spec_mask(q_seq, q.device)
        out = torch.empty_like(q)
        sem, scratch = self.allocate_workspace(q_seq, q.device)
        self._ops.xqa_bf16_fp8kv(
            q,
            k_cache,
            v_cache,
            page_table,
            seq_lens,
            mask,
            out,
            sem,
            scratch,
            pages * PAGE,
            1.0,
            1.0,
            True,
            0,
            PAGE * KVH * HD,
            KVH * HD,
            HD,
        )
        return out


def _current_arch_list() -> str:
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
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "fp8_kv_attention_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "xqa_mha_configured.cu"),
            str(PACKAGE / "csrc" / "xqa_bf16_fp8kv.cu"),
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


def make_inputs(q_seq: int, kv_seq: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    pages = (kv_seq + PAGE - 1) // PAGE
    q = (torch.randn((q_seq, QH, HD), device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    k_bf16 = (torch.randn((pages, PAGE, KVH, HD), device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    v_bf16 = (torch.randn((pages, PAGE, KVH, HD), device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    return q, k_bf16.to(torch.float8_e4m3fn), v_bf16.to(torch.float8_e4m3fn)


def reference(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, kv_seq: int) -> torch.Tensor:
    q_seq = q.shape[0]
    k = k_cache.reshape(-1, KVH, HD)[:kv_seq].float()
    v = v_cache.reshape(-1, KVH, HD)[:kv_seq].float()
    out = torch.empty_like(q)
    scale = HD ** -0.5
    for qi in range(q_seq):
        valid = kv_seq - q_seq + qi + 1
        valid = max(1, min(valid, kv_seq))
        for h in range(QH):
            kh = h // (QH // KVH)
            scores = (q[qi, h].float()[None, :] * k[:valid, kh]).sum(dim=1) * scale
            probs = torch.softmax(scores, dim=0)
            out[qi, h] = (probs[:, None] * v[:valid, kh]).sum(dim=0).to(torch.bfloat16)
    return out


def metrics(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float, float]:
    diff = (got.float() - ref.float()).abs()
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        float(torch.quantile(diff.flatten(), 0.99).item()),
        float(torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()),
    )


def run_shape(ops, name: str, q_seq: int, kv_seq: int) -> Metrics:
    q, k, v = make_inputs(q_seq, kv_seq, seed=1000 + q_seq * 17 + kv_seq)
    got = ops.xqa_bf16_fp8kv(q, k, v, kv_seq)
    torch.cuda.synchronize()
    ref = reference(q, k, v, kv_seq)
    max_abs, mean_abs, p99_abs, cos = metrics(got, ref)
    passed = max_abs <= 0.02 and mean_abs <= 0.0025 and cos >= 0.999
    return Metrics(
        shape=name,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=sorted(MODES), default="smoke")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, _ = torch.cuda.get_device_capability(0)
    if major < 12:
        raise RuntimeError("fp8-kv-attention v1 requires Blackwell-class CUDA capability")

    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    rows = []
    for name in MODES[args.mode]:
        q_seq, kv_seq = SHAPES[name]
        row = run_shape(ops, name, q_seq, kv_seq)
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
    print(f"PASS fp8-kv-attention {args.backend} mode={args.mode}: {len(rows)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
