#!/usr/bin/env python3
"""Benchmark XQA against native FlashRT and equivalent SDPA contracts."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
REGISTRATION = (
    ROOT.parent / "kernels/kernel-builder/src/pyproject/templates/torch"
)
sys.path.insert(0, str(PACKAGE / "tests"))
from test_fp8_kv_attention import (  # noqa: E402
    CONFIGS,
    PAGE,
    SourceOps,
    load_source_ops,
    make_inputs,
    metrics,
    reference,
)


WORKLOADS = {
    "qwen36_decode_1024": ("qwen36", 1, 1024),
    "qwen36_verify8_4096": ("qwen36", 8, 4096),
    "qwen3_vl_decode_1024": ("qwen3_vl", 1, 1024),
    "qwen3_vl_verify8_4096": ("qwen3_vl", 8, 4096),
    "qwen3_vl_verify8_32768": ("qwen3_vl", 8, 32768),
    "cosmos_edge_decode_1024": ("cosmos_edge", 1, 1024),
    "cosmos_edge_verify8_4096": ("cosmos_edge", 8, 4096),
    "cosmos_edge_verify8_32768": ("cosmos_edge", 8, 32768),
    "gqa32_kv16_decode_1024": ("gqa32_kv16", 1, 1024),
    "gqa32_kv16_verify8_4096": ("gqa32_kv16", 8, 4096),
    "gqa32_kv16_verify8_32768": ("gqa32_kv16", 8, 32768),
}
MODES = {
    "smoke": ["qwen3_vl_decode_1024"],
    "headline": [
        "qwen36_verify8_4096",
        "qwen3_vl_verify8_4096",
        "cosmos_edge_verify8_4096",
        "gqa32_kv16_verify8_4096",
    ],
    "full": list(WORKLOADS),
}


def time_us(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def build_native():
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    return load(
        name="fp8_kv_attention_raw_native",
        sources=[
            str(PACKAGE / "benchmarks/native_binding.cpp"),
            str(PACKAGE / "csrc/xqa_mha_configured.cu"),
            str(PACKAGE / "csrc/xqa_bf16_fp8kv.cu"),
            str(PACKAGE / "csrc/xqa_mha_d128.cu"),
        ],
        extra_include_paths=[
            str(PACKAGE / "csrc"),
            str(PACKAGE / "csrc/attention/flashinfer_xqa_src"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "--ftz=true",
            "--prec-div=false",
            "--prec-sqrt=false",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        verbose=False,
    )


def load_wrapper(backend: str, artifact: str | None):
    if backend == "source":
        return load_source_ops()
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("fp8_kv_attention")
    finally:
        if artifact:
            sys.path.remove(artifact)


def workspace(q, k, kv_seq):
    q_seq, q_heads, head_dim = q.shape
    kv_heads = k.shape[2]
    pages = k.shape[0]
    page_table = torch.arange(
        pages, device=q.device, dtype=torch.int32
    ).view(1, pages)
    seq_lens = torch.tensor([[kv_seq]], device=q.device, dtype=torch.int32)
    words = (q_seq + 31) // 32
    mask = torch.zeros((q_seq, words), device=q.device, dtype=torch.int32)
    for row in range(q_seq):
        upto = row + 1
        if upto // 32:
            mask[row, : upto // 32] = -1
        if upto % 32:
            mask[row, upto // 32] = (1 << (upto % 32)) - 1
    sem_count = kv_heads * (((q_seq * (q_heads // kv_heads)) + 31) // 32)
    sem = torch.zeros(max(256, sem_count), device=q.device, dtype=torch.int32)
    scratch = torch.empty(256 << 20, device=q.device, dtype=torch.uint8)
    out = torch.empty_like(q)
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    return (
        page_table, seq_lens, mask, sem, scratch, out, pages * PAGE, head_dim,
        sm_count,
    )


def wrapper_call(wrapper, q, k, v, ws, *, static_config: bool = True):
    page_table, seq_lens, mask, sem, scratch, out, max_seq, head_dim, sm_count = ws
    call_sm_count = sm_count if static_config else 0
    stride_page = PAGE * k.shape[2] * head_dim if static_config else 0
    stride_token = k.shape[2] * head_dim if static_config else 0
    stride_head = head_dim if static_config else 0
    if isinstance(wrapper, SourceOps):
        wrapper._ops.xqa_bf16_fp8kv(
            q, k, v, page_table, seq_lens, mask, out, sem, scratch,
            max_seq, 1.0, 1.0, True, call_sm_count,
            stride_page, stride_token, stride_head,
        )
        return out
    return wrapper.xqa_bf16_fp8kv(
        q, k, v, page_table, seq_lens, mask, out=out, semaphores=sem,
        scratch=scratch, max_seq_len=max_seq, sm_count=call_sm_count,
        k_stride_page=stride_page, k_stride_token=stride_token,
        k_stride_head=stride_head,
    )


def sdpa_inputs(q, k, v, kv_seq):
    q_seq, q_heads, _ = q.shape
    kv_heads = k.shape[2]
    group = q_heads // kv_heads
    kd = (
        k.reshape(-1, kv_heads, q.shape[-1])[:kv_seq]
        .to(torch.bfloat16)
        .repeat_interleave(group, dim=1)
        .permute(1, 0, 2)
        .unsqueeze(0)
    )
    vd = (
        v.reshape(-1, kv_heads, q.shape[-1])[:kv_seq]
        .to(torch.bfloat16)
        .repeat_interleave(group, dim=1)
        .permute(1, 0, 2)
        .unsqueeze(0)
    )
    qd = q.permute(1, 0, 2).unsqueeze(0)
    positions = torch.arange(kv_seq, device=q.device)
    valid = kv_seq - q_seq + torch.arange(1, q_seq + 1, device=q.device)
    attn_mask = positions.view(1, -1) < valid.view(-1, 1)
    return qd, kd, vd, attn_mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--mode", choices=MODES, default="smoke")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    wrapper = load_wrapper(args.backend, args.artifact)
    native = build_native()
    rows = []
    for name in MODES[args.mode]:
        config, q_seq, kv_seq = WORKLOADS[name]
        qh, kvh, hd = CONFIGS[config]
        q, k, v = make_inputs(config, q_seq, kv_seq, 7000 + q_seq + kv_seq)
        ws = workspace(q, k, kv_seq)
        page_table, seq_lens, mask, sem, scratch, out, max_seq, _, _ = ws
        raw = lambda: native.xqa(
            q, k, v, page_table, seq_lens, mask, out, sem, scratch,
            max_seq, 1.0, 1.0, True,
        )
        wrapped = lambda: wrapper_call(wrapper, q, k, v, ws)
        wrapped_default = lambda: wrapper_call(
            wrapper, q, k, v, ws, static_config=False
        )

        qd, kd, vd, attn_mask = sdpa_inputs(q, k, v, kv_seq)
        sdpa_out = torch.empty_like(q)

        def sdpa_predequant():
            y = F.scaled_dot_product_attention(
                qd, kd, vd, attn_mask=attn_mask, dropout_p=0.0
            )
            sdpa_out.copy_(y.squeeze(0).permute(1, 0, 2))

        def sdpa_equivalent():
            qdi, kdi, vdi, maski = sdpa_inputs(q, k, v, kv_seq)
            y = F.scaled_dot_product_attention(
                qdi, kdi, vdi, attn_mask=maski, dropout_p=0.0
            )
            sdpa_out.copy_(y.squeeze(0).permute(1, 0, 2))

        compiled_equivalent = torch.compile(sdpa_equivalent, fullgraph=True)
        compiled_predequant = torch.compile(sdpa_predequant, fullgraph=True)
        got = wrapped().clone()
        ref = reference(q, k, v, kv_seq)
        max_abs, mean_abs, p99_abs, cosine = metrics(got, ref)
        native_us = time_us(raw, args.warmup, args.iters)
        wrapper_us = time_us(wrapped, args.warmup, args.iters)
        wrapper_default_us = time_us(
            wrapped_default, args.warmup, args.iters
        )
        eager_us = time_us(sdpa_equivalent, args.warmup, args.iters)
        compile_us = time_us(compiled_equivalent, args.warmup, args.iters)
        sdpa_us = time_us(sdpa_predequant, args.warmup, args.iters)
        sdpa_compile_us = time_us(
            compiled_predequant, args.warmup, args.iters
        )
        accepted = (
            wrapper_us - native_us <= max(0.75, native_us * 0.05)
            and wrapper_default_us - native_us <= max(0.75, native_us * 0.05)
            and wrapper_us <= min(eager_us, compile_us) * 0.98
            and cosine >= 0.999
            and mean_abs <= 0.0025
        )
        row = {
            "workload": name,
            "config": config,
            "q_heads": qh,
            "kv_heads": kvh,
            "head_dim": hd,
            "q_seq": q_seq,
            "kv_seq": kv_seq,
            "native_us": native_us,
            "wrapper_us": wrapper_us,
            "wrapper_default_us": wrapper_default_us,
            "wrapper_native": wrapper_us / native_us,
            "wrapper_default_native": wrapper_default_us / native_us,
            "equivalent_eager_sdpa_us": eager_us,
            "equivalent_compile_sdpa_us": compile_us,
            "diagnostic_predequant_sdpa_us": sdpa_us,
            "diagnostic_predequant_compile_sdpa_us": sdpa_compile_us,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "p99_abs": p99_abs,
            "cosine": cosine,
            "accepted": accepted,
        }
        rows.append(row)
        print(
            f"{name}: native={native_us:.3f}us wrapper={wrapper_us:.3f}us "
            f"default={wrapper_default_us:.3f}us "
            f"equiv_compile={compile_us:.3f}us predequant_sdpa={sdpa_us:.3f}us "
            f"cos={cosine:.8f} accepted={accepted}"
        )
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(rows, indent=2) + "\n")
    if not all(row["accepted"] for row in rows):
        raise AssertionError("one or more XQA workloads failed acceptance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
