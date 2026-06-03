#!/usr/bin/env python3
"""Wan-style QKV postprocess benchmark for flashrt-vla-video."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-vla-video"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)


@dataclass
class Result:
    model_shape: str
    batch: int
    tokens: int
    heads: int
    head_dim: int
    dim: int
    fused_prealloc_us: float
    fused_alloc_us: float
    torch_eager_us: float
    torch_compile_us: float | None
    speedup_prealloc: float
    speedup_alloc: float
    speedup_vs_compile_prealloc: float | None
    compile_status: str | None
    q_max_abs: float
    q_p99_abs: float
    q_max_rel: float
    k_max_abs: float
    k_p99_abs: float
    k_max_rel: float
    status: str


@dataclass
class AttentionE2EResult:
    model_shape: str
    batch: int
    tokens: int
    heads: int
    head_dim: int
    dim: int
    grid: str
    fused_e2e_us: float
    torch_e2e_us: float
    torch_compile_e2e_us: float | None
    speedup_e2e: float
    speedup_vs_compile: float | None
    compile_status: str | None
    output_max_abs: float
    output_p99_abs: float
    output_max_rel: float
    status: str


@dataclass
class SelfAttentionE2EResult:
    model_shape: str
    batch: int
    tokens: int
    heads: int
    head_dim: int
    dim: int
    grid: str
    flashrt_us: float
    torch_us: float
    torch_compile_us: float | None
    speedup: float
    speedup_vs_compile: float | None
    compile_status: str | None
    output_max_abs: float
    output_p99_abs: float
    output_max_rel: float
    status: str


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def qkv_split_norm_rope_bf16(
        self,
        packed_qkv: torch.Tensor,
        norm_q_weight: torch.Tensor,
        norm_k_weight: torch.Tensor,
        freqs_re: torch.Tensor,
        freqs_im: torch.Tensor,
        *,
        heads: int,
        head_dim: int,
        seq_len: int | None = None,
        q_out: torch.Tensor | None = None,
        k_out: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = packed_qkv.shape[1]
        if q_out is None:
            q_out = torch.empty(
                (packed_qkv.shape[0], packed_qkv.shape[1], heads, head_dim),
                device=packed_qkv.device,
                dtype=packed_qkv.dtype,
            )
        if k_out is None:
            k_out = torch.empty_like(q_out)
        self._ops.qkv_split_norm_rope_bf16(
            packed_qkv,
            norm_q_weight,
            norm_k_weight,
            freqs_re,
            freqs_im,
            q_out,
            k_out,
            heads,
            head_dim,
            seq_len,
            eps,
        )
        return q_out, k_out


def _preload_cublaslt() -> None:
    for parent in Path(torch.__file__).resolve().parents:
        candidate = parent / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12"
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
    library = ctypes.util.find_library("cublasLt")
    if library:
        ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    _preload_cublaslt()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "flashrt_wan_qkv_demo"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "q_norm_rope_bf16.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("flashrt_vla_video")
    finally:
        if artifact:
            sys.path.remove(artifact)


def load_hub_ops(repo_id: str, version: int):
    from kernels import get_kernel

    return get_kernel(repo_id, version=version, trust_remote_code=True)


def load_ops(args):
    if args.backend == "source":
        return load_source_ops()
    if args.backend == "installed":
        return load_installed_ops(args.artifact)
    return load_hub_ops(args.repo_id, args.version)


def rope_params_complex(max_seq_len: int, dim: int, *, device: torch.device) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len, device=device, dtype=torch.float64),
        1.0 / torch.pow(
            torch.tensor(10000.0, device=device, dtype=torch.float64),
            torch.arange(0, dim, 2, device=device, dtype=torch.float64).div(dim),
        ),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def representative_wan_grid(tokens: int) -> tuple[int, int, int]:
    known = {
        256: (4, 8, 8),
        1024: (16, 8, 8),
        2520: (21, 12, 10),
        4096: (16, 16, 16),
    }
    if tokens in known:
        return known[tokens]
    return (1, 1, tokens)


def wan_freqs_re_im(tokens: int, head_dim: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    complex_dim = head_dim // 2
    c_t = complex_dim - 2 * (complex_dim // 3)
    c_h = complex_dim // 3
    c_w = complex_dim // 3
    f, h, w = representative_wan_grid(tokens)
    if f * h * w != tokens:
        raise ValueError(f"representative grid {f,h,w} does not match tokens={tokens}")

    ft = rope_params_complex(max(1024, f), c_t * 2, device=device)
    fh = rope_params_complex(max(1024, h), c_h * 2, device=device)
    fw = rope_params_complex(max(1024, w), c_w * 2, device=device)
    grid = torch.cat(
        [
            ft[:f].view(f, 1, 1, c_t).expand(f, h, w, c_t),
            fh[:h].view(1, h, 1, c_h).expand(f, h, w, c_h),
            fw[:w].view(1, 1, w, c_w).expand(f, h, w, c_w),
        ],
        dim=-1,
    ).reshape(tokens, complex_dim)
    return grid.real.float().contiguous(), grid.imag.float().contiguous(), (f, h, w)


def torch_qkv_split_norm_rope(
    packed_qkv: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    freqs_re: torch.Tensor,
    freqs_im: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, _ = packed_qkv.shape
    dim = heads * head_dim
    q = packed_qkv[..., :dim].reshape(batch, tokens, heads, head_dim)
    k = packed_qkv[..., dim : 2 * dim].reshape(batch, tokens, heads, head_dim)

    qf = q.float()
    kf = k.float()
    q_norm = qf * torch.rsqrt((qf * qf).mean(dim=(-2, -1), keepdim=True) + eps)
    k_norm = kf * torch.rsqrt((kf * kf).mean(dim=(-2, -1), keepdim=True) + eps)
    q_norm = q_norm * norm_q_weight.reshape(1, 1, heads, head_dim).float()
    k_norm = k_norm * norm_k_weight.reshape(1, 1, heads, head_dim).float()

    def rope(x: torch.Tensor) -> torch.Tensor:
        real = x[..., 0::2].float()
        imag = x[..., 1::2].float()
        fre = freqs_re[:tokens][None, :, None, :]
        fim = freqs_im[:tokens][None, :, None, :]
        out = torch.empty_like(x, dtype=torch.float32)
        out[..., 0::2] = real * fre - imag * fim
        out[..., 1::2] = real * fim + imag * fre
        return out.to(torch.bfloat16)

    return rope(q_norm), rope(k_norm)


def _time_us(fn, *, warmup: int, iters: int) -> float:
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


def _time_us_wall(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1_000_000.0)
    return statistics.mean(times)


def _compile_callable(fn, *, enabled: bool, mode: str):
    if not enabled:
        return None, "disabled"
    if not hasattr(torch, "compile"):
        return None, "unsupported"
    try:
        return torch.compile(fn, mode=mode, fullgraph=False), "ok"
    except Exception as exc:
        return None, f"failed:{type(exc).__name__}: {exc}"


def _time_compiled_us(fn, *, enabled: bool, mode: str, warmup: int, iters: int) -> tuple[float | None, str]:
    compiled, status = _compile_callable(fn, enabled=enabled, mode=mode)
    if compiled is None:
        return None, status
    try:
        return _time_us_wall(
            compiled,
            warmup=max(1, min(warmup, 5)),
            iters=max(1, min(iters, 20)),
        ), "ok"
    except Exception as exc:
        return None, f"failed:{type(exc).__name__}: {exc}"


def _error(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    abs_err = (got.float() - expected.float()).abs().flatten()
    rel_err = abs_err / expected.float().abs().clamp_min(1.0).flatten()
    max_abs = float(abs_err.max().item())
    kth = max(1, int(0.99 * abs_err.numel()))
    p99_abs = float(abs_err.kthvalue(kth).values.item())
    max_rel = float(rel_err.max().item())
    return max_abs, p99_abs, max_rel


def _model_shape(heads: int) -> str:
    if heads == 24:
        return "wan2.2-ti2v-5b"
    if heads == 40:
        return "wan-a14b-family"
    return f"heads{heads}"


def run_shape(
    ops,
    *,
    batch: int,
    tokens: int,
    heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
    compile_baseline: bool,
    compile_mode: str,
) -> Result:
    dim = heads * head_dim
    torch.manual_seed(20260603 + tokens + heads)
    packed_qkv = torch.randn((batch, tokens, 3 * dim), device="cuda", dtype=torch.bfloat16).contiguous()
    norm_q_weight = (torch.randn((dim,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
    norm_k_weight = (torch.randn((dim,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
    freqs_re, freqs_im, _ = wan_freqs_re_im(tokens, head_dim, device=torch.device("cuda"))
    q_out = torch.empty((batch, tokens, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k_out = torch.empty_like(q_out)

    q_fused, k_fused = ops.qkv_split_norm_rope_bf16(
        packed_qkv,
        norm_q_weight,
        norm_k_weight,
        freqs_re,
        freqs_im,
        heads=heads,
        head_dim=head_dim,
        q_out=q_out,
        k_out=k_out,
    )
    q_ref, k_ref = torch_qkv_split_norm_rope(
        packed_qkv,
        norm_q_weight,
        norm_k_weight,
        freqs_re,
        freqs_im,
        heads=heads,
        head_dim=head_dim,
    )
    torch.cuda.synchronize()
    q_max_abs, q_p99_abs, q_max_rel = _error(q_fused, q_ref)
    k_max_abs, k_p99_abs, k_max_rel = _error(k_fused, k_ref)
    status = "PASS" if max(q_max_abs, k_max_abs) <= 0.03125 and max(q_max_rel, k_max_rel) <= 0.05 else "FAIL"

    fused_prealloc_us = _time_us(
        lambda: ops.qkv_split_norm_rope_bf16(
            packed_qkv,
            norm_q_weight,
            norm_k_weight,
            freqs_re,
            freqs_im,
            heads=heads,
            head_dim=head_dim,
            q_out=q_out,
            k_out=k_out,
        ),
        warmup=warmup,
        iters=iters,
    )
    fused_alloc_us = _time_us(
        lambda: ops.qkv_split_norm_rope_bf16(
            packed_qkv,
            norm_q_weight,
            norm_k_weight,
            freqs_re,
            freqs_im,
            heads=heads,
            head_dim=head_dim,
        ),
        warmup=warmup,
        iters=iters,
    )
    torch_fn = lambda: torch_qkv_split_norm_rope(
            packed_qkv,
            norm_q_weight,
            norm_k_weight,
            freqs_re,
            freqs_im,
            heads=heads,
            head_dim=head_dim,
        )
    torch_eager_us = _time_us(
        torch_fn,
        warmup=warmup,
        iters=iters,
    )
    torch_compile_us, compile_status = _time_compiled_us(
        torch_fn,
        enabled=compile_baseline,
        mode=compile_mode,
        warmup=warmup,
        iters=iters,
    )

    return Result(
        model_shape=_model_shape(heads),
        batch=batch,
        tokens=tokens,
        heads=heads,
        head_dim=head_dim,
        dim=dim,
        fused_prealloc_us=fused_prealloc_us,
        fused_alloc_us=fused_alloc_us,
        torch_eager_us=torch_eager_us,
        torch_compile_us=torch_compile_us,
        speedup_prealloc=torch_eager_us / fused_prealloc_us,
        speedup_alloc=torch_eager_us / fused_alloc_us,
        speedup_vs_compile_prealloc=(
            torch_compile_us / fused_prealloc_us
            if torch_compile_us is not None
            else None
        ),
        compile_status=compile_status if compile_baseline else None,
        q_max_abs=q_max_abs,
        q_p99_abs=q_p99_abs,
        q_max_rel=q_max_rel,
        k_max_abs=k_max_abs,
        k_p99_abs=k_p99_abs,
        k_max_rel=k_max_rel,
        status=status,
    )


def run_attention_e2e_shape(
    ops,
    *,
    batch: int,
    tokens: int,
    heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
    compile_baseline: bool,
    compile_mode: str,
) -> AttentionE2EResult:
    dim = heads * head_dim
    torch.manual_seed(20260604 + tokens + heads)
    packed_qkv = torch.randn((batch, tokens, 3 * dim), device="cuda", dtype=torch.bfloat16).contiguous()
    norm_q_weight = (torch.randn((dim,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
    norm_k_weight = (torch.randn((dim,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
    freqs_re, freqs_im, grid = wan_freqs_re_im(tokens, head_dim, device=torch.device("cuda"))
    q_out = torch.empty((batch, tokens, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k_out = torch.empty_like(q_out)
    v = packed_qkv[..., 2 * dim :].reshape(batch, tokens, heads, head_dim)

    def attention(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
        )
        return out.permute(0, 2, 1, 3).contiguous()

    def fused_fn() -> torch.Tensor:
        q, k = ops.qkv_split_norm_rope_bf16(
            packed_qkv,
            norm_q_weight,
            norm_k_weight,
            freqs_re,
            freqs_im,
            heads=heads,
            head_dim=head_dim,
            q_out=q_out,
            k_out=k_out,
        )
        return attention(q, k)

    def torch_fn() -> torch.Tensor:
        q, k = torch_qkv_split_norm_rope(
            packed_qkv,
            norm_q_weight,
            norm_k_weight,
            freqs_re,
            freqs_im,
            heads=heads,
            head_dim=head_dim,
        )
        return attention(q, k)

    fused = fused_fn()
    expected = torch_fn()
    torch.cuda.synchronize()
    max_abs, p99_abs, max_rel = _error(fused, expected)
    status = "PASS" if max_abs <= 0.03125 and max_rel <= 0.05 else "FAIL"
    fused_us = _time_us_wall(fused_fn, warmup=warmup, iters=iters)
    torch_us = _time_us_wall(torch_fn, warmup=warmup, iters=iters)
    torch_compile_us, compile_status = _time_compiled_us(
        torch_fn,
        enabled=compile_baseline,
        mode=compile_mode,
        warmup=warmup,
        iters=iters,
    )
    return AttentionE2EResult(
        model_shape=_model_shape(heads),
        batch=batch,
        tokens=tokens,
        heads=heads,
        head_dim=head_dim,
        dim=dim,
        grid=f"{grid[0]}x{grid[1]}x{grid[2]}",
        fused_e2e_us=fused_us,
        torch_e2e_us=torch_us,
        torch_compile_e2e_us=torch_compile_us,
        speedup_e2e=torch_us / fused_us,
        speedup_vs_compile=(
            torch_compile_us / fused_us
            if torch_compile_us is not None
            else None
        ),
        compile_status=compile_status if compile_baseline else None,
        output_max_abs=max_abs,
        output_p99_abs=p99_abs,
        output_max_rel=max_rel,
        status=status,
    )


def run_self_attention_e2e_shape(
    ops,
    *,
    batch: int,
    tokens: int,
    heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
    compile_baseline: bool,
    compile_mode: str,
) -> SelfAttentionE2EResult:
    dim = heads * head_dim
    torch.manual_seed(20260605 + tokens + heads)
    x = (torch.randn((batch, tokens, dim), device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    q_w = (torch.randn((dim, dim), device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    k_w = (torch.randn((dim, dim), device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    v_w = (torch.randn((dim, dim), device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    o_w = (torch.randn((dim, dim), device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    q_b = (torch.randn((dim,), device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    k_b = (torch.randn((dim,), device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    v_b = (torch.randn((dim,), device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    o_b = (torch.randn((dim,), device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    packed_w = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
    packed_b = torch.cat([q_b, k_b, v_b], dim=0).contiguous()
    norm_q_weight = (torch.randn((dim,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
    norm_k_weight = (torch.randn((dim,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
    freqs_re, freqs_im, grid = wan_freqs_re_im(tokens, head_dim, device=torch.device("cuda"))
    q_out = torch.empty((batch, tokens, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k_out = torch.empty_like(q_out)

    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
        )
        return out.permute(0, 2, 1, 3).contiguous().flatten(2)

    def flashrt_fn() -> torch.Tensor:
        packed_qkv = torch.nn.functional.linear(x, packed_w, packed_b).contiguous()
        q, k = ops.qkv_split_norm_rope_bf16(
            packed_qkv,
            norm_q_weight,
            norm_k_weight,
            freqs_re,
            freqs_im,
            heads=heads,
            head_dim=head_dim,
            q_out=q_out,
            k_out=k_out,
        )
        v = packed_qkv[..., 2 * dim :].reshape(batch, tokens, heads, head_dim)
        return torch.nn.functional.linear(attention(q, k, v), o_w, o_b)

    def torch_fn() -> torch.Tensor:
        q_proj = torch.nn.functional.linear(x, q_w, q_b)
        k_proj = torch.nn.functional.linear(x, k_w, k_b)
        v = torch.nn.functional.linear(x, v_w, v_b).reshape(batch, tokens, heads, head_dim)
        packed_qkv = torch.cat([q_proj, k_proj, v.flatten(2)], dim=-1).contiguous()
        q, k = torch_qkv_split_norm_rope(
            packed_qkv,
            norm_q_weight,
            norm_k_weight,
            freqs_re,
            freqs_im,
            heads=heads,
            head_dim=head_dim,
        )
        return torch.nn.functional.linear(attention(q, k, v), o_w, o_b)

    got = flashrt_fn()
    expected = torch_fn()
    torch.cuda.synchronize()
    max_abs, p99_abs, max_rel = _error(got, expected)
    status = "PASS" if max_abs <= 0.125 and max_rel <= 0.05 else "FAIL"
    flashrt_us = _time_us_wall(flashrt_fn, warmup=warmup, iters=iters)
    torch_us = _time_us_wall(torch_fn, warmup=warmup, iters=iters)
    torch_compile_us, compile_status = _time_compiled_us(
        torch_fn,
        enabled=compile_baseline,
        mode=compile_mode,
        warmup=warmup,
        iters=iters,
    )
    return SelfAttentionE2EResult(
        model_shape=_model_shape(heads),
        batch=batch,
        tokens=tokens,
        heads=heads,
        head_dim=head_dim,
        dim=dim,
        grid=f"{grid[0]}x{grid[1]}x{grid[2]}",
        flashrt_us=flashrt_us,
        torch_us=torch_us,
        torch_compile_us=torch_compile_us,
        speedup=torch_us / flashrt_us,
        speedup_vs_compile=(
            torch_compile_us / flashrt_us
            if torch_compile_us is not None
            else None
        ),
        compile_status=compile_status if compile_baseline else None,
        output_max_abs=max_abs,
        output_p99_abs=p99_abs,
        output_max_rel=max_rel,
        status=status,
    )


def write_markdown(path: Path, payload: dict) -> None:
    rows: list[Result] = payload["results"]
    e2e_rows: list[AttentionE2EResult] = payload.get("attention_e2e_results", [])
    self_attn_rows: list[SelfAttentionE2EResult] = payload.get("self_attention_e2e_results", [])

    def fmt_us(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "n/a"

    def fmt_speed(value: float | None) -> str:
        return f"{value:.2f}x" if value is not None else "n/a"

    lines = [
        "# Wan QKV Postprocess Demo Results",
        "",
        "This benchmark measures the Wan-style packed-QKV split, Q/K RMSNorm, and RoPE block.",
        "",
        "## Environment",
        "",
    ]
    for key, value in payload["environment"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    if rows:
        lines.extend(
            [
                "",
                "## Postprocess Results",
                "",
            "| Shape | Fused prealloc us | Fused alloc us | PyTorch eager us | torch.compile us | Speedup vs eager | Speedup vs compile | Q max abs | K max abs | Status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in rows:
            shape = f"{item.model_shape} B={item.batch},T={item.tokens},H={item.heads},D={item.head_dim}"
            lines.append(
                f"| {shape} | {item.fused_prealloc_us:.3f} | {item.fused_alloc_us:.3f} | "
                f"{item.torch_eager_us:.3f} | {fmt_us(item.torch_compile_us)} | "
                f"{item.speedup_prealloc:.2f}x | {fmt_speed(item.speedup_vs_compile_prealloc)} | "
                f"{item.q_max_abs:.5f} | {item.k_max_abs:.5f} | {item.status} |"
            )
        lines.append("")
    if e2e_rows:
        lines.extend(
            [
                "## Attention E2E Results",
                "",
                "This path measures `packed_qkv -> Q/K postprocess -> scaled_dot_product_attention output`.",
                "",
                "| Shape | Grid | FlashRT E2E us | PyTorch E2E us | torch.compile E2E us | Speedup vs eager | Speedup vs compile | Output max abs | Status |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in e2e_rows:
            shape = f"{item.model_shape} B={item.batch},T={item.tokens},H={item.heads},D={item.head_dim}"
            lines.append(
                f"| {shape} | {item.grid} | {item.fused_e2e_us:.3f} | "
                f"{item.torch_e2e_us:.3f} | {fmt_us(item.torch_compile_e2e_us)} | "
                f"{item.speedup_e2e:.2f}x | {fmt_speed(item.speedup_vs_compile)} | "
                f"{item.output_max_abs:.5f} | {item.status} |"
            )
        lines.append("")
    if self_attn_rows:
        lines.extend(
            [
                "## Self-Attention E2E Results",
                "",
                "This path measures `x -> packed QKV projection -> Q/K postprocess -> scaled_dot_product_attention -> output projection`.",
                "The FlashRT path packs Wan's separate Q/K/V projection weights once and uses one mathematically equivalent packed QKV projection.",
                "",
                "| Shape | Grid | FlashRT self-attn us | PyTorch self-attn us | torch.compile self-attn us | Speedup vs eager | Speedup vs compile | Output max abs | Status |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in self_attn_rows:
            shape = f"{item.model_shape} B={item.batch},T={item.tokens},H={item.heads},D={item.head_dim}"
            lines.append(
                f"| {shape} | {item.grid} | {item.flashrt_us:.3f} | "
                f"{item.torch_us:.3f} | {fmt_us(item.torch_compile_us)} | "
                f"{item.speedup:.2f}x | {fmt_speed(item.speedup_vs_compile)} | "
                f"{item.output_max_abs:.5f} | {item.status} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed", "hub"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--repo-id", default="LiangSu8899/flashrt-vla-video")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=["postprocess", "attention-e2e", "self-attention-e2e", "both", "all"],
        default="postprocess",
    )
    parser.add_argument("--tokens", default="256,1024,2520,4096")
    parser.add_argument("--heads", default="24,40")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--output", default="internal-tests/demos/wan-qkv-postprocess/results.json")
    parser.add_argument("--markdown", default="internal-tests/demos/wan-qkv-postprocess/results.md")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.head_dim != 128:
        raise SystemExit("current FlashRT qkv_split_norm_rope_bf16 package expects head_dim=128")

    ops = load_ops(args)

    environment = {
        "backend": args.backend,
        "artifact": args.artifact or "",
        "repo_id": args.repo_id if args.backend == "hub" else "",
        "version": args.version if args.backend == "hub" else "",
        "mode": args.mode,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
        "warmup": args.warmup,
        "iters": args.iters,
        "compile_baseline": args.compile_baseline,
        "compile_mode": args.compile_mode,
    }
    results: list[Result] = []
    e2e_results: list[AttentionE2EResult] = []
    self_attn_results: list[SelfAttentionE2EResult] = []
    for heads in parse_csv_ints(args.heads):
        for tokens in parse_csv_ints(args.tokens):
            if args.mode in {"postprocess", "both", "all"}:
                start = time.time()
                result = run_shape(
                    ops,
                    batch=args.batch,
                    tokens=tokens,
                    heads=heads,
                    head_dim=args.head_dim,
                    warmup=args.warmup,
                    iters=args.iters,
                    compile_baseline=args.compile_baseline,
                    compile_mode=args.compile_mode,
                )
                results.append(result)
                print(
                    f"postprocess {result.model_shape} B={result.batch} T={result.tokens} H={result.heads} D={result.head_dim}: "
                    f"fused_prealloc={result.fused_prealloc_us:.3f}us "
                    f"fused_alloc={result.fused_alloc_us:.3f}us "
                    f"torch={result.torch_eager_us:.3f}us "
                    f"compile={result.torch_compile_us if result.torch_compile_us is not None else 'n/a'}us "
                    f"speedup={result.speedup_prealloc:.2f}x/{result.speedup_alloc:.2f}x "
                    f"status={result.status} elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
            if args.mode in {"attention-e2e", "both", "all"}:
                start = time.time()
                e2e = run_attention_e2e_shape(
                    ops,
                    batch=args.batch,
                    tokens=tokens,
                    heads=heads,
                    head_dim=args.head_dim,
                    warmup=args.warmup,
                    iters=args.iters,
                    compile_baseline=args.compile_baseline,
                    compile_mode=args.compile_mode,
                )
                e2e_results.append(e2e)
                print(
                    f"attention-e2e {e2e.model_shape} B={e2e.batch} T={e2e.tokens} H={e2e.heads} D={e2e.head_dim}: "
                    f"flashrt={e2e.fused_e2e_us:.3f}us torch={e2e.torch_e2e_us:.3f}us "
                    f"compile={e2e.torch_compile_e2e_us if e2e.torch_compile_e2e_us is not None else 'n/a'}us "
                    f"speedup={e2e.speedup_e2e:.2f}x status={e2e.status} elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
            if args.mode in {"self-attention-e2e", "all"}:
                start = time.time()
                self_attn = run_self_attention_e2e_shape(
                    ops,
                    batch=args.batch,
                    tokens=tokens,
                    heads=heads,
                    head_dim=args.head_dim,
                    warmup=args.warmup,
                    iters=args.iters,
                    compile_baseline=args.compile_baseline,
                    compile_mode=args.compile_mode,
                )
                self_attn_results.append(self_attn)
                print(
                    f"self-attention-e2e {self_attn.model_shape} B={self_attn.batch} T={self_attn.tokens} "
                    f"H={self_attn.heads} D={self_attn.head_dim}: "
                    f"flashrt={self_attn.flashrt_us:.3f}us torch={self_attn.torch_us:.3f}us "
                    f"compile={self_attn.torch_compile_us if self_attn.torch_compile_us is not None else 'n/a'}us "
                    f"speedup={self_attn.speedup:.2f}x status={self_attn.status} "
                    f"elapsed={time.time() - start:.1f}s",
                    flush=True,
                )

    payload = {
        "environment": environment,
        "results": results,
        "attention_e2e_results": e2e_results,
        "self_attention_e2e_results": self_attn_results,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "environment": environment,
                "results": [asdict(item) for item in results],
                "attention_e2e_results": [asdict(item) for item in e2e_results],
                "self_attention_e2e_results": [asdict(item) for item in self_attn_results],
            },
            indent=2,
        )
        + "\n"
    )
    write_markdown(ROOT / args.markdown, payload)
    print(f"wrote {output}")
    print(f"wrote {ROOT / args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
