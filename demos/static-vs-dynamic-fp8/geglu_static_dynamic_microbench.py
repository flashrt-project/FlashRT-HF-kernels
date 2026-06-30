#!/usr/bin/env python3
"""Static-fused vs dynamic-split FP8 GeGLU MLP microbench.

This is the standalone kernel-level companion for the PI0.5 e2e demo. It uses
random tensors with Gemma/PI0.5-like dimensions and public Hub kernels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from kernels import get_kernel


FP8_MAX = 448.0


@dataclass
class Result:
    M: int
    K: int
    H: int
    static_fused_ms: float
    dynamic_split_ms: float
    bf16_ms: float
    static_vs_dynamic: float
    bf16_vs_static: float
    static_max_abs: float
    static_p99_abs: float
    static_mse: float
    static_cosine: float
    dynamic_max_abs: float
    dynamic_p99_abs: float
    dynamic_mse: float
    dynamic_cosine: float


def scale_from_amax(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float().abs().max() / FP8_MAX, min=1e-12).reshape(1).to(
        device=x.device, dtype=torch.float32
    )


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float(), -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def dequant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.float() * scale.float()


def percentile_abs(diff: torch.Tensor, q: float) -> float:
    flat = diff.flatten()
    k = max(1, min(flat.numel(), int(q * flat.numel() + 0.999999)))
    return float(flat.kthvalue(k).values.item())


def compare(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float, float]:
    diff = (got.float() - ref.float()).abs()
    mse = (got.float() - ref.float()).pow(2).mean()
    cos = F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    return float(diff.max().item()), percentile_abs(diff, 0.99), float(mse.item()), float(cos.item())


def time_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        if not torch.isfinite(out).all():
            raise RuntimeError("non-finite benchmark output")
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def run_shape(M: int, K: int, H: int, *, warmup: int, iters: int) -> Result:
    device = torch.device("cuda")
    ffn = get_kernel("flashrt/flashrt-fp8-swiglu-ffn", version=1, trust_remote_code=True)
    gemm = get_kernel("flashrt/flashrt-gemm-epilogues", version=1, trust_remote_code=True)

    torch.manual_seed(1234 + M)
    x = torch.randn((M, K), device=device, dtype=torch.bfloat16)
    gate_up_w_bf16 = torch.randn((2 * H, K), device=device, dtype=torch.bfloat16) * 0.02
    down_w_bf16 = torch.randn((K, H), device=device, dtype=torch.bfloat16) * 0.02
    ones_k = torch.ones((K,), device=device, dtype=torch.bfloat16)
    ones_h = torch.ones((H,), device=device, dtype=torch.bfloat16)

    x_scale = scale_from_amax(x)
    gate_up_w_scale = scale_from_amax(gate_up_w_bf16)
    down_w_scale = scale_from_amax(down_w_bf16)
    gate_up_w_fp8 = quantize_fp8(gate_up_w_bf16, gate_up_w_scale).contiguous()
    down_w_fp8 = quantize_fp8(down_w_bf16, down_w_scale).contiguous()

    with torch.no_grad():
        gate_up_ref = x.float() @ gate_up_w_bf16.float().t()
        gate, up = gate_up_ref.chunk(2, dim=1)
        hidden_ref = F.gelu(gate, approximate="tanh") * up
        hidden_scale = scale_from_amax(hidden_ref)
        ref = (hidden_ref @ down_w_bf16.float().t()).to(torch.bfloat16)

    x_fp8 = torch.empty((M, K), device=device, dtype=torch.float8_e4m3fn)
    gate_up_bf16 = torch.empty((M, 2 * H), device=device, dtype=torch.bfloat16)
    hidden_bf16 = torch.empty((M, H), device=device, dtype=torch.bfloat16)
    hidden_fp8 = torch.empty((M, H), device=device, dtype=torch.float8_e4m3fn)
    out_static = torch.empty((M, K), device=device, dtype=torch.bfloat16)
    out_dynamic = torch.empty((M, K), device=device, dtype=torch.bfloat16)

    def static_fused():
        gemm.channel_scale_quantize_fp8_static_bf16(x, ones_k, x_scale, out=x_fp8)
        return ffn.fp8_geglu_mlp_bf16(
            x_fp8,
            gate_up_w_fp8,
            down_w_fp8,
            x_scale,
            gate_up_w_scale,
            hidden_scale,
            down_w_scale,
            gate_up_bf16,
            hidden_fp8,
            out_static,
        )

    def dynamic_split():
        dyn_x_scale = scale_from_amax(x)
        gemm.channel_scale_quantize_fp8_static_bf16(x, ones_k, dyn_x_scale, out=x_fp8)
        ffn.fp8_gemm_bf16(x_fp8, gate_up_w_fp8, dyn_x_scale, gate_up_w_scale, out=gate_up_bf16)
        gate, up = gate_up_bf16.float().chunk(2, dim=1)
        hidden_bf16.copy_((F.gelu(gate, approximate="tanh") * up).to(torch.bfloat16))
        dyn_hidden_scale = scale_from_amax(hidden_bf16)
        gemm.channel_scale_quantize_fp8_static_bf16(hidden_bf16, ones_h, dyn_hidden_scale, out=hidden_fp8)
        return ffn.fp8_gemm_bf16(hidden_fp8, down_w_fp8, dyn_hidden_scale, down_w_scale, out=out_dynamic)

    def bf16_ref():
        gate_up = x @ gate_up_w_bf16.t()
        gate, up = gate_up.chunk(2, dim=1)
        hidden = F.gelu(gate.float(), approximate="tanh") * up.float()
        return (hidden.to(torch.bfloat16) @ down_w_bf16.t()).to(torch.bfloat16)

    static_fused()
    dynamic_split()
    torch.cuda.synchronize()

    static_ms = time_ms(static_fused, warmup=warmup, iters=iters)
    dynamic_ms = time_ms(dynamic_split, warmup=warmup, iters=iters)
    bf16_ms = time_ms(bf16_ref, warmup=warmup, iters=iters)
    s_max, s_p99, s_mse, s_cos = compare(out_static, ref)
    d_max, d_p99, d_mse, d_cos = compare(out_dynamic, ref)
    return Result(
        M=M,
        K=K,
        H=H,
        static_fused_ms=static_ms,
        dynamic_split_ms=dynamic_ms,
        bf16_ms=bf16_ms,
        static_vs_dynamic=dynamic_ms / static_ms,
        bf16_vs_static=bf16_ms / static_ms,
        static_max_abs=s_max,
        static_p99_abs=s_p99,
        static_mse=s_mse,
        static_cosine=s_cos,
        dynamic_max_abs=d_max,
        dynamic_p99_abs=d_p99,
        dynamic_mse=d_mse,
        dynamic_cosine=d_cos,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", choices=("headline", "decode", "prefill"), default="headline")
    parser.add_argument("--K", type=int, default=2048)
    parser.add_argument("--H", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ms = {"headline": [50, 512], "decode": [50], "prefill": [512]}[args.shapes]
    results = [asdict(run_shape(M, args.K, args.H, warmup=args.warmup, iters=args.iters)) for M in ms]
    print(json.dumps({"name": "geglu_static_dynamic_microbench", "results": results}, indent=2))


if __name__ == "__main__":
    main()
