#!/usr/bin/env python3
"""Benchmark BF16-to-FP8 SwiGLU/GeGLU region boundaries."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import benchmark as base


SHAPES = {
    "decoder_m8": (8, 1024, 4096, 1024),
    "decoder_m51": (51, 1024, 4096, 1024),
    "decoder_m64": (64, 1024, 4096, 1024),
    "decoder_m105": (105, 1024, 4096, 1024),
    "decoder_m128": (128, 1024, 4096, 1024),
    "dit_m51": (51, 1536, 6144, 1536),
    "dit_m128": (128, 1536, 6144, 1536),
}


@dataclass
class Result:
    shape: str
    activation: str
    M: int
    K: int
    H: int
    N: int
    flashrt_bf16_entry_us: float
    flashrt_cuda_graph_us: float
    separate_quant_us: float
    fp8_kernel_only_us: float
    torch_bf16_eager_us: float
    torch_bf16_compile_us: float | None
    speedup_vs_separate_quant: float
    speedup_vs_eager: float
    speedup_vs_compile: float | None
    compile_status: str
    flashrt_compile_status: str
    input_quant_exact: bool
    output_dtype: str
    staged_max_abs: float
    staged_mean_abs: float
    staged_p99_abs: float
    staged_cosine: float
    bf16_max_abs: float
    bf16_mean_abs: float
    bf16_p99_abs: float
    bf16_cosine: float
    performance_status: str
    status: str


def percentile(x: torch.Tensor, q: float) -> float:
    flat = x.flatten()
    k = max(1, min(flat.numel(), int(q * flat.numel() + 0.999999)))
    return float(flat.kthvalue(k).values.item())


def metrics(got: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (got.float() - expected.float()).abs().flatten()
    cosine = F.cosine_similarity(
        got.float().flatten(), expected.float().flatten(), dim=0
    )
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": percentile(diff, 0.99),
        "cosine": float(cosine.item()),
    }


def quantize_input(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    inv_scale = 1.0 / scale.float()
    return torch.clamp(
        x.float() * inv_scale, -base.fp8_max(), base.fp8_max()
    ).to(base.fp8_dtype())


def midm_padded_rows(rows: int) -> int:
    if (
        torch.version.hip is None
        and torch.cuda.get_device_capability(0) == (11, 0)
        and 9 <= rows <= 128
    ):
        return ((rows + 63) // 64) * 64
    return rows


def make_case(M: int, K: int, H: int, N: int, activation: str):
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.25
    gate_up = torch.randn(
        (2 * H, K), device="cuda", dtype=torch.bfloat16
    ) * (K**-0.5)
    down = torch.randn((N, H), device="cuda", dtype=torch.bfloat16) * (H**-0.5)

    def scale_for(tensor: torch.Tensor) -> torch.Tensor:
        return (
            tensor.float().abs().max() / (0.9 * base.fp8_max())
        ).clamp_min(1e-6).reshape(1)

    x_scale = scale_for(x)
    gate_up_scale = scale_for(gate_up)
    down_scale = scale_for(down)
    x_fp8 = quantize_input(x, x_scale)
    gate_up_fp8 = base.quantize_fp8(gate_up, gate_up_scale)
    down_fp8 = base.quantize_fp8(down, down_scale)
    calibrated_gate_up = (
        (x_fp8.float() * x_scale) @ (gate_up_fp8.float() * gate_up_scale).T
    )
    gate, up = calibrated_gate_up.chunk(2, dim=1)
    if activation == "silu":
        calibrated_hidden = F.silu(gate) * up
    else:
        calibrated_hidden = F.gelu(gate, approximate="tanh") * up
    hidden_scale = scale_for(calibrated_hidden)
    return (
        x,
        gate_up,
        down,
        x_fp8,
        gate_up_fp8,
        down_fp8,
        x_scale,
        gate_up_scale,
        hidden_scale,
        down_scale,
    )


def run_shape(ops, name: str, shape, args) -> Result:
    M, K, H, N = shape
    (
        x,
        gate_up,
        down,
        x_fp8,
        gate_up_fp8,
        down_fp8,
        x_scale,
        gate_up_scale,
        hidden_scale,
        down_scale,
    ) = make_case(M, K, H, N, args.activation)

    padded_m = midm_padded_rows(M)
    input_fp8 = torch.empty((padded_m, K), device="cuda", dtype=base.fp8_dtype())
    gate_up_bf16 = torch.empty(
        (padded_m, 2 * H), device="cuda", dtype=torch.bfloat16
    )
    hidden_fp8 = torch.empty((padded_m, H), device="cuda", dtype=base.fp8_dtype())
    out = torch.empty((padded_m, N), device="cuda", dtype=torch.bfloat16)
    exact_gate_up = torch.empty_like(gate_up_bf16)
    exact_hidden = torch.empty_like(hidden_fp8)
    exact_out = torch.empty_like(out)
    staged_gate_up = torch.empty((M, 2 * H), device="cuda", dtype=torch.bfloat16)
    staged_hidden = torch.empty((M, H), device="cuda", dtype=base.fp8_dtype())
    staged_out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    op_prefix = "swi" if args.activation == "silu" else "ge"
    new_op = getattr(ops, f"bf16_fp8_{op_prefix}glu_mlp_bf16")
    staged_op = getattr(ops, f"fp8_{op_prefix}glu_mlp_bf16")

    def flashrt_call():
        return new_op(
            x,
            gate_up_fp8,
            down_fp8,
            x_scale,
            gate_up_scale,
            hidden_scale,
            down_scale,
            input_fp8=input_fp8,
            gate_up_bf16=gate_up_bf16,
            hidden_fp8=hidden_fp8,
            out=out,
            pad_to=padded_m,
        )

    def staged_call(input_arg=x_fp8):
        return staged_op(
            input_arg,
            gate_up_fp8,
            down_fp8,
            x_scale,
            gate_up_scale,
            hidden_scale,
            down_scale,
            gate_up_bf16=staged_gate_up,
            hidden_fp8=staged_hidden,
            out=staged_out,
        )

    def separate_quant_call():
        return staged_call(quantize_input(x, x_scale))

    def exact_staged_call():
        return staged_op(
            input_fp8,
            gate_up_fp8,
            down_fp8,
            x_scale,
            gate_up_scale,
            hidden_scale,
            down_scale,
            gate_up_bf16=exact_gate_up,
            hidden_fp8=exact_hidden,
            out=exact_out,
        )[:M]

    def torch_bf16_reference():
        gate, up = F.linear(x, gate_up).float().chunk(2, dim=1)
        if args.activation == "silu":
            hidden = F.silu(gate) * up
        else:
            hidden = F.gelu(gate, approximate="tanh") * up
        return F.linear(hidden.to(torch.bfloat16), down)

    got = flashrt_call().clone()
    staged = exact_staged_call().clone()
    torch.cuda.synchronize()
    quant_exact = bool(
        torch.equal(input_fp8[:M], x_fp8)
        and (padded_m == M or torch.count_nonzero(input_fp8[M:]).item() == 0)
    )
    staged_metrics = metrics(got, staged)
    bf16_expected = torch_bf16_reference()
    bf16_metrics = metrics(got, bf16_expected)
    staged_compatible = quant_exact and staged_metrics["max_abs"] == 0.0

    flashrt_us = base.time_us(flashrt_call, args.warmup, args.iters)
    graph = torch.cuda.CUDAGraph()
    flashrt_call()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        flashrt_call()
    graph_us = base.time_us(graph.replay, args.warmup, args.iters)
    separate_us = base.time_us(separate_quant_call, args.warmup, args.iters)
    kernel_us = base.time_us(staged_call, args.warmup, args.iters)
    eager_us = base.time_us(torch_bf16_reference, args.warmup, args.iters)

    compile_us = None
    compile_status = "not_requested"
    flashrt_compile_status = "not_requested"
    if args.compile_baseline:
        try:
            compiled = torch.compile(
                torch_bf16_reference, fullgraph=True, mode="reduce-overhead"
            )
            compiled_out = compiled()
            torch.cuda.synchronize()
            compiled_metrics = metrics(compiled_out, bf16_expected)
            if compiled_metrics["cosine"] < 0.9999:
                compile_status = (
                    "mismatch: cosine="
                    f"{compiled_metrics['cosine']:.8f}"
                )
            else:
                compile_us = base.time_us(
                    compiled, args.warmup, args.iters
                )
                compile_status = "fullgraph-ok"
        except Exception as exc:  # noqa: BLE001
            compile_status = f"failed: {type(exc).__name__}: {exc}"
        try:
            compiled_flashrt = torch.compile(
                flashrt_call, fullgraph=True, mode="reduce-overhead"
            )
            compiled_got = compiled_flashrt().clone()
            torch.cuda.synchronize()
            flashrt_compile_status = (
                "fullgraph-ok"
                if metrics(compiled_got, got)["max_abs"] == 0.0
                else "mismatch"
            )
        except Exception as exc:  # noqa: BLE001
            flashrt_compile_status = f"failed: {type(exc).__name__}: {exc}"

    speedup_eager = eager_us / flashrt_us
    speedup_separate = separate_us / flashrt_us
    perf_status = (
        ("PASS" if speedup_eager >= 1.3 else "FAIL")
        if M == 51
        else "DIAGNOSTIC"
    )
    status = (
        "PASS"
        if quant_exact and staged_compatible and perf_status != "FAIL"
        else "FAIL"
    )
    return Result(
        shape=name,
        activation=args.activation,
        M=M,
        K=K,
        H=H,
        N=N,
        flashrt_bf16_entry_us=flashrt_us,
        flashrt_cuda_graph_us=graph_us,
        separate_quant_us=separate_us,
        fp8_kernel_only_us=kernel_us,
        torch_bf16_eager_us=eager_us,
        torch_bf16_compile_us=compile_us,
        speedup_vs_separate_quant=speedup_separate,
        speedup_vs_eager=speedup_eager,
        speedup_vs_compile=compile_us / flashrt_us if compile_us else None,
        compile_status=compile_status,
        flashrt_compile_status=flashrt_compile_status,
        input_quant_exact=quant_exact,
        output_dtype=str(got.dtype),
        staged_max_abs=staged_metrics["max_abs"],
        staged_mean_abs=staged_metrics["mean_abs"],
        staged_p99_abs=staged_metrics["p99_abs"],
        staged_cosine=staged_metrics["cosine"],
        bf16_max_abs=bf16_metrics["max_abs"],
        bf16_mean_abs=bf16_metrics["mean_abs"],
        bf16_p99_abs=bf16_metrics["p99_abs"],
        bf16_cosine=bf16_metrics["cosine"],
        performance_status=perf_status,
        status=status,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed", "hub"], default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--repo-id", default="flashrt/flashrt-fp8-swiglu-ffn")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--activation", choices=["silu", "gelu"], default="silu")
    parser.add_argument("--shapes", default="all")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(23)
    if args.backend == "source":
        ops = base.load_source_ops()
    elif args.backend == "installed":
        ops = base.load_installed_ops(args.artifact)
    else:
        ops = base.load_hub_ops(args.repo_id, args.version)
    names = list(SHAPES) if args.shapes == "all" else args.shapes.split(",")
    unknown = [name for name in names if name not in SHAPES]
    if unknown:
        raise SystemExit(f"unknown shapes: {unknown}")

    results = []
    for name in names:
        result = run_shape(ops, name, SHAPES[name], args)
        results.append(result)
        compile_text = (
            f"{result.torch_bf16_compile_us:.3f}us"
            if result.torch_bf16_compile_us is not None
            else result.compile_status
        )
        print(
            f"{result.status} {name}/{args.activation}: "
            f"flashrt={result.flashrt_bf16_entry_us:.3f}us "
            f"graph={result.flashrt_cuda_graph_us:.3f}us "
            f"separate={result.separate_quant_us:.3f}us "
            f"kernel_only={result.fp8_kernel_only_us:.3f}us "
            f"eager={result.torch_bf16_eager_us:.3f}us "
            f"compile={compile_text} vs_eager={result.speedup_vs_eager:.2f}x "
            f"vs_separate={result.speedup_vs_separate_quant:.2f}x "
            f"staged_max={result.staged_max_abs:.6f} "
            f"bf16_cos={result.bf16_cosine:.8f} "
            f"perf={result.performance_status} "
            f"op_compile={result.flashrt_compile_status}"
        )
        torch.cuda.empty_cache()

    payload = {
        "backend": args.backend,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "activation": args.activation,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": [asdict(result) for result in results],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if any(result.status != "PASS" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
