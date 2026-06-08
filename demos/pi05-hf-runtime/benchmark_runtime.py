#!/usr/bin/env python3
"""PI0.5-shaped FlashRT Hub-kernel runtime overhead benchmark."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from kernels import get_kernel


SHAPES: dict[str, tuple[int, int, int, int, int]] = {
    "pi05_decoder": (10, 1024, 4096, 1024, 18),
    "pi05_vision": (512, 1152, 4304, 1152, 27),
    "groot_vit": (512, 1024, 4096, 1024, 24),
}


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float(), -448.0, 448.0).to(
        torch.float8_e4m3fn
    )


def dequant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.float() * scale.float()


def tensor_scale(x: torch.Tensor, *, floor: float = 1e-6, safety: float = 1.05) -> torch.Tensor:
    amax = x.detach().float().abs().max()
    return torch.clamp((amax / 448.0) * safety, min=floor).reshape(1).to(
        device=x.device,
        dtype=torch.float32,
    )


def percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def time_us(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
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


@dataclass
class RuntimeResult:
    shape: str
    m: int
    k: int
    h: int
    n: int
    layers: int
    torch_fp8_reference_us: float
    naive_hub_us: float
    runtime_prealloc_us: float
    runtime_cuda_graph_us: float | None
    runtime_cuda_graph_with_input_copy_us: float | None
    naive_vs_torch: float
    runtime_vs_torch: float
    graph_vs_torch: float | None
    runtime_vs_naive: float
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    graph_status: str


class TorchFP8Reference:
    def __init__(self, weights: "RuntimeWeights") -> None:
        self.weights = weights

    def __call__(self, x_bf16: torch.Tensor) -> torch.Tensor:
        x = x_bf16
        w = self.weights
        for i in range(w.layers):
            x_fp8 = quantize_fp8(x, w.input_scale)
            hidden = (
                dequant_fp8(x_fp8, w.input_scale)
                @ dequant_fp8(w.up_w_fp8[i], w.up_w_scale).T
            ).to(torch.bfloat16)
            hidden = torch.nn.functional.gelu(
                hidden.float() + w.up_bias[i].float(),
                approximate="tanh",
            )
            hidden_fp8 = quantize_fp8(hidden, w.hidden_scale)
            out = (
                dequant_fp8(hidden_fp8, w.hidden_scale)
                @ dequant_fp8(w.down_w_fp8[i], w.down_w_scale).T
            ).to(torch.bfloat16)
            x = (out.float() + w.down_bias[i].float()).to(torch.bfloat16)
        return x


class RuntimeWeights:
    def __init__(
        self,
        *,
        layers: int,
        k: int,
        h: int,
        n: int,
        device: torch.device,
    ) -> None:
        self.layers = layers
        self.input_scale = torch.tensor([0.01], device=device, dtype=torch.float32)
        self.hidden_scale = torch.tensor([0.01], device=device, dtype=torch.float32)
        self.channel_scale = torch.ones((k,), device=device, dtype=torch.bfloat16)

        up_w_bf16 = [
            (torch.randn((h, k), device=device, dtype=torch.float32) / math.sqrt(k))
            .to(torch.bfloat16)
            .contiguous()
            for _ in range(layers)
        ]
        down_w_bf16 = [
            (torch.randn((n, h), device=device, dtype=torch.float32) / math.sqrt(h))
            .to(torch.bfloat16)
            .contiguous()
            for _ in range(layers)
        ]
        self.up_w_scale = tensor_scale(torch.stack([w.abs().max() for w in up_w_bf16]))
        self.down_w_scale = tensor_scale(torch.stack([w.abs().max() for w in down_w_bf16]))
        self.up_w_fp8 = [
            quantize_fp8(w, self.up_w_scale).contiguous() for w in up_w_bf16
        ]
        self.down_w_fp8 = [
            quantize_fp8(w, self.down_w_scale).contiguous() for w in down_w_bf16
        ]
        self.up_bias = [
            (torch.randn((h,), device=device, dtype=torch.float32) * 0.01)
            .to(torch.bfloat16)
            .contiguous()
            for _ in range(layers)
        ]
        self.down_bias = [
            (torch.randn((n,), device=device, dtype=torch.float32) * 0.01)
            .to(torch.bfloat16)
            .contiguous()
            for _ in range(layers)
        ]


class NaiveHubChain:
    def __init__(self, weights: RuntimeWeights, *, version: int) -> None:
        self.weights = weights
        self.fp8_ops = get_kernel(
            "flashrt/flashrt-fp8-ffn",
            version=version,
            trust_remote_code=True,
        )
        self.quant_ops = get_kernel(
            "flashrt/flashrt-gemm-epilogues",
            version=version,
            trust_remote_code=True,
        )

    def __call__(self, x_bf16: torch.Tensor) -> torch.Tensor:
        x = x_bf16
        w = self.weights
        for i in range(w.layers):
            x_fp8 = self.quant_ops.channel_scale_quantize_fp8_static_bf16(
                x,
                w.channel_scale,
                w.input_scale,
            )
            x = self.fp8_ops.fp8_gelu_mlp_bf16(
                x_fp8,
                w.up_w_fp8[i],
                w.up_bias[i],
                w.down_w_fp8[i],
                w.down_bias[i],
                w.input_scale,
                w.up_w_scale,
                w.hidden_scale,
                w.down_w_scale,
            )
        return x


class FlashRTHubRuntime:
    def __init__(
        self,
        weights: RuntimeWeights,
        *,
        m: int,
        k: int,
        h: int,
        n: int,
        version: int,
        device: torch.device,
    ) -> None:
        self.weights = weights
        self.fp8_ops = get_kernel(
            "flashrt/flashrt-fp8-ffn",
            version=version,
            trust_remote_code=True,
        )
        self.quant_ops = get_kernel(
            "flashrt/flashrt-gemm-epilogues",
            version=version,
            trust_remote_code=True,
        )
        self.x_fp8 = [
            torch.empty((m, k), device=device, dtype=torch.float8_e4m3fn)
            for _ in range(weights.layers)
        ]
        self.hidden_bf16 = [
            torch.empty((m, h), device=device, dtype=torch.bfloat16)
            for _ in range(weights.layers)
        ]
        self.hidden_fp8 = [
            torch.empty((m, h), device=device, dtype=torch.float8_e4m3fn)
            for _ in range(weights.layers)
        ]
        self.out = [
            torch.empty((m, n), device=device, dtype=torch.bfloat16)
            for _ in range(weights.layers)
        ]

    def __call__(self, x_bf16: torch.Tensor) -> torch.Tensor:
        x = x_bf16
        w = self.weights
        for i in range(w.layers):
            self.quant_ops.channel_scale_quantize_fp8_static_bf16(
                x,
                w.channel_scale,
                w.input_scale,
                self.x_fp8[i],
            )
            self.fp8_ops.fp8_gelu_mlp_bf16(
                self.x_fp8[i],
                w.up_w_fp8[i],
                w.up_bias[i],
                w.down_w_fp8[i],
                w.down_bias[i],
                w.input_scale,
                w.up_w_scale,
                w.hidden_scale,
                w.down_w_scale,
                self.hidden_bf16[i],
                self.hidden_fp8[i],
                self.out[i],
            )
            x = self.out[i]
        return x


class CapturedRuntime:
    def __init__(self, runtime: FlashRTHubRuntime, x: torch.Tensor) -> None:
        self.runtime = runtime
        self.static_input = torch.empty_like(x)
        self.source_input = x.detach().clone()
        self.static_input.copy_(x)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.static_output = runtime(self.static_input)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = runtime(self.static_input)

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        return self.static_output

    def replay_with_input_copy(self) -> torch.Tensor:
        self.static_input.copy_(self.source_input)
        self.graph.replay()
        return self.static_output


def run(args: argparse.Namespace) -> RuntimeResult:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.shape not in SHAPES:
        raise SystemExit(f"unknown shape {args.shape!r}; choose one of {sorted(SHAPES)}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    m, k, h, n, layers = SHAPES[args.shape]
    weights = RuntimeWeights(layers=layers, k=k, h=h, n=n, device=device)
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16).contiguous()

    torch_ref = TorchFP8Reference(weights)
    naive = NaiveHubChain(weights, version=args.version)
    runtime = FlashRTHubRuntime(
        weights,
        m=m,
        k=k,
        h=h,
        n=n,
        version=args.version,
        device=device,
    )

    expected = torch_ref(x)
    got = runtime(x)
    torch.cuda.synchronize()
    diff = (got.float() - expected.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    p99_abs = float(percentile(diff, 0.99).item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            got.float().flatten(),
            expected.float().flatten(),
            dim=0,
        ).item()
    )
    if p99_abs > args.p99_abs_limit or cosine < args.cosine_limit:
        raise RuntimeError(
            "correctness gate failed: "
            f"p99_abs={p99_abs:.6f}, cosine={cosine:.8f}"
        )

    torch_us = time_us(lambda: torch_ref(x), warmup=args.warmup, iters=args.iters)
    naive_us = time_us(lambda: naive(x), warmup=args.warmup, iters=args.iters)
    runtime_us = time_us(lambda: runtime(x), warmup=args.warmup, iters=args.iters)

    graph_us = None
    graph_copy_us = None
    graph_status = "not_requested"
    if args.cuda_graph:
        try:
            captured = CapturedRuntime(runtime, x)
            graph_us = time_us(captured.replay, warmup=args.warmup, iters=args.iters)
            graph_copy_us = time_us(
                captured.replay_with_input_copy,
                warmup=args.warmup,
                iters=args.iters,
            )
            graph_status = "ok"
        except Exception as exc:  # noqa: BLE001
            graph_status = f"unsupported: {type(exc).__name__}: {exc}"

    return RuntimeResult(
        shape=args.shape,
        m=m,
        k=k,
        h=h,
        n=n,
        layers=layers,
        torch_fp8_reference_us=torch_us,
        naive_hub_us=naive_us,
        runtime_prealloc_us=runtime_us,
        runtime_cuda_graph_us=graph_us,
        runtime_cuda_graph_with_input_copy_us=graph_copy_us,
        naive_vs_torch=torch_us / naive_us,
        runtime_vs_torch=torch_us / runtime_us,
        graph_vs_torch=None if graph_us is None else torch_us / graph_us,
        runtime_vs_naive=naive_us / runtime_us,
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        cosine=cosine,
        graph_status=graph_status,
    )


def write_markdown(path: Path, result: RuntimeResult) -> None:
    graph_us = "n/a" if result.runtime_cuda_graph_us is None else f"{result.runtime_cuda_graph_us:.3f}"
    graph_speed = "n/a" if result.graph_vs_torch is None else f"{result.graph_vs_torch:.2f}x"
    graph_copy_us = (
        "n/a"
        if result.runtime_cuda_graph_with_input_copy_us is None
        else f"{result.runtime_cuda_graph_with_input_copy_us:.3f}"
    )
    text = f"""# PI0.5 HF Runtime Result

| Field | Value |
| --- | ---: |
| Shape | `{result.shape}` |
| M,K,H,N,layers | `{result.m},{result.k},{result.h},{result.n},{result.layers}` |
| Torch FP8 reference us | {result.torch_fp8_reference_us:.3f} |
| Naive Hub us | {result.naive_hub_us:.3f} |
| Runtime prealloc us | {result.runtime_prealloc_us:.3f} |
| Runtime CUDA Graph us | {graph_us} |
| Runtime CUDA Graph + input copy us | {graph_copy_us} |
| Naive vs torch | {result.naive_vs_torch:.2f}x |
| Runtime vs torch | {result.runtime_vs_torch:.2f}x |
| Graph vs torch | {graph_speed} |
| Runtime vs naive | {result.runtime_vs_naive:.2f}x |
| max_abs | {result.max_abs:.6f} |
| mean_abs | {result.mean_abs:.6f} |
| p99_abs | {result.p99_abs:.6f} |
| cosine | {result.cosine:.8f} |
| graph_status | `{result.graph_status}` |

"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="pi05_decoder", choices=sorted(SHAPES))
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--p99-abs-limit", type=float, default=0.25)
    parser.add_argument("--cosine-limit", type=float, default=0.999)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    result = run(args)
    print(json.dumps(asdict(result), indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2) + "\n")
    if args.markdown is not None:
        write_markdown(args.markdown, result)


if __name__ == "__main__":
    main()
