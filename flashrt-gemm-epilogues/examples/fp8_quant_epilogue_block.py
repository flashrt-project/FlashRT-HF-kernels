"""HF-style FP8 quantization epilogue block using flashrt-gemm-epilogues.

The fused kernels replace common post-projection epilogue sequences:

1. bias + GELU(tanh) + static-scale FP8 cast;
2. GELU(tanh) + static-scale FP8 cast;
3. per-channel scale + static-scale FP8 cast.

Run after publishing or installing the kernel package:

    python examples/fp8_quant_epilogue_block.py --m 64 --n 4096
"""

from __future__ import annotations

import argparse

import torch

try:
    from kernels import get_kernel
except ModuleNotFoundError:
    get_kernel = None


def torch_bias_gelu_fp8(
    x: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    y = x.float() + bias.float()
    y = torch.nn.functional.gelu(y, approximate="tanh")
    y = torch.clamp(y / scale.float(), -448.0, 448.0)
    return y.to(torch.float8_e4m3fn)


def torch_gelu_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    y = torch.nn.functional.gelu(x.float(), approximate="tanh")
    y = torch.clamp(y / scale.float(), -448.0, 448.0)
    return y.to(torch.float8_e4m3fn)


def torch_channel_scale_fp8(
    x: torch.Tensor,
    channel_scale: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    y = x.float() * channel_scale.float()
    y = torch.clamp(y / scale.float(), -448.0, 448.0)
    return y.to(torch.float8_e4m3fn)


class FlashRTFP8QuantEpilogue(torch.nn.Module):
    def __init__(self, *, repo_id: str, version: int) -> None:
        super().__init__()
        if get_kernel is not None:
            self.ops = get_kernel(repo_id, version=version, trust_remote_code=True)
        else:
            import flashrt_gemm_epilogues

            self.ops = flashrt_gemm_epilogues

    def bias_gelu(
        self,
        x: torch.Tensor,
        bias: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return self.ops.bias_gelu_quantize_fp8_static_bf16(x, bias, scale)

    def gelu(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return self.ops.gelu_quantize_fp8_static_bf16(x, scale)

    def channel_scale(
        self,
        x: torch.Tensor,
        channel_scale: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return self.ops.channel_scale_quantize_fp8_static_bf16(
            x, channel_scale, scale
        )


def _time_us(fn, warmup: int, iters: int) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="flashrt/flashrt-gemm-epilogues")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(0)
    x = torch.randn((args.m, args.n), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((args.n,), device="cuda", dtype=torch.bfloat16)
    channel_scale = torch.randn((args.n,), device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.25], device="cuda", dtype=torch.float32)

    fused = FlashRTFP8QuantEpilogue(
        repo_id=args.repo_id,
        version=args.version,
    ).cuda()

    y_bias = fused.bias_gelu(x, bias, scale)
    y_gelu = fused.gelu(x, scale)
    y_channel = fused.channel_scale(x, channel_scale, scale)
    torch.testing.assert_close(
        y_bias.float(), torch_bias_gelu_fp8(x, bias, scale).float(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        y_gelu.float(), torch_gelu_fp8(x, scale).float(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        y_channel.float(),
        torch_channel_scale_fp8(x, channel_scale, scale).float(),
        atol=0,
        rtol=0,
    )

    fused_us = _time_us(lambda: fused.bias_gelu(x, bias, scale), args.warmup, args.iters)
    torch_us = _time_us(
        lambda: torch_bias_gelu_fp8(x, bias, scale),
        args.warmup,
        args.iters,
    )
    print(
        f"bias_gelu_fp8 M={args.m} N={args.n}: "
        f"flashrt={fused_us:.3f}us torch={torch_us:.3f}us "
        f"speedup={torch_us / fused_us:.2f}x"
    )


if __name__ == "__main__":
    main()
