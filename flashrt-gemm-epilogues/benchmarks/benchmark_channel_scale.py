import argparse
import time

import torch

import flashrt_gemm_epilogues as flashrt_ops


def reference(input, channel_scale, scale):
    y = input.float() * channel_scale.float()
    y = torch.clamp(y / scale.float(), -448.0, 448.0)
    return y.to(torch.float8_e4m3fn)


def time_cuda(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(0)
    device = torch.device("cuda")
    input = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16)
    channel_scale = torch.randn((args.k,), device=device, dtype=torch.bfloat16)
    scale = torch.tensor([0.25], device=device, dtype=torch.float32)
    out = torch.empty(input.shape, device=device, dtype=torch.float8_e4m3fn)

    fused_us = time_cuda(
        lambda: flashrt_ops.channel_scale_quantize_fp8_static_bf16(
            input, channel_scale, scale, out=out
        ),
        args.warmup,
        args.iters,
    )
    eager_us = time_cuda(
        lambda: reference(input, channel_scale, scale),
        args.warmup,
        args.iters,
    )

    print(f"shape=({args.m}, {args.k})")
    print(f"fused_us={fused_us:.3f}")
    print(f"eager_us={eager_us:.3f}")
    print(f"speedup={eager_us / fused_us:.2f}x")


if __name__ == "__main__":
    main()
