import argparse
import time

import torch

import flashrt_gemm_epilogues as flashrt_ops


def reference(input, bias, scale):
    y = input.float() + bias.float()
    y = torch.nn.functional.gelu(y, approximate="tanh")
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
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(0)
    device = torch.device("cuda")
    input = torch.randn((args.m, args.n), device=device, dtype=torch.bfloat16)
    bias = torch.randn((args.n,), device=device, dtype=torch.bfloat16)
    scale = torch.tensor([0.25], device=device, dtype=torch.float32)
    out = torch.empty(input.shape, device=device, dtype=torch.float8_e4m3fn)

    fused_us = time_cuda(
        lambda: flashrt_ops.bias_gelu_quantize_fp8_static_bf16(
            input, bias, scale, out=out
        ),
        args.warmup,
        args.iters,
    )
    eager_us = time_cuda(lambda: reference(input, bias, scale), args.warmup, args.iters)

    print(f"shape=({args.m}, {args.n})")
    print(f"fused_us={fused_us:.3f}")
    print(f"eager_us={eager_us:.3f}")
    print(f"speedup={eager_us / fused_us:.2f}x")


if __name__ == "__main__":
    main()
