import argparse
import time

import torch

import flashrt_gemm_epilogues as flashrt_ops


def reference(a, b, bias):
    y = a @ b
    y = y + bias
    return torch.nn.functional.gelu(y).to(torch.bfloat16)


def bias_reference(a, b, bias):
    return ((a @ b) + bias).to(torch.bfloat16)


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
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--activation", choices=["gelu", "none"], default="gelu")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(0)
    device = torch.device("cuda")
    a = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16)
    b = torch.randn((args.k, args.n), device=device, dtype=torch.bfloat16)
    bias = torch.randn((args.n,), device=device, dtype=torch.bfloat16)
    out = torch.empty((args.m, args.n), device=device, dtype=torch.bfloat16)

    if args.activation == "gelu":
        fused_fn = lambda: flashrt_ops.bf16_gemm_bias_gelu(a, b, bias, out=out)
        eager_fn = lambda: reference(a, b, bias)
    else:
        fused_fn = lambda: flashrt_ops.bf16_gemm_bias(a, b, bias, out=out)
        eager_fn = lambda: bias_reference(a, b, bias)

    fused_us = time_cuda(fused_fn, args.warmup, args.iters)
    eager_us = time_cuda(eager_fn, args.warmup, args.iters)

    print(f"shape=({args.m}, {args.n}, {args.k})")
    print(f"activation={args.activation}")
    print(f"fused_us={fused_us:.3f}")
    print(f"eager_us={eager_us:.3f}")
    print(f"speedup={eager_us / fused_us:.2f}x")


if __name__ == "__main__":
    main()
