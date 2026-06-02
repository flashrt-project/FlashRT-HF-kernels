"""HF-style NVFP4 scale-factor layout example.

The helper converts linear scale-factor bytes with shape ``(rows, D / 16)`` to
the CUTLASS Sm1xx swizzled layout used by Blackwell NVFP4/FP4 GEMM paths.

Run after publishing or installing the kernel package:

    python examples/nvfp4_scale_factor_layout.py --rows 128 --d 4096
"""

from __future__ import annotations

import argparse

import torch
from kernels import get_kernel


def reference_swizzle(scales: torch.Tensor) -> torch.Tensor:
    rows, n_blocks = scales.shape
    n_col_super = (n_blocks + 3) // 4
    out = torch.zeros(
        ((rows + 127) // 128) * n_col_super * 512,
        dtype=torch.uint8,
    )
    src = scales.cpu()
    for row in range(rows):
        row_block = row // 128
        row_inner = row % 128
        for block in range(n_blocks):
            col_block = block // 4
            col_inner = block % 4
            super_idx = row_block * n_col_super + col_block
            inner_off = (row_inner % 32) * 16 + (row_inner // 32) * 4 + col_inner
            out[super_idx * 512 + inner_off] = src[row, block]
    return out.to(scales.device)


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
    parser.add_argument("--repo-id", default="flashrt/flashrt-nvfp4")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--d", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.rows <= 0:
        raise SystemExit("rows must be positive")
    if args.d <= 0 or args.d % 16 != 0:
        raise SystemExit("d must be positive and divisible by 16")

    ops = get_kernel(args.repo_id, version=args.version, trust_remote_code=True)
    scales = torch.randint(
        0,
        256,
        (args.rows, args.d // 16),
        device="cuda",
        dtype=torch.uint8,
    )
    out = ops.nvfp4_sf_linear_to_swizzled(scales)
    expected_bytes = ops.nvfp4_sf_swizzled_bytes(args.rows, args.d)
    if out.numel() != expected_bytes:
        raise RuntimeError(f"expected {expected_bytes} bytes, got {out.numel()}")
    torch.testing.assert_close(out, reference_swizzle(scales))

    out_reuse = torch.zeros((expected_bytes,), device="cuda", dtype=torch.uint8)
    fused_us = _time_us(
        lambda: ops.nvfp4_sf_linear_to_swizzled(scales, out=out_reuse),
        args.warmup,
        args.iters,
    )
    print(
        f"nvfp4_sf_linear_to_swizzled rows={args.rows} D={args.d}: "
        f"bytes={expected_bytes} latency={fused_us:.3f}us"
    )


if __name__ == "__main__":
    main()
