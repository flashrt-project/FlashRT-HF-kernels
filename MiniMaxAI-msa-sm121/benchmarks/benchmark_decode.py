import argparse
import time

import torch

from minimaxai_msa_sm121 import (
    flash_decode_with_gqa_share_sparse,
    has_native_ops,
    native_topk_from_scores,
)


def make_case(ctx: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch, hq, hkv, d = 1, 64, 4, 128
    block, topk = 128, 16
    q = torch.randn(batch, hq, d, device="cuda", dtype=torch.bfloat16, generator=g)
    k_cache = torch.randn(ctx, hkv, d, device="cuda", dtype=torch.bfloat16, generator=g)
    v_cache = torch.randn(ctx, hkv, d, device="cuda", dtype=torch.bfloat16, generator=g)
    req_to_token = torch.arange(ctx, device="cuda", dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([ctx], device="cuda", dtype=torch.int32)
    slot_ids = torch.zeros(batch, device="cuda", dtype=torch.int64)
    nb = (ctx + block - 1) // block
    n = min(topk, nb)
    topk_idx = torch.full((hkv, batch, topk), -1, device="cuda", dtype=torch.int32)
    topk_idx[:, :, :n] = torch.arange(n, device="cuda", dtype=torch.int32).view(1, 1, n)
    return q, None, k_cache, v_cache, req_to_token, seq_lens, slot_ids, block, topk_idx


def bench(ctx: int, warmup: int, iters: int) -> float:
    args = make_case(ctx)
    for _ in range(warmup):
        flash_decode_with_gqa_share_sparse(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        flash_decode_with_gqa_share_sparse(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def bench_native_topk(ctx: int, warmup: int, iters: int) -> float | None:
    if not has_native_ops():
        return None
    heads, batch, block, topk = 64, 1, 128, 16
    blocks = (ctx + block - 1) // block
    score = torch.randn(heads, batch, blocks, device="cuda", dtype=torch.float32)
    seq_lens = torch.tensor([ctx], device="cuda", dtype=torch.int32)
    for _ in range(warmup):
        native_topk_from_scores(score, seq_lens, block, topk)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        native_topk_from_scores(score, seq_lens, block, topk)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[2048, 4096, 32768])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    print("gpu:", torch.cuda.get_device_name())
    print("ctx,attention_mean_us,native_topk_mean_us")
    for ctx in args.ctx:
        attn_us = bench(ctx, args.warmup, args.iters)
        topk_us = bench_native_topk(ctx, args.warmup, args.iters)
        topk_text = "NA" if topk_us is None else f"{topk_us:.3f}"
        print(f"{ctx},{attn_us:.3f},{topk_text}")


if __name__ == "__main__":
    main()
