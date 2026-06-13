import argparse
import time

import torch

from minimaxai_msa_sm121 import flash_decode_with_gqa_share_sparse


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[2048, 4096, 32768])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    print("gpu:", torch.cuda.get_device_name())
    print("ctx,mean_us")
    for ctx in args.ctx:
        print(f"{ctx},{bench(ctx, args.warmup, args.iters):.3f}")


if __name__ == "__main__":
    main()
