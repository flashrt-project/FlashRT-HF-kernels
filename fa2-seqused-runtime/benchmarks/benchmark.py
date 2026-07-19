from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F

from fa2_seqused_runtime import allocate_outputs, allocate_workspace, forward_static


SHAPES = [
    (1, 1, 512, 8, 2, 128),
    (1, 16, 1024, 16, 4, 128),
    (1, 49, 2520, 24, 4, 128),
    (1, 64, 4096, 32, 8, 128),
    (1, 1024, 1024, 32, 8, 128),
]


def time_us(fn, warmup=50, repeats=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print("B,Sq,Sk,Hq,Hkv,D,FlashRT_us,SDPA_expandedGQA_us,Speedup")
    for batch, sq, sk, hq, hkv, dim in SHAPES:
        q = torch.randn(batch, sq, hq, dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, sk, hkv, dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        out, lse = allocate_outputs(q)
        workspace = allocate_workspace(q, k)
        kr = k.repeat_interleave(hq // hkv, dim=2)
        vr = v.repeat_interleave(hq // hkv, dim=2)

        def flashrt():
            forward_static(q, k, v, out=out, softmax_lse=lse, workspace=workspace)

        def sdpa():
            F.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3),
                kr.permute(0, 2, 1, 3),
                vr.permute(0, 2, 1, 3),
            )

        flashrt_us = time_us(flashrt)
        sdpa_us = time_us(sdpa)
        print(f"{batch},{sq},{sk},{hq},{hkv},{dim},{flashrt_us:.3f},{sdpa_us:.3f},{sdpa_us / flashrt_us:.3f}")


if __name__ == "__main__":
    main()
