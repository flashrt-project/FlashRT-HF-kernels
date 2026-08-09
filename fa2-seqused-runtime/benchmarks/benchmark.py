from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F

from fa2_seqused_runtime import allocate_outputs, allocate_workspace, forward_static


def _apply_mem_cap(max_mem_gb: float = 30.0) -> None:
    if not torch.cuda.is_available() or max_mem_gb <= 0:
        return
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(max_mem_gb * 1024**3)
    if total <= 0 or cap >= total:
        return
    torch.cuda.set_per_process_memory_fraction(cap / total)


SHAPES = [
    # GROOT DiT self/cross attention.
    ("groot-dit-self", 1, 51, 51, 32, 32, 48, False),
    ("groot-dit-cross", 1, 51, 1024, 32, 32, 48, False),
    # GROOT N1.7 ViT/VL and SigLIP vision attention.
    ("groot-n17-vit", 1, 256, 256, 16, 16, 64, False),
    ("groot-siglip", 2, 256, 256, 16, 16, 72, False),
    # Qwen2.5-VL/LingBot vision attention.
    ("vl-vision", 1, 256, 256, 16, 16, 80, False),
    # Generic runtime/GQA rows.
    ("gqa-decode", 1, 1, 512, 8, 2, 128, False),
    ("gqa-short", 1, 16, 1024, 16, 4, 128, False),
    ("vla-gqa", 1, 49, 2520, 24, 4, 128, False),
    ("gqa-long-kv", 1, 64, 4096, 32, 8, 128, False),
    ("qwen-causal", 1, 1024, 1024, 32, 8, 128, True),
    ("qwen36-causal", 1, 512, 512, 24, 4, 256, True),
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
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    args = parser.parse_args()
    _apply_mem_cap(args.max_mem_gb)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(
        "Workload,Mode,B,Sq,Sk,Hq,Hkv,D,FlashRT_us,SDPA_expandedGQA_us,Speedup,"
        "MaxAbs,P99Abs,MeanAbs,Cosine"
    )
    for name, batch, sq, sk, hq, hkv, dim, causal in SHAPES:
        if causal and dtype == torch.float16:
            continue
        q = torch.randn(batch, sq, hq, dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, sk, hkv, dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        out, lse = allocate_outputs(q)
        workspace = allocate_workspace(q, k)
        kr = k.repeat_interleave(hq // hkv, dim=2)
        vr = v.repeat_interleave(hq // hkv, dim=2)

        def flashrt():
            forward_static(
                q,
                k,
                v,
                out=out,
                softmax_lse=lse,
                workspace=workspace,
                causal=causal,
            )

        def sdpa():
            return F.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3),
                kr.permute(0, 2, 1, 3),
                vr.permute(0, 2, 1, 3),
                is_causal=causal,
            )

        flashrt_us = time_us(flashrt)
        sdpa_us = time_us(sdpa)
        actual = out.float()
        reference = sdpa().permute(0, 2, 1, 3).float()
        error = (actual - reference).abs()
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), reference.flatten(), dim=0
        ).item()
        print(
            f"{name},{'causal' if causal else 'noncausal'},"
            f"{batch},{sq},{sk},{hq},{hkv},{dim},"
            f"{flashrt_us:.3f},{sdpa_us:.3f},{sdpa_us / flashrt_us:.3f},"
            f"{error.max().item():.9f},"
            f"{torch.quantile(error, 0.99).item():.9f},"
            f"{error.mean().item():.9f},{cosine:.10f}"
        )


if __name__ == "__main__":
    main()
