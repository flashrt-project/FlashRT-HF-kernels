"""HF-style QKV post-processing block using flashrt-vla-video.

The fused kernel replaces this common sequence:

1. split packed QKV;
2. RMS-normalize Q and K;
3. apply interleaved RoPE to Q and K.

Run after publishing or installing the kernel package:

    python examples/qkv_postprocess_block.py --tokens 256 --heads 24
"""

from __future__ import annotations

import argparse

import torch

try:
    from kernels import get_kernel
except ModuleNotFoundError:
    get_kernel = None


def torch_qkv_split_norm_rope(
    packed_qkv: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    freqs_re: torch.Tensor,
    freqs_im: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, _ = packed_qkv.shape
    dim = heads * head_dim
    q = packed_qkv[..., :dim].reshape(batch, tokens, heads, head_dim)
    k = packed_qkv[..., dim : 2 * dim].reshape(batch, tokens, heads, head_dim)

    qf = q.float()
    kf = k.float()
    q_norm = qf * torch.rsqrt((qf * qf).mean(dim=(-2, -1), keepdim=True) + eps)
    k_norm = kf * torch.rsqrt((kf * kf).mean(dim=(-2, -1), keepdim=True) + eps)
    q_norm = q_norm * norm_q_weight.reshape(1, 1, heads, head_dim).float()
    k_norm = k_norm * norm_k_weight.reshape(1, 1, heads, head_dim).float()

    def rope(x: torch.Tensor) -> torch.Tensor:
        real = x[..., 0::2].float()
        imag = x[..., 1::2].float()
        fre = freqs_re[:tokens][None, :, None, :]
        fim = freqs_im[:tokens][None, :, None, :]
        out = torch.empty_like(x, dtype=torch.float32)
        out[..., 0::2] = real * fre - imag * fim
        out[..., 1::2] = real * fim + imag * fre
        return out.to(torch.bfloat16)

    return rope(q_norm), rope(k_norm)


class FlashRTQKVPostProcess(torch.nn.Module):
    def __init__(self, *, repo_id: str, version: int, heads: int, head_dim: int) -> None:
        super().__init__()
        if get_kernel is not None:
            self.ops = get_kernel(repo_id, version=version, trust_remote_code=True)
        else:
            import flashrt_vla_video

            self.ops = flashrt_vla_video
        self.heads = heads
        self.head_dim = head_dim

    def forward(
        self,
        packed_qkv: torch.Tensor,
        norm_q_weight: torch.Tensor,
        norm_k_weight: torch.Tensor,
        freqs_re: torch.Tensor,
        freqs_im: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ops.qkv_split_norm_rope_bf16(
            packed_qkv,
            norm_q_weight,
            norm_k_weight,
            freqs_re,
            freqs_im,
            heads=self.heads,
            head_dim=self.head_dim,
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
    parser.add_argument("--repo-id", default="flashrt/flashrt-vla-video")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.head_dim != 128:
        raise SystemExit("current kernel package expects head_dim=128")

    torch.manual_seed(0)
    dim = args.heads * args.head_dim
    packed_qkv = torch.randn(
        (1, args.tokens, 3 * dim), device="cuda", dtype=torch.bfloat16
    ).contiguous()
    norm_q_weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16).contiguous()
    norm_k_weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16).contiguous()
    freqs_re = torch.randn(
        (max(4096, args.tokens), args.head_dim // 2),
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    freqs_im = torch.randn_like(freqs_re)

    fused = FlashRTQKVPostProcess(
        repo_id=args.repo_id,
        version=args.version,
        heads=args.heads,
        head_dim=args.head_dim,
    ).cuda()

    q_fused, k_fused = fused(
        packed_qkv, norm_q_weight, norm_k_weight, freqs_re, freqs_im
    )
    q_ref, k_ref = torch_qkv_split_norm_rope(
        packed_qkv,
        norm_q_weight,
        norm_k_weight,
        freqs_re,
        freqs_im,
        heads=args.heads,
        head_dim=args.head_dim,
    )
    torch.testing.assert_close(q_fused.float(), q_ref.float(), atol=0.03125, rtol=0)
    torch.testing.assert_close(k_fused.float(), k_ref.float(), atol=0.03125, rtol=0)

    fused_us = _time_us(
        lambda: fused(packed_qkv, norm_q_weight, norm_k_weight, freqs_re, freqs_im),
        args.warmup,
        args.iters,
    )
    torch_us = _time_us(
        lambda: torch_qkv_split_norm_rope(
            packed_qkv,
            norm_q_weight,
            norm_k_weight,
            freqs_re,
            freqs_im,
            heads=args.heads,
            head_dim=args.head_dim,
        ),
        args.warmup,
        args.iters,
    )
    print(
        f"B=1 T={args.tokens} H={args.heads} D={args.head_dim}: "
        f"flashrt={fused_us:.3f}us torch={torch_us:.3f}us "
        f"speedup={torch_us / fused_us:.2f}x"
    )


if __name__ == "__main__":
    main()
