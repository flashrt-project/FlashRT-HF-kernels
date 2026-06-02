import pytest
import torch

from flashrt_vla_video import (
    k_norm_rope_v_cache_bf16,
    q_norm_rope_bf16,
    qkv_split_norm_rope_bf16,
)


def _reference_norm_rope(x, weight, cos, sin, eps=1e-6):
    half = x.shape[-1] // 2
    rstd = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    normed = x.float() * rstd * weight.float()
    lo = normed[..., :half]
    hi = normed[..., half:]
    out_lo = lo * cos.float() - hi * sin.float()
    out_hi = hi * cos.float() + lo * sin.float()
    return torch.cat([out_lo, out_hi], dim=-1).to(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(1, 128), (8, 128), (2, 4, 128)])
def test_q_norm_rope_bf16(shape):
    torch.manual_seed(0)
    q = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    weight = (torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
    cos = torch.randn(64, device="cuda", dtype=torch.bfloat16).contiguous()
    sin = torch.randn(64, device="cuda", dtype=torch.bfloat16).contiguous()

    out = q_norm_rope_bf16(q, weight, cos, sin)
    ref = _reference_norm_rope(q, weight, cos, sin)

    torch.testing.assert_close(out.float(), ref.float(), atol=0.03125, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(1, 128), (8, 128), (2, 4, 128)])
def test_k_norm_rope_v_cache_bf16(shape):
    torch.manual_seed(1)
    k = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    weight = (torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
    cos = torch.randn(64, device="cuda", dtype=torch.bfloat16).contiguous()
    sin = torch.randn(64, device="cuda", dtype=torch.bfloat16).contiguous()

    k_out, v_out = k_norm_rope_v_cache_bf16(k, v, weight, cos, sin)
    k_ref = _reference_norm_rope(k, weight, cos, sin)

    torch.testing.assert_close(k_out.float(), k_ref.float(), atol=0.03125, rtol=0)
    torch.testing.assert_close(v_out, v)


def _reference_qkv_split_norm_rope(
    packed_qkv, norm_q_weight, norm_k_weight, freqs_re, freqs_im, heads, head_dim, eps=1e-6
):
    batch, tokens, _ = packed_qkv.shape
    dim = heads * head_dim
    q = packed_qkv[..., :dim].reshape(batch, tokens, heads, head_dim)
    k = packed_qkv[..., dim : 2 * dim].reshape(batch, tokens, heads, head_dim)
    qf = q.float()
    kf = k.float()
    qn = qf * torch.rsqrt((qf * qf).mean(dim=(-2, -1), keepdim=True) + eps)
    kn = kf * torch.rsqrt((kf * kf).mean(dim=(-2, -1), keepdim=True) + eps)
    qn = qn * norm_q_weight.reshape(1, 1, heads, head_dim).float()
    kn = kn * norm_k_weight.reshape(1, 1, heads, head_dim).float()

    def rope(x):
        xr = x[..., 0::2].float()
        xi = x[..., 1::2].float()
        fr = freqs_re[:tokens][None, :, None, :]
        fi = freqs_im[:tokens][None, :, None, :]
        out = torch.empty_like(x, dtype=torch.float32)
        out[..., 0::2] = xr * fr - xi * fi
        out[..., 1::2] = xr * fi + xi * fr
        out = out.to(torch.bfloat16)
        return out

    return rope(qn), rope(kn)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tokens", [4, 64])
def test_qkv_split_norm_rope_bf16(tokens):
    torch.manual_seed(2)
    heads = 24
    head_dim = 128
    dim = heads * head_dim
    packed_qkv = (
        torch.randn((1, tokens, 3 * dim), device="cuda", dtype=torch.bfloat16) * 0.2
    ).contiguous()
    norm_q_weight = (
        torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.1 + 1
    ).contiguous()
    norm_k_weight = (
        torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.1 + 1
    ).contiguous()
    freqs_re = torch.randn((128, head_dim // 2), device="cuda", dtype=torch.float32).contiguous()
    freqs_im = torch.randn((128, head_dim // 2), device="cuda", dtype=torch.float32).contiguous()

    q_out, k_out = qkv_split_norm_rope_bf16(
        packed_qkv,
        norm_q_weight,
        norm_k_weight,
        freqs_re,
        freqs_im,
        heads=heads,
        head_dim=head_dim,
        seq_len=tokens,
    )
    q_ref, k_ref = _reference_qkv_split_norm_rope(
        packed_qkv, norm_q_weight, norm_k_weight, freqs_re, freqs_im, heads, head_dim
    )

    torch.testing.assert_close(q_out.float(), q_ref.float(), atol=0.03125, rtol=0)
    torch.testing.assert_close(k_out.float(), k_ref.float(), atol=0.03125, rtol=0)
