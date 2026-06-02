import pytest
import torch

from flashrt_vla_video import k_norm_rope_v_cache_bf16, q_norm_rope_bf16


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

    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


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

    torch.testing.assert_close(k_out.float(), k_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v_out, v)
