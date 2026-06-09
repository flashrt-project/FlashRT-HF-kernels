import math

import pytest
import torch

import flashrt_gemm_epilogues as flashrt_ops


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


LINEAR_SHAPES = [
    ("pi05_action_in", 10, 32, 1024),
    ("pi05_qkv", 10, 1024, 2560),
    ("pi05_o_proj", 10, 2048, 1024),
    ("pi05_action_out", 10, 1024, 32),
    ("decode_m1_1024", 1, 1024, 1024),
    ("decode_m1_qkv", 1, 1024, 2560),
    ("decode_m8_1024", 8, 1024, 1024),
    ("decode_m8_qkv", 8, 1024, 2560),
    ("decode_m10_1024", 10, 1024, 1024),
    ("decode_m10_qkv", 10, 1024, 2560),
    ("decode_m16_1024", 16, 1024, 1024),
    ("decode_m16_qkv", 16, 1024, 2560),
    ("vlm_m512_square", 512, 1152, 1152),
    ("vlm_m512_wide", 512, 1152, 4304),
    ("vla_m1024_square", 1024, 2048, 2048),
    ("vla_m1024_wide", 1024, 2048, 8192),
]


def _percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def _metrics(out: torch.Tensor, expected: torch.Tensor) -> dict[str, float | str]:
    diff = (out.float() - expected.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(_percentile(diff, 0.99).item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                out.float().flatten(), expected.float().flatten(), dim=0
            ).item()
        ),
        "dtype": str(out.dtype),
    }


def _print_result(
    name: str,
    op: str,
    shape: tuple[int, int, int],
    metrics: dict[str, float | str],
    *,
    p99_limit: float,
    cosine_limit: float,
) -> None:
    passed = metrics["p99_abs"] <= p99_limit and metrics["cosine"] >= cosine_limit
    print(
        "bf16_linear_correctness "
        f"name={name} op={op} shape={shape} dtype={metrics['dtype']} "
        f"max_abs={metrics['max_abs']:.6f} mean_abs={metrics['mean_abs']:.6f} "
        f"p99_abs={metrics['p99_abs']:.6f} cosine={metrics['cosine']:.8f} "
        f"tolerance=p99<={p99_limit},cosine>={cosine_limit} "
        f"verified={'PASS' if passed else 'FAIL'}"
    )


@pytest.mark.parametrize(("name", "m", "k", "n"), LINEAR_SHAPES)
def test_bf16_linear_correctness(name, m, k, n):
    torch.manual_seed(11)
    device = torch.device("cuda")
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16).contiguous()
    w = torch.randn((k, n), device=device, dtype=torch.bfloat16).contiguous()
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)

    returned = flashrt_ops.bf16_linear_bf16(x, w, out=out)
    expected = (x @ w).to(torch.bfloat16)

    assert returned is out
    metrics = _metrics(out, expected)
    p99_limit = 0.5
    cosine_limit = 0.999
    _print_result(name, "linear", (m, k, n), metrics, p99_limit=p99_limit, cosine_limit=cosine_limit)
    assert metrics["p99_abs"] <= p99_limit
    assert metrics["cosine"] >= cosine_limit


@pytest.mark.parametrize(("name", "m", "k", "n"), LINEAR_SHAPES)
def test_bf16_linear_bias_correctness(name, m, k, n):
    torch.manual_seed(17)
    device = torch.device("cuda")
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16).contiguous()
    w = torch.randn((k, n), device=device, dtype=torch.bfloat16).contiguous()
    bias = torch.randn((n,), device=device, dtype=torch.bfloat16).contiguous()
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)

    returned = flashrt_ops.bf16_linear_bias_bf16(x, w, bias, out=out)
    expected = torch.addmm(bias, x, w).to(torch.bfloat16)

    assert returned is out
    metrics = _metrics(out, expected)
    p99_limit = 0.5
    cosine_limit = 0.999
    _print_result(
        name, "linear_bias", (m, k, n), metrics, p99_limit=p99_limit, cosine_limit=cosine_limit
    )
    assert metrics["p99_abs"] <= p99_limit
    assert metrics["cosine"] >= cosine_limit


def test_bf16_linear_rejects_wrong_w_shape():
    device = torch.device("cuda")
    x = torch.randn((10, 32), device=device, dtype=torch.bfloat16).contiguous()
    w = torch.randn((31, 1024), device=device, dtype=torch.bfloat16).contiguous()

    with pytest.raises(RuntimeError, match="x.shape\\[1\\]"):
        flashrt_ops.bf16_linear_bf16(x, w)


def test_bf16_linear_bias_rejects_wrong_bias_shape():
    device = torch.device("cuda")
    x = torch.randn((10, 32), device=device, dtype=torch.bfloat16).contiguous()
    w = torch.randn((32, 1024), device=device, dtype=torch.bfloat16).contiguous()
    bias = torch.randn((1023,), device=device, dtype=torch.bfloat16).contiguous()

    with pytest.raises(RuntimeError, match="bias length"):
        flashrt_ops.bf16_linear_bias_bf16(x, w, bias)


def test_bf16_linear_rejects_noncontiguous_input():
    device = torch.device("cuda")
    x = torch.randn((32, 10), device=device, dtype=torch.bfloat16).t()
    w = torch.randn((32, 1024), device=device, dtype=torch.bfloat16).contiguous()

    with pytest.raises(RuntimeError, match="x must be contiguous"):
        flashrt_ops.bf16_linear_bf16(x, w)
