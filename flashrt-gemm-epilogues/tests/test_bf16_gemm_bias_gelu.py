import pytest
import torch

import flashrt_gemm_epilogues as flashrt_ops


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


def _reference(a, b, bias):
    y = a @ b
    y = y + bias
    return torch.nn.functional.gelu(y).to(torch.bfloat16)


def _bias_reference(a, b, bias):
    return ((a @ b) + bias).to(torch.bfloat16)


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (16, 64, 32),
        (32, 128, 64),
    ],
)
def test_bf16_gemm_bias(m, n, k):
    torch.manual_seed(2)
    device = torch.device("cuda")
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16).contiguous()
    b = torch.randn((k, n), device=device, dtype=torch.bfloat16).contiguous()
    bias = torch.randn((n,), device=device, dtype=torch.bfloat16).contiguous()

    out = flashrt_ops.bf16_gemm_bias(a, b, bias)
    expected = _bias_reference(a, b, bias)

    torch.testing.assert_close(out.float(), expected.float(), rtol=3e-2, atol=1.25e-1)


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (16, 64, 32),
        (32, 128, 64),
    ],
)
def test_bf16_gemm_bias_gelu(m, n, k):
    torch.manual_seed(3)
    device = torch.device("cuda")
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16).contiguous()
    b = torch.randn((k, n), device=device, dtype=torch.bfloat16).contiguous()
    bias = torch.randn((n,), device=device, dtype=torch.bfloat16).contiguous()

    out = flashrt_ops.bf16_gemm_bias_gelu(a, b, bias)
    expected = _reference(a, b, bias)

    torch.testing.assert_close(out.float(), expected.float(), rtol=3e-2, atol=1.25e-1)


def test_bf16_gemm_bias_gelu_out_tensor_is_reused():
    device = torch.device("cuda")
    a = torch.randn((16, 32), device=device, dtype=torch.bfloat16).contiguous()
    b = torch.randn((32, 64), device=device, dtype=torch.bfloat16).contiguous()
    bias = torch.randn((64,), device=device, dtype=torch.bfloat16).contiguous()
    out = torch.empty((16, 64), device=device, dtype=torch.bfloat16)

    returned = flashrt_ops.bf16_gemm_bias_gelu(a, b, bias, out=out)

    assert returned is out


def test_bf16_gemm_bias_gelu_rejects_wrong_b_shape():
    device = torch.device("cuda")
    a = torch.randn((16, 32), device=device, dtype=torch.bfloat16).contiguous()
    b = torch.randn((31, 64), device=device, dtype=torch.bfloat16).contiguous()
    bias = torch.randn((64,), device=device, dtype=torch.bfloat16).contiguous()

    with pytest.raises(RuntimeError, match="a.shape\\[1\\]"):
        flashrt_ops.bf16_gemm_bias_gelu(a, b, bias)


def test_bf16_gemm_bias_gelu_rejects_wrong_bias_shape():
    device = torch.device("cuda")
    a = torch.randn((16, 32), device=device, dtype=torch.bfloat16).contiguous()
    b = torch.randn((32, 64), device=device, dtype=torch.bfloat16).contiguous()
    bias = torch.randn((63,), device=device, dtype=torch.bfloat16).contiguous()

    with pytest.raises(RuntimeError, match="bias length"):
        flashrt_ops.bf16_gemm_bias_gelu(a, b, bias)
