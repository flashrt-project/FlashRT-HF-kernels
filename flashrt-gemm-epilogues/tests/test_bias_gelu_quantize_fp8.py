import pytest
import torch

import flashrt_gemm_epilogues as flashrt_ops


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not (hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz")),
    reason="CUDA/ROCm with FP8 support is required",
)


def _fp8_dtype():
    if torch.version.hip is not None and hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _fp8_max() -> float:
    return 240.0 if torch.version.hip is not None else 448.0


def _reference(input, bias, scale):
    y = input.float()
    if bias is not None:
        y = y + bias.float()
    y = torch.nn.functional.gelu(y, approximate="tanh")
    y = torch.clamp(y / scale.float(), -_fp8_max(), _fp8_max())
    return y.to(_fp8_dtype())


def _channel_scale_reference(input, channel_scale, scale):
    y = input.float() * channel_scale.float()
    y = torch.clamp(y / scale.float(), -_fp8_max(), _fp8_max())
    return y.to(_fp8_dtype())


@pytest.mark.parametrize("shape", [(4, 16), (2, 3, 32)])
def test_bias_gelu_quantize_fp8_static_bf16(shape):
    torch.manual_seed(0)
    device = torch.device("cuda")
    input = torch.randn(shape, device=device, dtype=torch.bfloat16).contiguous()
    bias = torch.randn((shape[-1],), device=device, dtype=torch.bfloat16).contiguous()
    scale = torch.tensor([0.25], device=device, dtype=torch.float32)

    out = flashrt_ops.bias_gelu_quantize_fp8_static_bf16(input, bias, scale)
    expected = _reference(input, bias, scale)

    torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)


def test_gelu_quantize_fp8_static_bf16_no_bias():
    torch.manual_seed(1)
    device = torch.device("cuda")
    input = torch.randn((8, 64), device=device, dtype=torch.bfloat16).contiguous()
    scale = torch.tensor([0.5], device=device, dtype=torch.float32)

    out = flashrt_ops.gelu_quantize_fp8_static_bf16(input, scale)
    expected = _reference(input, None, scale)

    torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)


def test_out_tensor_is_reused():
    device = torch.device("cuda")
    input = torch.randn((2, 16), device=device, dtype=torch.bfloat16).contiguous()
    bias = torch.randn((16,), device=device, dtype=torch.bfloat16).contiguous()
    scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    out = torch.empty(input.shape, device=device, dtype=_fp8_dtype())

    returned = flashrt_ops.bias_gelu_quantize_fp8_static_bf16(
        input, bias, scale, out=out
    )

    assert returned is out


def test_rejects_wrong_bias_shape():
    device = torch.device("cuda")
    input = torch.randn((2, 16), device=device, dtype=torch.bfloat16).contiguous()
    bias = torch.randn((15,), device=device, dtype=torch.bfloat16).contiguous()
    scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="bias length"):
        flashrt_ops.bias_gelu_quantize_fp8_static_bf16(input, bias, scale)


@pytest.mark.parametrize("shape", [(4, 16), (2, 3, 32)])
def test_channel_scale_quantize_fp8_static_bf16(shape):
    torch.manual_seed(2)
    device = torch.device("cuda")
    input = torch.randn(shape, device=device, dtype=torch.bfloat16).contiguous()
    channel_scale = torch.randn(
        (shape[-1],), device=device, dtype=torch.bfloat16
    ).contiguous()
    scale = torch.tensor([0.25], device=device, dtype=torch.float32)

    out = flashrt_ops.channel_scale_quantize_fp8_static_bf16(
        input, channel_scale, scale
    )
    expected = _channel_scale_reference(input, channel_scale, scale)

    torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)


def test_rejects_wrong_channel_scale_shape():
    device = torch.device("cuda")
    input = torch.randn((2, 16), device=device, dtype=torch.bfloat16).contiguous()
    channel_scale = torch.randn((15,), device=device, dtype=torch.bfloat16)
    scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="channel_scale length"):
        flashrt_ops.channel_scale_quantize_fp8_static_bf16(
            input, channel_scale, scale
        )
