#!/usr/bin/env python3
"""Replace a PyTorch GELU FFN with FlashRT Hub kernels.

This is a small integration skeleton for modules shaped like:

    Linear -> GELU(tanh) -> Linear

It keeps static scales and FP8 weights as buffers. The forward path starts from
BF16 activations for readability; a production FP8 model should pass FP8
activations directly between fused blocks.
"""

from __future__ import annotations

import torch
from kernels import get_kernel


def tensor_scale(x: torch.Tensor, *, floor: float = 1e-6, safety: float = 1.05) -> torch.Tensor:
    amax = x.detach().float().abs().max()
    return torch.clamp((amax / 448.0) * safety, min=floor).reshape(1).to(
        device=x.device,
        dtype=torch.float32,
    )


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float(), -448.0, 448.0).to(
        torch.float8_e4m3fn
    )


class TorchGeluFFN(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.up = torch.nn.Linear(hidden_size, intermediate_size)
        self.down = torch.nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        return self.down(x)


class FlashRTStaticFP8GeluFFN(torch.nn.Module):
    def __init__(
        self,
        torch_ffn: TorchGeluFFN,
        *,
        input_scale: torch.Tensor,
        hidden_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.fp8_ops = get_kernel(
            "flashrt/flashrt-fp8-ffn",
            version=1,
            trust_remote_code=True,
        )
        self.quant_ops = get_kernel(
            "flashrt/flashrt-gemm-epilogues",
            version=1,
            trust_remote_code=True,
        )

        up_w = torch_ffn.up.weight.detach().contiguous().to(torch.bfloat16)
        down_w = torch_ffn.down.weight.detach().contiguous().to(torch.bfloat16)
        up_w_scale = tensor_scale(up_w)
        down_w_scale = tensor_scale(down_w)

        self.register_buffer("up_w_fp8", quantize_fp8(up_w, up_w_scale).contiguous())
        self.register_buffer("down_w_fp8", quantize_fp8(down_w, down_w_scale).contiguous())
        self.register_buffer("up_w_scale", up_w_scale.contiguous())
        self.register_buffer("down_w_scale", down_w_scale.contiguous())
        self.register_buffer("input_scale", input_scale.detach().reshape(1).to(torch.float32))
        self.register_buffer("hidden_scale", hidden_scale.detach().reshape(1).to(torch.float32))
        self.register_buffer("up_bias", torch_ffn.up.bias.detach().contiguous().to(torch.bfloat16))
        self.register_buffer("down_bias", torch_ffn.down.bias.detach().contiguous().to(torch.bfloat16))
        self.register_buffer(
            "channel_scale",
            torch.ones((torch_ffn.up.in_features,), device=up_w.device, dtype=torch.bfloat16),
        )

        self._scratch: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}

    def _scratch_buffers(self, rows: int, in_features: int) -> tuple[torch.Tensor, ...]:
        key = (rows, in_features)
        cached = self._scratch.get(key)
        if cached is not None:
            return cached

        device = self.up_w_fp8.device
        intermediate = self.up_w_fp8.shape[0]
        out_features = self.down_w_fp8.shape[0]
        cached = (
            torch.empty((rows, in_features), device=device, dtype=torch.float8_e4m3fn),
            torch.empty((rows, intermediate), device=device, dtype=torch.bfloat16),
            torch.empty((rows, intermediate), device=device, dtype=torch.float8_e4m3fn),
            torch.empty((rows, out_features), device=device, dtype=torch.bfloat16),
        )
        self._scratch[key] = cached
        return cached

    def forward(self, x_bf16: torch.Tensor) -> torch.Tensor:
        shape = x_bf16.shape
        x = x_bf16.reshape(-1, shape[-1]).contiguous().to(torch.bfloat16)
        x_fp8, hidden_bf16, hidden_fp8, out = self._scratch_buffers(
            x.shape[0],
            x.shape[1],
        )

        self.quant_ops.channel_scale_quantize_fp8_static_bf16(
            x,
            self.channel_scale,
            self.input_scale,
            x_fp8,
        )
        self.fp8_ops.fp8_gelu_mlp_bf16(
            x_fp8,
            self.up_w_fp8,
            self.up_bias,
            self.down_w_fp8,
            self.down_bias,
            self.input_scale,
            self.up_w_scale,
            self.hidden_scale,
            self.down_w_scale,
            hidden_bf16,
            hidden_fp8,
            out,
        )
        return out.view(*shape[:-1], self.down_w_fp8.shape[0])


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(0)
    hidden_size = 1024
    intermediate_size = 4096
    rows = 512

    torch_ffn = TorchGeluFFN(hidden_size, intermediate_size).cuda().to(torch.bfloat16).eval()
    calibration_x = torch.randn((rows, hidden_size), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        hidden = torch.nn.functional.gelu(torch_ffn.up(calibration_x), approximate="tanh")

    flashrt_ffn = FlashRTStaticFP8GeluFFN(
        torch_ffn,
        input_scale=tensor_scale(calibration_x),
        hidden_scale=tensor_scale(hidden),
    ).cuda().eval()

    x = torch.randn((rows, hidden_size), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        y_ref = torch_ffn(x)
        y = flashrt_ffn(x)
    torch.cuda.synchronize()

    cos = torch.nn.functional.cosine_similarity(
        y_ref.float().flatten(),
        y.float().flatten(),
        dim=0,
    )
    max_abs = (y_ref.float() - y.float()).abs().max()
    print(f"output shape={tuple(y.shape)} dtype={y.dtype}")
    print(f"cos={float(cos):.6f} max_abs={float(max_abs):.6f}")


if __name__ == "__main__":
    main()
