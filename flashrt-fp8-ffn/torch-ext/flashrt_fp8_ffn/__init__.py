"""FlashRT FP8 FFN kernels."""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path
from typing import Optional

import torch


def _torch_bundled_cublaslt() -> Optional[Path]:
    for parent in Path(torch.__file__).resolve().parents:
        candidate = parent / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12"
        if candidate.exists():
            return candidate
    return None


def _preload_cublaslt() -> None:
    bundled = _torch_bundled_cublaslt()
    library = str(bundled) if bundled is not None else (
        ctypes.util.find_library("cublasLt") or "libcublasLt.so"
    )
    try:
        ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass


_preload_cublaslt()

from ._ops import add_op_namespace_prefix, ops


def _fp8_dtype() -> torch.dtype:
    if torch.version.hip is not None and hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


@torch.library.register_fake(add_op_namespace_prefix("fp8_gemm_bf16"))
def _fp8_gemm_bf16_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if input.dim() != 2 or weight.dim() != 2:
        raise RuntimeError("input and weight must be rank-2 tensors")
    if out.shape != (input.shape[0], weight.shape[0]):
        raise RuntimeError("out shape must be (input.shape[0], weight.shape[0])")
    return None


@torch.library.register_fake(add_op_namespace_prefix("fp8_linear_bias_gelu_quant_bf16"))
def _fp8_linear_bias_gelu_quant_bf16_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_scale: torch.Tensor,
    hidden_bf16: torch.Tensor,
    out_fp8: torch.Tensor,
) -> None:
    expected = (input.shape[0], weight.shape[0])
    if hidden_bf16.shape != expected or out_fp8.shape != expected:
        raise RuntimeError(
            "hidden_bf16 and out_fp8 shapes must be "
            "(input.shape[0], weight.shape[0])"
        )
    return None


@torch.library.register_fake(add_op_namespace_prefix("fp8_gelu_mlp_bf16"))
def _fp8_gelu_mlp_bf16_fake(
    input: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    input_scale: torch.Tensor,
    up_weight_scale: torch.Tensor,
    hidden_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    hidden_bf16: torch.Tensor,
    hidden_fp8: torch.Tensor,
    out: torch.Tensor,
) -> None:
    hidden_shape = (input.shape[0], up_weight.shape[0])
    out_shape = (input.shape[0], down_weight.shape[0])
    if hidden_bf16.shape != hidden_shape or hidden_fp8.shape != hidden_shape:
        raise RuntimeError(
            "hidden buffers must be (input.shape[0], up_weight.shape[0])"
        )
    if out.shape != out_shape:
        raise RuntimeError("out shape must be (input.shape[0], down_weight.shape[0])")
    return None


@torch.library.register_fake(add_op_namespace_prefix("bf16_fp8_gelu_mlp_bf16"))
def _bf16_fp8_gelu_mlp_bf16_fake(
    input: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    input_scale: torch.Tensor,
    up_weight_scale: torch.Tensor,
    hidden_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    input_fp8: torch.Tensor,
    hidden_bf16: torch.Tensor,
    hidden_fp8: torch.Tensor,
    out: torch.Tensor,
) -> None:
    padded_m = input_fp8.shape[0]
    if input_fp8.shape != (padded_m, input.shape[1]):
        raise RuntimeError("input_fp8 must have shape (padded_m, input.shape[1])")
    if hidden_bf16.shape != (padded_m, up_weight.shape[0]):
        raise RuntimeError("hidden_bf16 has an invalid padded shape")
    if hidden_fp8.shape != hidden_bf16.shape:
        raise RuntimeError("hidden_fp8 must match hidden_bf16")
    if out.shape != (padded_m, down_weight.shape[0]):
        raise RuntimeError("out has an invalid padded shape")
    return None


def _scalar_scale_like(input: torch.Tensor, value: float = 1.0) -> torch.Tensor:
    return torch.tensor([value], device=input.device, dtype=torch.float32)


def fp8_gemm_bf16(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``(input * input_scale) @ (weight * weight_scale).T``.

    ``input`` is FP8 E4M3 with shape ``(M, K)``. ``weight`` is FP8 E4M3 with
    shape ``(N, K)``. ``input_scale`` and ``weight_scale`` are CUDA float32
    scalar tensors. Output is BF16 with shape ``(M, N)``.
    """

    if out is None:
        out = torch.empty(
            (input.shape[0], weight.shape[0]),
            device=input.device,
            dtype=torch.bfloat16,
        )
    ops.fp8_gemm_bf16(input, weight, input_scale, weight_scale, out)
    return out


def fp8_linear_bias_gelu_quant_bf16(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_scale: torch.Tensor,
    hidden_bf16: torch.Tensor | None = None,
    out_fp8: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 linear + BF16 bias/GELU + FP8 quantized output.

    Returns ``(hidden_bf16, out_fp8)``. ``hidden_bf16`` is the post-GEMM
    pre-activation scratch; ``out_fp8`` is the quantized activation.
    """

    if hidden_bf16 is None:
        hidden_bf16 = torch.empty(
            (input.shape[0], weight.shape[0]),
            device=input.device,
            dtype=torch.bfloat16,
        )
    if out_fp8 is None:
        out_fp8 = torch.empty_like(hidden_bf16, dtype=_fp8_dtype())
    ops.fp8_linear_bias_gelu_quant_bf16(
        input,
        weight,
        bias,
        input_scale,
        weight_scale,
        output_scale,
        hidden_bf16,
        out_fp8,
    )
    return hidden_bf16, out_fp8


def fp8_gelu_mlp_bf16(
    input: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    input_scale: torch.Tensor,
    up_weight_scale: torch.Tensor,
    hidden_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    hidden_bf16: torch.Tensor | None = None,
    hidden_fp8: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP8 GELU MLP block with BF16 output.

    Computes:

    ``hidden = gelu(fp8_gemm(input, up_weight) + up_bias)``
    ``hidden_fp8 = quantize_fp8(hidden, hidden_scale)``
    ``out = fp8_gemm(hidden_fp8, down_weight) + down_bias``
    """

    if hidden_bf16 is None:
        hidden_bf16 = torch.empty(
            (input.shape[0], up_weight.shape[0]),
            device=input.device,
            dtype=torch.bfloat16,
        )
    if hidden_fp8 is None:
        hidden_fp8 = torch.empty_like(hidden_bf16, dtype=_fp8_dtype())
    if out is None:
        out = torch.empty(
            (input.shape[0], down_weight.shape[0]),
            device=input.device,
            dtype=torch.bfloat16,
        )
    ops.fp8_gelu_mlp_bf16(
        input,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        input_scale,
        up_weight_scale,
        hidden_scale,
        down_weight_scale,
        hidden_bf16,
        hidden_fp8,
        out,
    )
    return out


def _midm_padded_rows(input: torch.Tensor) -> int:
    rows = input.shape[0]
    if (
        torch.version.hip is None
        and input.is_cuda
        and torch.cuda.get_device_capability(input.device) == (11, 0)
        and 9 <= rows <= 128
    ):
        return ((rows + 63) // 64) * 64
    return rows


def bf16_fp8_gelu_mlp_bf16(
    input: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    input_scale: torch.Tensor,
    up_weight_scale: torch.Tensor,
    hidden_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    input_fp8: torch.Tensor | None = None,
    hidden_bf16: torch.Tensor | None = None,
    hidden_fp8: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    pad_to: int | None = None,
) -> torch.Tensor:
    """Run a static-scale FP8 GELU MLP from a BF16 region boundary.

    The op quantizes ``input`` and executes both FP8-weight GEMMs in one
    traceable package call. Thor SM110 pads rows 9..128 to a multiple of 64;
    other targets keep the logical M unless ``pad_to`` is supplied. Pass
    preallocated buffers for allocation-free CUDA Graph replay. This is one
    custom-op boundary, not a single CUDA launch.
    """

    logical_m = input.shape[0]
    padded_m = _midm_padded_rows(input) if pad_to is None else pad_to
    if padded_m < logical_m:
        raise ValueError("pad_to must be >= input.shape[0]")
    device = input.device
    if input_fp8 is None:
        input_fp8 = torch.empty(
            (padded_m, input.shape[1]), device=device, dtype=_fp8_dtype()
        )
    if hidden_bf16 is None:
        hidden_bf16 = torch.empty(
            (padded_m, up_weight.shape[0]), device=device, dtype=torch.bfloat16
        )
    if hidden_fp8 is None:
        hidden_fp8 = torch.empty_like(hidden_bf16, dtype=_fp8_dtype())
    if out is None:
        out = torch.empty(
            (padded_m, down_weight.shape[0]), device=device, dtype=torch.bfloat16
        )
    ops.bf16_fp8_gelu_mlp_bf16(
        input,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        input_scale,
        up_weight_scale,
        hidden_scale,
        down_weight_scale,
        input_fp8,
        hidden_bf16,
        hidden_fp8,
        out,
    )
    return out[:logical_m]


__all__ = [
    "bf16_fp8_gelu_mlp_bf16",
    "fp8_gemm_bf16",
    "fp8_gelu_mlp_bf16",
    "fp8_linear_bias_gelu_quant_bf16",
]
