"""FlashRT fused GEMM epilogue kernels."""

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


@torch.library.register_fake(add_op_namespace_prefix("bf16_linear_bf16"))
def _bf16_linear_bf16_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if x.dim() != 2 or w.dim() != 2:
        raise RuntimeError("x and w must be rank-2 tensors")
    if out.shape != (x.shape[0], w.shape[1]):
        raise RuntimeError("out shape must be (x.shape[0], w.shape[1])")
    return None


@torch.library.register_fake(add_op_namespace_prefix("bf16_linear_bias_bf16"))
def _bf16_linear_bias_bf16_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if x.dim() != 2 or w.dim() != 2:
        raise RuntimeError("x and w must be rank-2 tensors")
    if bias.dim() != 1:
        raise RuntimeError("bias must be rank-1")
    if out.shape != (x.shape[0], w.shape[1]):
        raise RuntimeError("out shape must be (x.shape[0], w.shape[1])")
    return None


@torch.library.register_fake(add_op_namespace_prefix("bf16_gemm_bias_gelu"))
def _bf16_gemm_bias_gelu_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if a.dim() != 2 or b.dim() != 2:
        raise RuntimeError("a and b must be rank-2 tensors")
    if out.shape != (a.shape[0], b.shape[1]):
        raise RuntimeError("out shape must be (a.shape[0], b.shape[1])")
    return None


@torch.library.register_fake(add_op_namespace_prefix("bf16_gemm_bias"))
def _bf16_gemm_bias_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if a.dim() != 2 or b.dim() != 2:
        raise RuntimeError("a and b must be rank-2 tensors")
    if out.shape != (a.shape[0], b.shape[1]):
        raise RuntimeError("out shape must be (a.shape[0], b.shape[1])")
    return None


@torch.library.register_fake(add_op_namespace_prefix("bias_gelu_quantize_fp8_static_bf16"))
def _bias_gelu_quantize_fp8_static_bf16_fake(
    input: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if out.shape != input.shape:
        raise RuntimeError("out shape must match input shape")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gelu_quantize_fp8_static_bf16"))
def _gelu_quantize_fp8_static_bf16_fake(
    input: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if out.shape != input.shape:
        raise RuntimeError("out shape must match input shape")
    return None


@torch.library.register_fake(add_op_namespace_prefix("channel_scale_quantize_fp8_static_bf16"))
def _channel_scale_quantize_fp8_static_bf16_fake(
    input: torch.Tensor,
    channel_scale: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if out.shape != input.shape:
        raise RuntimeError("out shape must match input shape")
    return None


def _fp8_dtype() -> torch.dtype:
    if torch.version.hip is not None and hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _allocate_fp8_like(input: torch.Tensor) -> torch.Tensor:
    return torch.empty(input.shape, device=input.device, dtype=_fp8_dtype())


def bf16_linear_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``x @ w`` and store BF16 output.

    ``x`` must be contiguous BF16 with shape ``(M, K)``. ``w`` must be
    contiguous BF16 with shape ``(K, N)``. If ``out`` is omitted, a BF16
    ``(M, N)`` tensor is allocated.
    """

    if out is None:
        out = torch.empty(
            (x.shape[0], w.shape[1]), device=x.device, dtype=torch.bfloat16
        )
    ops.bf16_linear_bf16(x, w, out)
    return out


def bf16_linear_bias_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``x @ w + bias`` and store BF16 output.

    ``x`` must be contiguous BF16 with shape ``(M, K)``. ``w`` must be
    contiguous BF16 with shape ``(K, N)``. ``bias`` must be contiguous BF16
    with shape ``(N,)``. If ``out`` is omitted, a BF16 ``(M, N)`` tensor is
    allocated.
    """

    if out is None:
        out = torch.empty(
            (x.shape[0], w.shape[1]), device=x.device, dtype=torch.bfloat16
        )
    ops.bf16_linear_bias_bf16(x, w, bias, out)
    return out


def bf16_gemm_bias_gelu(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``gelu(a @ b + bias)`` and store BF16 output.

    ``a`` must be contiguous BF16 with shape ``(M, K)``. ``b`` must be
    contiguous BF16 with shape ``(K, N)``. ``bias`` must be contiguous BF16
    with shape ``(N,)``. If ``out`` is omitted, a BF16 ``(M, N)`` tensor is
    allocated.
    """

    if out is None:
        out = torch.empty(
            (a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16
        )
    ops.bf16_gemm_bias_gelu(a, b, bias, out)
    return out


def bf16_gemm_bias(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``a @ b + bias`` and store BF16 output."""

    if out is None:
        out = torch.empty(
            (a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16
        )
    ops.bf16_gemm_bias(a, b, bias, out)
    return out


def bias_gelu_quantize_fp8_static_bf16(
    input: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``gelu(input + bias) / scale`` and store FP8 e4m3 output.

    ``input`` must be contiguous BF16 with shape ``(..., N)``. ``bias`` must be
    contiguous BF16 with shape ``(N,)``. ``scale`` must be a CUDA float32 scalar
    tensor. If ``out`` is omitted, an FP8 e4m3 output tensor is allocated.
    """

    if out is None:
        out = _allocate_fp8_like(input)
    ops.bias_gelu_quantize_fp8_static_bf16(input, bias, scale, out)
    return out


def gelu_quantize_fp8_static_bf16(
    input: torch.Tensor,
    scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``gelu(input) / scale`` and store FP8 e4m3 output."""

    if out is None:
        out = _allocate_fp8_like(input)
    ops.gelu_quantize_fp8_static_bf16(input, scale, out)
    return out


def channel_scale_quantize_fp8_static_bf16(
    input: torch.Tensor,
    channel_scale: torch.Tensor,
    scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``input * channel_scale / scale`` and store FP8 e4m3 output.

    ``channel_scale`` must be contiguous BF16 with shape ``(input.shape[-1],)``.
    It is broadcast over all leading input dimensions.
    """

    if out is None:
        out = _allocate_fp8_like(input)
    ops.channel_scale_quantize_fp8_static_bf16(input, channel_scale, scale, out)
    return out


__all__ = [
    "bf16_linear_bf16",
    "bf16_linear_bias_bf16",
    "bf16_gemm_bias_gelu",
    "bf16_gemm_bias",
    "bias_gelu_quantize_fp8_static_bf16",
    "channel_scale_quantize_fp8_static_bf16",
    "gelu_quantize_fp8_static_bf16",
]
