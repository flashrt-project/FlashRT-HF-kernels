"""FlashRT FP8 SwiGLU FFN kernels."""

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


@torch.library.register_fake(
    add_op_namespace_prefix("silu_mul_merged_quantize_fp8_static_bf16")
)
def _silu_mul_merged_quantize_fp8_static_bf16_fake(
    gate_up_bf16: torch.Tensor,
    output_scale: torch.Tensor,
    out_fp8: torch.Tensor,
) -> None:
    if gate_up_bf16.dim() != 2:
        raise RuntimeError("gate_up_bf16 must be rank-2")
    if gate_up_bf16.shape[1] % 2 != 0:
        raise RuntimeError("gate_up_bf16.shape[1] must be even")
    if out_fp8.shape != (gate_up_bf16.shape[0], gate_up_bf16.shape[1] // 2):
        raise RuntimeError(
            "out_fp8 must have shape "
            "(gate_up_bf16.shape[0], gate_up_bf16.shape[1] / 2)"
        )
    return None


@torch.library.register_fake(add_op_namespace_prefix("fp8_swiglu_mlp_bf16"))
def _fp8_swiglu_mlp_bf16_fake(
    input: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    hidden_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    gate_up_bf16: torch.Tensor,
    hidden_fp8: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if gate_up_weight.shape[0] % 2 != 0:
        raise RuntimeError("gate_up_weight.shape[0] must be even")
    hidden = gate_up_weight.shape[0] // 2
    if gate_up_bf16.shape != (input.shape[0], gate_up_weight.shape[0]):
        raise RuntimeError(
            "gate_up_bf16 must have shape "
            "(input.shape[0], gate_up_weight.shape[0])"
        )
    if hidden_fp8.shape != (input.shape[0], hidden):
        raise RuntimeError(
            "hidden_fp8 must have shape "
            "(input.shape[0], gate_up_weight.shape[0] / 2)"
        )
    if out.shape != (input.shape[0], down_weight.shape[0]):
        raise RuntimeError("out shape must be (input.shape[0], down_weight.shape[0])")
    return None


def fp8_gemm_bf16(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``(input * input_scale) @ (weight * weight_scale).T``.

    ``input`` is FP8 E4M3 with shape ``(M, K)``. ``weight`` is FP8 E4M3 with
    shape ``(N, K)``. Output is BF16 with shape ``(M, N)``.
    """

    if out is None:
        out = torch.empty(
            (input.shape[0], weight.shape[0]),
            device=input.device,
            dtype=torch.bfloat16,
        )
    ops.fp8_gemm_bf16(input, weight, input_scale, weight_scale, out)
    return out


def silu_mul_merged_quantize_fp8_static_bf16(
    gate_up_bf16: torch.Tensor,
    output_scale: torch.Tensor,
    out_fp8: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``SiLU(gate) * up / output_scale`` into FP8 E4M3.

    ``gate_up_bf16`` has shape ``(M, 2 * H)`` and row layout
    ``[gate | up]``. Output has shape ``(M, H)``.
    """

    if out_fp8 is None:
        out_fp8 = torch.empty(
            (gate_up_bf16.shape[0], gate_up_bf16.shape[1] // 2),
            device=gate_up_bf16.device,
            dtype=torch.float8_e4m3fn,
        )
    ops.silu_mul_merged_quantize_fp8_static_bf16(
        gate_up_bf16,
        output_scale,
        out_fp8,
    )
    return out_fp8


def fp8_swiglu_mlp_bf16(
    input: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    hidden_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    gate_up_bf16: torch.Tensor | None = None,
    hidden_fp8: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP8 SwiGLU MLP block with BF16 output.

    Computes:

    ``gate_up = fp8_gemm(input, gate_up_weight)``
    ``hidden_fp8 = quantize_fp8(SiLU(gate) * up, hidden_scale)``
    ``out = fp8_gemm(hidden_fp8, down_weight)``
    """

    hidden = gate_up_weight.shape[0] // 2
    if gate_up_bf16 is None:
        gate_up_bf16 = torch.empty(
            (input.shape[0], gate_up_weight.shape[0]),
            device=input.device,
            dtype=torch.bfloat16,
        )
    if hidden_fp8 is None:
        hidden_fp8 = torch.empty(
            (input.shape[0], hidden),
            device=input.device,
            dtype=torch.float8_e4m3fn,
        )
    if out is None:
        out = torch.empty(
            (input.shape[0], down_weight.shape[0]),
            device=input.device,
            dtype=torch.bfloat16,
        )
    ops.fp8_swiglu_mlp_bf16(
        input,
        gate_up_weight,
        down_weight,
        input_scale,
        gate_up_weight_scale,
        hidden_scale,
        down_weight_scale,
        gate_up_bf16,
        hidden_fp8,
        out,
    )
    return out


__all__ = [
    "fp8_gemm_bf16",
    "fp8_swiglu_mlp_bf16",
    "silu_mul_merged_quantize_fp8_static_bf16",
]
