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


@torch.library.register_fake(
    add_op_namespace_prefix("gelu_mul_merged_quantize_fp8_static_bf16")
)
def _gelu_mul_merged_quantize_fp8_static_bf16_fake(
    gate_up_bf16: torch.Tensor,
    output_scale: torch.Tensor,
    out_fp8: torch.Tensor,
) -> None:
    _silu_mul_merged_quantize_fp8_static_bf16_fake(
        gate_up_bf16, output_scale, out_fp8
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


@torch.library.register_fake(add_op_namespace_prefix("fp8_geglu_mlp_bf16"))
def _fp8_geglu_mlp_bf16_fake(
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
    _fp8_swiglu_mlp_bf16_fake(
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
    return None


def _bf16_fp8_glu_mlp_fake(
    input: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    hidden_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    input_fp8: torch.Tensor,
    gate_up_bf16: torch.Tensor,
    hidden_fp8: torch.Tensor,
    out: torch.Tensor,
) -> None:
    padded_m = input_fp8.shape[0]
    hidden = gate_up_weight.shape[0] // 2
    if input_fp8.shape != (padded_m, input.shape[1]):
        raise RuntimeError("input_fp8 has an invalid padded shape")
    if gate_up_bf16.shape != (padded_m, gate_up_weight.shape[0]):
        raise RuntimeError("gate_up_bf16 has an invalid padded shape")
    if hidden_fp8.shape != (padded_m, hidden):
        raise RuntimeError("hidden_fp8 has an invalid padded shape")
    if out.shape != (padded_m, down_weight.shape[0]):
        raise RuntimeError("out has an invalid padded shape")
    return None


@torch.library.register_fake(add_op_namespace_prefix("bf16_fp8_swiglu_mlp_bf16"))
def _bf16_fp8_swiglu_mlp_bf16_fake(
    input: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    hidden_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    input_fp8: torch.Tensor,
    gate_up_bf16: torch.Tensor,
    hidden_fp8: torch.Tensor,
    out: torch.Tensor,
) -> None:
    return _bf16_fp8_glu_mlp_fake(
        input, gate_up_weight, down_weight, input_scale,
        gate_up_weight_scale, hidden_scale, down_weight_scale, input_fp8,
        gate_up_bf16, hidden_fp8, out,
    )


@torch.library.register_fake(add_op_namespace_prefix("bf16_fp8_geglu_mlp_bf16"))
def _bf16_fp8_geglu_mlp_bf16_fake(
    input: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    hidden_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    input_fp8: torch.Tensor,
    gate_up_bf16: torch.Tensor,
    hidden_fp8: torch.Tensor,
    out: torch.Tensor,
) -> None:
    return _bf16_fp8_glu_mlp_fake(
        input, gate_up_weight, down_weight, input_scale,
        gate_up_weight_scale, hidden_scale, down_weight_scale, input_fp8,
        gate_up_bf16, hidden_fp8, out,
    )


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
            dtype=_fp8_dtype(),
        )
    ops.silu_mul_merged_quantize_fp8_static_bf16(
        gate_up_bf16,
        output_scale,
        out_fp8,
    )
    return out_fp8


def gelu_mul_merged_quantize_fp8_static_bf16(
    gate_up_bf16: torch.Tensor,
    output_scale: torch.Tensor,
    out_fp8: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``GELU_tanh(gate) * up / output_scale`` into FP8 E4M3."""

    if out_fp8 is None:
        out_fp8 = torch.empty(
            (gate_up_bf16.shape[0], gate_up_bf16.shape[1] // 2),
            device=gate_up_bf16.device,
            dtype=_fp8_dtype(),
        )
    ops.gelu_mul_merged_quantize_fp8_static_bf16(
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
            dtype=_fp8_dtype(),
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


def fp8_geglu_mlp_bf16(
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
    """FP8 GeGLU MLP block with BF16 output.

    Computes ``GELU_tanh(gate) * up`` between the two FP8 GEMMs.
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
            dtype=_fp8_dtype(),
        )
    if out is None:
        out = torch.empty(
            (input.shape[0], down_weight.shape[0]),
            device=input.device,
            dtype=torch.bfloat16,
        )
    ops.fp8_geglu_mlp_bf16(
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


def _bf16_fp8_glu_mlp_bf16(
    op,
    input: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    hidden_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    input_fp8: torch.Tensor | None,
    gate_up_bf16: torch.Tensor | None,
    hidden_fp8: torch.Tensor | None,
    out: torch.Tensor | None,
    pad_to: int | None,
) -> torch.Tensor:
    logical_m = input.shape[0]
    padded_m = _midm_padded_rows(input) if pad_to is None else pad_to
    if padded_m < logical_m:
        raise ValueError("pad_to must be >= input.shape[0]")
    hidden = gate_up_weight.shape[0] // 2
    device = input.device
    if input_fp8 is None:
        input_fp8 = torch.empty(
            (padded_m, input.shape[1]), device=device, dtype=_fp8_dtype()
        )
    if gate_up_bf16 is None:
        gate_up_bf16 = torch.empty(
            (padded_m, gate_up_weight.shape[0]),
            device=device,
            dtype=torch.bfloat16,
        )
    if hidden_fp8 is None:
        hidden_fp8 = torch.empty(
            (padded_m, hidden), device=device, dtype=_fp8_dtype()
        )
    if out is None:
        out = torch.empty(
            (padded_m, down_weight.shape[0]), device=device, dtype=torch.bfloat16
        )
    op(
        input,
        gate_up_weight,
        down_weight,
        input_scale,
        gate_up_weight_scale,
        hidden_scale,
        down_weight_scale,
        input_fp8,
        gate_up_bf16,
        hidden_fp8,
        out,
    )
    return out[:logical_m]


def bf16_fp8_swiglu_mlp_bf16(
    input: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    hidden_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    input_fp8: torch.Tensor | None = None,
    gate_up_bf16: torch.Tensor | None = None,
    hidden_fp8: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    pad_to: int | None = None,
) -> torch.Tensor:
    """Run a static-scale FP8 SwiGLU MLP from a BF16 region boundary.

    Supplying all padded scratch tensors makes the call allocation-free and
    CUDA-Graph safe. This is one custom-op boundary, not one CUDA launch.
    """

    return _bf16_fp8_glu_mlp_bf16(
        ops.bf16_fp8_swiglu_mlp_bf16,
        input,
        gate_up_weight,
        down_weight,
        input_scale,
        gate_up_weight_scale,
        hidden_scale,
        down_weight_scale,
        input_fp8,
        gate_up_bf16,
        hidden_fp8,
        out,
        pad_to,
    )


def bf16_fp8_geglu_mlp_bf16(
    input: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    hidden_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    input_fp8: torch.Tensor | None = None,
    gate_up_bf16: torch.Tensor | None = None,
    hidden_fp8: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    pad_to: int | None = None,
) -> torch.Tensor:
    """GeGLU counterpart of :func:`bf16_fp8_swiglu_mlp_bf16`."""

    return _bf16_fp8_glu_mlp_bf16(
        ops.bf16_fp8_geglu_mlp_bf16,
        input,
        gate_up_weight,
        down_weight,
        input_scale,
        gate_up_weight_scale,
        hidden_scale,
        down_weight_scale,
        input_fp8,
        gate_up_bf16,
        hidden_fp8,
        out,
        pad_to,
    )


__all__ = [
    "bf16_fp8_geglu_mlp_bf16",
    "bf16_fp8_swiglu_mlp_bf16",
    "fp8_gemm_bf16",
    "fp8_geglu_mlp_bf16",
    "fp8_swiglu_mlp_bf16",
    "gelu_mul_merged_quantize_fp8_static_bf16",
    "silu_mul_merged_quantize_fp8_static_bf16",
]
