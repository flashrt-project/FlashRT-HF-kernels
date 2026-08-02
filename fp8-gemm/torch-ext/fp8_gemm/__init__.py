"""FlashRT FP8 GEMM kernels."""

from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix, ops


@torch.library.register_fake(add_op_namespace_prefix("fp8_linear_bf16"))
def _fp8_linear_bf16_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    alpha: float,
    variant: int,
    out: torch.Tensor,
) -> None:
    if input.dim() != 2 or weight.dim() != 2:
        raise RuntimeError("input and weight must be rank-2 tensors")
    if out.shape != (input.shape[0], weight.shape[0]):
        raise RuntimeError("out must have shape (input.shape[0], weight.shape[0])")
    return None


@torch.library.register_fake(add_op_namespace_prefix("fp8_linear_residual_bf16"))
def _fp8_linear_residual_bf16_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    alpha: float,
    variant: int,
    residual: torch.Tensor,
) -> None:
    if input.shape[0] != 1:
        raise RuntimeError("residual path supports only M=1")
    if residual.shape != (1, weight.shape[0]):
        raise RuntimeError("residual must have shape (1, weight.shape[0])")
    return None


@torch.library.register_fake(add_op_namespace_prefix("fp8_blockwise_linear_bf16"))
def _fp8_blockwise_linear_bf16_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if input.dim() != 2 or weight.dim() != 2:
        raise RuntimeError("input and weight must be rank-2 tensors")
    m, k = input.shape
    n = weight.shape[0]
    if weight.shape[1] != k or n % 128 or k % 128:
        raise RuntimeError("weight shape is invalid or N/K are not divisible by 128")
    if input_scale.shape != (m, k // 128):
        raise RuntimeError("input_scale must have shape (M, K / 128)")
    if weight_scale.shape != (n // 128, k // 128):
        raise RuntimeError("weight_scale must have shape (N / 128, K / 128)")
    if out.shape != (m, n):
        raise RuntimeError("out must have shape (M, N)")
    return None


@torch.library.register_fake(
    add_op_namespace_prefix("fp8_blockwise_swiglu_quantize_fp8")
)
def _fp8_blockwise_swiglu_quantize_fp8_fake(
    input: torch.Tensor,
    gate_up_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
) -> None:
    m, k = input.shape
    if gate_up_weight.dim() != 2 or gate_up_weight.shape[0] % 2:
        raise RuntimeError("gate_up_weight must have shape (2*N, K)")
    n = gate_up_weight.shape[0] // 2
    if gate_up_weight.shape[1] != k or n % 128 or k % 128:
        raise RuntimeError("gate_up_weight shape is invalid or N/K are not divisible by 128")
    if input_scale.shape != (m, k // 128):
        raise RuntimeError("input_scale must have shape (M, K/128)")
    if gate_up_weight_scale.shape != (2 * n // 128, k // 128):
        raise RuntimeError("gate_up_weight_scale must have shape (2*N/128, K/128)")
    if output.shape != (m, n) or output_scale.shape != (m, n // 128):
        raise RuntimeError("output buffers have invalid shapes")
    return None


def select_fp8_linear_tile(m: int, n: int, k: int, variant: int = 0) -> str:
    """Return the FlashRT tile selected by the public dispatcher."""

    m = int(m)
    n = int(n)
    k = int(k)
    variant = int(variant)
    if m <= 0 or n <= 0 or k <= 0:
        raise RuntimeError("m, n, and k must be positive")
    if k % 32 != 0:
        raise RuntimeError("k must be divisible by 32")
    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    if capability == (11, 0):
        forced = {1: "sm110_sq_bf16", 2: "sm110_t1_bf16", 3: "sm110_wide_bf16"}
        if variant not in {0, *forced}:
            raise RuntimeError("SM110 variant must be 0 (auto), 1 (Sq), 2 (T1), or 3 (Wide)")
        if n % 16 or k % 16:
            raise RuntimeError("SM110 requires n and k divisible by 16")
        if variant:
            return forced[variant]
        if n >= 8 * k:
            return "sm110_wide_bf16"
        if m >= 128 and k >= 4 * n:
            return "sm110_sq_bf16"
        if n == k and m >= 512:
            return "sm110_sq_bf16" if k <= 1024 else "sm110_wide_bf16"
        if n == k and m >= 128:
            return "sm110_wide_bf16"
        return "sm110_t1_bf16"
    if m == 1:
        if variant == 4:
            return "gemv_fp8_m1_w4"
        if variant == 8:
            return "gemv_fp8_m1_w8"
        if variant == 16:
            return "gemv_fp8_m1_w16"
        if variant != 0:
            raise RuntimeError("M=1 variant must be 0, 4, 8, or 16")
        if n <= 2048:
            return "gemv_fp8_m1_w4"
        if n <= 8192:
            return "gemv_fp8_m1_w8"
        return "gemv_fp8_m1_w16"
    if variant != 0:
        raise RuntimeError("small-M dispatcher currently supports variant=0 only")
    if m <= 16:
        if k % 256 == 0:
            return "ld_fp8_gemm_16x128x256_w4" if n % 128 == 0 else "ld_fp8_gemm_16x64x256_w4"
        if n % 256 == 0:
            return "ld_fp8_gemm_16x256x128_w8"
        if n % 192 == 0:
            return "ld_fp8_gemm_16x192x128_w4"
        if n % 128 == 0:
            return "ld_fp8_gemm_16x128x128_w4"
        return "ld_fp8_gemm_16x64x128_w4"
    if m <= 32:
        if k % 256 == 0:
            return "ld_fp8_gemm_32x128x256_w4" if n % 128 == 0 else "ld_fp8_gemm_32x64x256_w4"
        if n % 192 == 0:
            return "ld_fp8_gemm_32x192x128_w4"
        if n % 128 == 0:
            return "ld_fp8_gemm_32x128x128_w4"
        return "ld_fp8_gemm_32x64x128_w4"
    if m <= 64:
        if k % 256 == 0:
            return "ld_fp8_gemm_64x128x256_w4" if n % 128 == 0 else "ld_fp8_gemm_64x64x256_w4"
        if n % 128 == 0:
            return "ld_fp8_gemm_64x128x128_w4"
        return "ld_fp8_gemm_64x64x128_w4"
    raise RuntimeError("only M=1 decode or 2 <= M <= 64 small-M rows are supported")


def fp8_linear_bf16(
    input: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 1.0,
    out: torch.Tensor | None = None,
    variant: int = 0,
) -> torch.Tensor:
    """Compute ``(input @ weight.T) * alpha`` with BF16 output.

    ``input`` and ``weight`` must be FP8 E4M3 CUDA tensors with shapes
    ``(M, K)`` and ``(N, K)``. ``alpha`` is a host float, normally the product
    of static per-tensor input and weight scales. SM110 uses the production
    CUTLASS Sq/T1/Wide dispatcher over full model row counts; SM120 uses the
    hand-tuned M<=64 path.
    """

    if out is None:
        out = torch.empty(
            (input.shape[0], weight.shape[0]),
            device=input.device,
            dtype=torch.bfloat16,
        )
    ops.fp8_linear_bf16(input, weight, float(alpha), int(variant), out)
    return out


def fp8_linear_residual_bf16(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    alpha: float = 1.0,
    variant: int = 0,
) -> torch.Tensor:
    """In-place ``residual += (input @ weight.T) * alpha`` for M=1 decode."""

    ops.fp8_linear_residual_bf16(input, weight, float(alpha), int(variant), residual)
    return residual


def fp8_blockwise_linear_bf16(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Block-128 scaled FP8 linear with BF16 output on SM89/SM120."""

    if out is None:
        out = torch.empty(
            (input.shape[0], weight.shape[0]),
            device=input.device,
            dtype=torch.bfloat16,
        )
    ops.fp8_blockwise_linear_bf16(
        input, weight, input_scale, weight_scale, out
    )
    return out


def fp8_blockwise_swiglu_quantize_fp8(
    input: torch.Tensor,
    gate_up_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SM89 block-128 FP8 gate/up GEMM + SiLU + FP8 requant producer."""

    n = gate_up_weight.shape[0] // 2
    if output is None:
        output = torch.empty(
            (input.shape[0], n), device=input.device, dtype=torch.float8_e4m3fn
        )
    if output_scale is None:
        output_scale = torch.empty(
            (input.shape[0], n // 128), device=input.device, dtype=torch.float32
        )
    ops.fp8_blockwise_swiglu_quantize_fp8(
        input, gate_up_weight, input_scale, gate_up_weight_scale,
        output, output_scale
    )
    return output, output_scale


__all__ = [
    "fp8_linear_bf16",
    "fp8_linear_residual_bf16",
    "fp8_blockwise_linear_bf16",
    "fp8_blockwise_swiglu_quantize_fp8",
    "select_fp8_linear_tile",
]
