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
    of static per-tensor input and weight scales.
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
