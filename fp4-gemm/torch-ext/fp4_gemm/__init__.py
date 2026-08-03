"""FlashRT FP4 GEMM kernels."""

from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix, ops


def sfa_size_bytes(rows: int, dim: int) -> int:
    if rows <= 0 or dim <= 0 or dim % 16 != 0:
        raise ValueError("rows must be positive and dim must be positive/divisible by 16")
    n_blocks = dim // 16
    n_row_super = (rows + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    return n_row_super * n_col_super * 512


def _alloc_fp4(rows: int, dim: int, device: torch.device | str):
    return (
        torch.empty((rows, dim // 2), device=device, dtype=torch.uint8),
        torch.empty((sfa_size_bytes(rows, dim),), device=device, dtype=torch.uint8),
    )


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_bf16"))
def _linear_fake(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    out: torch.Tensor,
    alpha: float = 1.0,
    variant: int = -1,
) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("fp4_w4a4_gemv_warpsplit_bf16"))
def _gemv_warpsplit_fake(a_packed, b_packed, sfa, sfb, out, alpha: float = 1.0, warps: int = 4, stages: int = 4) -> None:
    if a_packed.shape[0] != 1:
        raise RuntimeError("warp-split GEMV serves M=1 only")
    if out.shape != (1, b_packed.shape[0]):
        raise RuntimeError("out must have shape (1, N)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("fp4_w4a16_linear_bf16"))
def _legacy_linear_fake(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    out: torch.Tensor,
    alpha: float = 1.0,
    variant: int = -1,
) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_fp4_sfa_fp16"))
def _quant_fake(x: torch.Tensor, packed: torch.Tensor, sfa: torch.Tensor, is_sfb: bool = False) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_fp4_sfa_bf16"))
def _quant_bf16_fake(x: torch.Tensor, packed: torch.Tensor, sfa: torch.Tensor, is_sfb: bool = False) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("dequantize_fp4_sfa_fp16"))
def _dequant_fake(packed: torch.Tensor, sfa: torch.Tensor, out: torch.Tensor, is_sfb: bool = False) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_residual_bf16"))
def _residual_fake(a, b, sfa, sfb, residual, out, alpha: float = 1.0) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_bias_gelu_bf16"))
def _bias_gelu_fake(a, b, sfa, sfb, bias, out, alpha: float = 1.0) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_bias_gelu_nvfp4"))
def _bias_gelu_nvfp4_fake(
    a, b, sfa, sfb, bias, out_packed, out_sfa, alpha: float = 1.0
) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_streamk_bf16"))
def _streamk_fake(a, b, sfa, sfb, out, alpha: float = 1.0) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_streamk_bias_bf16"))
def _streamk_bias_fake(a, b, sfa, sfb, bias, out, alpha: float = 1.0) -> None:
    return None


def quantize_fp4_sfa_fp16(
    x: torch.Tensor,
    packed: torch.Tensor | None = None,
    sfa: torch.Tensor | None = None,
    is_sfb: bool = False,
):
    if packed is None or sfa is None:
        packed, sfa = _alloc_fp4(x.shape[0], x.shape[1], x.device)
    ops.quantize_fp4_sfa_fp16(x, packed, sfa, bool(is_sfb))
    return packed, sfa


def quantize_fp4_sfa_bf16(
    x: torch.Tensor,
    packed: torch.Tensor | None = None,
    sfa: torch.Tensor | None = None,
    is_sfb: bool = False,
):
    """Quantize BF16 directly to packed E2M1 and CUTLASS SFA/SFB."""
    if packed is None or sfa is None:
        packed, sfa = _alloc_fp4(x.shape[0], x.shape[1], x.device)
    ops.quantize_fp4_sfa_bf16(x, packed, sfa, bool(is_sfb))
    return packed, sfa


def dequantize_fp4_sfa_fp16(
    packed: torch.Tensor,
    sfa: torch.Tensor,
    out: torch.Tensor | None = None,
    is_sfb: bool = False,
) -> torch.Tensor:
    if out is None:
        out = torch.empty((packed.shape[0], packed.shape[1] * 2), device=packed.device, dtype=torch.float16)
    ops.dequantize_fp4_sfa_fp16(packed, sfa, out, bool(is_sfb))
    return out


def nvfp4_gemm_bf16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    alpha: float = 1.0,
    out: torch.Tensor | None = None,
    variant: int = -1,
) -> torch.Tensor:
    if out is None:
        out = torch.empty((a_packed.shape[0], b_packed.shape[0]), device=a_packed.device, dtype=torch.bfloat16)
    ops.nvfp4_gemm_bf16(a_packed, b_packed, sfa, sfb, out, float(alpha), int(variant))
    return out


def fp4_w4a4_gemv_warpsplit_bf16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    *,
    alpha: float = 1.0,
    warps: int = 4,
    stages: int = 4,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Warp-split-K NVFP4 W4A4 GEMV for the M=1 decode row (SM120).

    Splits K across warps inside one block with a shared-memory reduce -
    no cross-block intermediate, so it stays safe under CUDA-graph
    replay - and fills the SMs the tiled GEMM underfills at long-K
    small-M decode shapes. Same packed/scale layouts as the linear
    entry points."""
    if out is None:
        out = torch.empty((1, b_packed.shape[0]), device=a_packed.device, dtype=torch.bfloat16)
    ops.fp4_w4a4_gemv_warpsplit_bf16(a_packed, b_packed, sfa, sfb, out, float(alpha), int(warps), int(stages))
    return out


def fp4_w4a16_linear_bf16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    alpha: float = 1.0,
    out: torch.Tensor | None = None,
    variant: int = -1,
) -> torch.Tensor:
    """Compatibility alias for :func:`nvfp4_gemm_bf16`."""
    return nvfp4_gemm_bf16(
        a_packed, b_packed, sfa, sfb, alpha=alpha, out=out, variant=variant
    )


def nvfp4_gemm_residual_bf16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    residual: torch.Tensor,
    alpha: float = 1.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(residual)
    ops.nvfp4_gemm_residual_bf16(
        a_packed, b_packed, sfa, sfb, residual, out, float(alpha)
    )
    return out


def nvfp4_gemm_bias_gelu_bf16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    bias: torch.Tensor,
    alpha: float = 1.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            (a_packed.shape[0], b_packed.shape[0]),
            device=a_packed.device,
            dtype=torch.bfloat16,
        )
    ops.nvfp4_gemm_bias_gelu_bf16(
        a_packed, b_packed, sfa, sfb, bias, out, float(alpha)
    )
    return out


def nvfp4_gemm_bias_gelu_nvfp4(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    bias: torch.Tensor,
    alpha: float = 1.0,
    out_packed: torch.Tensor | None = None,
    out_sfa: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = a_packed.shape[0], b_packed.shape[0]
    if out_packed is None:
        out_packed = torch.empty((m, n // 2), device=a_packed.device, dtype=torch.uint8)
    if out_sfa is None:
        out_sfa = torch.empty((sfa_size_bytes(m, n),), device=a_packed.device, dtype=torch.uint8)
    ops.nvfp4_gemm_bias_gelu_nvfp4(
        a_packed, b_packed, sfa, sfb, bias, out_packed, out_sfa, float(alpha)
    )
    return out_packed, out_sfa


def nvfp4_gemm_streamk_bf16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    alpha: float = 1.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            (a_packed.shape[0], b_packed.shape[0]),
            device=a_packed.device,
            dtype=torch.bfloat16,
        )
    ops.nvfp4_gemm_streamk_bf16(
        a_packed, b_packed, sfa, sfb, out, float(alpha)
    )
    return out


def nvfp4_gemm_streamk_bias_bf16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    bias: torch.Tensor,
    alpha: float = 1.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            (a_packed.shape[0], b_packed.shape[0]),
            device=a_packed.device,
            dtype=torch.bfloat16,
        )
    ops.nvfp4_gemm_streamk_bias_bf16(
        a_packed, b_packed, sfa, sfb, bias, out, float(alpha)
    )
    return out


__all__ = [
    "dequantize_fp4_sfa_fp16",
    "fp4_w4a16_linear_bf16",
    "fp4_w4a4_gemv_warpsplit_bf16",
    "nvfp4_gemm_bf16",
    "nvfp4_gemm_bias_gelu_bf16",
    "nvfp4_gemm_bias_gelu_nvfp4",
    "nvfp4_gemm_residual_bf16",
    "nvfp4_gemm_streamk_bf16",
    "nvfp4_gemm_streamk_bias_bf16",
    "quantize_fp4_sfa_fp16",
    "quantize_fp4_sfa_bf16",
    "sfa_size_bytes",
]
