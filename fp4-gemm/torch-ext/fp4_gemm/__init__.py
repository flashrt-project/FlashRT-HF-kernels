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
        # Tile-layout padding entries are not written by every quantizer.
        torch.zeros((sfa_size_bytes(rows, dim),), device=device, dtype=torch.uint8),
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


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_fp16"))
def _linear_fp16_fake(a_packed, b_packed, sfa, sfb, out, alpha: float = 1.0, variant: int = -1) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_variant_bf16"))
def _linear_variant_bf16_fake(a, b, sfa, sfb, out, alpha: float = 1.0, variant: int = 7) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_nvfp4"))
def _linear_fp4out_fake(a, b, sfa, sfb, out_packed, out_sfa) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_geglu_nvfp4_fp16"))
def _geglu_fp4_fake(a, b, sfa, sfb, scratch, out_packed, out_sfa, skinny: bool = False) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_bias_gelu_nvfp4_fp16"))
def _bias_gelu_fp4_fp16_fake(a, b, sfa, sfb, bias, out_packed, out_sfa) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_bias_residual_fp16"))
def _bias_residual_fp16_fake(a, b, sfa, sfb, bias, residual, out) -> None:
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


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_bias_bf16"))
def _bias_fake(a, b, sfa, sfb, bias, out) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_bias_residual_bf16"))
def _bias_residual_fake(a, b, sfa, sfb, bias, residual, out) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_fp4_sfa_fp16"))
def _quant_fake(x: torch.Tensor, packed: torch.Tensor, sfa: torch.Tensor, is_sfb: bool = False) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_fp4_sfa_mse_fp16"))
def _quant_mse_fake(x, packed, sfa, is_sfb: bool = False) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_e0m3_sfa_fp16"))
def _quant_e0m3_fake(x, packed, sfa, is_sfb: bool = False) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("e0m3_weight_gemm_fp16"))
def _e0m3_gemm_fake(a, b, sfa, sfb, out, alpha: float = 1.0, a_format: int = 1) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_relu2_nvfp4"))
def _relu2_gemm_fake(a, b, sfa, sfb, out_packed, out_sfa) -> None:
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


def quantize_fp4_sfa_mse_fp16(
    x: torch.Tensor,
    packed: torch.Tensor | None = None,
    sfa: torch.Tensor | None = None,
    is_sfb: bool = False,
):
    """Pack FP16 weights using per-block reconstruction-MSE scale search."""
    if packed is None or sfa is None:
        packed, sfa = _alloc_fp4(x.shape[0], x.shape[1], x.device)
    ops.quantize_fp4_sfa_mse_fp16(x, packed, sfa, bool(is_sfb))
    return packed, sfa


def quantize_e0m3_sfa_fp16(
    x: torch.Tensor,
    packed: torch.Tensor | None = None,
    sfa: torch.Tensor | None = None,
    is_sfb: bool = False,
):
    """Quantize FP16 to packed E0M3 with CUTLASS SFA/SFB layout."""
    if packed is None or sfa is None:
        packed, sfa = _alloc_fp4(x.shape[0], x.shape[1], x.device)
    ops.quantize_e0m3_sfa_fp16(x, packed, sfa, bool(is_sfb))
    return packed, sfa


def e0m3_weight_gemm_fp16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    *,
    alpha: float = 1.0,
    a_format: int = 1,
    out: torch.Tensor | None = None,
):
    """SM110 GEMM with E0M3 weights and E2M1 or E0M3 activations."""
    if out is None:
        out = torch.empty(
            (a_packed.shape[0], b_packed.shape[0]),
            device=a_packed.device,
            dtype=torch.float16,
        )
    ops.e0m3_weight_gemm_fp16(
        a_packed, b_packed, sfa, sfb, out, float(alpha), int(a_format)
    )
    return out


def nvfp4_gemm_relu2_nvfp4(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    *,
    out_packed: torch.Tensor | None = None,
    out_sfa: torch.Tensor | None = None,
):
    """SM110 NVFP4 GEMM with fused ReLU-squared and NVFP4 output."""
    if out_packed is None or out_sfa is None:
        out_packed, out_sfa = _alloc_fp4(
            a_packed.shape[0], b_packed.shape[0], a_packed.device
        )
    ops.nvfp4_gemm_relu2_nvfp4(
        a_packed, b_packed, sfa, sfb, out_packed, out_sfa
    )
    return out_packed, out_sfa


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


def nvfp4_gemm_fp16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    alpha: float = 1.0,
    out: torch.Tensor | None = None,
    variant: int = -1,
) -> torch.Tensor:
    """SM110 native NVFP4 GEMM with FP16 output."""
    if out is None:
        out = torch.empty(
            (a_packed.shape[0], b_packed.shape[0]),
            device=a_packed.device,
            dtype=torch.float16,
        )
    ops.nvfp4_gemm_fp16(
        a_packed, b_packed, sfa, sfb, out, float(alpha), int(variant)
    )
    return out


def nvfp4_gemm_variant_bf16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    alpha: float = 1.0,
    out: torch.Tensor | None = None,
    variant: int = 7,
) -> torch.Tensor:
    """SM110 native NVFP4 GEMM with BF16 output.

    ``variant=7`` serves large-M encoder/SigLIP shapes; ``variant=10`` is
    the narrow-N PI0.5 decoder tile. The indices match FlashRT native.
    """
    if out is None:
        out = torch.empty(
            (a_packed.shape[0], b_packed.shape[0]),
            device=a_packed.device,
            dtype=torch.bfloat16,
        )
    ops.nvfp4_gemm_variant_bf16(
        a_packed, b_packed, sfa, sfb, out, float(alpha), int(variant)
    )
    return out


def nvfp4_gemm_nvfp4(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    *,
    out_packed: torch.Tensor | None = None,
    out_sfa: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SM110 NVFP4 GEMM with an NVFP4 output and CUTLASS SFA layout."""
    if out_packed is None or out_sfa is None:
        out_packed, out_sfa = _alloc_fp4(
            a_packed.shape[0], b_packed.shape[0], a_packed.device
        )
    ops.nvfp4_gemm_nvfp4(
        a_packed, b_packed, sfa, sfb, out_packed, out_sfa
    )
    return out_packed, out_sfa


def nvfp4_gemm_geglu_nvfp4_fp16(
    a_packed: torch.Tensor,
    b_interleaved_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    *,
    skinny: bool = False,
    scratch: torch.Tensor | None = None,
    out_packed: torch.Tensor | None = None,
    out_sfa: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GEMM with fused GeGLU and compact NVFP4 output on SM110.

    ``b_interleaved_packed`` stores gate/up rows pairwise, so its first
    dimension is twice the logical hidden width.
    """
    m, n_twice = a_packed.shape[0], b_interleaved_packed.shape[0]
    hidden = n_twice // 2
    if scratch is None:
        scratch = torch.empty((m, hidden), device=a_packed.device, dtype=torch.uint8)
    if out_packed is None:
        out_packed = torch.empty((m, hidden // 2), device=a_packed.device, dtype=torch.uint8)
    if out_sfa is None:
        out_sfa = torch.zeros((sfa_size_bytes(m, hidden),), device=a_packed.device, dtype=torch.uint8)
    ops.nvfp4_gemm_geglu_nvfp4_fp16(
        a_packed, b_interleaved_packed, sfa, sfb, scratch,
        out_packed, out_sfa, bool(skinny)
    )
    return out_packed, out_sfa


def nvfp4_gemm_bias_gelu_nvfp4_fp16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    bias: torch.Tensor,
    *,
    out_packed: torch.Tensor | None = None,
    out_sfa: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP16-bias GEMM with fused GELU and NVFP4 output on SM110."""
    m, n = a_packed.shape[0], b_packed.shape[0]
    if out_packed is None:
        out_packed = torch.empty((m, n // 2), device=a_packed.device, dtype=torch.uint8)
    if out_sfa is None:
        out_sfa = torch.zeros((sfa_size_bytes(m, n),), device=a_packed.device, dtype=torch.uint8)
    ops.nvfp4_gemm_bias_gelu_nvfp4_fp16(
        a_packed, b_packed, sfa, sfb, bias, out_packed, out_sfa
    )
    return out_packed, out_sfa


def nvfp4_gemm_bias_residual_fp16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP16-output GEMM with fused FP16 bias and residual on SM110."""
    if out is None:
        out = torch.empty_like(residual)
    ops.nvfp4_gemm_bias_residual_fp16(
        a_packed, b_packed, sfa, sfb, bias, residual, out
    )
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


def nvfp4_gemm_bias_bf16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    bias: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """SM110 NVFP4 GEMM with a fused per-column BF16 bias."""
    if out is None:
        out = torch.empty(
            (a_packed.shape[0], b_packed.shape[0]),
            device=a_packed.device,
            dtype=torch.bfloat16,
        )
    ops.nvfp4_gemm_bias_bf16(a_packed, b_packed, sfa, sfb, bias, out)
    return out


def nvfp4_gemm_bias_residual_bf16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """SM110 NVFP4 GEMM with fused BF16 bias and residual add."""
    if out is None:
        out = torch.empty_like(residual)
    ops.nvfp4_gemm_bias_residual_bf16(
        a_packed, b_packed, sfa, sfb, bias, residual, out
    )
    return out


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
        out_sfa = torch.zeros((sfa_size_bytes(m, n),), device=a_packed.device, dtype=torch.uint8)
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
    "e0m3_weight_gemm_fp16",
    "fp4_w4a16_linear_bf16",
    "fp4_w4a4_gemv_warpsplit_bf16",
    "nvfp4_gemm_bf16",
    "nvfp4_gemm_fp16",
    "nvfp4_gemm_variant_bf16",
    "nvfp4_gemm_nvfp4",
    "nvfp4_gemm_geglu_nvfp4_fp16",
    "nvfp4_gemm_bias_gelu_nvfp4_fp16",
    "nvfp4_gemm_bias_residual_fp16",
    "nvfp4_gemm_bias_bf16",
    "nvfp4_gemm_bias_gelu_bf16",
    "nvfp4_gemm_bias_gelu_nvfp4",
    "nvfp4_gemm_bias_residual_bf16",
    "nvfp4_gemm_residual_bf16",
    "nvfp4_gemm_relu2_nvfp4",
    "nvfp4_gemm_streamk_bf16",
    "nvfp4_gemm_streamk_bias_bf16",
    "quantize_fp4_sfa_fp16",
    "quantize_fp4_sfa_mse_fp16",
    "quantize_e0m3_sfa_fp16",
    "quantize_fp4_sfa_bf16",
    "sfa_size_bytes",
]
