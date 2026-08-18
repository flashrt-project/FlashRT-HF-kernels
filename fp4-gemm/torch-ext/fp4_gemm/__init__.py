"""FlashRT FP4 GEMM kernels."""

from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix, ops


NVFP4_BLOCK_SIZE = 16
NVFP4_SCALE_TILE_ROWS = 128
NVFP4_SCALE_TILE_BLOCKS = 4
SUPPORTED_LAYOUTS = ("row-major-packed-e2m1", "cutlass-sm1xx-blockscaled")
SUPPORTED_CUDA_CAPABILITIES = ("11.0a", "12.0a")


def capabilities() -> dict[str, object]:
    """Return the public Tensor API contract used by runtime dispatchers."""
    return {
        "quantization": "W4A4 NVFP4 E2M1",
        "block_size": NVFP4_BLOCK_SIZE,
        "scale_layout": "pad128(rows) x pad4(dim/16), 512 bytes per tile",
        "scale_size": "ceil(rows/128) * ceil((dim/16)/4) * 512 bytes",
        "layouts": SUPPORTED_LAYOUTS,
        "cuda_capabilities": SUPPORTED_CUDA_CAPABILITIES,
        "public_m_alignment": 1,
        "raw_sm120_tile_m": 128,
        "sm120_m256_min_m": 512,
        "sm120_m256_qualified_nk": (
            (17408, 5120),
            (5120, 17408),
            (12288, 5120),
        ),
        "sm120_m256_diagnostic_nk": ((16384, 5120),),
        "sm120_interleaved_gemv_m": 1,
        "sm120_interleaved_weight_layout": "groups-of-8 x K/64 x 8 x 32B",
        "sm120_warpsplit_mrows": (1, 16),
        "sm120_warpsplit_mrows_n_alignment": 8,
        "sm120_warpsplit_mrows_k_alignment": "64 * warps",
        "sm120_w4a16_marlin_m": (1, 16),
        "sm120_w4a16_marlin_n_alignment": 64,
        "sm120_w4a16_marlin_k_alignment": 128,
        "sm120_w4a16_marlin_weight_layout": "marlin-repacked-nvfp4",
        "sm120_w4a16_marlin_scale_layout": "marlin-permuted-fp8-e4m3-block16",
        "errors": "exceptions",
    }


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


@torch.library.register_fake(add_op_namespace_prefix("fp4_w4a4_gemm_warpsplit_mrows_bf16"))
def _gemm_warpsplit_mrows_fake(a_packed, b_packed, sfa, sfb, out, alpha: float = 1.0, warps: int = 2, stages: int = 6) -> None:
    if a_packed.dim() != 2 or b_packed.dim() != 2:
        raise RuntimeError("a_packed and b_packed must be rank-2")
    if a_packed.shape[1] != b_packed.shape[1]:
        raise RuntimeError("a_packed and b_packed must have the same K / 2")
    if not 1 <= a_packed.shape[0] <= 16:
        raise RuntimeError("the multi-row warp-split tier serves 1..16 rows")
    if b_packed.shape[0] % 8:
        raise RuntimeError("N must be a multiple of 8")
    if warps not in (2, 4, 8):
        raise RuntimeError("warps must be 2, 4 or 8")
    if stages not in (3, 4, 6) or (warps == 8 and stages == 6):
        raise RuntimeError("unsupported stages/warps combination")
    if (a_packed.shape[1] * 2) % (64 * warps):
        raise RuntimeError("K must be a multiple of 64*warps")
    if out.shape != (a_packed.shape[0], b_packed.shape[0]):
        raise RuntimeError("out must have shape (M, N)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("fp4_w4a4_gemm_warpsplit_mrows_pdl_bf16"))
def _gemm_warpsplit_mrows_pdl_fake(a_packed, b_packed, sfa, sfb, out, alpha: float = 1.0, warps: int = 2, stages: int = 6) -> None:
    return _gemm_warpsplit_mrows_fake(
        a_packed, b_packed, sfa, sfb, out, alpha, warps, stages
    )


@torch.library.register_fake(add_op_namespace_prefix("fp4_w4a4_gemv_warpsplit_bf16"))
def _gemv_warpsplit_fake(a_packed, b_packed, sfa, sfb, out, alpha: float = 1.0, warps: int = 4, stages: int = 4) -> None:
    if a_packed.shape[0] != 1:
        raise RuntimeError("warp-split GEMV serves M=1 only")
    if out.shape != (1, b_packed.shape[0]):
        raise RuntimeError("out must have shape (1, N)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("fp4_repack_b_interleaved_sm120"))
def _repack_b_interleaved_fake(b_packed, b_interleaved) -> None:
    if b_interleaved.shape != b_packed.shape:
        raise RuntimeError("b_interleaved must have the same shape as b_packed")
    return None


@torch.library.register_fake(add_op_namespace_prefix("fp4_w4a4_gemv_warpsplit_interleaved_bf16"))
def _gemv_warpsplit_interleaved_fake(
    a_packed, b_interleaved, sfa, sfb, out,
    alpha: float = 1.0, warps: int = 4, stages: int = 4,
) -> None:
    if a_packed.shape[0] != 1:
        raise RuntimeError("interleaved warp-split GEMV serves M=1 only")
    if out.shape != (1, b_interleaved.shape[0]):
        raise RuntimeError("out must have shape (1, N)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_m256_workspace_size"))
def _m256_workspace_size_fake(a_packed, b_packed, sfa, sfb) -> int:
    del sfa, sfb
    if a_packed.ndim != 2 or b_packed.ndim != 2:
        raise RuntimeError("a_packed and b_packed must be rank-2")
    if a_packed.shape[0] < 512:
        raise RuntimeError("the M256 tier requires M >= 512")
    if a_packed.shape[1] != b_packed.shape[1]:
        raise RuntimeError("packed A and B K dimensions must match")
    # The current persistent M256 CUTLASS schedule has no auxiliary storage.
    return 0


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemm_m256_bf16"))
def _m256_fake(a_packed, b_packed, sfa, sfb, workspace, out, alpha: float = 1.0) -> None:
    if a_packed.shape[0] < 512:
        raise RuntimeError("the M256 tier requires M >= 512")
    if out.shape != (a_packed.shape[0], b_packed.shape[0]):
        raise RuntimeError("out must have shape (M, N)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_w4a16_marlin_bf16"))
def _w4a16_marlin_fake(x, weight, weight_scale, global_scale, workspace, out) -> None:
    del weight, weight_scale, global_scale, workspace
    if x.ndim != 2 or out.ndim != 2 or out.shape[0] != x.shape[0]:
        raise RuntimeError("x/out must have shapes (M, K)/(M, N)")
    if not 1 <= x.shape[0] <= 16:
        raise RuntimeError("the Marlin W4A16 tier serves M in [1, 16]")
    if x.shape[1] % 128 or out.shape[1] % 64:
        raise RuntimeError("K/N must be divisible by 128/64")
    return None


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_w4a16_marlin_repack"))
def _w4a16_marlin_repack_fake(qweight_kn, weight_marlin) -> None:
    if qweight_kn.ndim != 2:
        raise RuntimeError("qweight_kn must have shape (K / 8, N)")
    k, n = qweight_kn.shape[0] * 8, qweight_kn.shape[1]
    if weight_marlin.shape != (k // 16, n * 2):
        raise RuntimeError("weight_marlin must have shape (K / 16, 2 * N)")
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


@torch.library.register_fake(add_op_namespace_prefix("quantize_fp4_sfa_mse_bf16"))
def _quant_mse_bf16_fake(x, packed, sfa, is_sfb: bool = False) -> None:
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


@torch.library.register_fake(add_op_namespace_prefix("quantize_fp4_sfa_bf16_pdl"))
def _quant_bf16_pdl_fake(x: torch.Tensor, packed: torch.Tensor, sfa: torch.Tensor, is_sfb: bool = False) -> None:
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


def quantize_fp4_sfa_mse_bf16(
    x: torch.Tensor,
    packed: torch.Tensor | None = None,
    sfa: torch.Tensor | None = None,
    is_sfb: bool = False,
):
    """Pack BF16 weights using per-block reconstruction-MSE scale search."""
    if packed is None or sfa is None:
        packed, sfa = _alloc_fp4(x.shape[0], x.shape[1], x.device)
    ops.quantize_fp4_sfa_mse_bf16(x, packed, sfa, bool(is_sfb))
    return packed, sfa


def aligned_fp4_dim(dim: int, alignment: int = 32) -> int:
    """Return the physical FP4 GEMM dimension for a logical dimension."""
    if dim <= 0 or alignment <= 0 or alignment % 16 != 0:
        raise ValueError("dim must be positive and alignment divisible by 16")
    return ((dim + alignment - 1) // alignment) * alignment


def quantize_fp4_sfa_padded_bf16(
    x: torch.Tensor,
    *,
    alignment: int = 32,
    is_sfb: bool = False,
):
    """Bind-time BF16 pack with zero padding on the final dimension.

    The returned packed tensor uses the physical aligned width. Keep the
    logical width returned as the third item for slicing final model outputs.
    This helper allocates and is intentionally not a runtime hot-path op.
    """
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError("x must be a 2-D BF16 tensor")
    logical_dim = int(x.shape[1])
    physical_dim = aligned_fp4_dim(logical_dim, alignment)
    if physical_dim == logical_dim:
        padded = x.contiguous()
    else:
        padded = torch.zeros(
            (x.shape[0], physical_dim), device=x.device, dtype=x.dtype
        )
        padded[:, :logical_dim].copy_(x)
    packed, sfa = quantize_fp4_sfa_bf16(padded, is_sfb=is_sfb)
    return packed, sfa, logical_dim


def pack_nvfp4_weight_bf16(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    alignment: int = 32,
    mse: bool = False,
):
    """Zero-pad and pack a logical ``(N, K)`` BF16 weight at bind time.

    Both physical ``N`` and ``K`` are aligned. This is the model-neutral
    contract used by widths such as SigLIP's logical 4304, which becomes 4320
    for Blackwell NVFP4 TMA. The hot path consumes the returned static tensors
    directly and performs no padding or allocation.
    """
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError("weight must be a 2-D BF16 tensor")
    logical_n, logical_k = map(int, weight.shape)
    physical_n = aligned_fp4_dim(logical_n, alignment)
    physical_k = aligned_fp4_dim(logical_k, alignment)
    padded_weight = torch.zeros(
        (physical_n, physical_k), device=weight.device, dtype=weight.dtype
    )
    padded_weight[:logical_n, :logical_k].copy_(weight)
    quantize = quantize_fp4_sfa_mse_bf16 if mse else quantize_fp4_sfa_bf16
    packed, sfb = quantize(padded_weight, is_sfb=True)
    padded_bias = None
    if bias is not None:
        if bias.ndim != 1 or bias.numel() != logical_n:
            raise ValueError("bias must have shape (N,)")
        if bias.dtype != torch.bfloat16 or bias.device != weight.device:
            raise ValueError("bias must match the weight dtype and device")
        padded_bias = torch.zeros(
            (physical_n,), device=bias.device, dtype=bias.dtype
        )
        padded_bias[:logical_n].copy_(bias)
    return packed, sfb, padded_bias, (logical_n, logical_k)


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


def quantize_fp4_sfa_bf16_pdl(
    x: torch.Tensor,
    packed: torch.Tensor | None = None,
    sfa: torch.Tensor | None = None,
    is_sfb: bool = False,
):
    """PDL-enabled twin of :func:`quantize_fp4_sfa_bf16`."""
    if packed is None or sfa is None:
        packed, sfa = _alloc_fp4(x.shape[0], x.shape[1], x.device)
    ops.quantize_fp4_sfa_bf16_pdl(x, packed, sfa, bool(is_sfb))
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


def cutlass_fp4_gemm_geglu_il_hw_v10(
    a_packed: torch.Tensor,
    b_interleaved_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    *,
    scratch: torch.Tensor | None = None,
    out_packed: torch.Tensor | None = None,
    out_sfa: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit public alias for the native SM110 v10 GeGLU schedule."""
    return nvfp4_gemm_geglu_nvfp4_fp16(
        a_packed,
        b_interleaved_packed,
        sfa,
        sfb,
        skinny=True,
        scratch=scratch,
        out_packed=out_packed,
        out_sfa=out_sfa,
    )


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


def fp4_w4a4_gemm_warpsplit_mrows_bf16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    *,
    alpha: float = 1.0,
    warps: int = 2,
    stages: int = 6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-row (M<=16) warp-split-K NVFP4 W4A4 GEMM (SM120).

    The 16x8x64 block-scaled MMA atom computes a full 16-row output
    tile, so up to sixteen rows ride one weight stream at near-GEMV
    cost - the spec-verify block and its re-advance prefixes are the
    customers. Same packed/scale layouts as the linear entry points;
    deeper default stages hide the strided-B latency the extra A-row
    loads expose."""
    if out is None:
        out = torch.empty((a_packed.shape[0], b_packed.shape[0]), device=a_packed.device, dtype=torch.bfloat16)
    ops.fp4_w4a4_gemm_warpsplit_mrows_bf16(a_packed, b_packed, sfa, sfb, out, float(alpha), int(warps), int(stages))
    return out


def fp4_w4a4_gemm_warpsplit_mrows_pdl_bf16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    *,
    alpha: float = 1.0,
    warps: int = 2,
    stages: int = 6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """PDL-enabled twin of the SM120 M<=16 warp-split GEMM."""
    if out is None:
        out = torch.empty(
            (a_packed.shape[0], b_packed.shape[0]),
            device=a_packed.device,
            dtype=torch.bfloat16,
        )
    ops.fp4_w4a4_gemm_warpsplit_mrows_pdl_bf16(
        a_packed, b_packed, sfa, sfb, out,
        float(alpha), int(warps), int(stages),
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


def fp4_repack_b_interleaved_sm120(
    b_packed: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Repack dense row-major packed FP4 weights for the SM120 M=1 GEMV.

    This is a bind-time operation. Cache the returned tensor with the packed
    weight; do not execute it in the decode hot path.
    """
    if out is None:
        out = torch.empty_like(b_packed)
    ops.fp4_repack_b_interleaved_sm120(b_packed, out)
    return out


def fp4_w4a4_gemv_warpsplit_interleaved_bf16(
    a_packed: torch.Tensor,
    b_interleaved: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    *,
    alpha: float = 1.0,
    warps: int = 8,
    stages: int = 3,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """M=1 SM120 W4A4 GEMV using bind-time interleaved FP4 weights."""
    if out is None:
        out = torch.empty(
            (1, b_interleaved.shape[0]),
            device=a_packed.device,
            dtype=torch.bfloat16,
        )
    ops.fp4_w4a4_gemv_warpsplit_interleaved_bf16(
        a_packed, b_interleaved, sfa, sfb, out,
        float(alpha), int(warps), int(stages),
    )
    return out


def nvfp4_gemm_m256_workspace_size(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
) -> int:
    """Return workspace bytes for the SM120 large-M tier.

    Query and allocate once before CUDA Graph capture.
    """
    return int(ops.nvfp4_gemm_m256_workspace_size(a_packed, b_packed, sfa, sfb))


def nvfp4_gemm_m256_bf16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    *,
    workspace: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SM120 M>=512 NVFP4 GEMM with a caller-owned graph-stable workspace."""
    if workspace is None:
        # Dynamo cannot place a custom operator returning a Python scalar in
        # an FX graph. This is exact for the current persistent M256 schedule;
        # eager setup still queries the native helper as a future-proof check.
        workspace_size = (
            0
            if torch.compiler.is_compiling()
            else nvfp4_gemm_m256_workspace_size(a_packed, b_packed, sfa, sfb)
        )
        workspace = torch.empty(
            workspace_size,
            device=a_packed.device,
            dtype=torch.uint8,
        )
    if out is None:
        out = torch.empty(
            (a_packed.shape[0], b_packed.shape[0]),
            device=a_packed.device,
            dtype=torch.bfloat16,
        )
    ops.nvfp4_gemm_m256_bf16(
        a_packed, b_packed, sfa, sfb, workspace, out, float(alpha)
    )
    return out


def allocate_w4a16_marlin_workspace(device: torch.device | str) -> torch.Tensor:
    """Allocate the persistent Marlin lock workspace before graph capture."""
    props = torch.cuda.get_device_properties(device)
    return torch.zeros(props.multi_processor_count, device=device, dtype=torch.int32)


def adopt_nvfp4_w4a16_marlin(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert standard ModelOpt NVFP4 tensors once at model bind time.

    Inputs use ``weight[N,K/2]``, ``weight_scale[N,K/16]`` and scalar
    ``weight_scale_2``. Returned tensors are the three runtime operands plus a
    zeroed lock workspace. This function is intentionally outside the graph
    hot path.
    """
    if weight_packed.dtype != torch.uint8 or weight_packed.ndim != 2:
        raise ValueError("weight_packed must be uint8 with shape (N, K / 2)")
    n, k = weight_packed.shape[0], weight_packed.shape[1] * 2
    if k % 128 or n % 64:
        raise ValueError("K/N must be divisible by 128/64")
    if weight_scale.shape != (n, k // NVFP4_BLOCK_SIZE):
        raise ValueError("weight_scale must have shape (N, K / 16)")
    if weight_scale_2.numel() != 1:
        raise ValueError("weight_scale_2 must contain one global scale")

    qweight_kn = weight_packed.view(torch.int32).T.contiguous()
    weight_marlin = torch.empty(
        (k // 16, n * 2), device=weight_packed.device, dtype=torch.int32
    )
    ops.nvfp4_w4a16_marlin_repack(qweight_kn, weight_marlin)

    scales = weight_scale.T.contiguous().to(torch.bfloat16)
    scale_perm = [i + 8 * j for i in range(8) for j in range(8)]
    scales = scales.reshape(-1, 64)[:, scale_perm].reshape(-1, n).contiguous()

    # Match vLLM's NVFP4 Marlin contract exactly: scale permutation uses the
    # model parameter dtype, then S0E5M3 encoding starts from FP16.
    scales = scales.to(torch.float16)
    scales = scales.view(-1, 4)[:, [0, 2, 1, 3]].reshape(scales.shape)
    scaled = scales.float() * (2**7)
    nonzero = scaled > 0
    scale_factor = 1.0
    if bool(nonzero.any()):
        max_value = scaled[nonzero].max()
        if bool(max_value < 448 * (2**7)):
            scale_factor = float((448 * (2**7) / max_value).log2().floor().exp2())
    if scale_factor > 1.0:
        scales = (scales.float() * scale_factor).to(torch.float16)
    scales = scales * (2**7)
    scales[scales < 2] = 0
    scales = (scales.view(torch.int16) << 1).view(torch.float8_e4m3fn)
    scales = scales[:, 1::2].contiguous()

    # Marlin's BF16 dequantizer uses exponent bias 126 and removes the seven
    # bits introduced while encoding the special unsigned FP8 scale format.
    global_scale = (
        weight_scale_2.to(device=weight_packed.device, dtype=torch.float32)
        * (2.0 ** 119)
        / scale_factor
    ).contiguous()
    workspace = allocate_w4a16_marlin_workspace(weight_packed.device)
    return weight_marlin, scales, global_scale, workspace


def nvfp4_w4a16_marlin_bf16(
    x: torch.Tensor,
    weight_marlin: torch.Tensor,
    weight_scale_marlin: torch.Tensor,
    weight_global_scale: torch.Tensor,
    *,
    workspace: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Small-M BF16 x NVFP4 linear using bind-time Marlin layouts.

    ``weight_marlin``, ``weight_scale_marlin`` and
    ``weight_global_scale`` are the tensors produced by a standard NVFP4
    Marlin adoption step. The operation supports M in [1, 16] and never
    allocates internally, making it suitable for CUDA Graph replay.
    """
    if out is None:
        n = weight_scale_marlin.numel() // (x.shape[1] // NVFP4_BLOCK_SIZE)
        out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    ops.nvfp4_w4a16_marlin_bf16(
        x, weight_marlin, weight_scale_marlin, weight_global_scale, workspace, out
    )
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
    "NVFP4_BLOCK_SIZE",
    "NVFP4_SCALE_TILE_ROWS",
    "NVFP4_SCALE_TILE_BLOCKS",
    "SUPPORTED_CUDA_CAPABILITIES",
    "SUPPORTED_LAYOUTS",
    "capabilities",
    "allocate_w4a16_marlin_workspace",
    "adopt_nvfp4_w4a16_marlin",
    "aligned_fp4_dim",
    "cutlass_fp4_gemm_geglu_il_hw_v10",
    "dequantize_fp4_sfa_fp16",
    "e0m3_weight_gemm_fp16",
    "fp4_w4a16_linear_bf16",
    "fp4_repack_b_interleaved_sm120",
    "fp4_w4a4_gemv_warpsplit_bf16",
    "fp4_w4a4_gemv_warpsplit_interleaved_bf16",
    "fp4_w4a4_gemm_warpsplit_mrows_bf16",
    "fp4_w4a4_gemm_warpsplit_mrows_pdl_bf16",
    "nvfp4_gemm_bf16",
    "nvfp4_gemm_fp16",
    "nvfp4_gemm_variant_bf16",
    "nvfp4_gemm_nvfp4",
    "nvfp4_gemm_m256_bf16",
    "nvfp4_gemm_m256_workspace_size",
    "nvfp4_w4a16_marlin_bf16",
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
    "quantize_fp4_sfa_mse_bf16",
    "quantize_fp4_sfa_padded_bf16",
    "pack_nvfp4_weight_bf16",
    "quantize_e0m3_sfa_fp16",
    "quantize_fp4_sfa_bf16",
    "quantize_fp4_sfa_bf16_pdl",
    "sfa_size_bytes",
]
