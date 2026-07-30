"""FlashRT spatiotemporal layout kernels."""

from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix, ops


def _check_ncdhw(x: torch.Tensor, name: str) -> None:
    if x.dim() != 5:
        raise RuntimeError(f"{name} must have shape (B, C, T, H, W)")


@torch.library.register_fake(add_op_namespace_prefix("ncdhw_to_blc_bf16"))
def _ncdhw_to_blc_bf16_fake(x: torch.Tensor, out: torch.Tensor) -> None:
    _check_ncdhw(x, "x")
    b, c, t, h, w = x.shape
    if out.shape != (b, t * h * w, c):
        raise RuntimeError("out must have shape (B, T * H * W, C)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("patch_im2col_bf16"))
def _patch_im2col_bf16_fake(x: torch.Tensor, out: torch.Tensor) -> None:
    if x.dim() != 4 or x.shape[1:] != (224, 224, 3):
        raise RuntimeError("x must have shape (num_views, 224, 224, 3)")
    if out.shape != (x.shape[0] * 256, 588):
        raise RuntimeError("out must have shape (num_views * 256, 588)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("time_unshuffle2_bf16"))
def _time_unshuffle2_bf16_fake(x: torch.Tensor, out: torch.Tensor) -> None:
    _check_ncdhw(x, "x")
    b, c2, t, h, w = x.shape
    if c2 % 2 != 0:
        raise RuntimeError("x.shape[1] must be even")
    if out.shape != (b, c2 // 2, 2 * t, h, w):
        raise RuntimeError("out must have shape (B, C / 2, 2 * T, H, W)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("add_bias_ncdhw_bf16"))
def _add_bias_ncdhw_bf16_fake(x: torch.Tensor, bias: torch.Tensor) -> None:
    _check_ncdhw(x, "x")
    if bias.shape != (x.shape[1],):
        raise RuntimeError("bias must have shape (C,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("update_cache2_ncdhw_bf16"))
def _update_cache2_ncdhw_bf16_fake(cur: torch.Tensor, prev: torch.Tensor, out: torch.Tensor) -> None:
    _check_ncdhw(cur, "cur")
    b, c, _, h, w = cur.shape
    if prev.shape != (b, c, 2, h, w) or out.shape != (b, c, 2, h, w):
        raise RuntimeError("prev and out must have shape (B, C, 2, H, W)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("channel_to_space3d_bf16"))
def _channel_to_space3d_bf16_fake(
    x: torch.Tensor,
    out_channels: int,
    temporal_factor: int,
    spatial_factor: int,
    repeats: int,
    first_chunk: bool,
    out: torch.Tensor,
) -> None:
    _check_ncdhw(x, "x")
    out_t = x.shape[2] * temporal_factor - (
        temporal_factor - 1 if first_chunk else 0
    )
    expected = (
        x.shape[0],
        out_channels,
        out_t,
        x.shape[3] * spatial_factor,
        x.shape[4] * spatial_factor,
    )
    if (
        out_channels <= 0
        or temporal_factor <= 0
        or spatial_factor <= 0
        or repeats <= 0
        or x.shape[1] * repeats
        < out_channels * temporal_factor * spatial_factor * spatial_factor
        or out.shape != expected
    ):
        raise RuntimeError("channel_to_space3d_bf16 shape contract failed")
    return None


@torch.library.register_fake(
    add_op_namespace_prefix("pack_causal_cache3_nhwc_bf16")
)
def _pack_causal_cache3_nhwc_bf16_fake(
    previous: torch.Tensor, current: torch.Tensor, out: torch.Tensor
) -> None:
    _check_ncdhw(previous, "previous")
    _check_ncdhw(current, "current")
    b, c, t, h, w = current.shape
    if (
        t != 1
        or previous.shape != (b, c, 2, h, w)
        or out.shape != (b, h, w, 3 * c)
    ):
        raise RuntimeError("causal cache pack shape contract failed")
    return None


@torch.library.register_fake(add_op_namespace_prefix("avg_pool3d_channels_bf16"))
def _avg_pool3d_channels_bf16_fake(
    x: torch.Tensor,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    group_size: int,
    out: torch.Tensor,
) -> None:
    _check_ncdhw(x, "x")
    expected = (
        x.shape[0],
        out_channels,
        (x.shape[2] + factor_t - 1) // factor_t,
        x.shape[3] // factor_s,
        x.shape[4] // factor_s,
    )
    if out.shape != expected:
        raise RuntimeError("out has the wrong pooled NCDHW shape")
    return None


@torch.library.register_fake(add_op_namespace_prefix("ndhwc_to_ncdhw_bf16"))
def _ndhwc_to_ncdhw_bf16_fake(x: torch.Tensor, out: torch.Tensor) -> None:
    if x.dim() != 5:
        raise RuntimeError("x must have shape (B, T, H, W, C)")
    expected = (x.shape[0], x.shape[4], x.shape[1], x.shape[2], x.shape[3])
    if out.shape != expected:
        raise RuntimeError("out must have shape (B, C, T, H, W)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("ndhwc_to_ncdhw_bias_bf16"))
def _ndhwc_to_ncdhw_bias_bf16_fake(
    x: torch.Tensor, bias: torch.Tensor, out: torch.Tensor
) -> None:
    _ndhwc_to_ncdhw_bf16_fake(x, out)
    if bias.shape != (x.shape[4],):
        raise RuntimeError("bias must have shape (C,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("ndhwc_to_ncdhw_add_bf16"))
def _ndhwc_to_ncdhw_add_bf16_fake(
    x: torch.Tensor, residual: torch.Tensor, out: torch.Tensor
) -> None:
    _ndhwc_to_ncdhw_bf16_fake(x, out)
    if residual.shape != out.shape:
        raise RuntimeError("residual must match the NCDHW output shape")
    return None


@torch.library.register_fake(
    add_op_namespace_prefix("ncdhw_quantize_fp8_static_ndhwc_bf16")
)
def _ncdhw_quantize_fp8_static_ndhwc_bf16_fake(
    x: torch.Tensor, scale: float, out: torch.Tensor
) -> None:
    _check_ncdhw(x, "x")
    expected = (x.shape[0], x.shape[2], x.shape[3], x.shape[4], x.shape[1])
    if out.shape != expected:
        raise RuntimeError("out must have shape (B, T, H, W, C)")
    return None


@torch.library.register_fake(
    add_op_namespace_prefix("upsample2x_quantize_fp8_static_nhwc_bf16")
)
def _upsample2x_quantize_fp8_static_nhwc_bf16_fake(
    x: torch.Tensor, scale: float, out: torch.Tensor
) -> None:
    if x.dim() != 4:
        raise RuntimeError("x must have shape (N, C, H, W)")
    expected = (x.shape[0], 2 * x.shape[2], 2 * x.shape[3], x.shape[1])
    if out.shape != expected:
        raise RuntimeError("out must have shape (N, 2H, 2W, C)")
    return None


def ncdhw_to_blc_bf16(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Convert BF16 NCDHW tensor to BLC where L = T * H * W."""

    if out is None:
        out = torch.empty((x.shape[0], x.shape[2] * x.shape[3] * x.shape[4], x.shape[1]), device=x.device, dtype=x.dtype)
    ops.ncdhw_to_blc_bf16(x, out)
    return out


def patch_im2col_bf16(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Convert BF16 NHWC images into flattened 14x14 patch rows."""

    if out is None:
        out = torch.empty((x.shape[0] * 256, 588), device=x.device, dtype=x.dtype)
    ops.patch_im2col_bf16(x, out)
    return out


def time_unshuffle2_bf16(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Convert BF16 (B, 2C, T, H, W) into (B, C, 2T, H, W)."""

    if out is None:
        out = torch.empty((x.shape[0], x.shape[1] // 2, 2 * x.shape[2], x.shape[3], x.shape[4]), device=x.device, dtype=x.dtype)
    ops.time_unshuffle2_bf16(x, out)
    return out


def add_bias_ncdhw_bf16(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Add channel bias to an NCDHW tensor in place."""

    ops.add_bias_ncdhw_bf16(x, bias)
    return x


def update_cache2_ncdhw_bf16(cur: torch.Tensor, prev: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Update two-frame latent cache from current and previous NCDHW tensors."""

    if out is None:
        out = torch.empty((cur.shape[0], cur.shape[1], 2, cur.shape[3], cur.shape[4]), device=cur.device, dtype=cur.dtype)
    ops.update_cache2_ncdhw_bf16(cur, prev, out)
    return out


def channel_to_space3d_bf16(
    x: torch.Tensor,
    out_channels: int,
    temporal_factor: int,
    spatial_factor: int,
    repeats: int = 1,
    first_chunk: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Move expanded channels into temporal/spatial dimensions."""

    out_t = x.shape[2] * temporal_factor - (
        temporal_factor - 1 if first_chunk else 0
    )
    if out is None:
        out = torch.empty(
            (
                x.shape[0],
                out_channels,
                out_t,
                x.shape[3] * spatial_factor,
                x.shape[4] * spatial_factor,
            ),
            device=x.device,
            dtype=x.dtype,
        )
    ops.channel_to_space3d_bf16(
        x,
        int(out_channels),
        int(temporal_factor),
        int(spatial_factor),
        int(repeats),
        bool(first_chunk),
        out,
    )
    return out


def pack_causal_cache3_nhwc_bf16(
    previous: torch.Tensor,
    current: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack two cached and one current NCDHW frame into NHWC with 3C."""

    if out is None:
        out = torch.empty(
            (
                current.shape[0],
                current.shape[3],
                current.shape[4],
                3 * current.shape[1],
            ),
            device=current.device,
            dtype=current.dtype,
        )
    ops.pack_causal_cache3_nhwc_bf16(previous, current, out)
    return out


def avg_pool3d_channels_bf16(
    x: torch.Tensor,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    group_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool NCDHW blocks while folding spatiotemporal factors into channels."""

    if out is None:
        out = torch.empty(
            (
                x.shape[0],
                out_channels,
                (x.shape[2] + factor_t - 1) // factor_t,
                x.shape[3] // factor_s,
                x.shape[4] // factor_s,
            ),
            device=x.device,
            dtype=x.dtype,
        )
    ops.avg_pool3d_channels_bf16(
        x, out_channels, factor_t, factor_s, group_size, out
    )
    return out


def ndhwc_to_ncdhw_bf16(
    x: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Convert contiguous BF16 NDHWC to contiguous BF16 NCDHW."""

    if out is None:
        out = torch.empty(
            (x.shape[0], x.shape[4], x.shape[1], x.shape[2], x.shape[3]),
            device=x.device,
            dtype=x.dtype,
        )
    ops.ndhwc_to_ncdhw_bf16(x, out)
    return out


def ndhwc_to_ncdhw_bias_bf16(
    x: torch.Tensor, bias: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Convert NDHWC to NCDHW and add a BF16 channel bias."""

    if out is None:
        out = torch.empty(
            (x.shape[0], x.shape[4], x.shape[1], x.shape[2], x.shape[3]),
            device=x.device,
            dtype=x.dtype,
        )
    ops.ndhwc_to_ncdhw_bias_bf16(x, bias, out)
    return out


def ndhwc_to_ncdhw_add_bf16(
    x: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert NDHWC to NCDHW and add a BF16 NCDHW residual."""

    if out is None:
        out = torch.empty_like(residual)
    ops.ndhwc_to_ncdhw_add_bf16(x, residual, out)
    return out


def ncdhw_quantize_fp8_static_ndhwc_bf16(
    x: torch.Tensor,
    scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize BF16 NCDHW to FP8 E4M3 NDHWC using a static scale."""

    if out is None:
        out = torch.empty(
            (x.shape[0], x.shape[2], x.shape[3], x.shape[4], x.shape[1]),
            device=x.device,
            dtype=torch.float8_e4m3fn,
        )
    ops.ncdhw_quantize_fp8_static_ndhwc_bf16(x, scale, out)
    return out


def upsample2x_quantize_fp8_static_nhwc_bf16(
    x: torch.Tensor,
    scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Nearest-neighbor 2x upsample NCHW and emit static-scaled FP8 NHWC."""

    if out is None:
        out = torch.empty(
            (x.shape[0], 2 * x.shape[2], 2 * x.shape[3], x.shape[1]),
            device=x.device,
            dtype=torch.float8_e4m3fn,
        )
    ops.upsample2x_quantize_fp8_static_nhwc_bf16(x, scale, out)
    return out


__all__ = [
    "ncdhw_to_blc_bf16",
    "patch_im2col_bf16",
    "time_unshuffle2_bf16",
    "add_bias_ncdhw_bf16",
    "update_cache2_ncdhw_bf16",
    "channel_to_space3d_bf16",
    "pack_causal_cache3_nhwc_bf16",
    "avg_pool3d_channels_bf16",
    "ndhwc_to_ncdhw_bf16",
    "ndhwc_to_ncdhw_bias_bf16",
    "ndhwc_to_ncdhw_add_bf16",
    "ncdhw_quantize_fp8_static_ndhwc_bf16",
    "upsample2x_quantize_fp8_static_nhwc_bf16",
]
