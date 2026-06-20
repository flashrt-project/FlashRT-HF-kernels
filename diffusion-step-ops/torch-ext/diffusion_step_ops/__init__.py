"""FlashRT diffusion step helper kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


def _check_same_shape(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor | None = None) -> None:
    if a.shape != b.shape:
        raise RuntimeError("input tensors must have the same shape")
    if c is not None and a.shape != c.shape:
        raise RuntimeError("output tensor must have the same shape as inputs")


@torch.library.register_fake(add_op_namespace_prefix("add_bf16_out"))
def _add_bf16_out_fake(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _check_same_shape(a, b, out)
    return None


@torch.library.register_fake(add_op_namespace_prefix("euler_step_bf16_out"))
def _euler_step_bf16_out_fake(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
    out: torch.Tensor,
) -> None:
    _check_same_shape(latent, velocity, out)
    return None


@torch.library.register_fake(add_op_namespace_prefix("cfg_combine_into_residual_bf16"))
def _cfg_combine_into_residual_bf16_fake(
    residual: torch.Tensor,
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    beta: float,
) -> None:
    _check_same_shape(residual, v_cond, v_uncond)
    return None


@torch.library.register_fake(add_op_namespace_prefix("cfg_combine_into_residual_fp16"))
def _cfg_combine_into_residual_fp16_fake(
    residual: torch.Tensor,
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    beta: float,
) -> None:
    _check_same_shape(residual, v_cond, v_uncond)
    return None


@torch.library.register_fake(add_op_namespace_prefix("teacher_force_first_frame_bf16"))
def _teacher_force_first_frame_bf16_fake(
    video_latent: torch.Tensor,
    cond_latent: torch.Tensor,
) -> None:
    if video_latent.dim() != 5:
        raise RuntimeError("video_latent must have shape (B, C, T, H, W)")
    if cond_latent.shape != (
        video_latent.shape[0],
        video_latent.shape[1],
        video_latent.shape[3],
        video_latent.shape[4],
    ):
        raise RuntimeError("cond_latent must have shape (B, C, H, W)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("motus_decode_postprocess_bf16_to_fp32"))
def _motus_decode_postprocess_bf16_to_fp32_fake(
    decoded: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if decoded.dim() != 5:
        raise RuntimeError("decoded must have shape (B, C, T_in, H, W)")
    if decoded.shape[2] < 2:
        raise RuntimeError("decoded T_in must be >= 2")
    expected = (decoded.shape[0], decoded.shape[1], decoded.shape[2] - 1, decoded.shape[3], decoded.shape[4])
    if out.shape != expected:
        raise RuntimeError("out must have shape (B, C, T_in - 1, H, W)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("cast_bf16_to_fp32"))
def _cast_bf16_to_fp32_fake(src: torch.Tensor, dst: torch.Tensor) -> None:
    if src.shape != dst.shape:
        raise RuntimeError("src and dst must have the same shape")
    return None


def add_bf16(a: torch.Tensor, b: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return ``a + b`` for contiguous BF16 CUDA tensors."""

    if out is None:
        out = torch.empty_like(a)
    ops.add_bf16_out(a, b, out)
    return out


def euler_step_bf16(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return ``latent + velocity * dt`` for BF16 CUDA tensors."""

    if out is None:
        out = torch.empty_like(latent)
    ops.euler_step_bf16_out(latent, velocity, float(dt), out)
    return out


def cfg_combine_into_residual_bf16(
    residual: torch.Tensor,
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """In-place ``residual += v_uncond + beta * (v_cond - v_uncond)``."""

    ops.cfg_combine_into_residual_bf16(residual, v_cond, v_uncond, float(beta))
    return residual


def cfg_combine_into_residual_fp16(
    residual: torch.Tensor,
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """FP16 variant of classifier-free guidance residual combine."""

    ops.cfg_combine_into_residual_fp16(residual, v_cond, v_uncond, float(beta))
    return residual


def teacher_force_first_frame_bf16(video_latent: torch.Tensor, cond_latent: torch.Tensor) -> torch.Tensor:
    """Copy ``cond_latent[:, :, :, :]`` into ``video_latent[:, :, 0, :, :]``."""

    ops.teacher_force_first_frame_bf16(video_latent, cond_latent)
    return video_latent


def motus_decode_postprocess_bf16_to_fp32(
    decoded: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Drop the first frame and map BF16 decoded latents from [-1, 1] to [0, 1]."""

    if out is None:
        out = torch.empty(
            (decoded.shape[0], decoded.shape[1], decoded.shape[2] - 1, decoded.shape[3], decoded.shape[4]),
            device=decoded.device,
            dtype=torch.float32,
        )
    ops.motus_decode_postprocess_bf16_to_fp32(decoded, out)
    return out


def cast_bf16_to_fp32(src: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cast a BF16 CUDA tensor to FP32."""

    if out is None:
        out = torch.empty_like(src, dtype=torch.float32)
    ops.cast_bf16_to_fp32(src, out)
    return out


__all__ = [
    "add_bf16",
    "cast_bf16_to_fp32",
    "cfg_combine_into_residual_bf16",
    "cfg_combine_into_residual_fp16",
    "euler_step_bf16",
    "motus_decode_postprocess_bf16_to_fp32",
    "teacher_force_first_frame_bf16",
]
