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


@torch.library.register_fake(add_op_namespace_prefix("pack_tail_bf16"))
def _pack_tail_bf16_fake(tail: torch.Tensor, flat_dim: int, out: torch.Tensor) -> None:
    if tail.dim() != 1 or out.shape != (flat_dim,) or flat_dim < tail.numel():
        raise RuntimeError("pack_tail_bf16 expects tail (N,), flat_dim >= N, out (flat_dim,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("add_bias_zero_tail_bf16"))
def _add_bias_zero_tail_bf16_fake(
    input: torch.Tensor,
    bias: torch.Tensor,
    valid_cols: int,
    out: torch.Tensor,
) -> None:
    if (
        input.dim() != 2
        or bias.shape != (input.shape[1],)
        or out.shape != input.shape
        or valid_cols < 0
        or valid_cols > input.shape[1]
    ):
        raise RuntimeError(
            "add_bias_zero_tail_bf16 expects input/out (rows, cols), "
            "bias (cols,), valid_cols in [0, cols]"
        )
    return None


@torch.library.register_fake(add_op_namespace_prefix("extract_tail_f32_to_bf16"))
def _extract_tail_f32_to_bf16_fake(
    flat: torch.Tensor,
    tail_numel: int,
    out: torch.Tensor,
) -> None:
    if flat.dim() != 1 or tail_numel <= 0 or tail_numel > flat.numel() or out.shape != (tail_numel,):
        raise RuntimeError(
            "extract_tail_f32_to_bf16 expects flat (N,), tail_numel in [1, N], out (tail_numel,)"
        )
    return None


@torch.library.register_fake(add_op_namespace_prefix("add_bias_pair_bf16"))
def _add_bias_pair_bf16_fake(
    input: torch.Tensor,
    bias_a: torch.Tensor,
    bias_b: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if (
        input.dim() != 2
        or bias_a.shape != (input.shape[1],)
        or bias_b.shape != bias_a.shape
        or out.shape != input.shape
    ):
        raise RuntimeError(
            "add_bias_pair_bf16 expects input/out (rows, hidden) and biases (hidden,)"
        )
    return None


@torch.library.register_fake(add_op_namespace_prefix("unipc_step_f32_bf16"))
def _unipc_step_f32_bf16_fake(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    prev_m1: torch.Tensor,
    prev_m2: torch.Tensor,
    prev_last_sample: torch.Tensor,
    sigma: float,
    corrector_order: int,
    predictor_order: int,
    c_sample: float,
    c_last: float,
    c_prev_m1: float,
    c_prev_m2: float,
    c_curr_m: float,
    p_sample: float,
    p_curr_m: float,
    p_prev_m1: float,
    next_sample: torch.Tensor,
    current_m: torch.Tensor,
    current_last_sample: torch.Tensor,
) -> None:
    del (
        sigma,
        corrector_order,
        predictor_order,
        c_sample,
        c_last,
        c_prev_m1,
        c_prev_m2,
        c_curr_m,
        p_sample,
        p_curr_m,
        p_prev_m1,
    )
    for tensor in (
        velocity,
        prev_m1,
        prev_m2,
        prev_last_sample,
        next_sample,
        current_m,
        current_last_sample,
    ):
        if tensor.shape != sample.shape:
            raise RuntimeError("all UniPC tensors must have the same shape")
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


def pack_tail_bf16(
    tail: torch.Tensor,
    flat_dim: int,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Place a BF16 tail at the end of a zero-filled flat BF16 tensor."""

    if out is None:
        out = torch.empty((flat_dim,), device=tail.device, dtype=tail.dtype)
    ops.pack_tail_bf16(tail, int(flat_dim), out)
    return out


def add_bias_zero_tail_bf16(
    input: torch.Tensor,
    bias: torch.Tensor,
    valid_cols: int,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Add a column bias and zero columns at or beyond ``valid_cols``."""

    if out is None:
        out = torch.empty_like(input)
    ops.add_bias_zero_tail_bf16(input, bias, int(valid_cols), out)
    return out


def extract_tail_f32_to_bf16(
    flat: torch.Tensor,
    tail_numel: int,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Extract the final ``tail_numel`` FP32 values and cast them to BF16."""

    if out is None:
        out = torch.empty((tail_numel,), device=flat.device, dtype=torch.bfloat16)
    ops.extract_tail_f32_to_bf16(flat, int(tail_numel), out)
    return out


def add_bias_pair_bf16(
    input: torch.Tensor,
    bias_a: torch.Tensor,
    bias_b: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Add two BF16 row-broadcast biases with BF16 rounding after each add."""

    if out is None:
        out = torch.empty_like(input)
    ops.add_bias_pair_bf16(input, bias_a, bias_b, out)
    return out


def unipc_step_f32_bf16(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    prev_m1: torch.Tensor,
    prev_m2: torch.Tensor,
    prev_last_sample: torch.Tensor,
    sigma: float,
    corrector_order: int,
    predictor_order: int,
    corrector_coefficients: tuple[float, float, float, float, float],
    predictor_coefficients: tuple[float, float, float],
    *,
    next_sample: Optional[torch.Tensor] = None,
    current_m: Optional[torch.Tensor] = None,
    current_last_sample: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one UniPC predictor/corrector update."""

    if len(corrector_coefficients) != 5:
        raise RuntimeError("corrector_coefficients must have five values")
    if len(predictor_coefficients) != 3:
        raise RuntimeError("predictor_coefficients must have three values")
    if next_sample is None:
        next_sample = torch.empty_like(sample)
    if current_m is None:
        current_m = torch.empty_like(sample)
    if current_last_sample is None:
        current_last_sample = torch.empty_like(sample)
    ops.unipc_step_f32_bf16(
        sample,
        velocity,
        prev_m1,
        prev_m2,
        prev_last_sample,
        float(sigma),
        int(corrector_order),
        int(predictor_order),
        *map(float, corrector_coefficients),
        *map(float, predictor_coefficients),
        next_sample,
        current_m,
        current_last_sample,
    )
    return next_sample, current_m, current_last_sample


__all__ = [
    "add_bf16",
    "add_bias_pair_bf16",
    "add_bias_zero_tail_bf16",
    "cast_bf16_to_fp32",
    "cfg_combine_into_residual_bf16",
    "cfg_combine_into_residual_fp16",
    "euler_step_bf16",
    "extract_tail_f32_to_bf16",
    "motus_decode_postprocess_bf16_to_fp32",
    "pack_tail_bf16",
    "teacher_force_first_frame_bf16",
    "unipc_step_f32_bf16",
]
