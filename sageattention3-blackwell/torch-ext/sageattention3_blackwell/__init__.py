"""SageAttention3 FP4 attention for Blackwell GPUs."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ._ops import add_op_namespace_prefix, ops


SUPPORTED_HEAD_DIMS = (64, 128)
SUPPORTED_LAYOUTS = ("NHD",)
TOKEN_ALIGNMENT = 128
ACCURACY_PROFILE = "speed-first"


def capabilities() -> dict[str, object]:
    """Return the public execution and accuracy contract."""
    return {
        "head_dims": SUPPORTED_HEAD_DIMS,
        "layouts": SUPPORTED_LAYOUTS,
        "token_alignment": TOKEN_ALIGNMENT,
        "attention": "self",
        "gqa": False,
        "accuracy_profile": ACCURACY_PROFILE,
        "caller_owned_workspace": True,
        "cuda_graph_safe": True,
    }


def _padded128(length: int) -> int:
    return ((int(length) + TOKEN_ALIGNMENT - 1) // TOKEN_ALIGNMENT) * TOKEN_ALIGNMENT


@dataclass(frozen=True)
class Sage3Workspace:
    q_packed: torch.Tensor
    k_packed: torch.Tensor
    v_packed: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    out_nhd: torch.Tensor
    out_hnd: torch.Tensor
    softmax_lse: torch.Tensor
    semaphore: torch.Tensor


def _check_nhd(x: torch.Tensor, name: str) -> None:
    if x.dim() != 4 or x.shape[-1] not in SUPPORTED_HEAD_DIMS:
        raise RuntimeError(f"{name} must have contiguous NHD shape [B,L,H,64|128]")
    if not x.is_cuda or not x.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous CUDA")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"{name} must be FP16 or BF16")


def allocate_workspace(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Sage3Workspace:
    """Allocate all quantized inputs and attention outputs before capture."""
    _check_nhd(q, "q")
    _check_nhd(k, "k")
    _check_nhd(v, "v")
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise RuntimeError("Sage3 currently requires self-attention heads and matching K/V")
    b, lq, h, d = q.shape
    lk = _padded128(k.shape[1])
    if lq % TOKEN_ALIGNMENT or k.shape[1] % TOKEN_ALIGNMENT:
        raise RuntimeError("preprocessed Sage3 Q/K/V lengths must be padded to 128")
    out_nhd = torch.empty((b, lq, h, d), device=q.device, dtype=q.dtype)
    return Sage3Workspace(
        q_packed=torch.empty((b, h, lq, d // 2), device=q.device, dtype=torch.uint8),
        k_packed=torch.empty((b, h, lk, d // 2), device=q.device, dtype=torch.uint8),
        v_packed=torch.empty((b, h, d, lk // 2), device=q.device, dtype=torch.uint8),
        q_scale=torch.empty((b, h, lq, d // 16), device=q.device, dtype=torch.float8_e4m3fn),
        k_scale=torch.empty((b, h, lk, d // 16), device=q.device, dtype=torch.float8_e4m3fn),
        v_scale=torch.empty((b, h, d, lk // 16), device=q.device, dtype=torch.float8_e4m3fn),
        out_nhd=out_nhd,
        out_hnd=out_nhd.transpose(1, 2),
        softmax_lse=torch.empty((b, h, lq), device=q.device, dtype=torch.float32),
        semaphore=torch.empty((1,), device=q.device, dtype=torch.int32),
    )


@torch.library.register_fake(add_op_namespace_prefix("quantize_q_fp4_nhd"))
def _q_fake(x, packed, sf):
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_k_fp4_nhd"))
def _k_fake(x, packed, sf):
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_v_fp4_nhd"))
def _v_fake(x, packed, sf):
    return None


@torch.library.register_fake(add_op_namespace_prefix("blockscaled_fp4_attention_static"))
def _attention_fake(q, k, v, sfq, sfk, sfv, delta_s, unpadded_k, softmax_scale,
                    causal, per_block_mean, bf16_output, out, softmax_lse, semaphore):
    return None


def prepare_qkv_fp4_nhd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    workspace: Sage3Workspace,
) -> Sage3Workspace:
    """Quantize centered/padded NHD Q/K/V into Sage3's blockscaled layouts."""
    ops.quantize_q_fp4_nhd(q, workspace.q_packed, workspace.q_scale)
    ops.quantize_k_fp4_nhd(k, workspace.k_packed, workspace.k_scale)
    ops.quantize_v_fp4_nhd(v, workspace.v_packed, workspace.v_scale)
    return workspace


def blockscaled_fp4_attention_static(
    workspace: Sage3Workspace,
    delta_s: torch.Tensor,
    *,
    unpadded_k: int,
    softmax_scale: float | None = None,
    causal: bool = False,
    per_block_mean: bool = True,
) -> torch.Tensor:
    """Run allocation-free Sage3 attention and return an NHD output view."""
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(workspace.out_hnd.shape[-1])
    ops.blockscaled_fp4_attention_static(
        workspace.q_packed, workspace.k_packed, workspace.v_packed,
        workspace.q_scale, workspace.k_scale, workspace.v_scale, delta_s,
        int(unpadded_k), float(softmax_scale), bool(causal),
        bool(per_block_mean), workspace.out_hnd.dtype == torch.bfloat16,
        workspace.out_hnd, workspace.softmax_lse, workspace.semaphore,
    )
    return workspace.out_nhd


__all__ = [
    "ACCURACY_PROFILE",
    "SUPPORTED_HEAD_DIMS",
    "SUPPORTED_LAYOUTS",
    "TOKEN_ALIGNMENT",
    "Sage3Workspace",
    "capabilities",
    "allocate_workspace",
    "prepare_qkv_fp4_nhd",
    "blockscaled_fp4_attention_static",
]
