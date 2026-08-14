"""SageAttention3 FP4 attention for Blackwell GPUs."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ._ops import add_op_namespace_prefix, ops


def _cuda_version_tuple() -> tuple[int, int]:
    version = torch.version.cuda
    if not version:
        return (0, 0)
    major, minor, *_ = version.split(".")
    return (int(major), int(minor))


CUDA_VERSION = _cuda_version_tuple()
SUPPORTED_HEAD_DIMS = (64, 128) if CUDA_VERSION >= (13, 0) else (64,)
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
        "fused_prep": True,
        "delta_dtypes": ("float32", "bfloat16"),
        "d128_min_cuda": "13.0",
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


@dataclass(frozen=True)
class Sage3FusedWorkspace:
    attention: Sage3Workspace
    k_centered: torch.Tensor
    q_groupmean: torch.Tensor
    k_mean: torch.Tensor
    delta_bf16: torch.Tensor
    original_length: int


def _check_nhd(x: torch.Tensor, name: str) -> None:
    if x.dim() != 4 or x.shape[-1] not in SUPPORTED_HEAD_DIMS:
        supported = "|".join(str(dim) for dim in SUPPORTED_HEAD_DIMS)
        raise RuntimeError(
            f"{name} must have contiguous NHD shape [B,L,H,{supported}] "
            f"for this CUDA {torch.version.cuda} artifact"
        )
    if not x.is_cuda or not x.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous CUDA")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"{name} must be FP16 or BF16")


def allocate_workspace(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> Sage3Workspace:
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
    expected_out = (b, lq, h, d)
    if out is None:
        out_nhd = torch.empty(expected_out, device=q.device, dtype=q.dtype)
    else:
        if (out.shape != expected_out or out.device != q.device or
                out.dtype != q.dtype or not out.is_contiguous()):
            raise RuntimeError("out must be contiguous NHD matching Q")
        out_nhd = out
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


def allocate_fused_workspace(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> Sage3FusedWorkspace:
    """Allocate the complete raw-BF16 Sage3 path before CUDA Graph capture."""
    _check_nhd(q, "q")
    _check_nhd(k, "k")
    _check_nhd(v, "v")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise RuntimeError("sage3_prefill_fp4_bf16 requires BF16 inputs")
    if q.shape != k.shape or q.shape != v.shape:
        raise RuntimeError("fused Sage3 currently requires self-attention Q/K/V shapes")
    b, length, h, d = q.shape
    padded = _padded128(length)
    if out is not None and padded != length:
        raise RuntimeError("out binding requires a 128-aligned sequence length")
    base_input = torch.empty((b, padded, h, d), device=q.device, dtype=q.dtype)
    padded_out = out if padded == length else None
    attention = allocate_workspace(
        base_input, base_input, base_input, out=padded_out,
    )
    groups = padded // TOKEN_ALIGNMENT
    return Sage3FusedWorkspace(
        attention=attention,
        k_centered=torch.empty((b, h, padded, d), device=q.device, dtype=q.dtype),
        q_groupmean=torch.empty((b, h, groups, d), device=q.device, dtype=q.dtype),
        k_mean=torch.empty((b, h, d), device=q.device, dtype=q.dtype),
        delta_bf16=torch.empty((b, h, groups, padded), device=q.device, dtype=q.dtype),
        original_length=length,
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


@torch.library.register_fake(add_op_namespace_prefix("quantize_q_fp4_centered_nhd"))
def _qc_fake(x, mean, packed, sf):
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_k_fp4_centered_nhd"))
def _kc_fake(x, mean, packed, sf, centered_hnd):
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


def sage3_prefill_fp4_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    per_block_mean: bool = True,
    out: torch.Tensor | None = None,
    workspace: Sage3FusedWorkspace | None = None,
) -> torch.Tensor:
    """Run fused Sage3 prep, FP4 attention, and crop from raw BF16 NHD inputs."""
    _check_nhd(q, "q")
    _check_nhd(k, "k")
    _check_nhd(v, "v")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise RuntimeError("sage3_prefill_fp4_bf16 requires BF16 inputs")
    if q.shape != k.shape or q.shape != v.shape:
        raise RuntimeError("fused Sage3 currently requires self-attention Q/K/V shapes")
    if not per_block_mean:
        raise RuntimeError("fused Sage3 currently requires per_block_mean=True")
    workspace = (
        allocate_fused_workspace(q, k, v, out=out)
        if workspace is None else workspace
    )
    if workspace.original_length != q.shape[1]:
        raise RuntimeError("workspace sequence length does not match this call")
    if workspace.attention.out_nhd.device != q.device:
        raise RuntimeError("workspace and inputs must be on the same device")
    if out is not None and out.data_ptr() != workspace.attention.out_nhd.data_ptr():
        raise RuntimeError("out must be bound when allocate_fused_workspace is called")
    b, length, h, d = q.shape
    groups = workspace.q_groupmean.shape[2]
    torch.mean(k, dim=1, out=workspace.k_mean)
    base = workspace.attention
    ops.quantize_q_fp4_centered_nhd(q, workspace.q_groupmean, base.q_packed, base.q_scale)
    ops.quantize_k_fp4_centered_nhd(
        k, workspace.k_mean, base.k_packed, base.k_scale, workspace.k_centered,
    )
    ops.quantize_v_fp4_nhd(v, base.v_packed, base.v_scale)
    b, h, groups, d = workspace.q_groupmean.shape
    padded = workspace.k_centered.shape[2]
    torch.bmm(
        workspace.q_groupmean.view(b * h, groups, d),
        workspace.k_centered.view(b * h, padded, d).transpose(1, 2),
        out=workspace.delta_bf16.view(b * h, groups, padded),
    )
    result = blockscaled_fp4_attention_static(
        base, workspace.delta_bf16, unpadded_k=q.shape[1],
        softmax_scale=softmax_scale, causal=causal, per_block_mean=True,
    )[:, : q.shape[1]]
    return result


__all__ = [
    "ACCURACY_PROFILE",
    "SUPPORTED_HEAD_DIMS",
    "SUPPORTED_LAYOUTS",
    "TOKEN_ALIGNMENT",
    "Sage3Workspace",
    "Sage3FusedWorkspace",
    "capabilities",
    "allocate_workspace",
    "allocate_fused_workspace",
    "prepare_qkv_fp4_nhd",
    "blockscaled_fp4_attention_static",
    "sage3_prefill_fp4_bf16",
]
