"""FlashRT transformer fused helper kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


def _same_shape_fake(name: str):
    @torch.library.register_fake(add_op_namespace_prefix(name))
    def _fake(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        if a.shape != b.shape or out.shape != a.shape:
            raise RuntimeError(f"{name} expects identical input/output shapes")
        return None


@torch.library.register_fake(add_op_namespace_prefix("rms_norm_gated_silu_bf16"))
def _rms_norm_gated_silu_fake(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> None:
    if x.dim() != 2 or gate.shape != x.shape or out.shape != x.shape or weight.shape != (x.shape[1],):
        raise RuntimeError("expected x/gate/out (rows,dim), weight (dim,)")
    if x.shape[1] != 128:
        raise RuntimeError("rms_norm_gated_silu_bf16 supports dim=128")
    return None


_same_shape_fake("silu_mul_bf16")
_same_shape_fake("sigmoid_mul_bf16")


@torch.library.register_fake(add_op_namespace_prefix("embedding_lookup_bf16"))
def _embedding_lookup_fake(token_ids: torch.Tensor, embed: torch.Tensor, out: torch.Tensor) -> None:
    if token_ids.dim() != 1 or embed.dim() != 2 or out.shape != (token_ids.shape[0], embed.shape[1]):
        raise RuntimeError("expected token_ids (rows,), embed (vocab,hidden), out (rows,hidden)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("partial_rope_qk_bf16"))
def _partial_rope_fake(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    rope_dim: int,
) -> None:
    if q_in.dim() != 3 or k_in.dim() != 3 or q_in.shape[0] != k_in.shape[0] or q_in.shape[2] != k_in.shape[2]:
        raise RuntimeError("q/k must be (rows,heads,head_dim) with shared rows/head_dim")
    if cos.shape != (q_in.shape[0], rope_dim) or sin.shape != cos.shape:
        raise RuntimeError("cos/sin shape mismatch")
    if q_out.shape != q_in.shape or k_out.shape != k_in.shape:
        raise RuntimeError("output shape mismatch")
    return None


@torch.library.register_fake(add_op_namespace_prefix("argmax_bf16"))
def _argmax_fake(logits: torch.Tensor, argmax_out: torch.Tensor) -> None:
    if logits.dim() != 2 or argmax_out.shape != (logits.shape[0],):
        raise RuntimeError("expected logits (rows,vocab), argmax_out (rows,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("spec_accept_greedy_bf16"))
def _spec_accept_fake(
    logits: torch.Tensor,
    drafts: torch.Tensor,
    argmax_out: torch.Tensor,
    accept_n: torch.Tensor,
    spec_k: int,
) -> None:
    if logits.dim() != 2 or argmax_out.shape != (logits.shape[0],) or accept_n.numel() < 1:
        raise RuntimeError("invalid spec accept shapes")
    return None


@torch.library.register_fake(add_op_namespace_prefix("nexn2_lin_split_qkv_broadcast_bf16"))
def _nexn2_lin_split_fake(conv_out: torch.Tensor, q32: torch.Tensor, k32: torch.Tensor, v32: torch.Tensor) -> None:
    if conv_out.dim() != 2 or conv_out.shape[1] != 8192:
        raise RuntimeError("conv_out must have shape (S,8192)")
    expected = (conv_out.shape[0], 32, 128)
    if q32.shape != expected or k32.shape != expected or v32.shape != expected:
        raise RuntimeError("q/k/v outputs must have shape (S,32,128)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("nexn2_split_q_gate_bf16"))
def _nexn2_split_q_gate_fake(q_proj: torch.Tensor, q_pre: torch.Tensor, gate: torch.Tensor) -> None:
    if q_proj.dim() != 3 or q_proj.shape[1:] != (16, 512):
        raise RuntimeError("q_proj must have shape (S,16,512)")
    if q_pre.shape != (q_proj.shape[0], 16, 256) or gate.shape != (q_proj.shape[0], 16 * 256):
        raise RuntimeError("q_pre/gate shape mismatch")
    return None


@torch.library.register_fake(add_op_namespace_prefix("nexn2_router_topk_bf16"))
def _nexn2_router_topk_fake(logits: torch.Tensor, out_idx: torch.Tensor, out_val: torch.Tensor, k: int) -> None:
    if logits.dim() != 1 or out_idx.shape != (k,) or out_val.shape != (k,):
        raise RuntimeError("expected logits (n_experts,), out_idx/out_val (k,)")
    return None


def rms_norm_gated_silu_bf16(x, gate, weight, *, eps: float = 1e-6, out: Optional[torch.Tensor] = None):
    if out is None:
        out = torch.empty_like(x)
    ops.rms_norm_gated_silu_bf16(x, gate, weight, float(eps), out)
    return out


def silu_mul_bf16(gate, up, *, out: Optional[torch.Tensor] = None):
    if out is None:
        out = torch.empty_like(gate)
    ops.silu_mul_bf16(gate, up, out)
    return out


def sigmoid_mul_bf16(gate, x, *, out: Optional[torch.Tensor] = None):
    if out is None:
        out = torch.empty_like(gate)
    ops.sigmoid_mul_bf16(gate, x, out)
    return out


def embedding_lookup_bf16(token_ids, embed, *, out: Optional[torch.Tensor] = None):
    if out is None:
        out = torch.empty((token_ids.shape[0], embed.shape[1]), device=embed.device, dtype=torch.bfloat16)
    ops.embedding_lookup_bf16(token_ids, embed, out)
    return out


def partial_rope_qk_bf16(q_in, k_in, cos, sin, rope_dim: int, *, q_out=None, k_out=None):
    if q_out is None:
        q_out = torch.empty_like(q_in)
    if k_out is None:
        k_out = torch.empty_like(k_in)
    ops.partial_rope_qk_bf16(q_in, k_in, cos, sin, q_out, k_out, int(rope_dim))
    return q_out, k_out


def argmax_bf16(logits, *, out: Optional[torch.Tensor] = None):
    if out is None:
        out = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
    ops.argmax_bf16(logits, out)
    return out


def spec_accept_greedy_bf16(logits, drafts, spec_k: int, *, argmax_out=None, accept_n=None):
    if argmax_out is None:
        argmax_out = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
    if accept_n is None:
        accept_n = torch.empty((1,), device=logits.device, dtype=torch.int32)
    ops.spec_accept_greedy_bf16(logits, drafts, argmax_out, accept_n, int(spec_k))
    return argmax_out, accept_n


def nexn2_lin_split_qkv_broadcast_bf16(conv_out, *, q32=None, k32=None, v32=None):
    shape = (conv_out.shape[0], 32, 128)
    if q32 is None:
        q32 = torch.empty(shape, device=conv_out.device, dtype=torch.bfloat16)
    if k32 is None:
        k32 = torch.empty(shape, device=conv_out.device, dtype=torch.bfloat16)
    if v32 is None:
        v32 = torch.empty(shape, device=conv_out.device, dtype=torch.bfloat16)
    ops.nexn2_lin_split_qkv_broadcast_bf16(conv_out, q32, k32, v32)
    return q32, k32, v32


def nexn2_split_q_gate_bf16(q_proj, *, q_pre=None, gate=None):
    if q_pre is None:
        q_pre = torch.empty((q_proj.shape[0], 16, 256), device=q_proj.device, dtype=torch.bfloat16)
    if gate is None:
        gate = torch.empty((q_proj.shape[0], 16 * 256), device=q_proj.device, dtype=torch.bfloat16)
    ops.nexn2_split_q_gate_bf16(q_proj, q_pre, gate)
    return q_pre, gate


def nexn2_router_topk_bf16(logits, k: int = 8, *, out_idx=None, out_val=None):
    if out_idx is None:
        out_idx = torch.empty((k,), device=logits.device, dtype=torch.int32)
    if out_val is None:
        out_val = torch.empty((k,), device=logits.device, dtype=torch.float32)
    ops.nexn2_router_topk_bf16(logits, out_idx, out_val, int(k))
    return out_idx, out_val


__all__ = [
    "argmax_bf16",
    "embedding_lookup_bf16",
    "nexn2_lin_split_qkv_broadcast_bf16",
    "nexn2_router_topk_bf16",
    "nexn2_split_q_gate_bf16",
    "partial_rope_qk_bf16",
    "rms_norm_gated_silu_bf16",
    "sigmoid_mul_bf16",
    "silu_mul_bf16",
    "spec_accept_greedy_bf16",
]
