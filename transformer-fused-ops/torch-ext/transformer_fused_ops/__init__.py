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


@torch.library.register_fake(add_op_namespace_prefix("rms_norm_gated_silu_quant_fp4_bf16"))
def _rms_norm_gated_silu_quant_fp4_fake(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    packed: torch.Tensor,
    sfa: torch.Tensor,
) -> None:
    del eps
    if x.dim() != 2 or x.shape[1] != 128:
        raise RuntimeError("x must have shape (rows,128)")
    if gate.shape != x.shape or out.shape != x.shape or weight.shape != (128,):
        raise RuntimeError("gate/out/weight shape contract failed")
    if packed.shape != (1, x.numel() // 2) or sfa.dim() != 1:
        raise RuntimeError("packed/SFA output shape contract failed")
    return None


_same_shape_fake("silu_mul_bf16")
_same_shape_fake("sigmoid_mul_bf16")


@torch.library.register_fake(add_op_namespace_prefix("per_head_sigmoid_gate_bf16"))
def _per_head_sigmoid_gate_fake(x, gate, out) -> None:
    if x.dim() != 4 or gate.shape != x.shape[:3] or out.shape != x.shape:
        raise RuntimeError(
            "expected x/out (batch,sequence,heads,head_dim) and "
            "gate (batch,sequence,heads)"
        )
    return None


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


@torch.library.register_fake(add_op_namespace_prefix("router_topk_bf16"))
def _router_topk_fake(logits: torch.Tensor, out_idx: torch.Tensor, out_val: torch.Tensor, k: int) -> None:
    if logits.dim() != 1 or out_idx.shape != (k,) or out_val.shape != (k,):
        raise RuntimeError("expected logits (n_experts,), out_idx/out_val (k,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("moe_weighted_sum_bf16_to_fp32"))
def _moe_weighted_sum_fake(
    expert_output: torch.Tensor,
    row_indices: torch.Tensor,
    router_weight: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if expert_output.dim() != 2 or row_indices.dim() != 2:
        raise RuntimeError("expected expert_output (routed_rows,stride) and row_indices (tokens,topk)")
    if router_weight.shape != row_indices.shape or out.shape[0] != row_indices.shape[0]:
        raise RuntimeError("router_weight or output shape mismatch")
    if out.dim() != 2 or expert_output.shape[1] < out.shape[1]:
        raise RuntimeError("expert output stride must cover output hidden size")
    return None


@torch.library.register_fake(add_op_namespace_prefix("relu2_quantize_fp8_static_bf16"))
def _relu2_quantize_fp8_static_fake(
    input: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if output.shape != input.shape or scale.numel() != 1:
        raise RuntimeError("output must match input and scale must be scalar")
    return None


@torch.library.register_fake(add_op_namespace_prefix("rms_norm_fp16"))
@torch.library.register_fake(add_op_namespace_prefix("rms_norm_fp16_vec"))
def _rms_norm_fp16_fake(x, weight, eps: float, out) -> None:
    if x.dim() != 2 or weight.shape != (x.shape[1],) or out.shape != x.shape:
        raise RuntimeError("expected x/out (rows,dim), weight (dim,)")


@torch.library.register_fake(add_op_namespace_prefix("layer_norm_fp16"))
@torch.library.register_fake(add_op_namespace_prefix("layer_norm_fp16_vec"))
def _layer_norm_fp16_fake(x, weight, bias, eps: float, out) -> None:
    if x.dim() != 2 or weight.shape != (x.shape[1],) or bias.shape != weight.shape or out.shape != x.shape:
        raise RuntimeError("expected x/out (rows,dim), weight/bias (dim,)")


@torch.library.register_fake(add_op_namespace_prefix("layer_norm_quant_fp8_static_fp16"))
@torch.library.register_fake(add_op_namespace_prefix("layer_norm_fp8_static_fp16_vec"))
def _layer_norm_quant_fp8_static_fp16_fake(x, weight, bias, scale, eps: float, out) -> None:
    if x.dim() != 2 or weight.shape != (x.shape[1],) or bias.shape != weight.shape or out.shape != x.shape:
        raise RuntimeError("expected x/out (rows,dim), weight/bias (dim,)")
    if scale.numel() != 1:
        raise RuntimeError("scale must contain one value")


@torch.library.register_fake(add_op_namespace_prefix("rope_rotate_half_fp16_"))
@torch.library.register_fake(add_op_namespace_prefix("rope_rotate_half_fp16_vec"))
def _rope_rotate_half_fp16_fake(x, cos, sin) -> None:
    if x.dim() != 3 or cos.shape != (x.shape[0], x.shape[2]) or sin.shape != cos.shape:
        raise RuntimeError("expected x (sequence,heads,head_dim), cos/sin (sequence,head_dim)")


@torch.library.register_fake(add_op_namespace_prefix("quantize_fp8_static_fp16"))
@torch.library.register_fake(add_op_namespace_prefix("quantize_fp8_static_fp16_vec"))
def _quantize_fp8_static_fp16_fake(x, scale, out) -> None:
    if scale.numel() != 1 or out.shape != x.shape:
        raise RuntimeError("output must match x and scale must contain one value")


@torch.library.register_fake(add_op_namespace_prefix("quantize_fp8_static_bf16"))
def _quantize_fp8_static_bf16_fake(x, scale, out) -> None:
    if scale.numel() != 1 or out.shape != x.shape:
        raise RuntimeError("output must match x and scale must contain one value")


@torch.library.register_fake(add_op_namespace_prefix("layer_norm_quant_fp8_static_bf16"))
def _layer_norm_quant_fp8_static_bf16_fake(x, weight, bias, scale, eps: float, out) -> None:
    if x.dim() != 2 or weight.shape != (x.shape[1],) or bias.shape != weight.shape:
        raise RuntimeError("expected x (rows,dim), weight/bias (dim,)")
    if scale.numel() != 1 or out.shape != x.shape:
        raise RuntimeError("output must match x and scale must contain one value")


@torch.library.register_fake(add_op_namespace_prefix("gate_geglu_merged_quant_fp8_static_bf16"))
def _gate_geglu_merged_quant_fp8_static_bf16_fake(merged, scale, out) -> None:
    if merged.dim() != 2 or merged.shape[1] % 2:
        raise RuntimeError("merged must have shape (rows, 2 * hidden)")
    if scale.numel() != 1 or out.shape != (merged.shape[0], merged.shape[1] // 2):
        raise RuntimeError("output must have shape (rows, hidden) and scale must be scalar")


@torch.library.register_fake(add_op_namespace_prefix("residual_add_fp16_"))
@torch.library.register_fake(add_op_namespace_prefix("residual_add_fp16_vec"))
def _residual_add_fp16_fake(residual, x) -> None:
    if residual.shape != x.shape:
        raise RuntimeError("residual and x must have the same shape")


@torch.library.register_fake(add_op_namespace_prefix("repeat_interleave_heads_fp16"))
@torch.library.register_fake(add_op_namespace_prefix("gpu_repeat_interleave_heads_vec"))
def _repeat_interleave_heads_fp16_fake(x, repeat: int, out) -> None:
    if x.dim() != 3 or out.shape != (x.shape[0], x.shape[1] * repeat, x.shape[2]):
        raise RuntimeError("output shape must be (sequence, heads * repeat, head_dim)")


def rms_norm_gated_silu_bf16(x, gate, weight, *, eps: float = 1e-6, out: Optional[torch.Tensor] = None):
    if out is None:
        out = torch.empty_like(x)
    ops.rms_norm_gated_silu_bf16(x, gate, weight, float(eps), out)
    return out


def rms_norm_gated_silu_quant_fp4_bf16(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    packed: Optional[torch.Tensor] = None,
    sfa: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = x.shape[0]
    if out is None:
        out = torch.empty_like(x)
    if packed is None:
        packed = torch.empty((1, x.numel() // 2), device=x.device, dtype=torch.uint8)
    if sfa is None:
        sfa = torch.zeros((rows * 1024,), device=x.device, dtype=torch.uint8)
    ops.rms_norm_gated_silu_quant_fp4_bf16(
        x, gate, weight, float(eps), out, packed, sfa
    )
    return out, packed, sfa


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


def per_head_sigmoid_gate_bf16(
    x: torch.Tensor,
    gate: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply ``x * (2 * sigmoid(gate))`` with a gate per NHD head."""
    if out is None:
        out = torch.empty_like(x)
    ops.per_head_sigmoid_gate_bf16(x, gate, out)
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


def router_topk_bf16(logits, k: int = 8, *, out_idx=None, out_val=None):
    """Return exact top-k expert logits and indices for one routing row."""
    if out_idx is None:
        out_idx = torch.empty((k,), device=logits.device, dtype=torch.int32)
    if out_val is None:
        out_val = torch.empty((k,), device=logits.device, dtype=torch.float32)
    ops.router_topk_bf16(logits, out_idx, out_val, int(k))
    return out_idx, out_val


def moe_weighted_sum_bf16_to_fp32(
    expert_output,
    row_indices,
    router_weight,
    *,
    hidden: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
):
    """Gather routed BF16 expert rows and reduce them in FP32 token order."""
    hidden = expert_output.shape[1] if hidden is None else int(hidden)
    if out is None:
        out = torch.empty(
            (row_indices.shape[0], hidden),
            device=expert_output.device,
            dtype=torch.float32,
        )
    ops.moe_weighted_sum_bf16_to_fp32(
        expert_output, row_indices, router_weight, out
    )
    return out


def relu2_quantize_fp8_static_bf16(
    input: torch.Tensor,
    scale: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``relu(input) ** 2 / scale`` and quantize to FP8 E4M3."""

    if out is None:
        out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    ops.relu2_quantize_fp8_static_bf16(input, scale, out)
    return out


def rms_norm_fp16(x, weight, *, eps: float = 1e-6, out=None):
    if out is None:
        out = torch.empty_like(x)
    ops.rms_norm_fp16(x, weight, float(eps), out)
    return out


def rms_norm_fp16_vec(x, weight, *, eps: float = 1e-6, out=None):
    if out is None:
        out = torch.empty_like(x)
    ops.rms_norm_fp16_vec(x, weight, float(eps), out)
    return out


def layer_norm_fp16(x, weight, bias, *, eps: float = 1e-6, out=None):
    if out is None:
        out = torch.empty_like(x)
    ops.layer_norm_fp16(x, weight, bias, float(eps), out)
    return out


def layer_norm_fp16_vec(x, weight, bias, *, eps: float = 1e-6, out=None):
    if out is None:
        out = torch.empty_like(x)
    ops.layer_norm_fp16_vec(x, weight, bias, float(eps), out)
    return out


def layer_norm_quant_fp8_static_fp16(
    x, weight, bias, scale, *, eps: float = 1e-6, out=None
):
    if out is None:
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    ops.layer_norm_quant_fp8_static_fp16(
        x, weight, bias, scale, float(eps), out
    )
    return out


def layer_norm_fp8_static_fp16_vec(
    x, weight, bias, scale, *, eps: float = 1e-6, out=None
):
    if out is None:
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    ops.layer_norm_fp8_static_fp16_vec(
        x, weight, bias, scale, float(eps), out
    )
    return out


def rope_rotate_half_fp16_(x, cos, sin):
    ops.rope_rotate_half_fp16_(x, cos, sin)
    return x


def rope_rotate_half_fp16_vec(x, cos, sin):
    ops.rope_rotate_half_fp16_vec(x, cos, sin)
    return x


def quantize_fp8_static_fp16(x, scale, *, out=None):
    if out is None:
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    ops.quantize_fp8_static_fp16(x, scale, out)
    return out


def quantize_fp8_static_fp16_vec(x, scale, *, out=None):
    if out is None:
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    ops.quantize_fp8_static_fp16_vec(x, scale, out)
    return out


def quantize_fp8_static_bf16(x, scale, *, out=None):
    """Quantize contiguous BF16 input with a static FP32 scale."""
    if out is None:
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    ops.quantize_fp8_static_bf16(x, scale, out)
    return out


def layer_norm_quant_fp8_static_bf16(
    x, weight, bias, scale, *, eps: float = 1e-6, out=None
):
    """LayerNorm BF16 input and emit static-scale FP8 without an intermediate."""
    if out is None:
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    ops.layer_norm_quant_fp8_static_bf16(
        x, weight, bias, scale, float(eps), out
    )
    return out


def gate_geglu_merged_quant_fp8_static_bf16(merged, scale, *, out=None):
    """Apply tanh-approximate GeGLU to merged BF16 gate/up and emit FP8."""
    if out is None:
        out = torch.empty(
            (merged.shape[0], merged.shape[1] // 2),
            device=merged.device,
            dtype=torch.float8_e4m3fn,
        )
    ops.gate_geglu_merged_quant_fp8_static_bf16(merged, scale, out)
    return out


def residual_add_fp16_(residual, x):
    ops.residual_add_fp16_(residual, x)
    return residual


def residual_add_fp16_vec(residual, x):
    ops.residual_add_fp16_vec(residual, x)
    return residual


def repeat_interleave_heads_fp16(x, repeat: int, *, out=None):
    if out is None:
        out = torch.empty(
            (x.shape[0], x.shape[1] * repeat, x.shape[2]),
            device=x.device,
            dtype=x.dtype,
        )
    ops.repeat_interleave_heads_fp16(x, int(repeat), out)
    return out


def gpu_repeat_interleave_heads_vec(x, repeat: int, *, out=None):
    if out is None:
        out = torch.empty(
            (x.shape[0], x.shape[1] * repeat, x.shape[2]),
            device=x.device,
            dtype=x.dtype,
        )
    ops.gpu_repeat_interleave_heads_vec(x, int(repeat), out)
    return out


__all__ = [
    "argmax_bf16",
    "embedding_lookup_bf16",
    "nexn2_lin_split_qkv_broadcast_bf16",
    "nexn2_router_topk_bf16",
    "nexn2_split_q_gate_bf16",
    "partial_rope_qk_bf16",
    "rms_norm_gated_silu_bf16",
    "rms_norm_gated_silu_quant_fp4_bf16",
    "sigmoid_mul_bf16",
    "per_head_sigmoid_gate_bf16",
    "silu_mul_bf16",
    "spec_accept_greedy_bf16",
    "relu2_quantize_fp8_static_bf16",
    "router_topk_bf16",
    "moe_weighted_sum_bf16_to_fp32",
    "layer_norm_fp16",
    "layer_norm_fp16_vec",
    "layer_norm_fp8_static_fp16_vec",
    "layer_norm_quant_fp8_static_fp16",
    "quantize_fp8_static_fp16",
    "quantize_fp8_static_fp16_vec",
    "quantize_fp8_static_bf16",
    "layer_norm_quant_fp8_static_bf16",
    "gate_geglu_merged_quant_fp8_static_bf16",
    "repeat_interleave_heads_fp16",
    "gpu_repeat_interleave_heads_vec",
    "residual_add_fp16_",
    "residual_add_fp16_vec",
    "rms_norm_fp16",
    "rms_norm_fp16_vec",
    "rope_rotate_half_fp16_",
    "rope_rotate_half_fp16_vec",
]
