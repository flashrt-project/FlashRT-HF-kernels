"""FlashRT Gated Delta attention kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


def _check_step(q, k, v, g, beta, out) -> None:
    if q.dim() != 3 or q.shape[2] != 128:
        raise RuntimeError("q must have shape (B,H,128)")
    if k.shape != q.shape or v.shape != q.shape:
        raise RuntimeError("k/v must match q")
    if g.shape != q.shape[:2] or beta.shape != g.shape:
        raise RuntimeError("g/beta must have shape (B,H)")
    if out.shape != q.shape:
        raise RuntimeError("out must match q")


def _check_chunk(q, k, v, g, beta, out) -> None:
    if q.dim() != 3 or q.shape[2] != 128:
        raise RuntimeError("q must have shape (S,H,128)")
    if k.shape != q.shape or v.shape != q.shape:
        raise RuntimeError("k/v must match q")
    if g.shape != q.shape[:2] or beta.shape != g.shape:
        raise RuntimeError("g/beta must have shape (S,H)")
    if out.shape != q.shape:
        raise RuntimeError("out must match q")


def _check_conv_out(conv_out) -> None:
    if conv_out.dim() != 2 or conv_out.shape[1] != 10240:
        raise RuntimeError("conv_out must have shape (S,10240)")


def _check_q16(x, S: int, name: str) -> None:
    if x.shape != (S, 16, 128):
        raise RuntimeError(f"{name} must have shape (S,16,128)")


def _check_v48(x, S: int, name: str) -> None:
    if x.shape != (S, 48, 128):
        raise RuntimeError(f"{name} must have shape (S,48,128)")


def _check_heads48(x, S: int, name: str) -> None:
    if x.shape != (S, 48):
        raise RuntimeError(f"{name} must have shape (S,48)")


def _chunks(S: int) -> int:
    return (S + 63) // 64


@torch.library.register_fake(add_op_namespace_prefix("gated_delta_recurrent_bf16"))
def _recurrent_fake(q, k, v, g, beta, state, out, use_qk_l2norm: bool = True) -> None:
    _check_step(q, k, v, g, beta, out)
    if state.shape != (q.shape[0], q.shape[1], 128, 128):
        raise RuntimeError("state must have shape (B,H,128,128)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gated_delta_recurrent_inout_bf16"))
def _recurrent_inout_fake(q, k, v, g, beta, state_in, state_out, out, use_qk_l2norm: bool = True) -> None:
    _check_step(q, k, v, g, beta, out)
    if state_in.shape != (q.shape[0], q.shape[1], 128, 128) or state_out.shape != state_in.shape:
        raise RuntimeError("state_in/state_out must have shape (B,H,128,128)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gated_delta_recurrent_f32state_bf16io"))
def _recurrent_f32_fake(q, k, v, g, beta, state_f32, out, use_qk_l2norm: bool = True) -> None:
    _check_step(q, k, v, g, beta, out)
    if state_f32.shape != (q.shape[0], q.shape[1], 128, 128):
        raise RuntimeError("state_f32 must have shape (B,H,128,128)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gated_delta_chunk_bf16"))
def _chunk_fake(q, k, v, g, beta, state, out, use_qk_l2norm: bool = True) -> None:
    _check_chunk(q, k, v, g, beta, out)
    if state.shape != (q.shape[1], 128, 128):
        raise RuntimeError("state must have shape (H,128,128)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gated_delta_chunk_smem_bf16"))
def _chunk_smem_fake(q, k, v, g, beta, state, out, use_qk_l2norm: bool = True) -> None:
    _chunk_fake(q, k, v, g, beta, state, out, use_qk_l2norm)
    return None


@torch.library.register_fake(add_op_namespace_prefix("lin_split_qkv_broadcast_bf16"))
def _split_broadcast_fake(conv_out, q48, k48, v48) -> None:
    _check_conv_out(conv_out)
    S = conv_out.shape[0]
    _check_v48(q48, S, "q48")
    _check_v48(k48, S, "k48")
    _check_v48(v48, S, "v48")
    return None


@torch.library.register_fake(add_op_namespace_prefix("lin_split_qkv_gqa_bf16"))
def _split_gqa_fake(conv_out, q16, k16, v48) -> None:
    _check_conv_out(conv_out)
    S = conv_out.shape[0]
    _check_q16(q16, S, "q16")
    _check_q16(k16, S, "k16")
    _check_v48(v48, S, "v48")
    return None


@torch.library.register_fake(add_op_namespace_prefix("split_q_gate_bf16"))
def _split_q_gate_fake(q_proj, q_pre, gate) -> None:
    if q_proj.dim() != 3 or q_proj.shape[1:] != (24, 512):
        raise RuntimeError("q_proj must have shape (S,24,512)")
    S = q_proj.shape[0]
    if q_pre.shape != (S, 24, 256) or gate.shape != (S, 24 * 256):
        raise RuntimeError("q_pre/gate shapes are invalid")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gdn_gating_bf16"))
def _gating_fake(a, b, neg_exp_A_log, dt_bias, g_out, beta_out) -> None:
    S = a.shape[0]
    _check_heads48(a, S, "a")
    _check_heads48(b, S, "b")
    _check_heads48(g_out, S, "g_out")
    _check_heads48(beta_out, S, "beta_out")
    if neg_exp_A_log.shape != (48,) or dt_bias.shape != (48,):
        raise RuntimeError("neg_exp_A_log/dt_bias must have shape (48)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gdn_gating_strided_bf16"))
def _gating_strided_fake(a, b, neg_exp_A_log, dt_bias, g_out, beta_out, a_stride: int, b_stride: int) -> None:
    _gating_fake(g_out, beta_out, neg_exp_A_log, dt_bias, g_out, beta_out)
    return None


@torch.library.register_fake(add_op_namespace_prefix("gdn_chunk_from_conv_smem_bf16"))
def _chunk_from_conv_fake(conv_out, a, b, neg_exp_A_log, dt_bias, state, out, use_qk_l2norm: bool = True) -> None:
    _check_conv_out(conv_out)
    S = conv_out.shape[0]
    _check_heads48(a, S, "a")
    _check_heads48(b, S, "b")
    if neg_exp_A_log.shape != (48,) or dt_bias.shape != (48,):
        raise RuntimeError("neg_exp_A_log/dt_bias must have shape (48)")
    if state.shape != (48, 128, 128):
        raise RuntimeError("state must have shape (48,128,128)")
    _check_v48(out, S, "out")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gdn_wy_norm_cumsum_pack_qk_bf16"))
def _wy_norm_cumsum_fake(q16, k16, g, q16_l2, k16_l2, q_pack_hv, k_pack_hk, g_cumsum) -> None:
    S = q16.shape[0]
    C = _chunks(S)
    _check_q16(q16, S, "q16")
    _check_q16(k16, S, "k16")
    _check_heads48(g, S, "g")
    _check_q16(q16_l2, S, "q16_l2")
    _check_q16(k16_l2, S, "k16_l2")
    if q_pack_hv.shape != (C, 48, 64, 128) or k_pack_hk.shape != (C, 16, 64, 128):
        raise RuntimeError("packed Q/K tensors have invalid WY shapes")
    _check_heads48(g_cumsum, S, "g_cumsum")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gdn_wy_kkt_b64_bf16"))
def _wy_kkt_fake(k16_l2, beta, g_cumsum, A) -> None:
    S = k16_l2.shape[0]
    _check_q16(k16_l2, S, "k16_l2")
    _check_heads48(beta, S, "beta")
    _check_heads48(g_cumsum, S, "g_cumsum")
    if A.shape != (_chunks(S), 48, 64, 64):
        raise RuntimeError("A must have shape (ceil(S/64),48,64,64)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gdn_wy_solve_tril_b64_f32"))
def _wy_solve_fake(A, Ai, S: int) -> None:
    if A.shape != (_chunks(S), 48, 64, 64) or Ai.shape != A.shape:
        raise RuntimeError("A/Ai must have shape (ceil(S/64),48,64,64)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gdn_wy_recompute_wu_b64_bf16"))
def _wy_recompute_fake(k16_l2, v48, beta, g_cumsum, Ai, w48, u48) -> None:
    S = k16_l2.shape[0]
    _check_q16(k16_l2, S, "k16_l2")
    _check_v48(v48, S, "v48")
    _check_heads48(beta, S, "beta")
    _check_heads48(g_cumsum, S, "g_cumsum")
    if Ai.shape != (_chunks(S), 48, 64, 64):
        raise RuntimeError("Ai must have shape (ceil(S/64),48,64,64)")
    _check_v48(w48, S, "w48")
    _check_v48(u48, S, "u48")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gdn_wy_chunk_h_b64_bf16"))
def _wy_chunk_h_fake(k16_l2, u48, w48, g_cumsum, state, h0, v_new) -> None:
    S = k16_l2.shape[0]
    _check_q16(k16_l2, S, "k16_l2")
    _check_v48(u48, S, "u48")
    _check_v48(w48, S, "w48")
    _check_heads48(g_cumsum, S, "g_cumsum")
    if state.shape != (48, 128, 128) or h0.shape != (_chunks(S), 48, 128, 128):
        raise RuntimeError("state/h0 shapes are invalid")
    _check_v48(v_new, S, "v_new")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gdn_wy_output_o_b64_bf16"))
def _wy_output_fake(q16_l2, k16_l2, v_new, h0, g_cumsum, out) -> None:
    S = q16_l2.shape[0]
    _check_q16(q16_l2, S, "q16_l2")
    _check_q16(k16_l2, S, "k16_l2")
    _check_v48(v_new, S, "v_new")
    if h0.shape != (_chunks(S), 48, 128, 128):
        raise RuntimeError("h0 must have shape (ceil(S/64),48,128,128)")
    _check_heads48(g_cumsum, S, "g_cumsum")
    _check_v48(out, S, "out")
    return None


def gated_delta_recurrent_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    use_qk_l2norm: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(q)
    ops.gated_delta_recurrent_bf16(q, k, v, g, beta, state, out, bool(use_qk_l2norm))
    return out


def gated_delta_recurrent_inout_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_in: torch.Tensor,
    *,
    use_qk_l2norm: bool = True,
    state_out: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out = torch.empty_like(q)
    if state_out is None:
        state_out = torch.empty_like(state_in)
    ops.gated_delta_recurrent_inout_bf16(q, k, v, g, beta, state_in, state_out, out, bool(use_qk_l2norm))
    return out, state_out


def gated_delta_recurrent_f32state_bf16io(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_f32: torch.Tensor,
    *,
    use_qk_l2norm: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(q)
    ops.gated_delta_recurrent_f32state_bf16io(q, k, v, g, beta, state_f32, out, bool(use_qk_l2norm))
    return out


def gated_delta_chunk_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    use_qk_l2norm: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(q)
    ops.gated_delta_chunk_bf16(q, k, v, g, beta, state, out, bool(use_qk_l2norm))
    return out


def gated_delta_chunk_smem_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    use_qk_l2norm: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(q)
    ops.gated_delta_chunk_smem_bf16(q, k, v, g, beta, state, out, bool(use_qk_l2norm))
    return out


def lin_split_qkv_broadcast_bf16(
    conv_out: torch.Tensor,
    *,
    q48: Optional[torch.Tensor] = None,
    k48: Optional[torch.Tensor] = None,
    v48: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    S = conv_out.shape[0]
    if q48 is None:
        q48 = torch.empty((S, 48, 128), device=conv_out.device, dtype=conv_out.dtype)
    if k48 is None:
        k48 = torch.empty_like(q48)
    if v48 is None:
        v48 = torch.empty_like(q48)
    ops.lin_split_qkv_broadcast_bf16(conv_out, q48, k48, v48)
    return q48, k48, v48


def lin_split_qkv_gqa_bf16(
    conv_out: torch.Tensor,
    *,
    q16: Optional[torch.Tensor] = None,
    k16: Optional[torch.Tensor] = None,
    v48: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    S = conv_out.shape[0]
    if q16 is None:
        q16 = torch.empty((S, 16, 128), device=conv_out.device, dtype=conv_out.dtype)
    if k16 is None:
        k16 = torch.empty_like(q16)
    if v48 is None:
        v48 = torch.empty((S, 48, 128), device=conv_out.device, dtype=conv_out.dtype)
    ops.lin_split_qkv_gqa_bf16(conv_out, q16, k16, v48)
    return q16, k16, v48


def split_q_gate_bf16(
    q_proj: torch.Tensor,
    *,
    q_pre: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    S = q_proj.shape[0]
    if q_pre is None:
        q_pre = torch.empty((S, 24, 256), device=q_proj.device, dtype=q_proj.dtype)
    if gate is None:
        gate = torch.empty((S, 24 * 256), device=q_proj.device, dtype=q_proj.dtype)
    ops.split_q_gate_bf16(q_proj, q_pre, gate)
    return q_pre, gate


def gdn_gating_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    neg_exp_A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    g_out: Optional[torch.Tensor] = None,
    beta_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if g_out is None:
        g_out = torch.empty_like(a)
    if beta_out is None:
        beta_out = torch.empty_like(a)
    ops.gdn_gating_bf16(a, b, neg_exp_A_log, dt_bias, g_out, beta_out)
    return g_out, beta_out


def gdn_gating_strided_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    neg_exp_A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    rows: int,
    a_stride: int,
    b_stride: int,
    g_out: Optional[torch.Tensor] = None,
    beta_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if g_out is None:
        g_out = torch.empty((rows, 48), device=a.device, dtype=a.dtype)
    if beta_out is None:
        beta_out = torch.empty_like(g_out)
    ops.gdn_gating_strided_bf16(a, b, neg_exp_A_log, dt_bias, g_out, beta_out, int(a_stride), int(b_stride))
    return g_out, beta_out


def gdn_chunk_from_conv_smem_bf16(
    conv_out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    neg_exp_A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    *,
    use_qk_l2norm: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty((conv_out.shape[0], 48, 128), device=conv_out.device, dtype=conv_out.dtype)
    ops.gdn_chunk_from_conv_smem_bf16(conv_out, a, b, neg_exp_A_log, dt_bias, state, out, bool(use_qk_l2norm))
    return out


def gdn_wy_norm_cumsum_pack_qk_bf16(
    q16: torch.Tensor,
    k16: torch.Tensor,
    g: torch.Tensor,
    *,
    q16_l2: Optional[torch.Tensor] = None,
    k16_l2: Optional[torch.Tensor] = None,
    q_pack_hv: Optional[torch.Tensor] = None,
    k_pack_hk: Optional[torch.Tensor] = None,
    g_cumsum: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    S = q16.shape[0]
    C = _chunks(S)
    if q16_l2 is None:
        q16_l2 = torch.empty_like(q16)
    if k16_l2 is None:
        k16_l2 = torch.empty_like(k16)
    if q_pack_hv is None:
        q_pack_hv = torch.empty((C, 48, 64, 128), device=q16.device, dtype=q16.dtype)
    if k_pack_hk is None:
        k_pack_hk = torch.empty((C, 16, 64, 128), device=q16.device, dtype=q16.dtype)
    if g_cumsum is None:
        g_cumsum = torch.empty_like(g)
    ops.gdn_wy_norm_cumsum_pack_qk_bf16(q16, k16, g, q16_l2, k16_l2, q_pack_hv, k_pack_hk, g_cumsum)
    return q16_l2, k16_l2, q_pack_hv, k_pack_hk, g_cumsum


def gdn_wy_kkt_b64_bf16(
    k16_l2: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    *,
    A: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    S = k16_l2.shape[0]
    if A is None:
        A = torch.empty((_chunks(S), 48, 64, 64), device=k16_l2.device, dtype=torch.float32)
    ops.gdn_wy_kkt_b64_bf16(k16_l2, beta, g_cumsum, A)
    return A


def gdn_wy_solve_tril_b64_f32(
    A: torch.Tensor,
    S: int,
    *,
    Ai: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if Ai is None:
        Ai = torch.empty_like(A)
    ops.gdn_wy_solve_tril_b64_f32(A, Ai, int(S))
    return Ai


def gdn_wy_recompute_wu_b64_bf16(
    k16_l2: torch.Tensor,
    v48: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    Ai: torch.Tensor,
    *,
    w48: Optional[torch.Tensor] = None,
    u48: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if w48 is None:
        w48 = torch.empty_like(v48)
    if u48 is None:
        u48 = torch.empty_like(v48)
    ops.gdn_wy_recompute_wu_b64_bf16(k16_l2, v48, beta, g_cumsum, Ai, w48, u48)
    return w48, u48


def gdn_wy_chunk_h_b64_bf16(
    k16_l2: torch.Tensor,
    u48: torch.Tensor,
    w48: torch.Tensor,
    g_cumsum: torch.Tensor,
    state: torch.Tensor,
    *,
    h0: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    S = k16_l2.shape[0]
    if h0 is None:
        h0 = torch.empty((_chunks(S), 48, 128, 128), device=k16_l2.device, dtype=k16_l2.dtype)
    if v_new is None:
        v_new = torch.empty_like(u48)
    ops.gdn_wy_chunk_h_b64_bf16(k16_l2, u48, w48, g_cumsum, state, h0, v_new)
    return h0, v_new


def gdn_wy_output_o_b64_bf16(
    q16_l2: torch.Tensor,
    k16_l2: torch.Tensor,
    v_new: torch.Tensor,
    h0: torch.Tensor,
    g_cumsum: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty((q16_l2.shape[0], 48, 128), device=q16_l2.device, dtype=q16_l2.dtype)
    ops.gdn_wy_output_o_b64_bf16(q16_l2, k16_l2, v_new, h0, g_cumsum, out)
    return out


__all__ = [
    "gated_delta_recurrent_bf16",
    "gated_delta_recurrent_inout_bf16",
    "gated_delta_recurrent_f32state_bf16io",
    "gated_delta_chunk_bf16",
    "gated_delta_chunk_smem_bf16",
    "lin_split_qkv_broadcast_bf16",
    "lin_split_qkv_gqa_bf16",
    "split_q_gate_bf16",
    "gdn_gating_bf16",
    "gdn_gating_strided_bf16",
    "gdn_chunk_from_conv_smem_bf16",
    "gdn_wy_norm_cumsum_pack_qk_bf16",
    "gdn_wy_kkt_b64_bf16",
    "gdn_wy_solve_tril_b64_f32",
    "gdn_wy_recompute_wu_b64_bf16",
    "gdn_wy_chunk_h_b64_bf16",
    "gdn_wy_output_o_b64_bf16",
]
