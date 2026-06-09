#!/usr/bin/env python3
"""PI0.5-shaped runtime demo using public FlashRT HF Kernel Hub packages."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from kernels import get_kernel


@dataclass(frozen=True)
class Profile:
    name: str
    video_tokens: int
    action_tokens: int
    text_tokens: int
    heads: int
    decoder_kv_heads: int
    head_dim: int
    latent_c: int
    latent_t: int
    latent_h: int
    latent_w: int
    ffn_hidden: int
    max_seq_len: int

    @property
    def dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def total_tokens(self) -> int:
        return self.video_tokens + self.action_tokens + self.text_tokens


PROFILES = {
    "small": Profile(
        name="small",
        video_tokens=64,
        action_tokens=8,
        text_tokens=8,
        heads=4,
        decoder_kv_heads=1,
        head_dim=128,
        latent_c=16,
        latent_t=4,
        latent_h=8,
        latent_w=8,
        ffn_hidden=1024,
        max_seq_len=256,
    ),
    "pi05_hotpath": Profile(
        name="pi05_hotpath",
        video_tokens=512,
        action_tokens=10,
        text_tokens=48,
        heads=8,
        decoder_kv_heads=1,
        head_dim=128,
        latent_c=64,
        latent_t=4,
        latent_h=32,
        latent_w=32,
        ffn_hidden=4096,
        max_seq_len=4096,
    ),
    "pi05_decoder_hotpath": Profile(
        name="pi05_decoder_hotpath",
        video_tokens=512,
        action_tokens=10,
        text_tokens=48,
        heads=8,
        decoder_kv_heads=1,
        head_dim=128,
        latent_c=64,
        latent_t=4,
        latent_h=32,
        latent_w=32,
        ffn_hidden=4096,
        max_seq_len=4096,
    ),
    "vla_video_hotpath": Profile(
        name="vla_video_hotpath",
        video_tokens=512,
        action_tokens=32,
        text_tokens=16,
        heads=24,
        decoder_kv_heads=1,
        head_dim=128,
        latent_c=64,
        latent_t=4,
        latent_h=32,
        latent_w=32,
        ffn_hidden=8192,
        max_seq_len=4096,
    ),
}


@dataclass
class Result:
    profile: str
    layers: int
    device: str
    torch_version: str
    cuda_version: str | None
    eager_us: float
    runtime_us: float
    graph_us: float | None
    graph_with_input_copy_us: float | None
    runtime_vs_eager: float
    graph_vs_eager: float | None
    graph_status: str
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    expected_rms: float
    cosine_gate_enabled: bool
    ffn_activation: str
    ffn_source: str
    qkv_source: str
    attention_backend: str
    decoder_qkv_backend: str


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float().reshape(()), -448.0, 448.0).to(
        torch.float8_e4m3fn
    )


def dequant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.float() * scale.float().reshape(())


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    return xf * inv * weight.float()


def pair_rope_rotate(x: torch.Tensor, freqs_re: torch.Tensor, freqs_im: torch.Tensor) -> torch.Tensor:
    # qkv_split_joint3_cat_bf16 uses adjacent even/odd RoPE pairs for the
    # video segment. Decode staging uses the rotate-half contract below.
    bsz, seq, heads, head_dim = x.shape
    xf = x.float().reshape(bsz, seq, heads, head_dim // 2, 2)
    re = xf[..., 0]
    im = xf[..., 1]
    cos = freqs_re[:seq].view(1, seq, 1, head_dim // 2).float()
    sin = freqs_im[:seq].view(1, seq, 1, head_dim // 2).float()
    out = torch.empty_like(xf)
    out[..., 0] = re * cos - im * sin
    out[..., 1] = re * sin + im * cos
    return out.reshape(bsz, seq, heads, head_dim).to(torch.bfloat16)


def interleaved_pair_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    bsz, seq, heads, head_dim = x.shape
    xf = x.float().reshape(bsz, seq, heads, head_dim // 2, 2)
    re = xf[..., 0]
    im = xf[..., 1]
    rope_pair = rope[:seq].float().reshape(seq, head_dim // 2, 2)
    cos = rope_pair[..., 0].view(1, seq, 1, head_dim // 2)
    sin = rope_pair[..., 1].view(1, seq, 1, head_dim // 2)
    out = torch.empty_like(xf)
    out[..., 0] = re * cos - im * sin
    out[..., 1] = re * sin + im * cos
    return out.reshape(bsz, seq, heads, head_dim).to(torch.bfloat16)


def decode_rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    xf = x.float()
    x0 = xf[:, :half]
    x1 = xf[:, half:]
    c = cos.view(1, half).float()
    s = sin.view(1, half).float()
    return torch.cat([x0 * c - x1 * s, x1 * c + x0 * s], dim=-1).to(torch.bfloat16)


def percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def time_us(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def sdpa_attention_bthd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Non-causal attention for tensors shaped [B, T, H, D]."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    )
    return out.transpose(1, 2).contiguous()


class RuntimeState:
    def __init__(self, profile: Profile, layers: int, device: torch.device) -> None:
        self.p = profile
        self.layers = layers
        d = profile.dim
        h = profile.ffn_hidden
        total = profile.total_tokens
        v = profile.video_tokens
        a = profile.action_tokens
        u = profile.text_tokens

        self.latent = torch.randn(
            (1, profile.latent_c, profile.latent_t, profile.latent_h, profile.latent_w),
            device=device,
            dtype=torch.bfloat16,
        )
        self.latent_2c = torch.randn(
            (1, 2 * profile.latent_c, profile.latent_t, profile.latent_h, profile.latent_w),
            device=device,
            dtype=torch.bfloat16,
        )
        self.latent_bias = torch.randn((profile.latent_c,), device=device, dtype=torch.bfloat16)
        self.latent_prev = torch.randn(
            (1, profile.latent_c, 2, profile.latent_h, profile.latent_w),
            device=device,
            dtype=torch.bfloat16,
        )

        self.packed_v = torch.randn((1, v, 3 * d), device=device, dtype=torch.bfloat16)
        self.packed_a = torch.randn((1, a, 3 * d), device=device, dtype=torch.bfloat16)
        self.packed_u = torch.randn((1, u, 3 * d), device=device, dtype=torch.bfloat16)
        self.qkv_v_bias = (0.01 * torch.randn((3 * d,), device=device)).to(torch.bfloat16)
        self.q_w_v = (1.0 + 0.01 * torch.randn((d,), device=device)).to(torch.bfloat16)
        self.k_w_v = (1.0 + 0.01 * torch.randn((d,), device=device)).to(torch.bfloat16)
        self.q_w_a = (1.0 + 0.01 * torch.randn((d,), device=device)).to(torch.bfloat16)
        self.k_w_a = (1.0 + 0.01 * torch.randn((d,), device=device)).to(torch.bfloat16)
        self.q_w_u = (1.0 + 0.01 * torch.randn((d,), device=device)).to(torch.bfloat16)
        self.k_w_u = (1.0 + 0.01 * torch.randn((d,), device=device)).to(torch.bfloat16)
        theta = torch.randn((max(v, 1), profile.head_dim // 2), device=device)
        self.freqs_re = torch.cos(theta).contiguous()
        self.freqs_im = torch.sin(theta).contiguous()

        self.q_pre = torch.randn((profile.heads, profile.head_dim), device=device, dtype=torch.bfloat16)
        self.k_pre = torch.randn_like(self.q_pre)
        self.v_pre = torch.randn_like(self.q_pre)
        self.q_norm_decode = (1.0 + 0.01 * torch.randn((profile.head_dim,), device=device)).to(torch.bfloat16)
        self.k_norm_decode = (1.0 + 0.01 * torch.randn((profile.head_dim,), device=device)).to(torch.bfloat16)
        theta_decode = torch.randn((profile.head_dim // 2,), device=device)
        self.cos_decode = torch.cos(theta_decode).to(torch.bfloat16).contiguous()
        self.sin_decode = torch.sin(theta_decode).to(torch.bfloat16).contiguous()
        self.cur_pos = torch.tensor([7], device=device, dtype=torch.int32)

        decoder_qkv_dim = (profile.heads + 2 * profile.decoder_kv_heads) * profile.head_dim
        self.decoder_packed_qkv = torch.randn(
            (1, a, decoder_qkv_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        theta_decoder_seq = torch.randn((a, profile.head_dim // 2), device=device)
        decoder_cos = torch.cos(theta_decoder_seq).to(torch.bfloat16)
        decoder_sin = torch.sin(theta_decoder_seq).to(torch.bfloat16)
        self.decoder_rope = torch.stack([decoder_cos, decoder_sin], dim=-1).reshape(
            a, profile.head_dim
        ).contiguous()
        self.decoder_cache_offset = 7

        self.v_res = torch.randn((v, d), device=device, dtype=torch.bfloat16)
        self.v_x = torch.randn_like(self.v_res)
        self.v_gate = torch.randn_like(self.v_res)
        self.v_bias = (0.01 * torch.randn((d,), device=device)).to(torch.bfloat16)
        self.a_res = torch.randn((a, d), device=device, dtype=torch.bfloat16)
        self.a_x = torch.randn_like(self.a_res)
        self.a_gate = torch.randn_like(self.a_res)
        self.a_bias = (0.01 * torch.randn((d,), device=device)).to(torch.bfloat16)
        self.u_res = torch.randn((u, d), device=device, dtype=torch.bfloat16)
        self.u_x = torch.randn_like(self.u_res)

        self.ada_weight = (1.0 + 0.01 * torch.randn((d,), device=device)).to(torch.bfloat16)
        self.ada_style = (0.02 * torch.randn((total, 3 * d), device=device)).to(torch.bfloat16)
        self.ada_x = torch.randn((total, d), device=device, dtype=torch.bfloat16)
        self.ada_gate_mul = torch.randn((total, d), device=device, dtype=torch.bfloat16)
        self.fp8_scale = torch.tensor([0.04], device=device, dtype=torch.float32)

        self.ffn_input_scale = torch.tensor([0.04], device=device, dtype=torch.float32)
        self.ffn_hidden_scale = torch.tensor([0.04], device=device, dtype=torch.float32)
        self.ffn_gate_up_w_scale = torch.tensor([0.035], device=device, dtype=torch.float32)
        self.ffn_down_w_scale = torch.tensor([0.035], device=device, dtype=torch.float32)
        self.ffn_channel_scale = torch.ones((d,), device=device, dtype=torch.bfloat16)
        self.gate_up_w = [
            quantize_fp8(
                (torch.randn((2 * h, d), device=device) / math.sqrt(d)).to(torch.bfloat16),
                self.ffn_gate_up_w_scale,
            ).contiguous()
            for _ in range(layers)
        ]
        self.down_w = [
            quantize_fp8(
                (torch.randn((d, h), device=device) / math.sqrt(h)).to(torch.bfloat16),
                self.ffn_down_w_scale,
            ).contiguous()
            for _ in range(layers)
        ]


class PyTorchPI05Reference:
    def __init__(
        self,
        state: RuntimeState,
        *,
        ffn_activation: str,
        attention_backend: str,
        decoder_qkv_backend: str,
    ) -> None:
        self.s = state
        self.ffn_activation = ffn_activation
        self.attention_backend = attention_backend
        self.decoder_qkv_backend = decoder_qkv_backend

    def _qkv_joint(self):
        s = self.s
        p = s.p
        d = p.dim

        def segment(packed, q_w, k_w, bias=None, rope=False):
            if bias is not None:
                packed = packed.float() + bias.float().view(1, 1, -1)
            q, k, v = packed.split(d, dim=2)
            q = q.view(1, q.shape[1], p.heads, p.head_dim)
            k = k.view(1, k.shape[1], p.heads, p.head_dim)
            v = v.to(torch.bfloat16).view(1, v.shape[1], p.heads, p.head_dim).contiguous()
            q = rms_norm(q, q_w.view(p.heads, p.head_dim)).to(torch.bfloat16)
            k = rms_norm(k, k_w.view(p.heads, p.head_dim)).to(torch.bfloat16)
            if rope:
                q = pair_rope_rotate(q, s.freqs_re, s.freqs_im)
                k = pair_rope_rotate(k, s.freqs_re, s.freqs_im)
            return q, k, v

        qv, kv, vv = segment(s.packed_v, s.q_w_v, s.k_w_v, s.qkv_v_bias, rope=True)
        qa, ka, va = segment(s.packed_a, s.q_w_a, s.k_w_a)
        qu, ku, vu = segment(s.packed_u, s.q_w_u, s.k_w_u)
        return torch.cat([qv, qa, qu], dim=1), torch.cat([kv, ka, ku], dim=1), torch.cat([vv, va, vu], dim=1)

    def _decoder_qkv_gqa(self):
        s = self.s
        p = s.p
        q_dim = p.heads * p.head_dim
        kv_dim = p.decoder_kv_heads * p.head_dim
        q = s.decoder_packed_qkv[:, :, :q_dim].view(1, p.action_tokens, p.heads, p.head_dim)
        k = s.decoder_packed_qkv[:, :, q_dim : q_dim + kv_dim].view(
            1, p.action_tokens, p.decoder_kv_heads, p.head_dim
        )
        v = s.decoder_packed_qkv[:, :, q_dim + kv_dim :].view(
            1, p.action_tokens, p.decoder_kv_heads, p.head_dim
        )
        return interleaved_pair_rope(q, s.decoder_rope), interleaved_pair_rope(k, s.decoder_rope), v

    def __call__(self) -> torch.Tensor:
        s = self.s
        p = s.p
        latent = s.latent.clone()
        latent = (latent.float() + s.latent_bias.float().view(1, -1, 1, 1, 1)).to(torch.bfloat16)
        _ = latent.permute(0, 2, 3, 4, 1).contiguous().view(1, p.latent_t * p.latent_h * p.latent_w, p.latent_c)
        _ = torch.empty((1, p.latent_c, 2 * p.latent_t, p.latent_h, p.latent_w), device=latent.device, dtype=torch.bfloat16)
        _[:, :, 0::2] = s.latent_2c[:, : p.latent_c]
        _[:, :, 1::2] = s.latent_2c[:, p.latent_c :]
        _ = latent[:, :, -2:] if latent.shape[2] >= 2 else torch.cat([s.latent_prev[:, :, 1:2], latent], dim=2)

        q_cat, k_cat, v_cat = self._qkv_joint()
        if self.decoder_qkv_backend == "gqa-cache":
            q_decode, k_decode, v_decode = self._decoder_qkv_gqa()
        else:
            q_decode = decode_rope_rotate(
                rms_norm(s.q_pre, s.q_norm_decode).to(torch.bfloat16),
                s.cos_decode,
                s.sin_decode,
            )
            k_decode = decode_rope_rotate(
                rms_norm(s.k_pre, s.k_norm_decode).to(torch.bfloat16),
                s.cos_decode,
                s.sin_decode,
            )
            v_decode = s.v_pre

        v_out = (s.v_res.float() + (s.v_x.float() + s.v_bias.float().view(1, -1)) * s.v_gate.float()).to(torch.bfloat16)
        a_out = (s.a_res.float() + (s.a_x.float() + s.a_bias.float().view(1, -1)) * s.a_gate.float()).to(torch.bfloat16)
        u_out = (s.u_res.float() + s.u_x.float()).to(torch.bfloat16)
        fused = torch.cat([v_out, a_out, u_out], dim=0)
        if self.attention_backend == "sdpa":
            attn = sdpa_attention_bthd(q_cat, k_cat, v_cat).view(p.total_tokens, p.dim)
            fused.add_(attn)

        updated = (fused.float() + s.ada_x.float() * s.ada_gate_mul.float()).to(torch.bfloat16)
        dim = updated.shape[1]
        scale = s.ada_style[:, :dim].float()
        shift = s.ada_style[:, dim : 2 * dim].float()
        normed = rms_norm(updated, s.ada_weight)
        x = quantize_fp8((normed * (1.0 + scale) + shift).to(torch.bfloat16), s.fp8_scale)

        for i in range(s.layers):
            gate_up = (
                dequant_fp8(x, s.ffn_input_scale)
                @ dequant_fp8(s.gate_up_w[i], s.ffn_gate_up_w_scale).T
            ).to(torch.bfloat16)
            gate, up = gate_up.float().chunk(2, dim=1)
            if self.ffn_activation == "gelu":
                activated = torch.nn.functional.gelu(gate, approximate="tanh")
            else:
                activated = torch.nn.functional.silu(gate)
            hidden = quantize_fp8((activated * up).to(torch.bfloat16), s.ffn_hidden_scale)
            x_bf16 = (
                dequant_fp8(hidden, s.ffn_hidden_scale)
                @ dequant_fp8(s.down_w[i], s.ffn_down_w_scale).T
            ).to(torch.bfloat16)
            if i != s.layers - 1:
                x = quantize_fp8(x_bf16, s.ffn_input_scale)
        keepalive = (
            q_cat.flatten()[0].to(torch.bfloat16) * 0
            + k_cat.flatten()[0].to(torch.bfloat16) * 0
            + v_cat.flatten()[0].to(torch.bfloat16) * 0
            + q_decode.flatten()[0].to(torch.bfloat16) * 0
            + k_decode.flatten()[0].to(torch.bfloat16) * 0
            + v_decode.flatten()[0].to(torch.bfloat16) * 0
        )
        if self.attention_backend == "none":
            return x_bf16 + keepalive
        return x_bf16


class HubPI05Runtime:
    def __init__(
        self,
        state: RuntimeState,
        *,
        version: int,
        ffn_activation: str,
        attention_backend: str,
        decoder_qkv_backend: str,
        local_qkv_artifact: str | None = None,
        local_ffn_artifact: str | None = None,
    ) -> None:
        self.s = state
        self.p = state.p
        self.ffn_activation = ffn_activation
        self.attention_backend = attention_backend
        self.decoder_qkv_backend = decoder_qkv_backend
        self.layout = get_kernel("flashrt/flashrt-spatiotemporal-layout", version=version, trust_remote_code=True)
        if local_qkv_artifact:
            sys.path.insert(0, local_qkv_artifact)
            self.qkv = importlib.import_module("flashrt_qkv_cache_rope")
        else:
            self.qkv = get_kernel("flashrt/flashrt-qkv-cache-rope", version=version, trust_remote_code=True)
        self.gates = get_kernel("flashrt/flashrt-vla-residual-gates", version=version, trust_remote_code=True)
        self.adapt = get_kernel("flashrt/flashrt-adaptive-norms", version=version, trust_remote_code=True)
        if local_ffn_artifact:
            sys.path.insert(0, local_ffn_artifact)
            self.ffn = importlib.import_module("flashrt_fp8_swiglu_ffn")
        else:
            self.ffn = get_kernel("flashrt/flashrt-fp8-swiglu-ffn", version=version, trust_remote_code=True)
        self.quant = get_kernel("flashrt/flashrt-gemm-epilogues", version=version, trust_remote_code=True)

        p = self.p
        total = p.total_tokens
        d = p.dim
        h = p.ffn_hidden
        self.latent_work = torch.empty_like(state.latent)
        self.latent_blc = torch.empty((1, p.latent_t * p.latent_h * p.latent_w, p.latent_c), device=state.latent.device, dtype=torch.bfloat16)
        self.latent_unshuffle = torch.empty((1, p.latent_c, 2 * p.latent_t, p.latent_h, p.latent_w), device=state.latent.device, dtype=torch.bfloat16)
        self.latent_cache = torch.empty_like(state.latent_prev)
        self.q_cat = torch.empty((1, total, p.heads, p.head_dim), device=state.latent.device, dtype=torch.bfloat16)
        self.k_cat = torch.empty_like(self.q_cat)
        self.v_cat = torch.empty_like(self.q_cat)
        self.attn_out = torch.empty_like(self.q_cat)
        self.q_decode = torch.empty((p.heads, p.head_dim), device=state.latent.device, dtype=torch.bfloat16)
        self.k_cache = torch.empty((p.max_seq_len, p.heads, p.head_dim), device=state.latent.device, dtype=torch.bfloat16)
        self.v_cache = torch.empty_like(self.k_cache)
        self.decoder_q = torch.empty(
            (1, p.action_tokens, p.heads, p.head_dim),
            device=state.latent.device,
            dtype=torch.bfloat16,
        )
        decoder_cache_shape = (
            1,
            p.max_seq_len,
            p.decoder_kv_heads,
            p.head_dim,
        )
        self.decoder_k_cache = torch.empty(decoder_cache_shape, device=state.latent.device, dtype=torch.bfloat16)
        self.decoder_v_cache = torch.empty_like(self.decoder_k_cache)
        self.v_out = torch.empty_like(state.v_res)
        self.a_out = torch.empty_like(state.a_res)
        self.u_out = torch.empty_like(state.u_res)
        self.fused = torch.empty((total, d), device=state.latent.device, dtype=torch.bfloat16)
        self.ada_fp8 = torch.empty((total, d), device=state.latent.device, dtype=torch.float8_e4m3fn)
        self.ada_gate = torch.empty((total, d), device=state.latent.device, dtype=torch.bfloat16)
        self.gate_up = [torch.empty((total, 2 * h), device=state.latent.device, dtype=torch.bfloat16) for _ in range(state.layers)]
        self.hidden_fp8 = [torch.empty((total, h), device=state.latent.device, dtype=torch.float8_e4m3fn) for _ in range(state.layers)]
        self.ffn_out = [torch.empty((total, d), device=state.latent.device, dtype=torch.bfloat16) for _ in range(state.layers)]
        self.next_fp8 = [torch.empty((total, d), device=state.latent.device, dtype=torch.float8_e4m3fn) for _ in range(max(0, state.layers - 1))]

    def __call__(self) -> torch.Tensor:
        s = self.s
        p = self.p
        self.latent_work.copy_(s.latent)
        self.layout.add_bias_ncdhw_bf16(self.latent_work, s.latent_bias)
        self.layout.ncdhw_to_blc_bf16(self.latent_work, self.latent_blc)
        self.layout.time_unshuffle2_bf16(s.latent_2c, self.latent_unshuffle)
        self.layout.update_cache2_ncdhw_bf16(self.latent_work, s.latent_prev, self.latent_cache)

        self.qkv.qkv_split_joint3_cat_bf16(
            s.packed_v,
            s.qkv_v_bias,
            s.q_w_v,
            s.k_w_v,
            s.freqs_re,
            s.freqs_im,
            s.packed_a,
            s.q_w_a,
            s.k_w_a,
            s.packed_u,
            s.q_w_u,
            s.k_w_u,
            p.heads,
            p.head_dim,
            self.q_cat,
            self.k_cat,
            self.v_cat,
        )
        if self.decoder_qkv_backend == "gqa-cache":
            if not hasattr(self.qkv, "qkv_split_rope_kvcache_bf16"):
                raise RuntimeError(
                    "qkv_split_rope_kvcache_bf16 is not available. "
                    "Pass --local-qkv-artifact pointing at a rebuilt "
                    "flashrt-qkv-cache-rope artifact."
                )
            self.qkv.qkv_split_rope_kvcache_bf16(
                s.decoder_packed_qkv,
                s.decoder_rope,
                p.heads,
                p.decoder_kv_heads,
                p.head_dim,
                s.decoder_cache_offset,
                self.decoder_q,
                self.decoder_k_cache,
                self.decoder_v_cache,
            )
        else:
            self.qkv.decode_q_norm_rope_stage_bf16(
                s.q_pre, s.q_norm_decode, s.cos_decode, s.sin_decode, q_out=self.q_decode
            )
            self.qkv.decode_k_norm_rope_kvwrite_devpos_bf16(
                s.k_pre,
                s.v_pre,
                s.k_norm_decode,
                s.cos_decode,
                s.sin_decode,
                s.cur_pos,
                self.k_cache,
                self.v_cache,
            )

        self.gates.joint3_bias_gate_residual_bf16(
            s.v_res,
            s.v_x,
            s.v_bias,
            s.v_gate,
            s.a_res,
            s.a_x,
            s.a_bias,
            s.a_gate,
            s.u_res,
            s.u_x,
            v_out=self.v_out,
            a_out=self.a_out,
            u_out=self.u_out,
        )
        self.fused[: p.video_tokens].copy_(self.v_out)
        self.fused[p.video_tokens : p.video_tokens + p.action_tokens].copy_(self.a_out)
        self.fused[p.video_tokens + p.action_tokens :].copy_(self.u_out)
        if self.attention_backend == "sdpa":
            self.attn_out.copy_(sdpa_attention_bthd(self.q_cat, self.k_cat, self.v_cat))
            self.fused.add_(self.attn_out.view(p.total_tokens, p.dim))
        _, x, _ = self.adapt.gate_residual_ada_norm_fp8_static_bf16(
            self.fused,
            s.ada_x,
            s.ada_gate_mul,
            s.ada_weight,
            s.ada_style,
            s.fp8_scale,
            out=self.ada_fp8,
            gate_out=self.ada_gate,
        )
        for i in range(s.layers):
            ffn_op_name = (
                "fp8_geglu_mlp_bf16"
                if self.ffn_activation == "gelu"
                else "fp8_swiglu_mlp_bf16"
            )
            if not hasattr(self.ffn, ffn_op_name):
                raise RuntimeError(
                    f"{ffn_op_name} is not available in flashrt-fp8-swiglu-ffn. "
                    "Rebuild/reupload the package with GeGLU support or run "
                    "--ffn-activation silu against the current Hub artifact."
                )
            out = getattr(self.ffn, ffn_op_name)(
                x,
                s.gate_up_w[i],
                s.down_w[i],
                s.ffn_input_scale if i > 0 else s.fp8_scale,
                s.ffn_gate_up_w_scale,
                s.ffn_hidden_scale,
                s.ffn_down_w_scale,
                gate_up_bf16=self.gate_up[i],
                hidden_fp8=self.hidden_fp8[i],
                out=self.ffn_out[i],
            )
            if i != s.layers - 1:
                self.quant.channel_scale_quantize_fp8_static_bf16(
                    out,
                    s.ffn_channel_scale,
                    s.ffn_input_scale,
                    self.next_fp8[i],
                )
                x = self.next_fp8[i]
        if self.decoder_qkv_backend == "gqa-cache":
            keepalive = (
                self.decoder_q.flatten()[0].to(torch.bfloat16) * 0
                + self.decoder_k_cache.flatten()[0].to(torch.bfloat16) * 0
                + self.decoder_v_cache.flatten()[0].to(torch.bfloat16) * 0
            )
            return self.ffn_out[-1] + keepalive
        return self.ffn_out[-1]


class CapturedRuntime:
    def __init__(self, runtime: HubPI05Runtime, state: RuntimeState) -> None:
        self.runtime = runtime
        self.state = state
        self.static_v = state.packed_v.detach().clone()
        self.source_v = state.packed_v.detach().clone()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.output = runtime()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = runtime()

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        return self.output

    def replay_with_input_copy(self) -> torch.Tensor:
        # Representative input refresh cost. In a real model this becomes the
        # camera/proprio/token staging copy into the static captured buffers.
        self.state.packed_v.copy_(self.source_v)
        self.graph.replay()
        return self.output


def compare(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float, float]:
    diff = (got.float() - expected.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    p99_abs = float(percentile(diff, 0.99).item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            got.float().flatten(), expected.float().flatten(), dim=0
        ).item()
    )
    return max_abs, mean_abs, p99_abs, cosine


def run(args: argparse.Namespace) -> Result:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(args.seed)
    profile = PROFILES[args.profile]
    device = torch.device("cuda")
    state = RuntimeState(profile, args.layers, device)
    eager = PyTorchPI05Reference(
        state,
        ffn_activation=args.ffn_activation,
        attention_backend=args.attention_backend,
        decoder_qkv_backend=args.decoder_qkv_backend,
    )
    runtime = HubPI05Runtime(
        state,
        version=args.version,
        ffn_activation=args.ffn_activation,
        attention_backend=args.attention_backend,
        decoder_qkv_backend=args.decoder_qkv_backend,
        local_qkv_artifact=args.local_qkv_artifact,
        local_ffn_artifact=args.local_ffn_artifact,
    )

    expected = eager()
    got = runtime()
    torch.cuda.synchronize()
    max_abs, mean_abs, p99_abs, cosine = compare(got, expected)
    expected_rms = float(torch.sqrt(torch.mean(expected.float() * expected.float())).item())
    cosine_gate_enabled = expected_rms >= args.cosine_min_rms
    if p99_abs > args.p99_abs_limit or (
        cosine_gate_enabled and cosine < args.cosine_limit
    ):
        raise RuntimeError(
            "correctness gate failed: "
            f"max_abs={max_abs:.6f}, p99_abs={p99_abs:.6f}, "
            f"cosine={cosine:.8f}, expected_rms={expected_rms:.8f}, "
            f"cosine_gate_enabled={cosine_gate_enabled}"
        )

    eager_us = time_us(eager, warmup=args.warmup, iters=args.iters)
    runtime_us = time_us(runtime, warmup=args.warmup, iters=args.iters)

    graph_us = None
    graph_copy_us = None
    graph_status = "not_requested"
    if args.cuda_graph:
        try:
            captured = CapturedRuntime(runtime, state)
            graph_us = time_us(captured.replay, warmup=args.warmup, iters=args.iters)
            graph_copy_us = time_us(
                captured.replay_with_input_copy, warmup=args.warmup, iters=args.iters
            )
            graph_status = "ok"
        except Exception as exc:  # noqa: BLE001
            graph_status = f"unsupported: {type(exc).__name__}: {exc}"

    return Result(
        profile=profile.name,
        layers=args.layers,
        device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        eager_us=eager_us,
        runtime_us=runtime_us,
        graph_us=graph_us,
        graph_with_input_copy_us=graph_copy_us,
        runtime_vs_eager=eager_us / runtime_us,
        graph_vs_eager=None if graph_us is None else eager_us / graph_us,
        graph_status=graph_status,
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        cosine=cosine,
        expected_rms=expected_rms,
        cosine_gate_enabled=cosine_gate_enabled,
        ffn_activation=args.ffn_activation,
        ffn_source=f"local:{args.local_ffn_artifact}" if args.local_ffn_artifact else "hub",
        qkv_source=f"local:{args.local_qkv_artifact}" if args.local_qkv_artifact else "hub",
        attention_backend=args.attention_backend,
        decoder_qkv_backend=args.decoder_qkv_backend,
    )


def write_markdown(path: Path, result: Result) -> None:
    graph_us = "n/a" if result.graph_us is None else f"{result.graph_us:.3f}"
    graph_copy = (
        "n/a"
        if result.graph_with_input_copy_us is None
        else f"{result.graph_with_input_copy_us:.3f}"
    )
    graph_speed = "n/a" if result.graph_vs_eager is None else f"{result.graph_vs_eager:.2f}x"
    text = f"""# FlashRT HF Runtime Demo Result

| Field | Value |
| --- | ---: |
| Profile | `{result.profile}` |
| Layers | {result.layers} |
| Device | `{result.device}` |
| PyTorch | `{result.torch_version}` |
| CUDA runtime | `{result.cuda_version}` |
| PyTorch eager reference us | {result.eager_us:.3f} |
| Hub runtime prealloc us | {result.runtime_us:.3f} |
| Hub runtime CUDA Graph us | {graph_us} |
| Hub runtime CUDA Graph + input copy us | {graph_copy} |
| Runtime vs eager | {result.runtime_vs_eager:.2f}x |
| Graph vs eager | {graph_speed} |
| max_abs | {result.max_abs:.6f} |
| mean_abs | {result.mean_abs:.6f} |
| p99_abs | {result.p99_abs:.6f} |
| cosine | {result.cosine:.8f} |
| expected_rms | {result.expected_rms:.8f} |
| cosine_gate_enabled | `{result.cosine_gate_enabled}` |
| FFN activation | `{result.ffn_activation}` |
| FFN source | `{result.ffn_source}` |
| QKV source | `{result.qkv_source}` |
| Attention backend | `{result.attention_backend}` |
| Decoder QKV backend | `{result.decoder_qkv_backend}` |
| graph_status | `{result.graph_status}` |

"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="pi05_hotpath")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument(
        "--ffn-activation",
        choices=("gelu", "silu"),
        default="gelu",
        help="PI0.5/Gemma uses gelu. Current published Hub artifacts may only "
        "have silu until flashrt-fp8-swiglu-ffn is rebuilt.",
    )
    parser.add_argument(
        "--local-ffn-artifact",
        default=None,
        help="Optional local build/<variant> directory for validating an unreleased "
        "flashrt-fp8-swiglu-ffn artifact. Other packages are still loaded from Hub.",
    )
    parser.add_argument(
        "--local-qkv-artifact",
        default=None,
        help="Optional local build/<variant> directory for validating an unreleased "
        "flashrt-qkv-cache-rope artifact.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("none", "sdpa"),
        default="none",
        help="`none` preserves the package-composition hotpath. `sdpa` runs real "
        "non-causal attention on the Q/K/V produced by flashrt-qkv-cache-rope.",
    )
    parser.add_argument(
        "--decoder-qkv-backend",
        choices=("decode-stage", "gqa-cache"),
        default="decode-stage",
        help="`decode-stage` uses the published single-token decode staging APIs. "
        "`gqa-cache` uses qkv_split_rope_kvcache_bf16 for PI0.5-style decoder "
        "packed GQA QKV plus K/V cache writes.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--p99-abs-limit", type=float, default=0.25)
    parser.add_argument("--cosine-limit", type=float, default=0.995)
    parser.add_argument("--cosine-min-rms", type=float, default=1e-2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    result = run(args)
    print(json.dumps(asdict(result), indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2) + "\n")
    if args.markdown is not None:
        write_markdown(args.markdown, result)


if __name__ == "__main__":
    main()
