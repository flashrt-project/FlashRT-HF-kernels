#!/usr/bin/env python3
"""Checkpoint-backed PI0.5 decoder-loop staging demo with HF Kernel Hub ops.

This is not a full policy E2E benchmark. It isolates the Gemma-300M decoder
denoise loop and uses real PI0.5 safetensors weights plus the official
FlashRT RTX frontend's conversion contracts:

- decoder Q/K are adjacent-pair interleaved before the QKV GEMM;
- decoder action_out projection is pre-scaled by ``-1 / num_steps``;
- time embeddings and AdaRMSNorm style tensors are precomputed per denoise step;
- FP8 activation scales are calibrated once before runtime, statically aggregated
  across all denoise steps, and then held fixed inside the CUDA Graph hot path.

The kernel-covered path uses BF16 linear projections, AdaRMSNorm, GQA QKV split
+ RoPE + K/V cache write, FA2 attention, FP8 GeGLU FFN, gated residual updates,
and CUDA Graph replay.
"""

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
from safetensors import safe_open


DEC_L = 18
DEC_D = 1024
DEC_H = 4096
DEC_NH = 8
DEC_NKV = 1
DEC_HD = 256
ACTION_DIM = 32
FP8_MAX = 448.0


@dataclass
class Result:
    checkpoint: str
    layers: int
    steps: int
    device: str
    torch_version: str
    cuda_version: str | None
    eager_us: float
    runtime_us: float
    graph_us: float | None
    runtime_vs_eager: float
    graph_vs_eager: float | None
    graph_status: str
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    gemm_source: str
    attention_source: str
    qkv_source: str
    ffn_source: str
    residual_source: str
    calibration_mode: str
    calibration_source: str
    calibration_path: str | None
    calibration_steps: int
    scale_safety: float
    kernel_coverage: list[str]
    torch_gaps: list[str]


def _resolve_weight_path(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "model.safetensors"
    if not p.exists():
        raise FileNotFoundError(f"checkpoint weights not found: {p}")
    return p


def _interleave_qk(w: torch.Tensor, num_heads: int) -> torch.Tensor:
    out_dim, in_dim = w.shape
    head_dim = out_dim // num_heads
    return (
        w.reshape(num_heads, head_dim, in_dim)
        .reshape(num_heads, 2, head_dim // 2, in_dim)
        .permute(0, 2, 1, 3)
        .reshape(out_dim, in_dim)
    )


def _scale_from_amax(x: torch.Tensor, *, safety: float = 1.05) -> torch.Tensor:
    amax = x.float().abs().max()
    scale = torch.clamp(amax / FP8_MAX * float(safety), min=1e-12)
    return scale.reshape(1).to(device=x.device, dtype=torch.float32).contiguous()


def _quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float().reshape(()), -FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn
    ).contiguous()


def _dequant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.float() * scale.float().reshape(())


def _rms_norm_style_ref(
    x: torch.Tensor,
    style: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = x.shape[-1]
    scale = style[:, :dim].float()
    shift = style[:, dim : 2 * dim].float()
    gate = style[:, 2 * dim :].to(torch.bfloat16)
    xf = x.float()
    inv = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    out = xf * inv * weight.float().view(1, -1)
    out = out * (1.0 + scale) + shift
    return out.to(torch.bfloat16), gate


def _qkv_split_rope_ref(
    packed_qkv: torch.Tensor,
    rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_dim = DEC_NH * DEC_HD
    kv_dim = DEC_NKV * DEC_HD
    q = packed_qkv[:, :q_dim].view(1, packed_qkv.shape[0], DEC_NH, DEC_HD)
    k = packed_qkv[:, q_dim : q_dim + kv_dim].view(1, packed_qkv.shape[0], DEC_NKV, DEC_HD)
    v = packed_qkv[:, q_dim + kv_dim :].view(1, packed_qkv.shape[0], DEC_NKV, DEC_HD)

    def apply(x: torch.Tensor) -> torch.Tensor:
        bsz, seq, heads, head_dim = x.shape
        pair = x.float().reshape(bsz, seq, heads, head_dim // 2, 2)
        re = pair[..., 0]
        im = pair[..., 1]
        rope_pair = rope[:seq].float().reshape(seq, head_dim // 2, 2)
        cos = rope_pair[..., 0].view(1, seq, 1, head_dim // 2)
        sin = rope_pair[..., 1].view(1, seq, 1, head_dim // 2)
        out = torch.empty_like(pair.float())
        out[..., 0] = re * cos - im * sin
        out[..., 1] = re * sin + im * cos
        return out.reshape(bsz, seq, heads, head_dim).to(torch.bfloat16)

    return apply(q), apply(k), v.contiguous()


def _sdpa_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    k_rep = k.repeat_interleave(DEC_NH // DEC_NKV, dim=2)
    v_rep = v.repeat_interleave(DEC_NH // DEC_NKV, dim=2)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k_rep.transpose(1, 2),
        v_rep.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    )
    return out.transpose(1, 2).contiguous().view(q.shape[1], DEC_NH * DEC_HD)


def _layer_kv(cache: torch.Tensor, layer: int, *, name: str) -> torch.Tensor:
    """Return one layer's encoder KV as (1, seq, n_kv, head_dim)."""
    if cache.dim() != 4:
        raise ValueError(f"{name} must have rank 4, got shape={tuple(cache.shape)}")
    if cache.shape[0] == 1:
        return cache
    if layer >= cache.shape[0]:
        raise ValueError(f"{name} layer {layer} out of range for shape={tuple(cache.shape)}")
    return cache[layer : layer + 1]


def _make_rope(chunk_size: int, encoder_seq_len: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(
        encoder_seq_len, encoder_seq_len + chunk_size, device=device, dtype=torch.float64
    )
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, DEC_HD, 2, device=device, dtype=torch.float64) / DEC_HD)
    )
    phase = positions[:, None] * inv_freq[None, :]
    cos = torch.cos(phase).to(torch.bfloat16)
    sin = torch.sin(phase).to(torch.bfloat16)
    return torch.stack([cos, sin], dim=-1).reshape(chunk_size, DEC_HD).contiguous()


def _load_module(local_artifact: str | None, package: str, module: str):
    if local_artifact:
        sys.path.insert(0, local_artifact)
        return importlib.import_module(module), f"local:{local_artifact}"
    return get_kernel(package, version=1, trust_remote_code=True), "hub"


def _scale_payload_to_tensors(
    payload: dict,
    *,
    layers: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    input_values = payload.get("input_scales")
    hidden_values = payload.get("hidden_scales")
    if not isinstance(input_values, list) or not isinstance(hidden_values, list):
        raise ValueError("calibration JSON must contain input_scales and hidden_scales lists")
    if len(input_values) != layers or len(hidden_values) != layers:
        raise ValueError(
            "calibration scale count mismatch: "
            f"expected {layers}, got input={len(input_values)} hidden={len(hidden_values)}"
        )

    def one(value: object, name: str) -> torch.Tensor:
        scale = float(value)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"invalid calibration scale {name}={scale}")
        return torch.tensor([scale], device=device, dtype=torch.float32).contiguous()

    return (
        [one(v, f"input_scales[{i}]") for i, v in enumerate(input_values)],
        [one(v, f"hidden_scales[{i}]") for i, v in enumerate(hidden_values)],
    )


class DecoderWeights:
    def __init__(self, checkpoint: Path, *, layers: int, steps: int, device: torch.device) -> None:
        self.layers = layers
        self.steps = steps
        self.device = device
        self.ones = torch.ones((DEC_D,), device=device, dtype=torch.bfloat16)
        with safe_open(str(checkpoint), framework="pt") as f:
            keys = set(f.keys())
            strip = "model." if any(k.startswith("model.") for k in keys) else ""

            def get(key: str, *, dtype=torch.bfloat16) -> torch.Tensor:
                return f.get_tensor(strip + key).to(device=device, dtype=dtype).contiguous()

            dp = "paligemma_with_expert.gemma_expert.model.layers"
            self.action_in_w = get("action_in_proj.weight").t().contiguous()
            self.action_in_b = get("action_in_proj.bias")
            self.action_out_w = (get("action_out_proj.weight").t() * (-1.0 / steps)).contiguous()
            self.action_out_b = (get("action_out_proj.bias") * (-1.0 / steps)).contiguous()
            self.qkv_w = []
            self.o_w = []
            self.gate_up_w = []
            self.down_w = []
            self.attn_mod_w = []
            self.attn_mod_b = []
            self.ffn_mod_w = []
            self.ffn_mod_b = []
            for i in range(layers):
                q_w = _interleave_qk(get(f"{dp}.{i}.self_attn.q_proj.weight").float(), DEC_NH)
                k_w = _interleave_qk(get(f"{dp}.{i}.self_attn.k_proj.weight").float(), DEC_NKV)
                v_w = get(f"{dp}.{i}.self_attn.v_proj.weight").float()
                self.qkv_w.append(torch.cat([q_w, k_w, v_w], dim=0).t().to(torch.bfloat16).contiguous())
                self.o_w.append(get(f"{dp}.{i}.self_attn.o_proj.weight").t().contiguous())

                gate = get(f"{dp}.{i}.mlp.gate_proj.weight")
                up = get(f"{dp}.{i}.mlp.up_proj.weight")
                self.gate_up_w.append(torch.cat([gate, up], dim=0).contiguous())
                self.down_w.append(get(f"{dp}.{i}.mlp.down_proj.weight").contiguous())

                self.attn_mod_w.append(get(f"{dp}.{i}.input_layernorm.dense.weight").t().contiguous())
                self.attn_mod_b.append(get(f"{dp}.{i}.input_layernorm.dense.bias"))
                self.ffn_mod_w.append(get(f"{dp}.{i}.post_attention_layernorm.dense.weight").t().contiguous())
                self.ffn_mod_b.append(get(f"{dp}.{i}.post_attention_layernorm.dense.bias"))

            self.final_mod_w = get("paligemma_with_expert.gemma_expert.model.norm.dense.weight").t()
            self.final_mod_b = get("paligemma_with_expert.gemma_expert.model.norm.dense.bias")
            self.time_in_w = get("time_mlp_in.weight").t()
            self.time_in_b = get("time_mlp_in.bias")
            self.time_out_w = get("time_mlp_out.weight").t()
            self.time_out_b = get("time_mlp_out.bias")

        self._quantize_ffn_weights()
        self._precompute_styles(chunk_size=10)

    def _quantize_ffn_weights(self) -> None:
        self.gate_up_w_scale = []
        self.down_w_scale = []
        self.gate_up_w_fp8 = []
        self.down_w_fp8 = []
        for gate_up, down in zip(self.gate_up_w, self.down_w):
            gu_s = _scale_from_amax(gate_up, safety=1.0)
            dn_s = _scale_from_amax(down, safety=1.0)
            self.gate_up_w_scale.append(gu_s)
            self.down_w_scale.append(dn_s)
            self.gate_up_w_fp8.append(_quantize_fp8(gate_up, gu_s))
            self.down_w_fp8.append(_quantize_fp8(down, dn_s))

    def _time_embedding_rows(self) -> torch.Tensor:
        dt = -1.0 / self.steps
        t = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        fraction = torch.linspace(0.0, 1.0, DEC_D // 2, dtype=torch.float32, device=self.device)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        rows = []
        for _ in range(self.steps):
            sinusoid = t * (1.0 / period) * 2 * math.pi
            rows.append(torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1).to(torch.bfloat16))
            t = t + dt
        return torch.stack(rows, dim=0)

    def _precompute_styles(self, *, chunk_size: int) -> None:
        time_rows = self._time_embedding_rows()
        self.style_attn = torch.empty(
            (self.steps, self.layers, chunk_size, 3 * DEC_D),
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.style_ffn = torch.empty_like(self.style_attn)
        self.style_final = torch.empty(
            (self.steps, chunk_size, 3 * DEC_D),
            device=self.device,
            dtype=torch.bfloat16,
        )
        for step in range(self.steps):
            te = time_rows[step : step + 1]
            tmp = te @ self.time_in_w + self.time_in_b.view(1, -1)
            tmp = (tmp.float() * torch.sigmoid(tmp.float())).to(torch.bfloat16)
            tmp2 = tmp @ self.time_out_w + self.time_out_b.view(1, -1)
            tmp2 = (tmp2.float() * torch.sigmoid(tmp2.float())).to(torch.bfloat16)
            expanded = tmp2.expand(chunk_size, -1).contiguous()
            for i in range(self.layers):
                self.style_attn[step, i] = expanded @ self.attn_mod_w[i] + self.attn_mod_b[i].view(1, -1)
                self.style_ffn[step, i] = expanded @ self.ffn_mod_w[i] + self.ffn_mod_b[i].view(1, -1)
            self.style_final[step] = expanded @ self.final_mod_w + self.final_mod_b.view(1, -1)


class DecoderState:
    def __init__(
        self,
        *,
        weights: DecoderWeights,
        chunk_size: int,
        encoder_seq_len: int,
        seed: int,
    ) -> None:
        torch.manual_seed(seed)
        device = weights.device
        self.noise0 = torch.randn((chunk_size, ACTION_DIM), device=device, dtype=torch.bfloat16)
        self.encoder_k = torch.randn(
            (1, encoder_seq_len, DEC_NKV, DEC_HD), device=device, dtype=torch.bfloat16
        )
        self.encoder_v = torch.randn_like(self.encoder_k)
        self.rope = _make_rope(chunk_size, encoder_seq_len, device)
        self.chunk_size = chunk_size
        self.encoder_seq_len = encoder_seq_len


class TorchDecoderReference:
    def __init__(
        self,
        weights: DecoderWeights,
        state: DecoderState,
        *,
        scale_safety: float,
        calibration_input: Path | None,
    ) -> None:
        self.w = weights
        self.s = state
        self.scale_safety = scale_safety
        if calibration_input is None:
            self.input_scale, self.hidden_scale = self._calibrate_scales()
            self.calibration_mode = "static_all_steps_amax"
            self.calibration_source = "computed"
            self.calibration_path = None
        else:
            payload = json.loads(calibration_input.read_text())
            self.input_scale, self.hidden_scale = _scale_payload_to_tensors(
                payload, layers=weights.layers, device=weights.device
            )
            self.calibration_mode = str(payload.get("mode", "static_loaded"))
            self.calibration_source = f"file:{calibration_input}"
            self.calibration_path = str(calibration_input)

    def save_calibration(
        self,
        path: Path,
        *,
        checkpoint: Path,
        encoder_seq_len: int,
    ) -> None:
        payload = {
            "version": 1,
            "checkpoint": str(checkpoint),
            "layers": self.w.layers,
            "steps": self.w.steps,
            "encoder_seq_len": encoder_seq_len,
            "mode": self.calibration_mode,
            "source": self.calibration_source,
            "scale_safety": self.scale_safety,
            "input_scales": [float(s.item()) for s in self.input_scale],
            "hidden_scales": [float(s.item()) for s in self.hidden_scale],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def _calibrate_scales(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        input_amax = torch.zeros((self.w.layers,), device=self.w.device, dtype=torch.float32)
        hidden_amax = torch.zeros_like(input_amax)
        noise = self.s.noise0.clone()
        for step in range(self.w.steps):
            x = (noise @ self.w.action_in_w + self.w.action_in_b.view(1, -1)).to(torch.bfloat16)
            for i in range(self.w.layers):
                normed, gate = _rms_norm_style_ref(x, self.w.style_attn[step, i], self.w.ones)
                qkv = (normed @ self.w.qkv_w[i]).to(torch.bfloat16)
                q, k_dec, v_dec = _qkv_split_rope_ref(qkv, self.s.rope)
                enc_k = _layer_kv(self.s.encoder_k, i, name="encoder_k")
                enc_v = _layer_kv(self.s.encoder_v, i, name="encoder_v")
                k = torch.cat([enc_k, k_dec], dim=1)
                v = torch.cat([enc_v, v_dec], dim=1)
                attn = _sdpa_gqa(q, k, v)
                attn_o = (attn @ self.w.o_w[i]).to(torch.bfloat16)
                x = (x.float() + attn_o.float() * gate.float()).to(torch.bfloat16)
                ffn_normed, ffn_gate = _rms_norm_style_ref(x, self.w.style_ffn[step, i], self.w.ones)
                input_amax[i] = torch.maximum(input_amax[i], ffn_normed.float().abs().max())
                x_scale = _scale_from_amax(ffn_normed, safety=self.scale_safety)
                x_fp8 = _quantize_fp8(ffn_normed, x_scale)
                gate_up = _dequant_fp8(x_fp8, x_scale) @ _dequant_fp8(
                    self.w.gate_up_w_fp8[i], self.w.gate_up_w_scale[i]
                ).t()
                gate_part, up_part = gate_up.chunk(2, dim=1)
                hidden = F.gelu(gate_part, approximate="tanh") * up_part
                hidden_bf16 = hidden.to(torch.bfloat16)
                hidden_amax[i] = torch.maximum(hidden_amax[i], hidden_bf16.float().abs().max())
                hidden_scale = _scale_from_amax(hidden_bf16, safety=self.scale_safety)
                hidden_fp8 = _quantize_fp8(hidden_bf16, hidden_scale)
                ffn_out = (
                    _dequant_fp8(hidden_fp8, hidden_scale)
                    @ _dequant_fp8(self.w.down_w_fp8[i], self.w.down_w_scale[i]).t()
                ).to(torch.bfloat16)
                x = (x.float() + ffn_out.float() * ffn_gate.float()).to(torch.bfloat16)
            final_normed, _ = _rms_norm_style_ref(x, self.w.style_final[step], self.w.ones)
            action = (
                final_normed @ self.w.action_out_w + self.w.action_out_b.view(1, -1)
            ).to(torch.bfloat16)
            noise = (noise.float() + action.float()).to(torch.bfloat16)
        input_scales = [
            torch.clamp(input_amax[i] / FP8_MAX * self.scale_safety, min=1e-12)
            .reshape(1)
            .to(device=self.w.device, dtype=torch.float32)
            .contiguous()
            for i in range(self.w.layers)
        ]
        hidden_scales = [
            torch.clamp(hidden_amax[i] / FP8_MAX * self.scale_safety, min=1e-12)
            .reshape(1)
            .to(device=self.w.device, dtype=torch.float32)
            .contiguous()
            for i in range(self.w.layers)
        ]
        return input_scales, hidden_scales

    def __call__(self) -> torch.Tensor:
        w = self.w
        s = self.s
        noise = s.noise0.clone()
        for step in range(w.steps):
            x = (noise @ w.action_in_w + w.action_in_b.view(1, -1)).to(torch.bfloat16)
            for i in range(w.layers):
                normed, gate = _rms_norm_style_ref(x, w.style_attn[step, i], w.ones)
                qkv = (normed @ w.qkv_w[i]).to(torch.bfloat16)
                q, k_dec, v_dec = _qkv_split_rope_ref(qkv, s.rope)
                enc_k = _layer_kv(s.encoder_k, i, name="encoder_k")
                enc_v = _layer_kv(s.encoder_v, i, name="encoder_v")
                k = torch.cat([enc_k, k_dec], dim=1)
                v = torch.cat([enc_v, v_dec], dim=1)
                attn = _sdpa_gqa(q, k, v)
                attn_o = (attn @ w.o_w[i]).to(torch.bfloat16)
                x = (x.float() + attn_o.float() * gate.float()).to(torch.bfloat16)
                ffn_normed, ffn_gate = _rms_norm_style_ref(x, w.style_ffn[step, i], w.ones)
                x_fp8 = _quantize_fp8(ffn_normed, self.input_scale[i])
                gate_up = _dequant_fp8(x_fp8, self.input_scale[i]) @ _dequant_fp8(
                    w.gate_up_w_fp8[i], w.gate_up_w_scale[i]
                ).t()
                gate_part, up_part = gate_up.chunk(2, dim=1)
                hidden = F.gelu(gate_part, approximate="tanh") * up_part
                hidden_fp8 = _quantize_fp8(hidden.to(torch.bfloat16), self.hidden_scale[i])
                ffn_out = (
                    _dequant_fp8(hidden_fp8, self.hidden_scale[i])
                    @ _dequant_fp8(w.down_w_fp8[i], w.down_w_scale[i]).t()
                ).to(torch.bfloat16)
                x = (x.float() + ffn_out.float() * ffn_gate.float()).to(torch.bfloat16)
            final_normed, _ = _rms_norm_style_ref(x, w.style_final[step], w.ones)
            action = (final_normed @ w.action_out_w + w.action_out_b.view(1, -1)).to(torch.bfloat16)
            noise = (noise.float() + action.float()).to(torch.bfloat16)
        return noise


class HubDecoderLoop:
    def __init__(
        self,
        weights: DecoderWeights,
        state: DecoderState,
        reference: TorchDecoderReference,
        *,
        local_gemm_artifact: str | None,
        local_qkv_artifact: str | None,
        local_ffn_artifact: str | None,
        local_residual_artifact: str | None,
        attention_backend: str,
    ) -> None:
        self.w = weights
        self.s = state
        self.ref = reference
        self.attention_backend = attention_backend
        self.adapt = get_kernel("flashrt/flashrt-adaptive-norms", version=1, trust_remote_code=True)
        self.gemm, self.gemm_source = _load_module(
            local_gemm_artifact,
            "flashrt/flashrt-gemm-epilogues",
            "flashrt_gemm_epilogues",
        )
        if attention_backend == "fa2":
            self.attn_kernel = get_kernel(
                "kernels-community/flash-attn2", version=1, trust_remote_code=True
            )
            self.attention_source = "hub:kernels-community/flash-attn2"
        else:
            self.attn_kernel = None
            self.attention_source = "torch:sdpa"
        self.qkv, self.qkv_source = _load_module(
            local_qkv_artifact,
            "flashrt/flashrt-qkv-cache-rope",
            "flashrt_qkv_cache_rope",
        )
        self.ffn, self.ffn_source = _load_module(
            local_ffn_artifact,
            "flashrt/flashrt-fp8-swiglu-ffn",
            "flashrt_fp8_swiglu_ffn",
        )
        self.residual, self.residual_source = _load_module(
            local_residual_artifact,
            "flashrt/flashrt-vla-residual-gates",
            "flashrt_vla_residual_gates",
        )
        if not hasattr(self.gemm, "bf16_linear_bf16"):
            raise RuntimeError("gemm artifact missing bf16_linear_bf16")
        if not hasattr(self.gemm, "bf16_linear_bias_bf16"):
            raise RuntimeError("gemm artifact missing bf16_linear_bias_bf16")
        if not hasattr(self.qkv, "qkv_split_rope_kvcache_bf16"):
            raise RuntimeError("qkv artifact missing qkv_split_rope_kvcache_bf16")
        if not hasattr(self.ffn, "fp8_geglu_mlp_bf16"):
            raise RuntimeError("ffn artifact missing fp8_geglu_mlp_bf16")
        if not hasattr(self.residual, "gate_residual_bf16"):
            raise RuntimeError("residual artifact missing gate_residual_bf16")

        cs = state.chunk_size
        max_seq = state.encoder_seq_len + cs
        device = weights.device
        self.x = torch.empty((cs, DEC_D), device=device, dtype=torch.bfloat16)
        self.noise = torch.empty((cs, ACTION_DIM), device=device, dtype=torch.bfloat16)
        self.noise_gate = torch.ones_like(self.noise)
        self.normed = torch.empty_like(self.x)
        self.gate = torch.empty_like(self.x)
        self.ffn_gate = torch.empty_like(self.x)
        self.zero_x = torch.zeros_like(self.x)
        self.zero_gate = torch.zeros_like(self.x)
        self.qkv_buf = torch.empty(
            (cs, (DEC_NH + 2 * DEC_NKV) * DEC_HD), device=device, dtype=torch.bfloat16
        )
        self.q = torch.empty((1, cs, DEC_NH, DEC_HD), device=device, dtype=torch.bfloat16)
        self.k_cache = torch.empty((1, max_seq, DEC_NKV, DEC_HD), device=device, dtype=torch.bfloat16)
        self.v_cache = torch.empty_like(self.k_cache)
        self.attn_bthd = torch.empty((1, cs, DEC_NH, DEC_HD), device=device, dtype=torch.bfloat16)
        self.attn = self.attn_bthd.view(cs, DEC_NH * DEC_HD)
        self.attn_o = torch.empty((cs, DEC_D), device=device, dtype=torch.bfloat16)
        self.ffn_fp8 = torch.empty((cs, DEC_D), device=device, dtype=torch.float8_e4m3fn)
        self.gate_up = [torch.empty((cs, 2 * DEC_H), device=device, dtype=torch.bfloat16) for _ in range(weights.layers)]
        self.hidden_fp8 = [torch.empty((cs, DEC_H), device=device, dtype=torch.float8_e4m3fn) for _ in range(weights.layers)]
        self.ffn_out = [torch.empty((cs, DEC_D), device=device, dtype=torch.bfloat16) for _ in range(weights.layers)]
        self.final_normed = torch.empty_like(self.x)
        self.final_gate = torch.empty_like(self.x)
        self.action = torch.empty((cs, ACTION_DIM), device=device, dtype=torch.bfloat16)

    def __call__(self) -> torch.Tensor:
        w = self.w
        s = self.s
        self.noise.copy_(s.noise0)
        for step in range(w.steps):
            self.gemm.bf16_linear_bias_bf16(
                self.noise, w.action_in_w, w.action_in_b, out=self.x
            )
            for i in range(w.layers):
                self.adapt.ada_rms_norm_style_bf16(
                    self.x, w.ones, w.style_attn[step, i], out=self.normed, gate_out=self.gate
                )
                self.gemm.bf16_linear_bf16(self.normed, w.qkv_w[i], out=self.qkv_buf)
                self.k_cache[:, : s.encoder_seq_len].copy_(
                    _layer_kv(s.encoder_k, i, name="encoder_k")
                )
                self.v_cache[:, : s.encoder_seq_len].copy_(
                    _layer_kv(s.encoder_v, i, name="encoder_v")
                )
                self.qkv.qkv_split_rope_kvcache_bf16(
                    self.qkv_buf.view(1, s.chunk_size, -1),
                    s.rope,
                    DEC_NH,
                    DEC_NKV,
                    DEC_HD,
                    s.encoder_seq_len,
                    self.q,
                    self.k_cache,
                    self.v_cache,
                )
                if self.attention_backend == "fa2":
                    self.attn_kernel.fwd(
                        self.q,
                        self.k_cache,
                        self.v_cache,
                        out=self.attn_bthd,
                        p_dropout=0.0,
                        is_causal=False,
                    )
                else:
                    self.attn.copy_(_sdpa_gqa(self.q, self.k_cache, self.v_cache))
                self.gemm.bf16_linear_bf16(self.attn, w.o_w[i], out=self.attn_o)
                self.residual.gate_residual_bf16(self.x, self.attn_o, self.gate, out=self.x)
                _, ffn_fp8, ffn_gate = self.adapt.gate_residual_ada_norm_fp8_static_bf16(
                    self.x,
                    self.zero_x,
                    self.zero_gate,
                    w.ones,
                    w.style_ffn[step, i],
                    self.ref.input_scale[i],
                    out=self.ffn_fp8,
                    gate_out=self.ffn_gate,
                )
                self.ffn.fp8_geglu_mlp_bf16(
                    ffn_fp8,
                    w.gate_up_w_fp8[i],
                    w.down_w_fp8[i],
                    self.ref.input_scale[i],
                    w.gate_up_w_scale[i],
                    self.ref.hidden_scale[i],
                    w.down_w_scale[i],
                    self.gate_up[i],
                    self.hidden_fp8[i],
                    self.ffn_out[i],
                )
                self.residual.gate_residual_bf16(self.x, self.ffn_out[i], ffn_gate, out=self.x)
            self.adapt.ada_rms_norm_style_bf16(
                self.x, w.ones, w.style_final[step], out=self.final_normed, gate_out=self.final_gate
            )
            self.gemm.bf16_linear_bias_bf16(
                self.final_normed, w.action_out_w, w.action_out_b, out=self.action
            )
            self.residual.gate_residual_bf16(
                self.noise, self.action, self.noise_gate, out=self.noise
            )
        return self.noise


def _quant_proj_weight(w_kn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a stored (K, N) projection weight to FP8 (N, K) + per-tensor scale.

    Stored decoder/encoder projection weights are ``x @ w`` layout ``(K, N)``;
    ``fp8_gemm_bf16`` expects ``(N, K)``.
    """
    nk = w_kn.t().contiguous()
    scale = _scale_from_amax(nk, safety=1.0)
    return _quantize_fp8(nk, scale), scale


def calibrate_decoder_proj(
    weights: "DecoderWeights",
    state: "DecoderState",
    *,
    scale_safety: float,
    calibration_input,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Static amax calibration of decoder QKV/O projection input scales.

    Mirrors ``TorchDecoderReference`` so the FP8-projection runtime uses scales
    consistent with the BF16-projection reference. Returns ``(qkv_in, o_in)``.
    """
    dev = weights.device
    ref = TorchDecoderReference(
        weights, state, scale_safety=scale_safety, calibration_input=calibration_input
    )
    qa = torch.zeros((weights.layers,), device=dev)
    oa = torch.zeros((weights.layers,), device=dev)
    noise = state.noise0.clone()
    for step in range(weights.steps):
        x = (noise @ weights.action_in_w + weights.action_in_b.view(1, -1)).to(torch.bfloat16)
        for i in range(weights.layers):
            nm, gate = _rms_norm_style_ref(x, weights.style_attn[step, i], weights.ones)
            qa[i] = torch.maximum(qa[i], nm.float().abs().max())
            q, kd, vd = _qkv_split_rope_ref((nm @ weights.qkv_w[i]).to(torch.bfloat16), state.rope)
            k = torch.cat([_layer_kv(state.encoder_k, i, name="encoder_k"), kd], dim=1)
            v = torch.cat([_layer_kv(state.encoder_v, i, name="encoder_v"), vd], dim=1)
            attn = _sdpa_gqa(q, k, v)
            oa[i] = torch.maximum(oa[i], attn.float().abs().max())
            x = (x.float() + (attn @ weights.o_w[i]).float() * gate.float()).to(torch.bfloat16)
            fn, fg = _rms_norm_style_ref(x, weights.style_ffn[step, i], weights.ones)
            xf = _quantize_fp8(fn, ref.input_scale[i])
            gu = _dequant_fp8(xf, ref.input_scale[i]) @ _dequant_fp8(
                weights.gate_up_w_fp8[i], weights.gate_up_w_scale[i]
            ).t()
            gp, up = gu.chunk(2, dim=1)
            hidden = F.gelu(gp, approximate="tanh") * up
            hf = _quantize_fp8(hidden.to(torch.bfloat16), ref.hidden_scale[i])
            fo = (_dequant_fp8(hf, ref.hidden_scale[i]) @ _dequant_fp8(
                weights.down_w_fp8[i], weights.down_w_scale[i]
            ).t()).to(torch.bfloat16)
            x = (x.float() + fo.float() * fg.float()).to(torch.bfloat16)
        fnl, _ = _rms_norm_style_ref(x, weights.style_final[step], weights.ones)
        noise = (noise.float() + (fnl @ weights.action_out_w + weights.action_out_b.view(1, -1)).float()).to(torch.bfloat16)

    def mk(a):
        return torch.clamp(a / FP8_MAX * scale_safety, min=1e-12).reshape(1).to(dev, torch.float32).contiguous()

    return [mk(qa[i]) for i in range(weights.layers)], [mk(oa[i]) for i in range(weights.layers)]


class Fp8HubDecoderLoop(HubDecoderLoop):
    """Decoder loop with QKV/O projections run in FP8 (published Hub kernels only).

    Replaces the BF16 QKV/O ``bf16_linear`` calls with
    ``channel_scale_quantize_fp8_static_bf16`` (per-tensor activation quant) plus
    ``fp8_gemm_bf16``. FFN already runs in FP8 in the base class.
    """

    def enable_fp8_projections(self, qkv_in_scale, o_in_scale) -> None:
        self._qi = qkv_in_scale
        self._oi = o_in_scale
        self._qkv_w_fp8, self._qkv_w_scale, self._o_w_fp8, self._o_w_scale = [], [], [], []
        for i in range(self.w.layers):
            a, b = _quant_proj_weight(self.w.qkv_w[i])
            self._qkv_w_fp8.append(a)
            self._qkv_w_scale.append(b)
            a, b = _quant_proj_weight(self.w.o_w[i])
            self._o_w_fp8.append(a)
            self._o_w_scale.append(b)
        cs = self.x.shape[0]
        self._nf = torch.empty((cs, DEC_D), device=self.w.device, dtype=torch.float8_e4m3fn)
        self._af = torch.empty((cs, DEC_NH * DEC_HD), device=self.w.device, dtype=torch.float8_e4m3fn)
        self._ones_d = torch.ones((DEC_D,), device=self.w.device, dtype=torch.bfloat16)
        self._ones_a = torch.ones((DEC_NH * DEC_HD,), device=self.w.device, dtype=torch.bfloat16)

    def __call__(self) -> torch.Tensor:
        w = self.w
        s = self.s
        self.noise.copy_(s.noise0)
        for step in range(w.steps):
            self.gemm.bf16_linear_bias_bf16(self.noise, w.action_in_w, w.action_in_b, out=self.x)
            for i in range(w.layers):
                self.adapt.ada_rms_norm_style_bf16(
                    self.x, w.ones, w.style_attn[step, i], out=self.normed, gate_out=self.gate
                )
                self.gemm.channel_scale_quantize_fp8_static_bf16(self.normed, self._ones_d, self._qi[i], out=self._nf)
                self.ffn.fp8_gemm_bf16(self._nf, self._qkv_w_fp8[i], self._qi[i], self._qkv_w_scale[i], out=self.qkv_buf)
                self.k_cache[:, : s.encoder_seq_len].copy_(_layer_kv(s.encoder_k, i, name="encoder_k"))
                self.v_cache[:, : s.encoder_seq_len].copy_(_layer_kv(s.encoder_v, i, name="encoder_v"))
                self.qkv.qkv_split_rope_kvcache_bf16(
                    self.qkv_buf.view(1, s.chunk_size, -1), s.rope, DEC_NH, DEC_NKV, DEC_HD,
                    s.encoder_seq_len, self.q, self.k_cache, self.v_cache,
                )
                self.attn_kernel.fwd(self.q, self.k_cache, self.v_cache, out=self.attn_bthd, p_dropout=0.0, is_causal=False)
                self.gemm.channel_scale_quantize_fp8_static_bf16(self.attn, self._ones_a, self._oi[i], out=self._af)
                self.ffn.fp8_gemm_bf16(self._af, self._o_w_fp8[i], self._oi[i], self._o_w_scale[i], out=self.attn_o)
                self.residual.gate_residual_bf16(self.x, self.attn_o, self.gate, out=self.x)
                _, ffn_fp8, ffn_gate = self.adapt.gate_residual_ada_norm_fp8_static_bf16(
                    self.x, self.zero_x, self.zero_gate, w.ones, w.style_ffn[step, i], self.ref.input_scale[i],
                    out=self.ffn_fp8, gate_out=self.ffn_gate,
                )
                self.ffn.fp8_geglu_mlp_bf16(
                    ffn_fp8, w.gate_up_w_fp8[i], w.down_w_fp8[i], self.ref.input_scale[i], w.gate_up_w_scale[i],
                    self.ref.hidden_scale[i], w.down_w_scale[i], self.gate_up[i], self.hidden_fp8[i], self.ffn_out[i],
                )
                self.residual.gate_residual_bf16(self.x, self.ffn_out[i], ffn_gate, out=self.x)
            self.adapt.ada_rms_norm_style_bf16(self.x, w.ones, w.style_final[step], out=self.final_normed, gate_out=self.final_gate)
            self.gemm.bf16_linear_bias_bf16(self.final_normed, w.action_out_w, w.action_out_b, out=self.action)
            self.residual.gate_residual_bf16(self.noise, self.action, self.noise_gate, out=self.noise)
        return self.noise


class Captured:
    def __init__(self, runtime: HubDecoderLoop) -> None:
        self.runtime = runtime
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


def _time_us(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
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


def _percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def _compare(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float, float]:
    diff = (got.float() - expected.float()).abs()
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        float(_percentile(diff, 0.99).item()),
        float(F.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item()),
    )


def run(args: argparse.Namespace) -> Result:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    checkpoint = _resolve_weight_path(args.checkpoint)
    weights = DecoderWeights(checkpoint, layers=args.layers, steps=args.steps, device=device)
    state = DecoderState(
        weights=weights,
        chunk_size=10,
        encoder_seq_len=args.encoder_seq_len,
        seed=args.seed,
    )
    reference = TorchDecoderReference(
        weights,
        state,
        scale_safety=args.scale_safety,
        calibration_input=args.calibration_input,
    )
    if args.calibration_output is not None:
        reference.save_calibration(
            args.calibration_output,
            checkpoint=checkpoint,
            encoder_seq_len=args.encoder_seq_len,
        )
    runtime = HubDecoderLoop(
        weights,
        state,
        reference,
        local_gemm_artifact=args.local_gemm_artifact,
        local_qkv_artifact=args.local_qkv_artifact,
        local_ffn_artifact=args.local_ffn_artifact,
        local_residual_artifact=args.local_residual_artifact,
        attention_backend=args.attention_backend,
    )

    expected = reference()
    got = runtime()
    torch.cuda.synchronize()
    max_abs, mean_abs, p99_abs, cosine = _compare(got, expected)
    if p99_abs > args.p99_abs_limit or cosine < args.cosine_limit:
        raise RuntimeError(
            f"correctness failed: max_abs={max_abs:.6f} p99_abs={p99_abs:.6f} cosine={cosine:.8f}"
        )

    eager_us = _time_us(reference, warmup=args.warmup, iters=args.iters)
    runtime_us = _time_us(runtime, warmup=args.warmup, iters=args.iters)
    graph_us = None
    graph_status = "not_requested"
    if args.cuda_graph:
        try:
            captured = Captured(runtime)
            graph_us = _time_us(captured.replay, warmup=args.warmup, iters=args.iters)
            graph_status = "ok"
        except Exception as exc:  # noqa: BLE001
            graph_status = f"unsupported: {type(exc).__name__}: {exc}"

    return Result(
        checkpoint=str(checkpoint),
        layers=args.layers,
        steps=args.steps,
        device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        eager_us=eager_us,
        runtime_us=runtime_us,
        graph_us=graph_us,
        runtime_vs_eager=eager_us / runtime_us,
        graph_vs_eager=None if graph_us is None else eager_us / graph_us,
        graph_status=graph_status,
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        cosine=cosine,
        gemm_source=runtime.gemm_source,
        attention_source=runtime.attention_source,
        qkv_source=runtime.qkv_source,
        ffn_source=runtime.ffn_source,
        residual_source=runtime.residual_source,
        calibration_mode=reference.calibration_mode,
        calibration_source=reference.calibration_source,
        calibration_path=reference.calibration_path,
        calibration_steps=args.steps,
        scale_safety=args.scale_safety,
        kernel_coverage=[
            "bf16_linear_bf16",
            "bf16_linear_bias_bf16",
            "ada_rms_norm_style_bf16",
            "gate_residual_ada_norm_fp8_static_bf16",
            "qkv_split_rope_kvcache_bf16",
            "flash_attn2.fwd" if args.attention_backend == "fa2" else "torch SDPA",
            "fp8_geglu_mlp_bf16",
            "gate_residual_bf16",
            "CUDA Graph replay",
        ],
        torch_gaps=[],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/heima/suliang/PI/checkpoints/pi05_libero_pytorch")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--encoder-seq-len", type=int, default=560)
    parser.add_argument("--scale-safety", type=float, default=1.05)
    parser.add_argument("--local-gemm-artifact", default=None)
    parser.add_argument("--local-qkv-artifact", default=None)
    parser.add_argument("--local-ffn-artifact", default=None)
    parser.add_argument("--local-residual-artifact", default=None)
    parser.add_argument("--calibration-input", type=Path)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--attention-backend", choices=("fa2", "sdpa"), default="fa2")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--p99-abs-limit", type=float, default=0.5)
    parser.add_argument("--cosine-limit", type=float, default=0.9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.layers <= DEC_L:
        raise ValueError(f"--layers must be in [1, {DEC_L}]")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    result = run(args)
    payload = asdict(result)
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
