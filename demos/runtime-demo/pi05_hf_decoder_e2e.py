#!/usr/bin/env python3
"""PI0.5 real-input E2E bridge and HF-kernel runtime path.

This script removes the synthetic encoder-KV boundary from
``pi05_decoder_loop_hub.py`` without mixing dependency stacks:

1. ``export-encoder`` runs in the OpenPI/official-FlashRT environment. It loads
   a real LIBERO frame, runs FlashRT once for reference tensors, and writes a
   torch bundle containing normalized images, projected encoder input, real
   decoder K/V cache inputs, and initial noise.
2. ``run`` runs the checkpoint-backed HF-kernel decoder against official
   encoder K/V.
3. ``run-encoder-decoder`` runs the HF-kernel Gemma encoder + decoder starting
   from official projected encoder input.
4. ``run-vision-encoder-decoder`` runs the HF-kernel SigLIP vision/projector,
   Gemma encoder, and decoder starting from normalized LIBERO images.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
PI_ROOT = ROOT.parent
DEFAULT_CKPT = PI_ROOT / "checkpoints" / "pi05_libero_pytorch"
DEFAULT_FLASHRT_ROOT = PI_ROOT / "official" / "FlashRT"
DEFAULT_LIBERO_ROOT = PI_ROOT / "openpi-compiler" / "RL" / "data" / "libero_rollouts"

ACTION_DIM = 32
VIS_L = 27
VIS_D = 1152
VIS_H = 4304
VIS_NH = 16
VIS_HD = 72
VIS_SEQ_PER_VIEW = 256
VIS_PATCH_FLAT = 14 * 14 * 3
DEC_L = 18
ENC_D = 2048
ENC_H = 16384
ENC_NH = 8
DEC_NKV = 1
DEC_HD = 256
LIBERO_TASKS = {
    8: "put both moka pots on the stove",
    9: "put the yellow and white mug in the microwave and close it",
}


@dataclass
class BridgeResult:
    name: str
    status: str
    scope: str
    checkpoint: str
    encoder_bundle: str
    input_source: str
    prompt: str
    frame_index: int
    layers: int
    steps: int
    encoder_seq_len: int
    device: str
    torch_version: str
    cuda_version: str | None
    first_us: float
    runtime_us: float
    graph_us: float | None
    graph_status: str
    runtime_vs_first: float
    graph_vs_first: float | None
    max_abs: float
    mean_abs: float
    p99_abs: float
    mse: float
    cosine: float
    official_flashrt_max_abs: float | None
    official_flashrt_mean_abs: float | None
    official_flashrt_p99_abs: float | None
    official_flashrt_mse: float | None
    official_flashrt_cosine: float | None
    encoder_reference_p99_abs: float | None
    encoder_reference_cosine: float | None
    encoder_official_p99_abs: float | None
    encoder_official_cosine: float | None
    vision_projector_p99_abs: float | None
    vision_projector_cosine: float | None
    calibration_mode: str
    calibration_source: str
    encoder_calibration_mode: str | None
    encoder_calibration_source: str | None
    kernel_coverage: list[str]
    torch_gaps: list[str]


def _resolve_weight_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "model.safetensors"
    if not p.exists():
        raise FileNotFoundError(f"checkpoint weights not found: {p}")
    return p


def _stats_us(times: list[float]) -> float:
    return statistics.mean(float(t) for t in times)


def _percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def _compare(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float, float, float]:
    diff = (got.float() - expected.float()).abs()
    sq = (got.float() - expected.float()).pow(2)
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        float(_percentile(diff, 0.99).item()),
        float(F.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item()),
        float(sq.mean().item()),
    )


def _scale_from_amax(x: torch.Tensor, *, safety: float = 1.0) -> torch.Tensor:
    amax = float(x.float().abs().max().item())
    scale = max(amax * safety / 448.0, 1e-12)
    return torch.tensor([scale], device=x.device, dtype=torch.float32).contiguous()


def _quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float(), -448.0, 448.0).to(torch.float8_e4m3fn)


def _dequant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.float() * scale.float()


def _time_us(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        out = fn()
        if not torch.isfinite(out).all():
            raise RuntimeError("non-finite output during warmup")
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        if not torch.isfinite(out).all():
            raise RuntimeError("non-finite output during benchmark")
        times.append(start.elapsed_time(end) * 1000.0)
    return _stats_us(times)


def _load_libero_frame(data_root: Path, frame_index: int) -> tuple[dict[str, Any], str, int]:
    from PIL import Image
    import numpy as np
    import pandas as pd

    base = np.array(Image.open(data_root / "images" / f"base_{frame_index:06d}.png"))
    wrist = np.array(Image.open(data_root / "images" / f"wrist_{frame_index:06d}.png"))
    df = pd.read_parquet(data_root / "data" / "chunk-000" / "file-000.parquet")
    row = df[df["index"] == frame_index].iloc[0]
    state_raw = np.asarray(row["observation.state"], dtype=np.float32)
    state = np.zeros(ACTION_DIM, dtype=np.float32)
    state[: state_raw.shape[0]] = state_raw
    task_index = int(row["task_index"])
    prompt = LIBERO_TASKS.get(task_index, f"libero task {task_index}")
    return {"image": base, "wrist_image": wrist, "state": state}, prompt, task_index


def _import_official_flashrt(root: Path) -> None:
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def export_encoder(args: argparse.Namespace) -> None:
    _import_official_flashrt(args.flashrt_root)
    import numpy as np
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    obs, prompt, task_index = _load_libero_frame(args.libero_root, args.frame_index)
    checkpoint = _resolve_weight_path(args.checkpoint)

    frontend = Pi05TorchFrontendRtx(
        checkpoint.parent,
        num_views=args.num_views,
        num_steps=args.steps,
        use_fp8=args.encoder_use_fp8,
    )
    frontend.set_prompt(prompt, state=obs["state"])
    if frontend.pipeline is None:
        raise RuntimeError("set_prompt did not build a pipeline")

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream_int = stream.cuda_stream
        frontend._fill_img_buf(obs)
        noise = torch.randn(
            frontend.chunk_size, ACTION_DIM, device="cuda", dtype=torch.bfloat16
        )
        frontend._copy_tensor_to_pipeline_buf_stream(
            frontend._img_buf, frontend.pipeline.input_images_buf, stream_int
        )
        frontend._copy_tensor_to_pipeline_buf_stream(
            noise, frontend.pipeline.input_noise_buf, stream_int
        )
        frontend.pipeline._copy_lang_embeds_to_encoder_x(stream=stream_int)
        frontend.pipeline.vision_encoder(stream=stream_int)
        pipe = frontend.pipeline
        pipe.fvk.layer_norm(
            pipe.bufs["vision_x_pooled"].ptr.value,
            pipe.weights["vision_final_norm_w"],
            pipe.weights["vision_final_norm_b"],
            pipe.bufs["vision_x_norm"].ptr.value,
            pipe.vision_seq_enc,
            VIS_D,
            1e-5,
            stream=stream_int,
        )
        pipe.gemm.bf16_nn(
            pipe.bufs["vision_x_norm"].ptr.value,
            pipe.weights["encoder_multi_modal_projector_w"],
            pipe.bufs["encoder_x"].ptr.value,
            pipe.vision_seq_enc,
            ENC_D,
            VIS_D,
            stream=stream_int,
        )
        pipe.fvk.add_bias_bf16(
            pipe.bufs["encoder_x"].ptr.value,
            pipe.weights["encoder_multi_modal_projector_b"],
            pipe.vision_seq_enc,
            ENC_D,
            stream=stream_int,
        )
        encoder_x_input = torch.empty(
            (frontend.pipeline.encoder_seq_len, ENC_D),
            device="cuda",
            dtype=torch.bfloat16,
        )
        frontend._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(encoder_x_input.data_ptr()),
            frontend.pipeline.bufs["encoder_x"].ptr,
            encoder_x_input.numel() * encoder_x_input.element_size(),
            3,
            stream_int,
        )
        frontend.pipeline.transformer_encoder(stream=stream_int)
        official_out = None
        if args.include_official_decoder:
            frontend.pipeline.transformer_decoder(stream=stream_int)
            official_out = torch.empty_like(noise)
            frontend._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(official_out.data_ptr()),
                frontend.pipeline.input_noise_buf.ptr,
                official_out.numel() * official_out.element_size(),
                3,
                stream_int,
            )
    frontend._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream.cuda_stream))

    encoder_seq_len = int(frontend.pipeline.encoder_seq_len)
    enc_k = frontend.attn_backend.enc_K[
        : args.layers, :encoder_seq_len, :DEC_NKV, :DEC_HD
    ].detach().contiguous().cpu()
    enc_v = frontend.attn_backend.enc_V[
        : args.layers, :encoder_seq_len, :DEC_NKV, :DEC_HD
    ].detach().contiguous().cpu()

    bundle = {
        "version": 1,
        "kind": "pi05_real_encoder_kv",
        "checkpoint": str(checkpoint),
        "input_source": "libero_rollout",
        "libero_root": str(args.libero_root),
        "frame_index": int(args.frame_index),
        "task_index": task_index,
        "prompt": prompt,
        "num_views": int(args.num_views),
        "layers": int(args.layers),
        "steps": int(args.steps),
        "encoder_seq_len": encoder_seq_len,
        "chunk_size": int(frontend.chunk_size),
        "input_images": frontend._img_buf.detach().contiguous().cpu(),
        "noise0": noise.detach().contiguous().cpu(),
        "encoder_x_input": encoder_x_input.detach().contiguous().cpu(),
        "encoder_k": enc_k,
        "encoder_v": enc_v,
        "official_decoder_out": (
            None if official_out is None else official_out.detach().contiguous().cpu()
        ),
        "state": torch.from_numpy(obs["state"]).contiguous(),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "encoder_source": "official_flashrt_rtx",
        "encoder_use_fp8": bool(args.encoder_use_fp8),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    print(json.dumps({k: v for k, v in bundle.items() if not torch.is_tensor(v)}, indent=2))


def _load_decoder_module():
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pi05_decoder_loop_hub as dec

    return dec


def _make_encoder_rope(seq_len: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device, dtype=torch.float64)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, DEC_HD, 2, device=device, dtype=torch.float64) / DEC_HD)
    )
    phase = positions[:, None] * inv_freq[None, :]
    cos = torch.cos(phase).to(torch.bfloat16)
    sin = torch.sin(phase).to(torch.bfloat16)
    return torch.stack([cos, sin], dim=-1).reshape(seq_len, DEC_HD).contiguous()


def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    return (xf * inv * weight.float().view(1, -1)).to(torch.bfloat16)


def _layer_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = ((xf - mean) * (xf - mean)).mean(dim=-1, keepdim=True)
    return ((xf - mean) * torch.rsqrt(var + eps) * weight.float().view(1, -1) + bias.float().view(1, -1)).to(
        torch.bfloat16
    )


def _enc_qkv_split_rope_ref(
    packed_qkv: torch.Tensor,
    rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_dim = ENC_NH * DEC_HD
    kv_dim = DEC_NKV * DEC_HD
    q = packed_qkv[:, :q_dim].view(1, packed_qkv.shape[0], ENC_NH, DEC_HD)
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


def _enc_sdpa_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    k_rep = k.repeat_interleave(ENC_NH // DEC_NKV, dim=2)
    v_rep = v.repeat_interleave(ENC_NH // DEC_NKV, dim=2)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k_rep.transpose(1, 2),
        v_rep.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    )
    return out.transpose(1, 2).contiguous().view(q.shape[1], ENC_NH * DEC_HD)


class VisionWeights:
    def __init__(self, checkpoint: Path, *, num_views: int, device: torch.device) -> None:
        from safetensors import safe_open

        self.num_views = int(num_views)
        self.device = device
        with safe_open(str(checkpoint), framework="pt") as f:
            keys = set(f.keys())
            strip = "model." if any(k.startswith("model.") for k in keys) else ""

            def get_raw(key: str) -> torch.Tensor:
                return f.get_tensor(strip + key)

            def get_bf16(key: str) -> torch.Tensor:
                return get_raw(key).to(device=device, dtype=torch.bfloat16).contiguous()

            vp = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
            pe_w = get_raw(f"{vp}.embeddings.patch_embedding.weight").to(
                device=device,
                dtype=torch.bfloat16,
            )
            self.patch_w = pe_w.permute(2, 3, 1, 0).contiguous().view(
                VIS_PATCH_FLAT,
                VIS_D,
            )
            self.patch_b = get_bf16(f"{vp}.embeddings.patch_embedding.bias")
            pos = get_bf16(f"{vp}.embeddings.position_embedding.weight")
            self.pos_embed = pos.repeat(self.num_views, 1).contiguous()

            self.qkv_w: list[torch.Tensor] = []
            self.qkv_b: list[torch.Tensor] = []
            self.o_w: list[torch.Tensor] = []
            self.o_b: list[torch.Tensor] = []
            self.up_w: list[torch.Tensor] = []
            self.up_b: list[torch.Tensor] = []
            self.down_w: list[torch.Tensor] = []
            self.down_b: list[torch.Tensor] = []
            self.ln1_w: list[torch.Tensor] = []
            self.ln1_b: list[torch.Tensor] = []
            self.ln2_w: list[torch.Tensor] = []
            self.ln2_b: list[torch.Tensor] = []
            for i in range(VIS_L):
                lp = f"{vp}.encoder.layers.{i}"
                q_w = get_bf16(f"{lp}.self_attn.q_proj.weight")
                k_w = get_bf16(f"{lp}.self_attn.k_proj.weight")
                v_w = get_bf16(f"{lp}.self_attn.v_proj.weight")
                self.qkv_w.append(torch.cat([q_w, k_w, v_w], dim=0).t().contiguous())
                q_b = get_bf16(f"{lp}.self_attn.q_proj.bias")
                k_b = get_bf16(f"{lp}.self_attn.k_proj.bias")
                v_b = get_bf16(f"{lp}.self_attn.v_proj.bias")
                self.qkv_b.append(torch.cat([q_b, k_b, v_b], dim=0).contiguous())
                self.o_w.append(get_bf16(f"{lp}.self_attn.out_proj.weight").t().contiguous())
                self.o_b.append(get_bf16(f"{lp}.self_attn.out_proj.bias"))
                self.up_w.append(get_bf16(f"{lp}.mlp.fc1.weight").t().contiguous())
                self.up_b.append(get_bf16(f"{lp}.mlp.fc1.bias"))
                self.down_w.append(get_bf16(f"{lp}.mlp.fc2.weight").t().contiguous())
                self.down_b.append(get_bf16(f"{lp}.mlp.fc2.bias"))
                self.ln1_w.append(get_bf16(f"{lp}.layer_norm1.weight"))
                self.ln1_b.append(get_bf16(f"{lp}.layer_norm1.bias"))
                self.ln2_w.append(get_bf16(f"{lp}.layer_norm2.weight"))
                self.ln2_b.append(get_bf16(f"{lp}.layer_norm2.bias"))

            self.final_norm_w = get_bf16(f"{vp}.post_layernorm.weight")
            self.final_norm_b = get_bf16(f"{vp}.post_layernorm.bias")
            mp = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
            self.projector_w = get_bf16(f"{mp}.weight").t().contiguous()
            self.projector_b = get_bf16(f"{mp}.bias")


class EncoderWeights:
    def __init__(self, checkpoint: Path, *, layers: int, device: torch.device) -> None:
        from safetensors import safe_open

        dec = _load_decoder_module()
        self.layers = layers
        self.device = device
        self.ones = torch.ones((ENC_D,), device=device, dtype=torch.bfloat16)
        self.qkv_w: list[torch.Tensor] = []
        self.o_w: list[torch.Tensor] = []
        self.gate_up_w_fp8: list[torch.Tensor] = []
        self.down_w_fp8: list[torch.Tensor] = []
        self.gate_up_w_scale: list[torch.Tensor] = []
        self.down_w_scale: list[torch.Tensor] = []
        with safe_open(str(checkpoint), framework="pt") as f:
            keys = set(f.keys())
            strip = "model." if any(k.startswith("model.") for k in keys) else ""

            def get_raw(key: str) -> torch.Tensor:
                return f.get_tensor(strip + key)

            def get_bf16(key: str) -> torch.Tensor:
                return get_raw(key).to(device=device, dtype=torch.bfloat16).contiguous()

            ep = "paligemma_with_expert.paligemma.model.language_model.layers"
            for i in range(layers):
                attn_scale = get_raw(f"{ep}.{i}.input_layernorm.weight").to(
                    device=device, dtype=torch.float32
                )
                fuse_attn = 1.0 + attn_scale
                q_w = get_raw(f"{ep}.{i}.self_attn.q_proj.weight").to(device=device, dtype=torch.float32)
                k_w = get_raw(f"{ep}.{i}.self_attn.k_proj.weight").to(device=device, dtype=torch.float32)
                v_w = get_raw(f"{ep}.{i}.self_attn.v_proj.weight").to(device=device, dtype=torch.float32)
                q_w = dec._interleave_qk(q_w, ENC_NH) * fuse_attn.unsqueeze(0)
                k_w = dec._interleave_qk(k_w, DEC_NKV) * fuse_attn.unsqueeze(0)
                v_w = v_w * fuse_attn.unsqueeze(0)
                self.qkv_w.append(
                    torch.cat([q_w, k_w, v_w], dim=0).t().to(torch.bfloat16).contiguous()
                )
                self.o_w.append(get_bf16(f"{ep}.{i}.self_attn.o_proj.weight").t().contiguous())

                ffn_scale = get_raw(f"{ep}.{i}.post_attention_layernorm.weight").to(
                    device=device, dtype=torch.float32
                )
                fuse_ffn = 1.0 + ffn_scale
                gate = (
                    get_raw(f"{ep}.{i}.mlp.gate_proj.weight").to(device=device, dtype=torch.float32)
                    * fuse_ffn.unsqueeze(0)
                ).to(torch.bfloat16)
                up = (
                    get_raw(f"{ep}.{i}.mlp.up_proj.weight").to(device=device, dtype=torch.float32)
                    * fuse_ffn.unsqueeze(0)
                ).to(torch.bfloat16)
                gate_up = torch.cat([gate, up], dim=0).contiguous()
                down = get_bf16(f"{ep}.{i}.mlp.down_proj.weight")
                gu_s = _scale_from_amax(gate_up, safety=1.0)
                dn_s = _scale_from_amax(down, safety=1.0)
                self.gate_up_w_scale.append(gu_s)
                self.down_w_scale.append(dn_s)
                self.gate_up_w_fp8.append(_quantize_fp8(gate_up, gu_s))
                self.down_w_fp8.append(_quantize_fp8(down, dn_s))


class TorchEncoderReference:
    def __init__(
        self,
        weights: EncoderWeights,
        encoder_x_input: torch.Tensor,
        *,
        scale_safety: float,
        calibration_input: Path | None = None,
    ) -> None:
        self.w = weights
        self.encoder_x_input = encoder_x_input
        self.rope = _make_encoder_rope(encoder_x_input.shape[0], weights.device)
        self.scale_safety = scale_safety
        if calibration_input is not None:
            payload = json.loads(calibration_input.read_text())
            self.input_scale, self.hidden_scale = self._scales_from_payload(payload)
            self.calibration_mode = str(payload.get("mode", "static_all_layers_amax"))
            self.calibration_source = f"file:{calibration_input}"
        else:
            self.input_scale, self.hidden_scale = self._calibrate()
            self.calibration_mode = "static_all_layers_amax"
            self.calibration_source = "computed"

    def _scales_from_payload(self, payload: dict[str, Any]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        values_in = payload.get("input_scales")
        values_hid = payload.get("hidden_scales")
        if not isinstance(values_in, list) or not isinstance(values_hid, list):
            raise ValueError("encoder calibration JSON must contain input_scales and hidden_scales")
        if len(values_in) != self.w.layers or len(values_hid) != self.w.layers:
            raise ValueError("encoder calibration scale count mismatch")

        def one(value: object, name: str) -> torch.Tensor:
            scale = float(value)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"invalid encoder scale {name}={scale}")
            return torch.tensor([scale], device=self.w.device, dtype=torch.float32).contiguous()

        return (
            [one(v, f"input_scales[{i}]") for i, v in enumerate(values_in)],
            [one(v, f"hidden_scales[{i}]") for i, v in enumerate(values_hid)],
        )

    def _calibrate(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        input_scales: list[torch.Tensor] = []
        hidden_scales: list[torch.Tensor] = []
        x = self.encoder_x_input.clone()
        for i in range(self.w.layers):
            normed = _rms_norm_ref(x, self.w.ones)
            qkv = normed @ self.w.qkv_w[i]
            q, k, v = _enc_qkv_split_rope_ref(qkv, self.rope)
            if i == self.w.layers - 1:
                break
            attn = _enc_sdpa_gqa(q, k, v)
            attn_o = attn @ self.w.o_w[i]
            x = (x.float() + attn_o.float()).to(torch.bfloat16)
            ffn_normed = _rms_norm_ref(x, self.w.ones)
            x_scale = _scale_from_amax(ffn_normed, safety=self.scale_safety)
            x_fp8 = _quantize_fp8(ffn_normed, x_scale)
            gate_up = (
                _dequant_fp8(x_fp8, x_scale)
                @ _dequant_fp8(self.w.gate_up_w_fp8[i], self.w.gate_up_w_scale[i]).t()
            ).to(torch.bfloat16)
            gate, up = gate_up.float().chunk(2, dim=1)
            hidden = F.gelu(gate, approximate="tanh") * up
            hidden_scale = _scale_from_amax(hidden.to(torch.bfloat16), safety=self.scale_safety)
            hidden_fp8 = _quantize_fp8(hidden.to(torch.bfloat16), hidden_scale)
            down = (
                _dequant_fp8(hidden_fp8, hidden_scale)
                @ _dequant_fp8(self.w.down_w_fp8[i], self.w.down_w_scale[i]).t()
            ).to(torch.bfloat16)
            x = (x.float() + down.float()).to(torch.bfloat16)
            input_scales.append(x_scale)
            hidden_scales.append(hidden_scale)
        if len(input_scales) != max(0, self.w.layers - 1):
            raise RuntimeError("encoder calibration did not visit expected FFN layers")
        input_scales.append(torch.tensor([1.0], device=self.w.device, dtype=torch.float32))
        hidden_scales.append(torch.tensor([1.0], device=self.w.device, dtype=torch.float32))
        return input_scales, hidden_scales

    def save_calibration(self, path: Path, *, checkpoint: Path, encoder_seq_len: int) -> None:
        payload = {
            "mode": self.calibration_mode,
            "checkpoint": str(checkpoint),
            "layers": self.w.layers,
            "encoder_seq_len": encoder_seq_len,
            "scale_safety": self.scale_safety,
            "input_scales": [float(s.item()) for s in self.input_scale],
            "hidden_scales": [float(s.item()) for s in self.hidden_scale],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.encoder_x_input.shape[0]
        x = self.encoder_x_input.clone()
        k_all = torch.empty(
            (self.w.layers, seq, DEC_NKV, DEC_HD), device=self.w.device, dtype=torch.bfloat16
        )
        v_all = torch.empty_like(k_all)
        for i in range(self.w.layers):
            normed = _rms_norm_ref(x, self.w.ones)
            qkv = normed @ self.w.qkv_w[i]
            q, k, v = _enc_qkv_split_rope_ref(qkv, self.rope)
            k_all[i].copy_(k[0])
            v_all[i].copy_(v[0])
            if i == self.w.layers - 1:
                break
            attn = _enc_sdpa_gqa(q, k, v)
            attn_o = attn @ self.w.o_w[i]
            x = (x.float() + attn_o.float()).to(torch.bfloat16)
            ffn_normed = _rms_norm_ref(x, self.w.ones)
            x_fp8 = _quantize_fp8(ffn_normed, self.input_scale[i])
            gate_up = (
                _dequant_fp8(x_fp8, self.input_scale[i])
                @ _dequant_fp8(self.w.gate_up_w_fp8[i], self.w.gate_up_w_scale[i]).t()
            ).to(torch.bfloat16)
            gate, up = gate_up.float().chunk(2, dim=1)
            hidden = F.gelu(gate, approximate="tanh") * up
            hidden_fp8 = _quantize_fp8(hidden.to(torch.bfloat16), self.hidden_scale[i])
            down = (
                _dequant_fp8(hidden_fp8, self.hidden_scale[i])
                @ _dequant_fp8(self.w.down_w_fp8[i], self.w.down_w_scale[i]).t()
            ).to(torch.bfloat16)
            x = (x.float() + down.float()).to(torch.bfloat16)
        return k_all, v_all


class HubVisionRuntime:
    def __init__(
        self,
        weights: VisionWeights,
        input_images: torch.Tensor,
        encoder_x_template: torch.Tensor,
        *,
        local_gemm_artifact: str | None,
        local_qkv_artifact: str | None,
        local_residual_artifact: str | None,
        local_norm_artifact: str | None,
        local_layout_artifact: str | None,
    ) -> None:
        dec = _load_decoder_module()
        self.w = weights
        self.num_views = weights.num_views
        self.seq = self.num_views * VIS_SEQ_PER_VIEW
        self.images = input_images
        self.encoder_x_input = encoder_x_template.clone()
        self.encoder_x_vision = self.encoder_x_input[: self.seq]
        self.gemm, self.gemm_source = dec._load_module(
            local_gemm_artifact,
            "flashrt/flashrt-gemm-epilogues",
            "flashrt_gemm_epilogues",
        )
        self.qkv, self.qkv_source = dec._load_module(
            local_qkv_artifact,
            "flashrt/flashrt-qkv-cache-rope",
            "flashrt_qkv_cache_rope",
        )
        self.residual, self.residual_source = dec._load_module(
            local_residual_artifact,
            "flashrt/flashrt-vla-residual-gates",
            "flashrt_vla_residual_gates",
        )
        self.norm, self.norm_source = dec._load_module(
            local_norm_artifact,
            "flashrt/flashrt-residual-norm-quant",
            "flashrt_residual_norm_quant",
        )
        self.layout, self.layout_source = dec._load_module(
            local_layout_artifact,
            "flashrt/flashrt-spatiotemporal-layout",
            "flashrt_spatiotemporal_layout",
        )
        self.attn_kernel = __import__("kernels").get_kernel(
            "kernels-community/flash-attn2",
            version=1,
            trust_remote_code=True,
        )
        device = weights.device
        self.patches = torch.empty((self.seq, VIS_PATCH_FLAT), device=device, dtype=torch.bfloat16)
        self.x = torch.empty((self.seq, VIS_D), device=device, dtype=torch.bfloat16)
        self.normed = torch.empty_like(self.x)
        self.qkv_buf = torch.empty((self.seq, 3 * VIS_D), device=device, dtype=torch.bfloat16)
        self.q = torch.empty(
            (self.num_views, VIS_SEQ_PER_VIEW, VIS_NH, VIS_HD),
            device=device,
            dtype=torch.bfloat16,
        )
        self.k = torch.empty_like(self.q)
        self.v = torch.empty_like(self.q)
        self.attn_bthd = torch.empty_like(self.q)
        self.attn = self.attn_bthd.view(self.seq, VIS_D)
        self.hidden = torch.empty((self.seq, VIS_H), device=device, dtype=torch.bfloat16)

    def __call__(self) -> torch.Tensor:
        self.layout.patch_im2col_bf16(self.images, out=self.patches)
        self.gemm.bf16_linear_bf16(self.patches, self.w.patch_w, out=self.x)
        self.residual.bias_residual_bf16(self.x, self.w.pos_embed, self.w.patch_b, out=self.x)
        self.norm.layer_norm_bf16(self.x, self.w.ln1_w[0], self.w.ln1_b[0], out=self.normed)
        for i in range(VIS_L):
            self.gemm.bf16_linear_bias_bf16(
                self.normed,
                self.w.qkv_w[i],
                self.w.qkv_b[i],
                out=self.qkv_buf,
            )
            self.qkv.qkv_split_bf16(
                self.qkv_buf.view(self.num_views, VIS_SEQ_PER_VIEW, 3 * VIS_D),
                VIS_NH,
                VIS_HD,
                self.q,
                self.k,
                self.v,
            )
            self.attn_kernel.fwd(
                self.q,
                self.k,
                self.v,
                out=self.attn_bthd,
                p_dropout=0.0,
                is_causal=False,
            )
            self.gemm.bf16_linear_bf16(self.attn, self.w.o_w[i], out=self.normed)
            self.residual.bias_residual_bf16(self.x, self.normed, self.w.o_b[i], out=self.x)
            self.norm.layer_norm_bf16(self.x, self.w.ln2_w[i], self.w.ln2_b[i], out=self.normed)
            self.gemm.bf16_gemm_bias_gelu(
                self.normed,
                self.w.up_w[i],
                self.w.up_b[i],
                out=self.hidden,
            )
            self.gemm.bf16_linear_bf16(self.hidden, self.w.down_w[i], out=self.normed)
            self.residual.bias_residual_bf16(
                self.x,
                self.normed,
                self.w.down_b[i],
                out=self.x,
            )
            if i != VIS_L - 1:
                self.norm.layer_norm_bf16(
                    self.x,
                    self.w.ln1_w[i + 1],
                    self.w.ln1_b[i + 1],
                    out=self.normed,
                )
        self.norm.layer_norm_bf16(
            self.x,
            self.w.final_norm_w,
            self.w.final_norm_b,
            out=self.normed,
        )
        self.gemm.bf16_linear_bias_bf16(
            self.normed,
            self.w.projector_w,
            self.w.projector_b,
            out=self.encoder_x_vision,
        )
        return self.encoder_x_input


class HubEncoderRuntime:
    def __init__(
        self,
        weights: EncoderWeights,
        encoder_x_input: torch.Tensor,
        reference: TorchEncoderReference,
        *,
        local_gemm_artifact: str | None,
        local_qkv_artifact: str | None,
        local_ffn_artifact: str | None,
        local_residual_artifact: str | None,
        local_norm_artifact: str | None,
    ) -> None:
        dec = _load_decoder_module()
        self.w = weights
        self.reference = reference
        self.gemm, self.gemm_source = dec._load_module(
            local_gemm_artifact,
            "flashrt/flashrt-gemm-epilogues",
            "flashrt_gemm_epilogues",
        )
        self.qkv, self.qkv_source = dec._load_module(
            local_qkv_artifact,
            "flashrt/flashrt-qkv-cache-rope",
            "flashrt_qkv_cache_rope",
        )
        self.ffn, self.ffn_source = dec._load_module(
            local_ffn_artifact,
            "flashrt/flashrt-fp8-swiglu-ffn",
            "flashrt_fp8_swiglu_ffn",
        )
        self.residual, self.residual_source = dec._load_module(
            local_residual_artifact,
            "flashrt/flashrt-vla-residual-gates",
            "flashrt_vla_residual_gates",
        )
        self.norm, self.norm_source = dec._load_module(
            local_norm_artifact,
            "flashrt/flashrt-residual-norm-quant",
            "flashrt_residual_norm_quant",
        )
        self.attn_kernel = __import__("kernels").get_kernel(
            "kernels-community/flash-attn2", version=1, trust_remote_code=True
        )
        seq = encoder_x_input.shape[0]
        device = weights.device
        self.encoder_x_input = encoder_x_input
        self.x = torch.empty_like(encoder_x_input)
        self.ones_gate = torch.ones_like(encoder_x_input)
        self.normed = torch.empty_like(encoder_x_input)
        self.qkv_buf = torch.empty(
            (seq, (ENC_NH + 2 * DEC_NKV) * DEC_HD), device=device, dtype=torch.bfloat16
        )
        self.q = torch.empty((1, seq, ENC_NH, DEC_HD), device=device, dtype=torch.bfloat16)
        self.k_all = torch.empty(
            (weights.layers, seq, DEC_NKV, DEC_HD), device=device, dtype=torch.bfloat16
        )
        self.v_all = torch.empty_like(self.k_all)
        self.attn_bthd = torch.empty((1, seq, ENC_NH, DEC_HD), device=device, dtype=torch.bfloat16)
        self.attn = self.attn_bthd.view(seq, ENC_NH * DEC_HD)
        self.attn_o = torch.empty_like(encoder_x_input)
        self.ffn_fp8 = torch.empty((seq, ENC_D), device=device, dtype=torch.float8_e4m3fn)
        self.gate_up = [
            torch.empty((seq, 2 * ENC_H), device=device, dtype=torch.bfloat16)
            for _ in range(weights.layers)
        ]
        self.hidden_fp8 = [
            torch.empty((seq, ENC_H), device=device, dtype=torch.float8_e4m3fn)
            for _ in range(weights.layers)
        ]
        self.ffn_out = [
            torch.empty((seq, ENC_D), device=device, dtype=torch.bfloat16)
            for _ in range(weights.layers)
        ]
        self.rope = reference.rope

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.x.copy_(self.encoder_x_input)
        for i in range(self.w.layers):
            self.norm.rms_norm_bf16(self.x, self.w.ones, out=self.normed)
            self.gemm.bf16_linear_bf16(self.normed, self.w.qkv_w[i], out=self.qkv_buf)
            self.qkv.qkv_split_rope_kvcache_bf16(
                self.qkv_buf.view(1, self.x.shape[0], -1),
                self.rope,
                ENC_NH,
                DEC_NKV,
                DEC_HD,
                0,
                self.q,
                self.k_all[i : i + 1],
                self.v_all[i : i + 1],
            )
            if i == self.w.layers - 1:
                break
            self.attn_kernel.fwd(
                self.q,
                self.k_all[i : i + 1],
                self.v_all[i : i + 1],
                out=self.attn_bthd,
                p_dropout=0.0,
                is_causal=False,
            )
            self.gemm.bf16_linear_bf16(self.attn, self.w.o_w[i], out=self.attn_o)
            self.residual.gate_residual_bf16(self.x, self.attn_o, self.ones_gate, out=self.x)
            self.norm.rms_norm_quant_fp8_static_bf16(
                self.x,
                self.w.ones,
                self.reference.input_scale[i],
                out=self.ffn_fp8,
            )
            self.ffn.fp8_geglu_mlp_bf16(
                self.ffn_fp8,
                self.w.gate_up_w_fp8[i],
                self.w.down_w_fp8[i],
                self.reference.input_scale[i],
                self.w.gate_up_w_scale[i],
                self.reference.hidden_scale[i],
                self.w.down_w_scale[i],
                self.gate_up[i],
                self.hidden_fp8[i],
                self.ffn_out[i],
            )
            self.residual.gate_residual_bf16(self.x, self.ffn_out[i], self.ones_gate, out=self.x)
        return self.k_all, self.v_all


class EncoderDecoderRuntime:
    def __init__(self, encoder: HubEncoderRuntime, decoder: Any) -> None:
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self) -> torch.Tensor:
        self.encoder()
        return self.decoder()


class VisionEncoderDecoderRuntime:
    def __init__(self, vision: HubVisionRuntime, encoder: HubEncoderRuntime, decoder: Any) -> None:
        self.vision = vision
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self) -> torch.Tensor:
        self.vision()
        self.encoder()
        return self.decoder()


def run_hf_decoder(args: argparse.Namespace) -> BridgeResult:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dec = _load_decoder_module()
    bundle = torch.load(args.encoder_bundle, map_location="cpu", weights_only=False)
    checkpoint = _resolve_weight_path(args.checkpoint or bundle["checkpoint"])
    layers = int(args.layers or bundle["layers"])
    steps = int(args.steps or bundle["steps"])
    if layers > int(bundle["layers"]):
        raise ValueError(f"requested layers={layers} exceeds bundle layers={bundle['layers']}")

    weights = dec.DecoderWeights(checkpoint, layers=layers, steps=steps, device=device)
    state = dec.DecoderState.__new__(dec.DecoderState)
    state.noise0 = bundle["noise0"].to(device=device, dtype=torch.bfloat16).contiguous()
    state.encoder_k = bundle["encoder_k"][:layers].to(device=device, dtype=torch.bfloat16)
    state.encoder_v = bundle["encoder_v"][:layers].to(device=device, dtype=torch.bfloat16)
    state.encoder_k = state.encoder_k.contiguous()
    state.encoder_v = state.encoder_v.contiguous()
    state.chunk_size = int(bundle["chunk_size"])
    state.encoder_seq_len = int(bundle["encoder_seq_len"])
    state.rope = dec._make_rope(state.chunk_size, state.encoder_seq_len, device)

    reference = dec.TorchDecoderReference(
        weights,
        state,
        scale_safety=args.scale_safety,
        calibration_input=args.calibration_input,
    )
    if args.calibration_output is not None:
        reference.save_calibration(
            args.calibration_output,
            checkpoint=checkpoint,
            encoder_seq_len=state.encoder_seq_len,
        )
    runtime = dec.HubDecoderLoop(
        weights,
        state,
        reference,
        local_gemm_artifact=args.local_gemm_artifact,
        local_qkv_artifact=args.local_qkv_artifact,
        local_ffn_artifact=args.local_ffn_artifact,
        local_residual_artifact=args.local_residual_artifact,
        attention_backend=args.attention_backend,
    )

    t0 = time.perf_counter()
    expected = reference()
    got = runtime()
    torch.cuda.synchronize()
    first_us = (time.perf_counter() - t0) * 1_000_000.0
    max_abs, mean_abs, p99_abs, cosine, mse = _compare(got, expected)
    if p99_abs > args.p99_abs_limit or cosine < args.cosine_limit:
        raise RuntimeError(
            f"correctness failed: max_abs={max_abs:.6f} "
            f"p99_abs={p99_abs:.6f} cosine={cosine:.8f}"
        )

    runtime_us = _time_us(runtime, warmup=args.warmup, iters=args.iters)
    graph_us = None
    graph_status = "not_requested"
    if args.cuda_graph:
        try:
            captured = dec.Captured(runtime)
            graph_us = _time_us(captured.replay, warmup=args.warmup, iters=args.iters)
            graph_status = "ok"
        except Exception as exc:  # noqa: BLE001
            graph_status = f"unsupported: {type(exc).__name__}: {exc}"

    official_metrics = (None, None, None, None, None)
    official_out = bundle.get("official_decoder_out")
    if torch.is_tensor(official_out):
        official_metrics = _compare(
            got, official_out.to(device=device, dtype=torch.bfloat16).contiguous()
        )

    return BridgeResult(
        name="pi05_hf_decoder_e2e",
        status="pass",
        scope=(
            "Real LIBERO input -> official FlashRT encoder KV -> "
            "HF-kernel PI0.5 decoder action-expert runtime"
        ),
        checkpoint=str(checkpoint),
        encoder_bundle=str(args.encoder_bundle),
        input_source=str(bundle.get("input_source", "unknown")),
        prompt=str(bundle.get("prompt", "")),
        frame_index=int(bundle.get("frame_index", -1)),
        layers=layers,
        steps=steps,
        encoder_seq_len=state.encoder_seq_len,
        device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        first_us=first_us,
        runtime_us=runtime_us,
        graph_us=graph_us,
        graph_status=graph_status,
        runtime_vs_first=first_us / runtime_us,
        graph_vs_first=None if graph_us is None else first_us / graph_us,
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        mse=mse,
        cosine=cosine,
        official_flashrt_max_abs=official_metrics[0],
        official_flashrt_mean_abs=official_metrics[1],
        official_flashrt_p99_abs=official_metrics[2],
        official_flashrt_mse=official_metrics[4],
        official_flashrt_cosine=official_metrics[3],
        encoder_reference_p99_abs=None,
        encoder_reference_cosine=None,
        encoder_official_p99_abs=None,
        encoder_official_cosine=None,
        vision_projector_p99_abs=None,
        vision_projector_cosine=None,
        calibration_mode=reference.calibration_mode,
        calibration_source=reference.calibration_source,
        encoder_calibration_mode=None,
        encoder_calibration_source=None,
        kernel_coverage=[
            "official_flashrt_vision_encoder_to_real_kv",
            "bf16_linear_bf16",
            "bf16_linear_bias_bf16",
            "ada_rms_norm_style_bf16",
            "qkv_split_rope_kvcache_bf16",
            "flash_attn2.fwd" if args.attention_backend == "fa2" else "torch SDPA",
            "fp8_geglu_mlp_bf16",
            "gate_residual_bf16",
            "CUDA Graph replay",
        ],
        torch_gaps=[],
    )


def run_hf_encoder_decoder(args: argparse.Namespace) -> BridgeResult:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dec = _load_decoder_module()
    bundle = torch.load(args.encoder_bundle, map_location="cpu", weights_only=False)
    if "encoder_x_input" not in bundle:
        raise RuntimeError(
            "encoder bundle does not contain encoder_x_input; re-run export-encoder "
            "with the current script"
        )
    checkpoint = _resolve_weight_path(args.checkpoint or bundle["checkpoint"])
    layers = int(args.layers or bundle["layers"])
    steps = int(args.steps or bundle["steps"])
    if layers > DEC_L:
        raise ValueError(f"requested layers={layers} exceeds PI0.5 encoder layers={DEC_L}")

    encoder_x_input = bundle["encoder_x_input"].to(
        device=device, dtype=torch.bfloat16
    ).contiguous()
    enc_weights = EncoderWeights(checkpoint, layers=layers, device=device)
    enc_ref = TorchEncoderReference(
        enc_weights,
        encoder_x_input,
        scale_safety=args.encoder_scale_safety,
        calibration_input=args.encoder_calibration_input,
    )
    if args.encoder_calibration_output is not None:
        enc_ref.save_calibration(
            args.encoder_calibration_output,
            checkpoint=checkpoint,
            encoder_seq_len=encoder_x_input.shape[0],
        )
    enc_runtime = HubEncoderRuntime(
        enc_weights,
        encoder_x_input,
        enc_ref,
        local_gemm_artifact=args.local_gemm_artifact,
        local_qkv_artifact=args.local_qkv_artifact,
        local_ffn_artifact=args.local_ffn_artifact,
        local_residual_artifact=args.local_residual_artifact,
        local_norm_artifact=args.local_norm_artifact,
    )

    ref_k, ref_v = enc_ref()
    rt_k, rt_v = enc_runtime()
    torch.cuda.synchronize()
    enc_ref_metrics_k = _compare(rt_k, ref_k)
    enc_ref_metrics_v = _compare(rt_v, ref_v)
    encoder_reference_p99 = max(enc_ref_metrics_k[2], enc_ref_metrics_v[2])
    encoder_reference_cos = min(enc_ref_metrics_k[3], enc_ref_metrics_v[3])
    if (
        encoder_reference_p99 > args.encoder_p99_abs_limit
        or encoder_reference_cos < args.encoder_cosine_limit
    ):
        raise RuntimeError(
            "encoder correctness failed vs HF reference: "
            f"p99_abs={encoder_reference_p99:.6f} cosine={encoder_reference_cos:.8f}"
        )

    encoder_official_p99 = None
    encoder_official_cos = None
    full_official_shape = layers == int(bundle["layers"])
    if full_official_shape:
        official_k = bundle["encoder_k"][:layers].to(device=device, dtype=torch.bfloat16).contiguous()
        official_v = bundle["encoder_v"][:layers].to(device=device, dtype=torch.bfloat16).contiguous()
        enc_official_metrics_k = _compare(rt_k, official_k)
        enc_official_metrics_v = _compare(rt_v, official_v)
        encoder_official_p99 = max(enc_official_metrics_k[2], enc_official_metrics_v[2])
        encoder_official_cos = min(enc_official_metrics_k[3], enc_official_metrics_v[3])

    dec_weights = dec.DecoderWeights(checkpoint, layers=layers, steps=steps, device=device)
    runtime_state = dec.DecoderState.__new__(dec.DecoderState)
    runtime_state.noise0 = bundle["noise0"].to(device=device, dtype=torch.bfloat16).contiguous()
    runtime_state.encoder_k = enc_runtime.k_all
    runtime_state.encoder_v = enc_runtime.v_all
    runtime_state.chunk_size = int(bundle["chunk_size"])
    runtime_state.encoder_seq_len = int(bundle["encoder_seq_len"])
    runtime_state.rope = dec._make_rope(
        runtime_state.chunk_size,
        runtime_state.encoder_seq_len,
        device,
    )

    ref_state = dec.DecoderState.__new__(dec.DecoderState)
    ref_state.noise0 = runtime_state.noise0
    ref_state.encoder_k = ref_k
    ref_state.encoder_v = ref_v
    ref_state.chunk_size = runtime_state.chunk_size
    ref_state.encoder_seq_len = runtime_state.encoder_seq_len
    ref_state.rope = runtime_state.rope
    dec_ref = dec.TorchDecoderReference(
        dec_weights,
        ref_state,
        scale_safety=args.scale_safety,
        calibration_input=args.calibration_input,
    )
    if args.calibration_output is not None:
        dec_ref.save_calibration(
            args.calibration_output,
            checkpoint=checkpoint,
            encoder_seq_len=runtime_state.encoder_seq_len,
        )
    dec_runtime = dec.HubDecoderLoop(
        dec_weights,
        runtime_state,
        dec_ref,
        local_gemm_artifact=args.local_gemm_artifact,
        local_qkv_artifact=args.local_qkv_artifact,
        local_ffn_artifact=args.local_ffn_artifact,
        local_residual_artifact=args.local_residual_artifact,
        attention_backend=args.attention_backend,
    )
    combined = EncoderDecoderRuntime(enc_runtime, dec_runtime)

    t0 = time.perf_counter()
    expected = dec_ref()
    got = combined()
    torch.cuda.synchronize()
    first_us = (time.perf_counter() - t0) * 1_000_000.0
    max_abs, mean_abs, p99_abs, cosine, mse = _compare(got, expected)
    if p99_abs > args.p99_abs_limit or cosine < args.cosine_limit:
        raise RuntimeError(
            f"combined correctness failed: max_abs={max_abs:.6f} "
            f"p99_abs={p99_abs:.6f} cosine={cosine:.8f}"
        )

    runtime_us = _time_us(combined, warmup=args.warmup, iters=args.iters)
    graph_us = None
    graph_status = "not_requested"
    if args.cuda_graph:
        try:
            captured = dec.Captured(combined)
            graph_us = _time_us(captured.replay, warmup=args.warmup, iters=args.iters)
            graph_status = "ok"
        except Exception as exc:  # noqa: BLE001
            graph_status = f"unsupported: {type(exc).__name__}: {exc}"

    official_metrics = (None, None, None, None, None)
    official_out = bundle.get("official_decoder_out")
    if torch.is_tensor(official_out) and full_official_shape and steps == int(bundle["steps"]):
        official_metrics = _compare(
            got, official_out.to(device=device, dtype=torch.bfloat16).contiguous()
        )

    return BridgeResult(
        name="pi05_hf_encoder_decoder_e2e",
        status="pass",
        scope=(
            "Real LIBERO input -> official FlashRT vision/projector encoder_x -> "
            "HF-kernel Gemma encoder KV -> HF-kernel PI0.5 decoder action-expert runtime"
        ),
        checkpoint=str(checkpoint),
        encoder_bundle=str(args.encoder_bundle),
        input_source=str(bundle.get("input_source", "unknown")),
        prompt=str(bundle.get("prompt", "")),
        frame_index=int(bundle.get("frame_index", -1)),
        layers=layers,
        steps=steps,
        encoder_seq_len=runtime_state.encoder_seq_len,
        device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        first_us=first_us,
        runtime_us=runtime_us,
        graph_us=graph_us,
        graph_status=graph_status,
        runtime_vs_first=first_us / runtime_us,
        graph_vs_first=None if graph_us is None else first_us / graph_us,
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        mse=mse,
        cosine=cosine,
        official_flashrt_max_abs=official_metrics[0],
        official_flashrt_mean_abs=official_metrics[1],
        official_flashrt_p99_abs=official_metrics[2],
        official_flashrt_mse=official_metrics[4],
        official_flashrt_cosine=official_metrics[3],
        encoder_reference_p99_abs=encoder_reference_p99,
        encoder_reference_cosine=encoder_reference_cos,
        encoder_official_p99_abs=encoder_official_p99,
        encoder_official_cosine=encoder_official_cos,
        vision_projector_p99_abs=None,
        vision_projector_cosine=None,
        calibration_mode=dec_ref.calibration_mode,
        calibration_source=dec_ref.calibration_source,
        encoder_calibration_mode=enc_ref.calibration_mode,
        encoder_calibration_source=enc_ref.calibration_source,
        kernel_coverage=[
            "official_flashrt_vision_projector_to_encoder_x",
            "rms_norm_bf16",
            "rms_norm_quant_fp8_static_bf16",
            "bf16_linear_bf16",
            "qkv_split_rope_kvcache_bf16",
            "flash_attn2.fwd",
            "fp8_geglu_mlp_bf16",
            "gate_residual_bf16",
            "ada_rms_norm_style_bf16",
            "bf16_linear_bias_bf16",
            "CUDA Graph replay",
        ],
        torch_gaps=["official FlashRT SigLIP vision/projector frontend"],
    )


def _quant_proj_weight(w_kn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a stored ``(K, N)`` projection weight to FP8 ``(N, K)`` + scale."""
    nk = w_kn.t().contiguous()
    scale = _scale_from_amax(nk, safety=1.0)
    return _quantize_fp8(nk, scale), scale


class Fp8HubEncoderRuntime(HubEncoderRuntime):
    """Encoder runtime with QKV/O projections in FP8 (published Hub kernels only)."""

    def calibrate(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        ref = self.reference
        x = self.encoder_x_input.clone()
        qi: list[torch.Tensor] = []
        oi: list[torch.Tensor] = []
        for i in range(self.w.layers):
            normed = _rms_norm_ref(x, self.w.ones)
            qi.append(_scale_from_amax(normed, safety=ref.scale_safety))
            q, k, v = _enc_qkv_split_rope_ref(normed @ self.w.qkv_w[i], ref.rope)
            if i == self.w.layers - 1:
                break
            attn = _enc_sdpa_gqa(q, k, v)
            oi.append(_scale_from_amax(attn, safety=ref.scale_safety))
            attn_o = (attn @ self.w.o_w[i]).to(torch.bfloat16)
            x = (x.float() + attn_o.float()).to(torch.bfloat16)
            ffn_normed = _rms_norm_ref(x, self.w.ones)
            gate_up = (
                _dequant_fp8(_quantize_fp8(ffn_normed, ref.input_scale[i]), ref.input_scale[i])
                @ _dequant_fp8(self.w.gate_up_w_fp8[i], self.w.gate_up_w_scale[i]).t()
            ).to(torch.bfloat16)
            gate, up = gate_up.float().chunk(2, dim=1)
            hidden = F.gelu(gate, approximate="tanh") * up
            down = (
                _dequant_fp8(_quantize_fp8(hidden.to(torch.bfloat16), ref.hidden_scale[i]), ref.hidden_scale[i])
                @ _dequant_fp8(self.w.down_w_fp8[i], self.w.down_w_scale[i]).t()
            ).to(torch.bfloat16)
            x = (x.float() + down.float()).to(torch.bfloat16)
        oi.append(torch.tensor([1.0], device=self.w.device, dtype=torch.float32))
        return qi, oi

    def enable_fp8(self, qkv_in_scale, o_in_scale) -> None:
        self._qi, self._oi = qkv_in_scale, o_in_scale
        self._qkv_w_fp8, self._qkv_w_scale, self._o_w_fp8, self._o_w_scale = [], [], [], []
        for i in range(self.w.layers):
            a, b = _quant_proj_weight(self.w.qkv_w[i])
            self._qkv_w_fp8.append(a)
            self._qkv_w_scale.append(b)
            a, b = _quant_proj_weight(self.w.o_w[i])
            self._o_w_fp8.append(a)
            self._o_w_scale.append(b)
        seq = self.x.shape[0]
        self._xf = torch.empty((seq, ENC_D), device=self.w.device, dtype=torch.float8_e4m3fn)
        self._af = torch.empty((seq, ENC_NH * DEC_HD), device=self.w.device, dtype=torch.float8_e4m3fn)
        self._oh = torch.ones((ENC_NH * DEC_HD,), device=self.w.device, dtype=torch.bfloat16)

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.x.copy_(self.encoder_x_input)
        for i in range(self.w.layers):
            self.norm.rms_norm_quant_fp8_static_bf16(self.x, self.w.ones, self._qi[i], out=self._xf)
            self.ffn.fp8_gemm_bf16(self._xf, self._qkv_w_fp8[i], self._qi[i], self._qkv_w_scale[i], out=self.qkv_buf)
            self.qkv.qkv_split_rope_kvcache_bf16(
                self.qkv_buf.view(1, self.x.shape[0], -1), self.rope, ENC_NH, DEC_NKV, DEC_HD, 0,
                self.q, self.k_all[i:i + 1], self.v_all[i:i + 1],
            )
            if i == self.w.layers - 1:
                break
            self.attn_kernel.fwd(self.q, self.k_all[i:i + 1], self.v_all[i:i + 1], out=self.attn_bthd, p_dropout=0.0, is_causal=False)
            self.gemm.channel_scale_quantize_fp8_static_bf16(self.attn, self._oh, self._oi[i], out=self._af)
            self.ffn.fp8_gemm_bf16(self._af, self._o_w_fp8[i], self._oi[i], self._o_w_scale[i], out=self.attn_o)
            self.residual.gate_residual_bf16(self.x, self.attn_o, self.ones_gate, out=self.x)
            self.norm.rms_norm_quant_fp8_static_bf16(self.x, self.w.ones, self.reference.input_scale[i], out=self.ffn_fp8)
            self.ffn.fp8_geglu_mlp_bf16(
                self.ffn_fp8, self.w.gate_up_w_fp8[i], self.w.down_w_fp8[i], self.reference.input_scale[i],
                self.w.gate_up_w_scale[i], self.reference.hidden_scale[i], self.w.down_w_scale[i],
                self.gate_up[i], self.hidden_fp8[i], self.ffn_out[i],
            )
            self.residual.gate_residual_bf16(self.x, self.ffn_out[i], self.ones_gate, out=self.x)
        return self.k_all, self.v_all


class Fp8HubVisionRuntime(HubVisionRuntime):
    """SigLIP vision runtime with QKV/O/FFN projections in FP8 (published kernels)."""

    def _setup_fp8(self) -> None:
        dec = _load_decoder_module()
        self._fp8ffn, _ = dec._load_module(None, "flashrt/flashrt-fp8-ffn", "flashrt_fp8_ffn")
        self._ones = torch.ones((VIS_D,), device=self.w.device, dtype=torch.bfloat16)
        self._zeros = torch.zeros((VIS_D,), device=self.w.device, dtype=torch.bfloat16)
        self._nf = torch.empty((self.seq, VIS_D), device=self.w.device, dtype=torch.float8_e4m3fn)
        self._af = torch.empty((self.seq, VIS_D), device=self.w.device, dtype=torch.float8_e4m3fn)
        self._hbf = torch.empty((self.seq, VIS_H), device=self.w.device, dtype=torch.bfloat16)
        self._hf = torch.empty((self.seq, VIS_H), device=self.w.device, dtype=torch.float8_e4m3fn)

    def calibrate(self, *, scale_safety: float):
        qi = [None] * VIS_L
        oi = [None] * VIS_L
        fi = [None] * VIS_L
        hi = [None] * VIS_L

        def sc(a):
            return _scale_from_amax_value(a, safety=scale_safety, device=self.w.device)

        self.layout.patch_im2col_bf16(self.images, out=self.patches)
        self.gemm.bf16_linear_bf16(self.patches, self.w.patch_w, out=self.x)
        self.residual.bias_residual_bf16(self.x, self.w.pos_embed, self.w.patch_b, out=self.x)
        self.norm.layer_norm_bf16(self.x, self.w.ln1_w[0], self.w.ln1_b[0], out=self.normed)
        for i in range(VIS_L):
            qi[i] = sc(self.normed.float().abs().max())
            self.gemm.bf16_linear_bias_bf16(self.normed, self.w.qkv_w[i], self.w.qkv_b[i], out=self.qkv_buf)
            self.qkv.qkv_split_bf16(self.qkv_buf.view(self.num_views, VIS_SEQ_PER_VIEW, 3 * VIS_D), VIS_NH, VIS_HD, self.q, self.k, self.v)
            self.attn_kernel.fwd(self.q, self.k, self.v, out=self.attn_bthd, p_dropout=0.0, is_causal=False)
            oi[i] = sc(self.attn.float().abs().max())
            self.gemm.bf16_linear_bf16(self.attn, self.w.o_w[i], out=self.normed)
            self.residual.bias_residual_bf16(self.x, self.normed, self.w.o_b[i], out=self.x)
            self.norm.layer_norm_bf16(self.x, self.w.ln2_w[i], self.w.ln2_b[i], out=self.normed)
            fi[i] = sc(self.normed.float().abs().max())
            self.gemm.bf16_gemm_bias_gelu(self.normed, self.w.up_w[i], self.w.up_b[i], out=self.hidden)
            hi[i] = sc(self.hidden.float().abs().max())
            self.gemm.bf16_linear_bf16(self.hidden, self.w.down_w[i], out=self.normed)
            self.residual.bias_residual_bf16(self.x, self.normed, self.w.down_b[i], out=self.x)
            if i != VIS_L - 1:
                self.norm.layer_norm_bf16(self.x, self.w.ln1_w[i + 1], self.w.ln1_b[i + 1], out=self.normed)
        return qi, oi, fi, hi

    def enable_fp8(self, scales) -> None:
        self._qi, self._oi, self._fi, self._hi = scales
        self._setup_fp8()
        self._qkv_f, self._qkv_s, self._o_f, self._o_s = [], [], [], []
        self._up_f, self._up_s, self._dn_f, self._dn_s = [], [], [], []
        for i in range(VIS_L):
            a, b = _quant_proj_weight(self.w.qkv_w[i]); self._qkv_f.append(a); self._qkv_s.append(b)
            a, b = _quant_proj_weight(self.w.o_w[i]); self._o_f.append(a); self._o_s.append(b)
            a, b = _quant_proj_weight(self.w.up_w[i]); self._up_f.append(a); self._up_s.append(b)
            a, b = _quant_proj_weight(self.w.down_w[i]); self._dn_f.append(a); self._dn_s.append(b)

    def __call__(self) -> torch.Tensor:
        ff = self._fp8ffn
        self.layout.patch_im2col_bf16(self.images, out=self.patches)
        self.gemm.bf16_linear_bf16(self.patches, self.w.patch_w, out=self.x)
        self.residual.bias_residual_bf16(self.x, self.w.pos_embed, self.w.patch_b, out=self.x)
        self.norm.layer_norm_bf16(self.x, self.w.ln1_w[0], self.w.ln1_b[0], out=self.normed)
        for i in range(VIS_L):
            self.gemm.channel_scale_quantize_fp8_static_bf16(self.normed, self._ones, self._qi[i], out=self._nf)
            ff.fp8_gemm_bf16(self._nf, self._qkv_f[i], self._qi[i], self._qkv_s[i], out=self.qkv_buf)
            self.qkv_buf.add_(self.w.qkv_b[i])
            self.qkv.qkv_split_bf16(self.qkv_buf.view(self.num_views, VIS_SEQ_PER_VIEW, 3 * VIS_D), VIS_NH, VIS_HD, self.q, self.k, self.v)
            self.attn_kernel.fwd(self.q, self.k, self.v, out=self.attn_bthd, p_dropout=0.0, is_causal=False)
            self.gemm.channel_scale_quantize_fp8_static_bf16(self.attn, self._ones, self._oi[i], out=self._af)
            ff.fp8_gemm_bf16(self._af, self._o_f[i], self._oi[i], self._o_s[i], out=self.normed)
            self.residual.bias_residual_bf16(self.x, self.normed, self.w.o_b[i], out=self.x)
            self.norm.layer_norm_bf16(self.x, self.w.ln2_w[i], self.w.ln2_b[i], out=self.normed)
            self.gemm.channel_scale_quantize_fp8_static_bf16(self.normed, self._ones, self._fi[i], out=self._nf)
            ff.fp8_gelu_mlp_bf16(self._nf, self._up_f[i], self.w.up_b[i], self._dn_f[i], self._zeros, self._fi[i],
                                 self._up_s[i], self._hi[i], self._dn_s[i], self._hbf, self._hf, out=self.normed)
            self.residual.bias_residual_bf16(self.x, self.normed, self.w.down_b[i], out=self.x)
            if i != VIS_L - 1:
                self.norm.layer_norm_bf16(self.x, self.w.ln1_w[i + 1], self.w.ln1_b[i + 1], out=self.normed)
        self.norm.layer_norm_bf16(self.x, self.w.final_norm_w, self.w.final_norm_b, out=self.normed)
        self.gemm.bf16_linear_bias_bf16(self.normed, self.w.projector_w, self.w.projector_b, out=self.encoder_x_vision)
        return self.encoder_x_input


def _scale_from_amax_value(amax: torch.Tensor, *, safety: float, device: torch.device) -> torch.Tensor:
    return torch.clamp(amax.float() / 448.0 * safety, min=1e-12).reshape(1).to(device=device, dtype=torch.float32).contiguous()


def run_hf_vision_encoder_decoder(args: argparse.Namespace) -> BridgeResult:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dec = _load_decoder_module()
    bundle = torch.load(args.encoder_bundle, map_location="cpu", weights_only=False)
    if "input_images" not in bundle or "encoder_x_input" not in bundle:
        raise RuntimeError(
            "encoder bundle does not contain input_images/encoder_x_input; "
            "re-run export-encoder with the current script"
        )
    checkpoint = _resolve_weight_path(args.checkpoint or bundle["checkpoint"])
    layers = int(args.layers or bundle["layers"])
    steps = int(args.steps or bundle["steps"])
    if layers > DEC_L:
        raise ValueError(f"requested layers={layers} exceeds PI0.5 encoder layers={DEC_L}")

    input_images = bundle["input_images"].to(device=device, dtype=torch.bfloat16).contiguous()
    encoder_x_template = bundle["encoder_x_input"].to(
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    vision_weights = VisionWeights(
        checkpoint,
        num_views=int(bundle.get("num_views", input_images.shape[0])),
        device=device,
    )
    fp8_proj = bool(getattr(args, "fp8_projections", False))
    VisionCls = Fp8HubVisionRuntime if fp8_proj else HubVisionRuntime
    vision_runtime = VisionCls(
        vision_weights,
        input_images,
        encoder_x_template,
        local_gemm_artifact=args.local_gemm_artifact,
        local_qkv_artifact=args.local_qkv_artifact,
        local_residual_artifact=args.local_residual_artifact,
        local_norm_artifact=args.local_norm_artifact,
        local_layout_artifact=args.local_layout_artifact,
    )
    if fp8_proj:
        vision_runtime.enable_fp8(vision_runtime.calibrate(scale_safety=args.encoder_scale_safety))
    produced_encoder_x = vision_runtime()
    torch.cuda.synchronize()
    vision_metrics = _compare(
        produced_encoder_x[: vision_runtime.seq],
        encoder_x_template[: vision_runtime.seq],
    )
    if (
        vision_metrics[2] > args.vision_p99_abs_limit
        or vision_metrics[3] < args.vision_cosine_limit
    ):
        raise RuntimeError(
            "vision/projector correctness failed vs official encoder_x prefix: "
            f"p99_abs={vision_metrics[2]:.6f} cosine={vision_metrics[3]:.8f}"
        )

    enc_weights = EncoderWeights(checkpoint, layers=layers, device=device)
    enc_ref = TorchEncoderReference(
        enc_weights,
        produced_encoder_x,
        scale_safety=args.encoder_scale_safety,
        calibration_input=args.encoder_calibration_input,
    )
    if args.encoder_calibration_output is not None:
        enc_ref.save_calibration(
            args.encoder_calibration_output,
            checkpoint=checkpoint,
            encoder_seq_len=produced_encoder_x.shape[0],
        )
    EncoderCls = Fp8HubEncoderRuntime if fp8_proj else HubEncoderRuntime
    enc_runtime = EncoderCls(
        enc_weights,
        vision_runtime.encoder_x_input,
        enc_ref,
        local_gemm_artifact=args.local_gemm_artifact,
        local_qkv_artifact=args.local_qkv_artifact,
        local_ffn_artifact=args.local_ffn_artifact,
        local_residual_artifact=args.local_residual_artifact,
        local_norm_artifact=args.local_norm_artifact,
    )
    if fp8_proj:
        enc_runtime.enable_fp8(*enc_runtime.calibrate())

    ref_k, ref_v = enc_ref()
    rt_k, rt_v = enc_runtime()
    torch.cuda.synchronize()
    enc_ref_metrics_k = _compare(rt_k, ref_k)
    enc_ref_metrics_v = _compare(rt_v, ref_v)
    encoder_reference_p99 = max(enc_ref_metrics_k[2], enc_ref_metrics_v[2])
    encoder_reference_cos = min(enc_ref_metrics_k[3], enc_ref_metrics_v[3])
    if (
        encoder_reference_p99 > args.encoder_p99_abs_limit
        or encoder_reference_cos < args.encoder_cosine_limit
    ):
        raise RuntimeError(
            "encoder correctness failed vs HF reference: "
            f"p99_abs={encoder_reference_p99:.6f} cosine={encoder_reference_cos:.8f}"
        )

    encoder_official_p99 = None
    encoder_official_cos = None
    full_official_shape = layers == int(bundle["layers"])
    if full_official_shape:
        official_k = bundle["encoder_k"][:layers].to(
            device=device,
            dtype=torch.bfloat16,
        ).contiguous()
        official_v = bundle["encoder_v"][:layers].to(
            device=device,
            dtype=torch.bfloat16,
        ).contiguous()
        enc_official_metrics_k = _compare(rt_k, official_k)
        enc_official_metrics_v = _compare(rt_v, official_v)
        encoder_official_p99 = max(enc_official_metrics_k[2], enc_official_metrics_v[2])
        encoder_official_cos = min(enc_official_metrics_k[3], enc_official_metrics_v[3])

    dec_weights = dec.DecoderWeights(checkpoint, layers=layers, steps=steps, device=device)
    runtime_state = dec.DecoderState.__new__(dec.DecoderState)
    runtime_state.noise0 = bundle["noise0"].to(device=device, dtype=torch.bfloat16).contiguous()
    runtime_state.encoder_k = enc_runtime.k_all
    runtime_state.encoder_v = enc_runtime.v_all
    runtime_state.chunk_size = int(bundle["chunk_size"])
    runtime_state.encoder_seq_len = int(bundle["encoder_seq_len"])
    runtime_state.rope = dec._make_rope(
        runtime_state.chunk_size,
        runtime_state.encoder_seq_len,
        device,
    )

    ref_state = dec.DecoderState.__new__(dec.DecoderState)
    ref_state.noise0 = runtime_state.noise0
    ref_state.encoder_k = ref_k
    ref_state.encoder_v = ref_v
    ref_state.chunk_size = runtime_state.chunk_size
    ref_state.encoder_seq_len = runtime_state.encoder_seq_len
    ref_state.rope = runtime_state.rope
    dec_ref = dec.TorchDecoderReference(
        dec_weights,
        ref_state,
        scale_safety=args.scale_safety,
        calibration_input=args.calibration_input,
    )
    if args.calibration_output is not None:
        dec_ref.save_calibration(
            args.calibration_output,
            checkpoint=checkpoint,
            encoder_seq_len=runtime_state.encoder_seq_len,
        )
    DecoderCls = dec.Fp8HubDecoderLoop if fp8_proj else dec.HubDecoderLoop
    dec_runtime = DecoderCls(
        dec_weights,
        runtime_state,
        dec_ref,
        local_gemm_artifact=args.local_gemm_artifact,
        local_qkv_artifact=args.local_qkv_artifact,
        local_ffn_artifact=args.local_ffn_artifact,
        local_residual_artifact=args.local_residual_artifact,
        attention_backend=args.attention_backend,
    )
    if fp8_proj:
        dec_qi, dec_oi = dec.calibrate_decoder_proj(
            dec_weights, ref_state,
            scale_safety=args.scale_safety, calibration_input=args.calibration_input,
        )
        dec_runtime.enable_fp8_projections(dec_qi, dec_oi)
    combined = VisionEncoderDecoderRuntime(vision_runtime, enc_runtime, dec_runtime)

    t0 = time.perf_counter()
    expected = dec_ref()
    got = combined()
    torch.cuda.synchronize()
    first_us = (time.perf_counter() - t0) * 1_000_000.0
    max_abs, mean_abs, p99_abs, cosine, mse = _compare(got, expected)
    if p99_abs > args.p99_abs_limit or cosine < args.cosine_limit:
        raise RuntimeError(
            f"combined correctness failed: max_abs={max_abs:.6f} "
            f"p99_abs={p99_abs:.6f} cosine={cosine:.8f}"
        )

    runtime_us = _time_us(combined, warmup=args.warmup, iters=args.iters)
    graph_us = None
    graph_status = "not_requested"
    if args.cuda_graph:
        try:
            captured = dec.Captured(combined)
            graph_us = _time_us(captured.replay, warmup=args.warmup, iters=args.iters)
            graph_status = "ok"
        except Exception as exc:  # noqa: BLE001
            graph_status = f"unsupported: {type(exc).__name__}: {exc}"

    official_metrics = (None, None, None, None, None)
    official_out = bundle.get("official_decoder_out")
    if torch.is_tensor(official_out) and full_official_shape and steps == int(bundle["steps"]):
        official_metrics = _compare(
            got,
            official_out.to(device=device, dtype=torch.bfloat16).contiguous(),
        )

    return BridgeResult(
        name="pi05_hf_vision_encoder_decoder_e2e",
        status="pass",
        scope=(
            "Real LIBERO normalized images -> HF-kernel SigLIP vision/projector -> "
            "HF-kernel Gemma encoder KV -> HF-kernel PI0.5 decoder action-expert runtime"
        ),
        checkpoint=str(checkpoint),
        encoder_bundle=str(args.encoder_bundle),
        input_source=str(bundle.get("input_source", "unknown")),
        prompt=str(bundle.get("prompt", "")),
        frame_index=int(bundle.get("frame_index", -1)),
        layers=layers,
        steps=steps,
        encoder_seq_len=runtime_state.encoder_seq_len,
        device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        first_us=first_us,
        runtime_us=runtime_us,
        graph_us=graph_us,
        graph_status=graph_status,
        runtime_vs_first=first_us / runtime_us,
        graph_vs_first=None if graph_us is None else first_us / graph_us,
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        mse=mse,
        cosine=cosine,
        official_flashrt_max_abs=official_metrics[0],
        official_flashrt_mean_abs=official_metrics[1],
        official_flashrt_p99_abs=official_metrics[2],
        official_flashrt_mse=official_metrics[4],
        official_flashrt_cosine=official_metrics[3],
        encoder_reference_p99_abs=encoder_reference_p99,
        encoder_reference_cosine=encoder_reference_cos,
        encoder_official_p99_abs=encoder_official_p99,
        encoder_official_cosine=encoder_official_cos,
        vision_projector_p99_abs=vision_metrics[2],
        vision_projector_cosine=vision_metrics[3],
        calibration_mode=dec_ref.calibration_mode,
        calibration_source=dec_ref.calibration_source,
        encoder_calibration_mode=enc_ref.calibration_mode,
        encoder_calibration_source=enc_ref.calibration_source,
        kernel_coverage=[
            "patch_im2col_bf16",
            "layer_norm_bf16",
            "bf16_linear_bf16",
            "bf16_linear_bias_bf16",
            "qkv_split_bf16",
            "flash_attn2.fwd",
            "bf16_gemm_bias_gelu",
            "bias_residual_bf16",
            "rms_norm_bf16",
            "rms_norm_quant_fp8_static_bf16",
            "qkv_split_rope_kvcache_bf16",
            "fp8_geglu_mlp_bf16",
            "gate_residual_bf16",
            "ada_rms_norm_style_bf16",
            "CUDA Graph replay",
        ],
        torch_gaps=[],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    exp = sub.add_parser("export-encoder")
    exp.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    exp.add_argument("--flashrt-root", type=Path, default=DEFAULT_FLASHRT_ROOT)
    exp.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT)
    exp.add_argument("--frame-index", type=int, default=50)
    exp.add_argument("--num-views", type=int, default=2)
    exp.add_argument("--layers", type=int, default=DEC_L)
    exp.add_argument("--steps", type=int, default=10)
    exp.add_argument("--seed", type=int, default=0)
    exp.add_argument("--encoder-use-fp8", action=argparse.BooleanOptionalAction, default=False)
    exp.add_argument("--include-official-decoder", action=argparse.BooleanOptionalAction, default=True)
    exp.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run")
    run.add_argument("--encoder-bundle", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--layers", type=int)
    run.add_argument("--steps", type=int)
    run.add_argument("--scale-safety", type=float, default=1.05)
    run.add_argument("--calibration-input", type=Path)
    run.add_argument("--calibration-output", type=Path)
    run.add_argument("--local-gemm-artifact")
    run.add_argument("--local-qkv-artifact")
    run.add_argument("--local-ffn-artifact")
    run.add_argument("--local-residual-artifact")
    run.add_argument("--attention-backend", choices=("fa2", "sdpa"), default="fa2")
    run.add_argument("--warmup", type=int, default=2)
    run.add_argument("--iters", type=int, default=10)
    run.add_argument("--cuda-graph", action="store_true")
    run.add_argument("--p99-abs-limit", type=float, default=0.5)
    run.add_argument("--cosine-limit", type=float, default=0.9)
    run.add_argument("--output", type=Path)

    encdec = sub.add_parser("run-encoder-decoder")
    encdec.add_argument("--encoder-bundle", type=Path, required=True)
    encdec.add_argument("--checkpoint", type=Path)
    encdec.add_argument("--layers", type=int)
    encdec.add_argument("--steps", type=int)
    encdec.add_argument("--scale-safety", type=float, default=1.05)
    encdec.add_argument("--encoder-scale-safety", type=float, default=1.05)
    encdec.add_argument("--calibration-input", type=Path)
    encdec.add_argument("--calibration-output", type=Path)
    encdec.add_argument("--encoder-calibration-input", type=Path)
    encdec.add_argument("--encoder-calibration-output", type=Path)
    encdec.add_argument("--local-gemm-artifact")
    encdec.add_argument("--local-qkv-artifact")
    encdec.add_argument("--local-ffn-artifact")
    encdec.add_argument("--local-residual-artifact")
    encdec.add_argument("--local-norm-artifact")
    encdec.add_argument("--attention-backend", choices=("fa2",), default="fa2")
    encdec.add_argument("--warmup", type=int, default=2)
    encdec.add_argument("--iters", type=int, default=10)
    encdec.add_argument("--cuda-graph", action="store_true")
    encdec.add_argument("--p99-abs-limit", type=float, default=0.5)
    encdec.add_argument("--cosine-limit", type=float, default=0.9)
    encdec.add_argument("--encoder-p99-abs-limit", type=float, default=0.5)
    encdec.add_argument("--encoder-cosine-limit", type=float, default=0.9)
    encdec.add_argument("--output", type=Path)

    visencdec = sub.add_parser("run-vision-encoder-decoder")
    visencdec.add_argument("--encoder-bundle", type=Path, required=True)
    visencdec.add_argument("--checkpoint", type=Path)
    visencdec.add_argument("--layers", type=int)
    visencdec.add_argument("--steps", type=int)
    visencdec.add_argument("--scale-safety", type=float, default=1.05)
    visencdec.add_argument("--encoder-scale-safety", type=float, default=1.05)
    visencdec.add_argument("--calibration-input", type=Path)
    visencdec.add_argument("--calibration-output", type=Path)
    visencdec.add_argument("--encoder-calibration-input", type=Path)
    visencdec.add_argument("--encoder-calibration-output", type=Path)
    visencdec.add_argument("--local-gemm-artifact")
    visencdec.add_argument("--local-qkv-artifact")
    visencdec.add_argument("--local-ffn-artifact")
    visencdec.add_argument("--local-residual-artifact")
    visencdec.add_argument("--local-norm-artifact")
    visencdec.add_argument("--local-layout-artifact")
    visencdec.add_argument("--attention-backend", choices=("fa2",), default="fa2")
    visencdec.add_argument(
        "--fp8-projections",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run QKV/O/vision projections in FP8 (published Hub kernels only). "
        "Default ON: ~21.6 ms on RTX 5090. Use --no-fp8-projections for the "
        "BF16-projection path (~22.5 ms, slightly higher cosine).",
    )
    visencdec.add_argument("--warmup", type=int, default=2)
    visencdec.add_argument("--iters", type=int, default=10)
    visencdec.add_argument("--cuda-graph", action="store_true")
    visencdec.add_argument("--p99-abs-limit", type=float, default=0.5)
    visencdec.add_argument("--cosine-limit", type=float, default=0.9)
    visencdec.add_argument("--encoder-p99-abs-limit", type=float, default=0.9)
    visencdec.add_argument("--encoder-cosine-limit", type=float, default=0.9)
    visencdec.add_argument("--vision-p99-abs-limit", type=float, default=0.5)
    visencdec.add_argument("--vision-cosine-limit", type=float, default=0.9)
    visencdec.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.cmd == "export-encoder":
        export_encoder(args)
        return
    if args.cmd == "run-vision-encoder-decoder":
        result = run_hf_vision_encoder_decoder(args)
    elif args.cmd == "run-encoder-decoder":
        result = run_hf_encoder_decoder(args)
    else:
        result = run_hf_decoder(args)
    payload = asdict(result)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
