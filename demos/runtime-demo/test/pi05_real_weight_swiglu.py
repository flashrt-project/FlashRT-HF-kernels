#!/usr/bin/env python3
"""Real-weight PI0.5 FP8 SwiGLU FFN island using HF Kernel Hub ops.

This script is the first checkpoint-backed bridge between the public HF kernel
packages and the PI0.5 runtime plan. It loads real PI0.5 safetensors weights,
applies the same per-tensor FP8 weight quantization contract used by FlashRT,
calibrates static activation scales, and runs one FFN island through
`flashrt/flashrt-fp8-swiglu-ffn`.

It is not a full policy benchmark. It validates the real-weight/static-scale
contract for one hot model island before wiring the whole runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from kernels import get_kernel
from safetensors import safe_open


FP8_MAX = 448.0


def _stats(xs: list[float]) -> dict[str, float]:
    ys = sorted(float(x) for x in xs)
    return {
        "n": float(len(ys)),
        "p50_us": ys[int(0.50 * (len(ys) - 1))],
        "p90_us": ys[int(0.90 * (len(ys) - 1))],
        "p95_us": ys[int(0.95 * (len(ys) - 1))],
        "mean_us": statistics.mean(ys),
        "min_us": ys[0],
        "max_us": ys[-1],
    }


def _time_us(fn: Callable[[], object], *, warmup: int, iters: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.mean(times), times


def _scale_from_amax(x: torch.Tensor, *, safety: float) -> torch.Tensor:
    amax = x.float().abs().max()
    scale = torch.clamp(amax / FP8_MAX * float(safety), min=1e-12)
    return scale.reshape(1).to(device=x.device, dtype=torch.float32).contiguous()


def _quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float().reshape(()), -FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn
    ).contiguous()


def _dequant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.float() * scale.float().reshape(())


def _resolve_weight_path(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "model.safetensors"
    if not p.exists():
        raise FileNotFoundError(f"checkpoint weights not found: {p}")
    return p


def _load_activation_file(
    path: str,
    *,
    family: str,
    layer: int,
    rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    # Internal calibration artifact produced by
    # pi05_capture_openpi_ffn_activations.py. PyTorch >=2.6 defaults to
    # weights_only=True, which rejects the TorchVersion object in metadata.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "activations" not in payload:
        raise RuntimeError(f"activation file missing 'activations': {path}")
    meta = dict(payload.get("metadata", {}))
    if meta.get("family") not in (None, family):
        raise RuntimeError(
            f"activation family mismatch: file has {meta.get('family')}, expected {family}"
        )
    if meta.get("layer") not in (None, layer):
        raise RuntimeError(
            f"activation layer mismatch: file has {meta.get('layer')}, expected {layer}"
        )
    acts = payload["activations"].to(device=device, dtype=torch.bfloat16).contiguous()
    if acts.dim() != 2:
        raise RuntimeError(f"activations must be rank-2, got shape {tuple(acts.shape)}")
    if rows > 0:
        acts = acts[:rows].contiguous()
    meta["used_shape"] = list(acts.shape)
    return acts, meta


def _load_pi05_ffn_weights(
    checkpoint: Path,
    *,
    family: str,
    layer: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    with safe_open(str(checkpoint), framework="pt") as f:
        keys = set(f.keys())
        strip = "model." if "model.paligemma_with_expert.paligemma.lm_head.weight" in keys else ""

        def get(key: str) -> torch.Tensor:
            return f.get_tensor(strip + key).to(device=device)

        if family == "encoder":
            prefix = f"paligemma_with_expert.paligemma.model.language_model.layers.{layer}"
            ffn_scale = get(f"{prefix}.post_attention_layernorm.weight").float()
            fuse = 1.0 + ffn_scale
            gate = get(f"{prefix}.mlp.gate_proj.weight").float() * fuse.unsqueeze(0)
            up = get(f"{prefix}.mlp.up_proj.weight").float() * fuse.unsqueeze(0)
            down = get(f"{prefix}.mlp.down_proj.weight").float()
            weight_contract = (
                "gate/up are concatenated as [2H, D]; down is [D, H]. "
                "Encoder gate/up include FlashRT's RMSNorm weight fold. "
                "Use representative post-RMSNorm-without-weight activations "
                "for this contract; OpenPI layer.mlp hook activations already "
                "include the norm weight and should use a separate raw-weight "
                "contract in a future script revision."
            )
        elif family == "decoder":
            prefix = f"paligemma_with_expert.gemma_expert.model.layers.{layer}"
            gate = get(f"{prefix}.mlp.gate_proj.weight").float()
            up = get(f"{prefix}.mlp.up_proj.weight").float()
            down = get(f"{prefix}.mlp.down_proj.weight").float()
            weight_contract = (
                "gate/up are concatenated as [2H, D]; down is [D, H]. "
                "Decoder uses raw MLP weights because AdaRMSNorm is runtime."
            )
        else:
            raise ValueError(f"unsupported family: {family}")

    gate_up = torch.cat([gate, up], dim=0).to(torch.bfloat16).contiguous()
    down = down.to(torch.bfloat16).contiguous()
    meta = {
        "family": family,
        "layer": layer,
        "gate_up_shape": list(gate_up.shape),
        "down_shape": list(down.shape),
        "weight_contract": weight_contract,
    }
    return gate_up, down, meta


def _reference_fp8(
    x_fp8: torch.Tensor,
    gate_up_w_fp8: torch.Tensor,
    down_w_fp8: torch.Tensor,
    input_scale: torch.Tensor,
    gate_up_w_scale: torch.Tensor,
    hidden_scale: torch.Tensor,
    down_w_scale: torch.Tensor,
) -> torch.Tensor:
    x = _dequant_fp8(x_fp8, input_scale)
    gate_up_w = _dequant_fp8(gate_up_w_fp8, gate_up_w_scale)
    down_w = _dequant_fp8(down_w_fp8, down_w_scale)
    gate_up = x @ gate_up_w.t()
    gate, up = gate_up.chunk(2, dim=1)
    hidden = F.silu(gate) * up
    hidden_fp8 = _quantize_fp8(hidden.to(torch.bfloat16), hidden_scale)
    return (_dequant_fp8(hidden_fp8, hidden_scale) @ down_w.t()).to(torch.bfloat16)


def _reference_bf16(
    x: torch.Tensor,
    gate_up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    gate_up = x.float() @ gate_up_w.float().t()
    gate, up = gate_up.chunk(2, dim=1)
    hidden = F.silu(gate) * up
    return (hidden @ down_w.float().t()).to(torch.bfloat16)


def _correctness(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    flat = diff.flatten()
    p99 = flat.kthvalue(max(1, math.ceil(0.99 * flat.numel()))).values
    cosine = F.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(p99.item()),
        "cosine": float(cosine.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--family", choices=("encoder", "decoder"), default="encoder")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, default=968)
    parser.add_argument(
        "--activation-file",
        help=(
            "Optional .pt file from pi05_capture_openpi_ffn_activations.py. "
            "If provided, static scales and timed input use real OpenPI activations."
        ),
    )
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument("--activation-std", type=float, default=0.25)
    parser.add_argument("--scale-safety", type=float, default=1.05)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    checkpoint = _resolve_weight_path(args.checkpoint)
    gate_up_w, down_w, weight_meta = _load_pi05_ffn_weights(
        checkpoint, family=args.family, layer=args.layer, device=device
    )
    dim = gate_up_w.shape[1]

    activation_meta: dict[str, object] | None = None
    if args.activation_file:
        x, activation_meta = _load_activation_file(
            args.activation_file,
            family=args.family,
            layer=args.layer,
            rows=args.rows,
            device=device,
        )
        if x.shape[1] != dim:
            raise RuntimeError(
                f"activation hidden dim {x.shape[1]} does not match weight dim {dim}"
            )
        calib = x.reshape(1, x.shape[0], x.shape[1])
    else:
        calib = (
            torch.randn(
                (args.calibration_samples, args.rows, dim),
                device=device,
                dtype=torch.bfloat16,
            )
            * args.activation_std
        ).to(torch.bfloat16)
        x = calib[0].contiguous()
    input_scale = _scale_from_amax(calib, safety=args.scale_safety)

    x_fp8 = _quantize_fp8(x, input_scale)

    gate_up_w_scale = _scale_from_amax(gate_up_w, safety=1.0)
    down_w_scale = _scale_from_amax(down_w, safety=1.0)
    gate_up_w_fp8 = _quantize_fp8(gate_up_w, gate_up_w_scale)
    down_w_fp8 = _quantize_fp8(down_w, down_w_scale)

    gate_up_for_hidden = _dequant_fp8(x_fp8, input_scale) @ _dequant_fp8(
        gate_up_w_fp8, gate_up_w_scale
    ).t()
    gate, up = gate_up_for_hidden.chunk(2, dim=1)
    hidden_calib = F.silu(gate) * up
    hidden_scale = _scale_from_amax(hidden_calib, safety=args.scale_safety)

    ops = get_kernel("flashrt/flashrt-fp8-swiglu-ffn", version=1, trust_remote_code=True)
    gate_up_bf16 = torch.empty(
        (args.rows, gate_up_w_fp8.shape[0]), device=device, dtype=torch.bfloat16
    )
    hidden_fp8 = torch.empty(
        (args.rows, gate_up_w_fp8.shape[0] // 2),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    out = torch.empty((args.rows, down_w_fp8.shape[0]), device=device, dtype=torch.bfloat16)

    def kernel_call() -> torch.Tensor:
        return ops.fp8_swiglu_mlp_bf16(
            x_fp8,
            gate_up_w_fp8,
            down_w_fp8,
            input_scale,
            gate_up_w_scale,
            hidden_scale,
            down_w_scale,
            gate_up_bf16,
            hidden_fp8,
            out,
        )

    actual = kernel_call()
    torch.cuda.synchronize()
    ref_fp8 = _reference_fp8(
        x_fp8,
        gate_up_w_fp8,
        down_w_fp8,
        input_scale,
        gate_up_w_scale,
        hidden_scale,
        down_w_scale,
    )
    ref_bf16 = _reference_bf16(x, gate_up_w, down_w)

    kernel_mean_us, kernel_times = _time_us(kernel_call, warmup=args.warmup, iters=args.iters)

    def ref_call() -> torch.Tensor:
        return _reference_bf16(x, gate_up_w, down_w)

    ref_mean_us, ref_times = _time_us(ref_call, warmup=max(2, args.warmup // 2), iters=args.iters)

    payload = {
        "name": "pi05_real_weight_swiglu_ffn",
        "status": "pass",
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "checkpoint": str(checkpoint),
        "weights": weight_meta,
        "rows": args.rows,
        "calibration": {
            "source": (
                "OpenPI captured BF16 activations"
                if args.activation_file
                else "deterministic representative BF16 activations"
            ),
            "activation_file": args.activation_file,
            "activation_metadata": activation_meta,
            "samples": args.calibration_samples,
            "activation_std": args.activation_std,
            "scale_safety": args.scale_safety,
            "input_scale": float(input_scale.item()),
            "hidden_scale": float(hidden_scale.item()),
            "gate_up_weight_scale": float(gate_up_w_scale.item()),
            "down_weight_scale": float(down_w_scale.item()),
        },
        "correctness_vs_quantized_reference": _correctness(actual, ref_fp8),
        "quantization_error_vs_bf16_reference": _correctness(ref_fp8, ref_bf16),
        "latency": {
            "kernel": _stats(kernel_times),
            "torch_bf16_reference": _stats(ref_times),
            "speedup_vs_torch_bf16_reference": ref_mean_us / kernel_mean_us,
        },
        "note": (
            "Checkpoint-backed FFN island. This validates the real-weight "
            "static-scale Hub kernel contract; full PI0.5 E2E still requires "
            "wiring all model islands and real activation calibration samples."
        ),
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
