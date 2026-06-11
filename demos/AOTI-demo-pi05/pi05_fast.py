#!/usr/bin/env python3
"""pi05-fast: a flux-fast-style optimization stack for LeRobot pi05.

Takes an unmodified LeRobot pi05 policy and applies a toggleable ladder of
inference optimizations, then benchmarks against the BF16 eager baseline:

    bf16 eager (baseline)
      + inductor tuning flags
      + FlashRT fused FP8 GeGLU MLP   (action expert + prefix language model)
      + torch.compile  OR  torch.export + AOTInductor

The FP8 MLP swap uses the Hub kernels ``flashrt-fp8-swiglu-ffn`` and
``flashrt-gemm-epilogues``; static scales are calibrated on a real observation
(random inputs break pi05's wide-magnitude prefix). Compilation is the runtime
layer -- ``compile`` for a warm persistent process, ``export-aoti`` for an
ahead-of-time artifact that loads without re-tuning.

This is a recipe, not a wrapper: every step is a small, optional change to the
stock policy. See run_benchmark.py for the ladder and README.md for results.
"""

from __future__ import annotations

import glob
from pathlib import Path

import torch
import torch.nn as nn

FFN_REPO = "flashrt/flashrt-fp8-swiglu-ffn"
GEMM_REPO = "flashrt/flashrt-gemm-epilogues"


_KERNEL_CACHE: dict[str, object] = {}


def load_kernel(repo: str):
    """Load a published FlashRT Hub package, falling back to a local cache.

    The public path is ``get_kernel``. If the package is not reachable (e.g. a
    private staging repo), fall back to the newest local snapshot via
    ``get_local_kernel`` so the demo also runs against pre-pulled kernels.
    Cached per repo: re-importing a kernel module re-runs ``register_fake`` and
    raises "already registered".
    """
    if repo in _KERNEL_CACHE:
        return _KERNEL_CACHE[repo]

    from kernels import get_kernel

    try:
        ops = get_kernel(repo, version=1, trust_remote_code=True)
    except Exception:
        from kernels import get_local_kernel

        name = repo.split("/")[-1]
        module = name.replace("-", "_")
        pattern = str(Path.home() / f".cache/huggingface/hub/kernels--flashrt--{name}/snapshots/*")
        snapshots = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime)
        if not snapshots:
            raise
        ops = get_local_kernel(Path(snapshots[-1]), module)
    _KERNEL_CACHE[repo] = ops
    return ops


def quantize_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = max(weight.detach().float().abs().max().item(), 1e-12) / 448.0
    fp8 = torch.clamp(weight.float() / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
    return fp8.contiguous(), torch.tensor([scale], dtype=torch.float32)


def static_scale(amax: float, safety: float = 1.0) -> torch.Tensor:
    return torch.tensor([max(amax, 1e-12) / 448.0 * safety], dtype=torch.float32)


class FlashRTGeGLU(nn.Module):
    """FP8 drop-in for a Gemma GeGLU MLP (gate/up/down, gelu_pytorch_tanh, no bias)."""

    def __init__(self, mlp, in_amax: float, hid_amax: float, ffn_ops, quant_ops, safety: float = 1.0) -> None:
        super().__init__()
        self.ffn_ops = ffn_ops
        self.quant_ops = quant_ops
        self.in_features = mlp.gate_proj.weight.shape[1]
        device = mlp.gate_proj.weight.device

        gate_up = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0).contiguous()
        gate_up_fp8, gate_up_scale = quantize_fp8(gate_up)
        down_fp8, down_scale = quantize_fp8(mlp.down_proj.weight)
        self.register_buffer("gate_up_fp8", gate_up_fp8.to(device))
        self.register_buffer("down_fp8", down_fp8.to(device))
        self.register_buffer("gate_up_scale", gate_up_scale.to(device))
        self.register_buffer("down_scale", down_scale.to(device))
        self.register_buffer("input_scale", static_scale(in_amax, safety).to(device))
        self.register_buffer("hidden_scale", static_scale(hid_amax, safety).to(device))
        self.register_buffer("channel_scale", torch.ones(self.in_features, device=device, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.in_features).to(torch.bfloat16)
        x_fp8 = self.quant_ops.channel_scale_quantize_fp8_static_bf16(flat, self.channel_scale, self.input_scale)
        out = self.ffn_ops.fp8_geglu_mlp_bf16(
            x_fp8, self.gate_up_fp8, self.down_fp8,
            self.input_scale, self.gate_up_scale, self.hidden_scale, self.down_scale,
        )
        return out.reshape(shape)


def gemma_layers(model) -> list:
    """The Gemma decoder layers in pi05: action expert + prefix language model."""
    expert = list(model.paligemma_with_expert.gemma_expert.model.layers)
    prefix = list(model.paligemma_with_expert.paligemma.model.language_model.layers)
    return expert + prefix


def calibrate_mlps(policy, batch, layers) -> list[tuple[float, float]]:
    """Capture per-MLP input/hidden amax on a real observation, in EAGER mode.

    pi05 wraps ``sample_actions`` in ``torch.compile`` by default; a compiled
    graph does not fire Python forward hooks, so calibration drops the compiled
    methods, runs once eagerly, and restores them.
    """
    model = policy.model
    stats = [[0.0, 0.0] for _ in layers]
    handles = []
    for idx, layer in enumerate(layers):
        mlp = layer.mlp

        def hook(mod, inputs, idx=idx):
            x = inputs[0]
            stats[idx][0] = max(stats[idx][0], x.float().abs().max().item())
            hidden = mod.act_fn(mod.gate_proj(x)) * mod.up_proj(x)
            stats[idx][1] = max(stats[idx][1], hidden.float().abs().max().item())

        handles.append(mlp.register_forward_pre_hook(hook))

    saved = {name: vars(model).pop(name) for name in ("sample_actions", "forward") if name in vars(model)}
    with torch.inference_mode():
        policy.predict_action_chunk(dict(batch))
    torch.cuda.synchronize()
    vars(model).update(saved)
    for handle in handles:
        handle.remove()
    return [(a, b) for a, b in stats]


def apply_fp8_mlp(policy, batch, ffn_ops, quant_ops, safety: float = 1.0) -> None:
    """Replace every Gemma GeGLU MLP with the fused FP8 kernel (calibrated)."""
    layers = gemma_layers(policy.model)
    stats = calibrate_mlps(policy, batch, layers)
    device = next(policy.parameters()).device
    for layer, (in_amax, hid_amax) in zip(layers, stats):
        layer.mlp = FlashRTGeGLU(layer.mlp, in_amax, hid_amax, ffn_ops, quant_ops, safety).to(device)


def apply_inductor_flags() -> None:
    """flux-fast-style inductor tuning flags."""
    import torch._inductor.config as cfg

    cfg.coordinate_descent_tuning = True
    cfg.coordinate_descent_check_all_directions = True
    cfg.epilogue_fusion = False


def force_eager(policy) -> None:
    """Strip pi05's built-in ``compile_model=True`` wrapper to get a true eager
    baseline (from_pretrained compiles ``sample_actions`` in __init__)."""
    model = policy.model
    for name in ("sample_actions", "forward"):
        vars(model).pop(name, None)


def apply_compile(policy, mode: str = "max-autotune") -> None:
    """Compile the denoise hot path. pi05 already compiles ``sample_actions`` when
    ``compile_model=True``; we (re)apply explicitly so the recipe is self-contained."""
    model = policy.model
    import types

    base = vars(model).pop("sample_actions", None)
    fn = base if base is not None else types.MethodType(type(model).sample_actions, model)
    model.sample_actions = torch.compile(fn, mode=mode)


def optimize(policy, batch, *, fp8: bool = True, inductor_flags: bool = True,
             compile_mode: str = "compile", safety: float = 1.0):
    """Apply the optimization ladder in place and return the policy.

    compile_mode: "disabled" | "compile" | "export-aoti".
    """
    if inductor_flags:
        apply_inductor_flags()
    if fp8:
        ffn_ops = load_kernel(FFN_REPO)
        quant_ops = load_kernel(GEMM_REPO)
        apply_fp8_mlp(policy, batch, ffn_ops, quant_ops, safety)
    if compile_mode == "disabled":
        force_eager(policy)
    elif compile_mode == "compile":
        apply_compile(policy)
    elif compile_mode == "export-aoti":
        from pi05_aoti import apply_export_aoti

        apply_export_aoti(policy, batch)
    return policy
