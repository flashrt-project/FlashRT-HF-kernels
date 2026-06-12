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
GELU_FFN_REPO = "flashrt/flashrt-fp8-ffn"
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


class FlashRTGeluMLP(nn.Module):
    """FP8 drop-in for a SigLIP MLP (fc1 -> gelu_tanh -> fc2, with bias).

    The vision tower is kept in fp32; this casts to bf16 for the FP8 path and
    back to the input dtype so the SigLIP residual stays unchanged.
    """

    def __init__(self, mlp, in_amax: float, hid_amax: float, ffn_ops, quant_ops, safety: float = 1.0) -> None:
        super().__init__()
        self.ffn_ops = ffn_ops
        self.quant_ops = quant_ops
        self.in_features = mlp.fc1.weight.shape[1]
        self.out_features = mlp.fc2.weight.shape[0]
        device = mlp.fc1.weight.device

        up_fp8, up_scale = quantize_fp8(mlp.fc1.weight)
        down_fp8, down_scale = quantize_fp8(mlp.fc2.weight)
        self.register_buffer("up_fp8", up_fp8.to(device))
        self.register_buffer("down_fp8", down_fp8.to(device))
        self.register_buffer("up_scale", up_scale.to(device))
        self.register_buffer("down_scale", down_scale.to(device))
        self.register_buffer("up_bias", mlp.fc1.bias.detach().to(torch.bfloat16))
        self.register_buffer("down_bias", mlp.fc2.bias.detach().to(torch.bfloat16))
        self.register_buffer("input_scale", static_scale(in_amax, safety).to(device))
        self.register_buffer("hidden_scale", static_scale(hid_amax, safety).to(device))
        self.register_buffer("channel_scale", torch.ones(self.in_features, device=device, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        dtype = x.dtype
        flat = x.reshape(-1, self.in_features).to(torch.bfloat16)
        x_fp8 = self.quant_ops.channel_scale_quantize_fp8_static_bf16(flat, self.channel_scale, self.input_scale)
        out = self.ffn_ops.fp8_gelu_mlp_bf16(
            x_fp8, self.up_fp8, self.up_bias, self.down_fp8, self.down_bias,
            self.input_scale, self.up_scale, self.hidden_scale, self.down_scale,
        )
        return out.reshape(*shape[:-1], self.out_features).to(dtype)


def siglip_mlps(model) -> list:
    """The SigLIP vision-tower MLP modules."""
    vision_tower = model.paligemma_with_expert.paligemma.model.vision_tower
    return [m for _, m in vision_tower.named_modules() if type(m).__name__ == "SiglipMLP"]


def calibrate_siglip(policy, batch, mlps) -> list[tuple[float, float]]:
    """Per-MLP input/hidden amax for the SigLIP MLPs, eager (see calibrate_mlps)."""
    model = policy.model
    stats = [[0.0, 0.0] for _ in mlps]
    handles = []
    for idx, mlp in enumerate(mlps):
        def hook(mod, inputs, idx=idx):
            x = inputs[0]
            stats[idx][0] = max(stats[idx][0], x.float().abs().max().item())
            hidden = mod.activation_fn(mod.fc1(x))
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


def apply_fp8_vision_mlp(policy, batch, ffn_ops, quant_ops, safety: float = 1.0) -> None:
    """Replace every SigLIP MLP with the fused FP8 GELU kernel (calibrated)."""
    mlps = siglip_mlps(policy.model)
    stats = calibrate_siglip(policy, batch, mlps)
    vision_tower = policy.model.paligemma_with_expert.paligemma.model.vision_tower
    device = next(policy.parameters()).device
    layers = vision_tower.vision_model.encoder.layers
    for layer, (in_amax, hid_amax) in zip(layers, stats):
        layer.mlp = FlashRTGeluMLP(layer.mlp, in_amax, hid_amax, ffn_ops, quant_ops, safety).to(device)


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


def apply_sync_free_loop(policy) -> None:
    """Remove per-denoise-step CPU->GPU syncs by hoisting constant tensor builds
    out of the loop. The stock loop calls ``torch.tensor(time, device=cuda)`` and
    ``torch.tensor(att_masks, device=cuda)`` every step (~2 host syncs/step); the
    timestep schedule is precomputed once and the suffix attention mask is built
    on-device instead of from a Python list.
    It also replaces the per-step ``copy.deepcopy`` of the prefix KV cache with a
    shallow copy (new cache + layer objects, shared tensors), avoiding a copy of
    every KV tensor each step — correctness-equivalent because the joint forward's
    ``cache.update()`` reassigns via ``cat`` rather than writing in place.

    Applied as a runtime monkeypatch (LeRobot source is untouched). Covers the
    non-RTC inference path.
    """
    import copy
    import types

    import torch.nn.functional as F
    from lerobot.policies.pi05 import modeling_pi05 as M

    model = policy.model

    def embed_suffix(self, noisy_actions, timestep):
        time_emb = M.create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features,
            min_period=self.config.min_period, max_period=self.config.max_period,
            device=timestep.device,
        ).type(dtype=timestep.dtype)
        action_emb = self.action_in_proj(noisy_actions)
        x = F.silu(self.time_mlp_in(time_emb))
        adarms_cond = F.silu(self.time_mlp_out(x))
        bsize, action_time_dim = action_emb.shape[:2]
        pad = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        # suffix attention-mask pattern [1, 0, 0, ...] built on-device: no host
        # sync (the stock torch.tensor(python_list) does), and a fresh tensor each
        # step so CUDA graphs don't see a reused persistent buffer.
        am = torch.zeros(self.config.chunk_size, dtype=action_emb.dtype, device=action_emb.device)
        am[0] = 1
        att_masks = am[None, :].expand(bsize, self.config.chunk_size)
        return action_emb, pad, att_masks, adarms_cond

    def sample_actions(self, images, img_masks, tokens, masks, noise=None, num_steps=None, **kwargs):
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        bsize = tokens.shape[0]
        device = tokens.device
        if noise is None:
            noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), device)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = M.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_4d, position_ids=prefix_position_ids,
            past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True,
        )
        dt = -1.0 / num_steps
        times = torch.tensor([1.0 + s * dt for s in range(num_steps)], dtype=torch.float32, device=device)
        x_t = noise
        for step in range(num_steps):
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values,
                x_t=x_t, timestep=times[step].expand(bsize),
            )
            x_t = x_t + dt * v_t
        return x_t

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        bsize = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d = M.make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        pos = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_4d = self._prepare_attention_masks_4d(full)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        kv = copy.copy(past_key_values)                              # shallow cache
        kv.layers = [copy.copy(layer) for layer in past_key_values.layers]   # new layer objs, shared tensors
        outs, _ = self.paligemma_with_expert.forward(
            attention_mask=full_4d, position_ids=pos, past_key_values=kv,
            inputs_embeds=[None, suffix_embs], use_cache=False, adarms_cond=[None, adarms_cond],
        )
        suffix_out = outs[1][:, -self.config.chunk_size:].to(torch.float32)
        return self.action_out_proj(suffix_out)

    model.embed_suffix = types.MethodType(embed_suffix, model)
    model.sample_actions = types.MethodType(sample_actions, model)
    model.denoise_step = types.MethodType(denoise_step, model)


def apply_compile(policy, mode: str = "max-autotune") -> None:
    """Compile the denoise hot path. pi05 already compiles ``sample_actions`` when
    ``compile_model=True``; we (re)apply explicitly so the recipe is self-contained."""
    model = policy.model
    import types

    base = vars(model).pop("sample_actions", None)
    fn = base if base is not None else types.MethodType(type(model).sample_actions, model)
    model.sample_actions = torch.compile(fn, mode=mode)


def optimize(policy, batch, *, fp8: bool = True, vision_fp8: bool = False,
             sync_free: bool = False, inductor_flags: bool = True,
             compile_mode: str = "compile", safety: float = 1.0):
    """Apply the optimization ladder in place and return the policy.

    compile_mode: "disabled" | "compile" | "export-aoti".
    """
    if inductor_flags:
        apply_inductor_flags()
    if sync_free:
        apply_sync_free_loop(policy)
    quant_ops = load_kernel(GEMM_REPO) if (fp8 or vision_fp8) else None
    if fp8:
        ffn_ops = load_kernel(FFN_REPO)
        apply_fp8_mlp(policy, batch, ffn_ops, quant_ops, safety)
    if vision_fp8:
        gelu_ops = load_kernel(GELU_FFN_REPO)
        apply_fp8_vision_mlp(policy, batch, gelu_ops, quant_ops, safety)
    if compile_mode == "disabled":
        force_eager(policy)
    elif compile_mode == "compile":
        apply_compile(policy)
    elif compile_mode == "export-aoti":
        from pi05_aoti import apply_export_aoti

        apply_export_aoti(policy, batch)
    return policy
