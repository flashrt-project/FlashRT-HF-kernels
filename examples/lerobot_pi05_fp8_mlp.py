#!/usr/bin/env python3
"""Hot-plug FlashRT FP8 GeGLU MLP kernels into a LeRobot pi05 policy.

This is a *partial* model-integration example: it replaces only the Gemma
GeGLU MLP blocks (gate/up/down with gelu_pytorch_tanh) in pi05's two Gemma
stacks -- the action expert and the prefix language model -- with the fused
``fp8_geglu_mlp_bf16`` kernel. Attention projections are left untouched (a
naive per-projection FP8 swap re-quantizes activations four times per layer
and loses to cuBLAS at the small token counts pi05 runs; that path needs
quantize fusion, not a drop-in).

The model keeps its own ``torch.compile`` as the runtime layer -- the swapped
modules are compile-clean, so the model recompiles around them and FP8 runs
inside the existing CUDA graph. No custom runtime is needed.

Key lesson encoded here: FP8 static scales MUST be calibrated on a *real*
observation. pi05's prefix mixes image and text tokens with very wide
activation magnitudes; calibrating on random inputs produces broken per-tensor
scales and destroys the action output. Calibrate on a real frame from the
dataset the checkpoint targets.

Measured on RTX 5090 (LIBERO finetune, 10 denoise steps), action expert +
prefix MLPs swapped: ~62 ms -> ~53 ms end to end (about 1.17x) with action
cosine similarity ~0.998 versus the BF16 baseline.

Requirements:
    pip install kernels
    pip install "lerobot[pi,dataset]"        # pi05 policy + dataset frame loader
    huggingface-cli login                    # PaliGemma tokenizer is gated

Run:
    python examples/lerobot_pi05_fp8_mlp.py
"""

from __future__ import annotations

import time

import torch
from kernels import get_kernel

CHECKPOINT = "lerobot/pi05_libero_finetuned_v044"
CALIB_DATASET = "physical-intelligence/libero"


def tensor_scale(amax: float, *, safety: float = 1.0, floor: float = 1e-12) -> torch.Tensor:
    return torch.tensor([max(amax, floor) / 448.0 * safety], dtype=torch.float32)


def quantize_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = max(weight.detach().float().abs().max().item(), 1e-12) / 448.0
    fp8 = torch.clamp(weight.float() / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
    return fp8.contiguous(), torch.tensor([scale], dtype=torch.float32)


class FlashRTGeGLU(torch.nn.Module):
    """Drop-in FP8 replacement for a Gemma GeGLU MLP (gate/up/down, no bias).

    ``ffn_ops`` / ``quant_ops`` are the Hub kernel modules, loaded once and
    shared across all swapped layers. ``in_amax`` / ``hid_amax`` come from
    real-observation calibration.
    """

    def __init__(self, mlp, in_amax: float, hid_amax: float, ffn_ops, quant_ops, *, safety: float = 1.0) -> None:
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
        self.register_buffer("input_scale", tensor_scale(in_amax, safety=safety).to(device))
        self.register_buffer("hidden_scale", tensor_scale(hid_amax, safety=safety).to(device))
        self.register_buffer(
            "channel_scale",
            torch.ones(self.in_features, device=device, dtype=torch.bfloat16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.in_features).to(torch.bfloat16)
        x_fp8 = self.quant_ops.channel_scale_quantize_fp8_static_bf16(
            flat, self.channel_scale, self.input_scale
        )
        out = self.ffn_ops.fp8_geglu_mlp_bf16(
            x_fp8,
            self.gate_up_fp8,
            self.down_fp8,
            self.input_scale,
            self.gate_up_scale,
            self.hidden_scale,
            self.down_scale,
        )
        return out.reshape(shape)


def gemma_mlps(model) -> list:
    """The GeGLU MLP modules in pi05's action expert + prefix language model."""
    expert = model.paligemma_with_expert.gemma_expert.model.layers
    prefix = model.paligemma_with_expert.paligemma.model.language_model.layers
    return [layer.mlp for layer in expert] + [layer.mlp for layer in prefix]


def calibrate(policy, batch, mlps) -> list[tuple[float, float]]:
    """Capture per-MLP input/hidden amax on a real observation.

    NOTE: pi05 wraps ``sample_actions`` in ``torch.compile`` by default, and a
    compiled graph does NOT fire Python forward hooks. Calibration must run in
    eager mode, so we temporarily drop the compiled methods and restore them
    afterwards.
    """
    model = policy.model
    stats = [[0.0, 0.0] for _ in mlps]
    handles = []
    for idx, mlp in enumerate(mlps):
        def hook(mod, inputs, idx=idx, mod_ref=mlp):
            x = inputs[0]
            stats[idx][0] = max(stats[idx][0], x.float().abs().max().item())
            hidden = mod_ref.act_fn(mod_ref.gate_proj(x)) * mod_ref.up_proj(x)
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


def build_calibration_batch(policy) -> dict:
    """A single real observation, run through the policy's own preprocessor.

    Uses the first frame of the dataset the checkpoint targets so the
    activation distribution matches inference. Replace this with your own
    deployment observation if you have one.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    dataset = LeRobotDataset(CALIB_DATASET)
    frame = dataset[0]
    preprocessor, _ = make_pi05_pre_post_processors(policy.config, dataset.meta.stats)

    observation = {key: value for key, value in frame.items() if key.startswith("observation.")}
    observation["task"] = frame["task"]
    batch = preprocessor(observation)
    device = next(policy.parameters()).device
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def median_latency_ms(fn, *, warmup: int = 4, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.time()
        fn()
        torch.cuda.synchronize()
        samples.append((time.time() - start) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    ffn_ops = get_kernel("flashrt/flashrt-fp8-swiglu-ffn", version=1, trust_remote_code=True)
    quant_ops = get_kernel("flashrt/flashrt-gemm-epilogues", version=1, trust_remote_code=True)

    policy = PI05Policy.from_pretrained(CHECKPOINT).eval().cuda()
    batch = build_calibration_batch(policy)

    def run() -> torch.Tensor:
        with torch.inference_mode():
            return policy.predict_action_chunk(dict(batch))

    baseline = run().float().clone()
    torch.cuda.synchronize()
    baseline_ms = median_latency_ms(run)

    mlps = gemma_mlps(policy.model)
    stats = calibrate(policy, batch, mlps)

    expert = policy.model.paligemma_with_expert.gemma_expert.model.layers
    prefix = policy.model.paligemma_with_expert.paligemma.model.language_model.layers
    layers = list(expert) + list(prefix)
    for layer, (in_amax, hid_amax) in zip(layers, stats):
        layer.mlp = FlashRTGeGLU(layer.mlp, in_amax, hid_amax, ffn_ops, quant_ops).cuda()

    run()  # the model's torch.compile recompiles around the swapped modules
    torch.cuda.synchronize()
    fp8_out = run().float().clone()
    fp8_ms = median_latency_ms(run)

    cos = torch.nn.functional.cosine_similarity(
        fp8_out.flatten(), baseline.flatten(), dim=0
    )
    print(f"baseline      {baseline_ms:6.2f} ms")
    print(f"+ fp8 MLP     {fp8_ms:6.2f} ms   ({baseline_ms / fp8_ms:.2f}x)")
    print(f"action cosine {float(cos):.5f}  (vs BF16 baseline)")


if __name__ == "__main__":
    main()
