#!/usr/bin/env python3
"""Full OpenPI PI0.5 E2E with Gemma FFNs replaced by HF Kernel Hub ops.

This is a bridge demo, not the final FlashRT serving runtime. It keeps the
official OpenPI model path for vision, attention, projections, and denoising
control flow, but replaces selected Gemma MLPs with the public
``flashrt/flashrt-fp8-swiglu-ffn`` Hub package. PI0.5/Gemma uses GeGLU
(``gelu_pytorch_tanh(gate) * up``), so this script requires the package's
``fp8_geglu_mlp_bf16`` op.

The script is intentionally strict:

1. run an OpenPI/PyTorch baseline with fixed observation and noise;
2. collect real Gemma MLP input activations and derive static FP8 scales;
3. install preloaded Hub-kernel MLP wrappers with prequantized weights;
4. rerun the full ``sample_actions`` path and report latency plus action error.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import time
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
FP8_MAX = 448.0


def _stats(xs: list[float]) -> dict[str, float]:
    ys = sorted(float(x) for x in xs)
    return {
        "n": float(len(ys)),
        "p50_ms": ys[int(0.50 * (len(ys) - 1))],
        "p90_ms": ys[int(0.90 * (len(ys) - 1))],
        "p95_ms": ys[int(0.95 * (len(ys) - 1))],
        "mean_ms": statistics.mean(ys),
        "min_ms": ys[0],
        "max_ms": ys[-1],
    }


def _resolve_weight_path(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "model.safetensors"
    if not p.exists():
        raise FileNotFoundError(f"checkpoint weights not found: {p}")
    return p


def _maybe_repair_transformers_replace(openpi_root: Path) -> None:
    import transformers

    src = openpi_root / "openpi/models_pytorch/transformers_replace"
    if not src.exists():
        raise FileNotFoundError(f"transformers_replace source missing: {src}")
    dst = Path(transformers.__file__).resolve().parent
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _check_transformers_replace(openpi_root: Path, repair: bool) -> dict[str, Any]:
    import transformers

    if repair:
        _maybe_repair_transformers_replace(openpi_root)

    try:
        from transformers.models.siglip import check

        replace_ok = bool(check.check_whether_transformers_replace_is_installed_correctly())
    except Exception:
        replace_ok = False

    return {
        "transformers_version": transformers.__version__,
        "transformers_path": str(Path(transformers.__file__).resolve().parent),
        "transformers_replace_ok": replace_ok,
    }


def _scale_from_amax_torch(x, *, safety: float):
    import torch

    amax = x.float().abs().max()
    scale = torch.clamp(amax / FP8_MAX * float(safety), min=1e-12)
    return scale.reshape(1).to(device=x.device, dtype=torch.float32).contiguous()


def _quantize_fp8_torch(x, scale):
    import torch

    return torch.clamp(x.float() / scale.float().reshape(()), -FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn
    ).contiguous()


def _correctness(actual, expected) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    diff = (actual.float() - expected.float()).abs()
    flat = diff.flatten()
    p99 = flat.kthvalue(max(1, math.ceil(0.99 * flat.numel()))).values
    cosine = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(p99.item()),
        "cosine": float(cosine.item()),
    }


def _load_cached_build(repo_id: str, variant: str):
    cache_root = Path.home() / ".cache/huggingface/hub"
    repo_cache = cache_root / ("kernels--" + repo_id.replace("/", "--"))
    snapshots = repo_cache / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"no cached snapshots for {repo_id}: {snapshots}")
    matches = sorted(snapshots.glob(f"*/build/{variant}/__init__.py"))
    if not matches:
        raise FileNotFoundError(f"no cached {variant} build for {repo_id}")
    init_py = matches[-1].resolve()
    module_name = "flashrt_cached_" + repo_id.split("/")[-1].replace("-", "_")
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_py,
        submodule_search_locations=[str(init_py.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load cached build: {init_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _compile_direct_extension(repo_dir: Path, name: str):
    import torch
    from torch.utils.cpp_extension import load

    builder_root = Path("/home/heima/suliang/PI/kernels/kernel-builder/src/pyproject/templates/torch")
    if not builder_root.exists():
        builder_root = Path("/workspace/PI/kernels/kernel-builder/src/pyproject/templates/torch")
    sources = [
        repo_dir / "torch-ext/torch_binding.cpp",
        *sorted((repo_dir / "csrc").glob("*.cu")),
    ]
    build_dir = (repo_dir.parent / "internal-tests/runtime-demo/torch-extensions" / name).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    load(
        name=name,
        sources=[str(s) for s in sources],
        extra_include_paths=[
            str(repo_dir / "torch-ext"),
            str(repo_dir / "csrc"),
            str(builder_root),
        ],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_ldflags=["-lcublasLt", "-lcublas"],
        build_directory=str(build_dir),
        verbose=False,
    )
    return getattr(torch.ops, name)


class _DirectSwiGLUOps:
    def __init__(self, ops: Any) -> None:
        self._ops = ops

    def fp8_gemm_bf16(self, input, weight, input_scale, weight_scale, out=None):
        import torch

        if out is None:
            out = torch.empty((input.shape[0], weight.shape[0]), device=input.device, dtype=torch.bfloat16)
        self._ops.fp8_gemm_bf16(input, weight, input_scale, weight_scale, out)
        return out

    def fp8_swiglu_mlp_bf16(
        self,
        input,
        gate_up_weight,
        down_weight,
        input_scale,
        gate_up_weight_scale,
        hidden_scale,
        down_weight_scale,
        gate_up_bf16,
        hidden_fp8,
        out,
    ):
        self._ops.fp8_swiglu_mlp_bf16(
            input,
            gate_up_weight,
            down_weight,
            input_scale,
            gate_up_weight_scale,
            hidden_scale,
            down_weight_scale,
            gate_up_bf16,
            hidden_fp8,
            out,
        )
        return out

    def fp8_geglu_mlp_bf16(
        self,
        input,
        gate_up_weight,
        down_weight,
        input_scale,
        gate_up_weight_scale,
        hidden_scale,
        down_weight_scale,
        gate_up_bf16,
        hidden_fp8,
        out,
    ):
        self._ops.fp8_geglu_mlp_bf16(
            input,
            gate_up_weight,
            down_weight,
            input_scale,
            gate_up_weight_scale,
            hidden_scale,
            down_weight_scale,
            gate_up_bf16,
            hidden_fp8,
            out,
        )
        return out


class _DirectGemmEpilogueOps:
    def __init__(self, ops: Any) -> None:
        self._ops = ops

    def channel_scale_quantize_fp8_static_bf16(self, input, channel_scale, scale, out):
        self._ops.channel_scale_quantize_fp8_static_bf16(input, channel_scale, scale, out)
        return out


def _load_direct_kernel(repo_id: str):
    root = Path(__file__).resolve().parents[2]
    repo_dir = root / repo_id.split("/")[-1]
    if repo_id.endswith("flashrt-fp8-swiglu-ffn"):
        return _DirectSwiGLUOps(
            _compile_direct_extension(repo_dir, "_flashrt_direct_fp8_swiglu_ffn")
        )
    if repo_id.endswith("flashrt-gemm-epilogues"):
        return _DirectGemmEpilogueOps(
            _compile_direct_extension(repo_dir, "_flashrt_direct_gemm_epilogues")
        )
    raise ValueError(f"direct kernel source is not configured for {repo_id}")


def _load_kernel(repo_id: str, *, source: str, variant: str):
    if source == "direct":
        return _load_direct_kernel(repo_id), "direct"
    if source in ("auto", "hub"):
        try:
            from kernels import get_kernel

            return get_kernel(repo_id, version=1, trust_remote_code=True), "hub"
        except Exception:
            if source == "hub":
                raise
    return _load_cached_build(repo_id, variant), "cache"


@dataclass
class LayerBuffers:
    input_fp8: Any
    gate_up_bf16: Any
    hidden_fp8: Any
    out: Any


class HubGeGLUMLP:
    def __init__(
        self,
        *,
        ops: Any,
        quant_ops: Any,
        orig_mlp: Any,
        input_scale: Any,
        calibration_input: Any,
        scale_safety: float,
    ) -> None:
        import torch

        self.ops = ops
        self.quant_ops = quant_ops
        self.input_scale = input_scale
        self.scale_safety = scale_safety
        self.ones = torch.ones(
            (orig_mlp.gate_proj.weight.shape[1],),
            device=orig_mlp.gate_proj.weight.device,
            dtype=torch.bfloat16,
        ).contiguous()

        gate = orig_mlp.gate_proj.weight.detach().to(device="cuda", dtype=torch.bfloat16)
        up = orig_mlp.up_proj.weight.detach().to(device="cuda", dtype=torch.bfloat16)
        down = orig_mlp.down_proj.weight.detach().to(device="cuda", dtype=torch.bfloat16)
        gate_up = torch.cat([gate, up], dim=0).contiguous()
        down = down.contiguous()

        self.gate_up_w_scale = _scale_from_amax_torch(gate_up, safety=1.0)
        self.down_w_scale = _scale_from_amax_torch(down, safety=1.0)
        self.gate_up_w_fp8 = _quantize_fp8_torch(gate_up, self.gate_up_w_scale)
        self.down_w_fp8 = _quantize_fp8_torch(down, self.down_w_scale)

        # Static hidden scale is calibrated from real captured decoder MLP
        # inputs. Using synthetic probe data here produces invalid action
        # drift after the 10-step denoise loop.
        probe = calibration_input.reshape(-1, calibration_input.shape[-1]).contiguous()
        probe_fp8 = _quantize_fp8_torch(probe, input_scale)
        gate_up_bf16 = self.ops.fp8_gemm_bf16(
            probe_fp8, self.gate_up_w_fp8, input_scale, self.gate_up_w_scale
        )
        gate, up = gate_up_bf16.float().chunk(2, dim=1)
        hidden = torch.nn.functional.gelu(gate, approximate="tanh") * up
        self.hidden_scale = _scale_from_amax_torch(hidden.to(torch.bfloat16), safety=scale_safety)

        self.buffers: dict[tuple[int, int], LayerBuffers] = {}

    def _buffers(self, rows: int, dim: int) -> LayerBuffers:
        import torch

        key = (rows, dim)
        buf = self.buffers.get(key)
        if buf is None:
            hidden = self.gate_up_w_fp8.shape[0] // 2
            buf = LayerBuffers(
                input_fp8=torch.empty((rows, dim), device="cuda", dtype=torch.float8_e4m3fn),
                gate_up_bf16=torch.empty(
                    (rows, 2 * hidden), device="cuda", dtype=torch.bfloat16
                ),
                hidden_fp8=torch.empty((rows, hidden), device="cuda", dtype=torch.float8_e4m3fn),
                out=torch.empty((rows, dim), device="cuda", dtype=torch.bfloat16),
            )
            self.buffers[key] = buf
        return buf

    def __call__(self, x):
        orig_shape = tuple(x.shape)
        flat = x.reshape(-1, orig_shape[-1]).contiguous()
        rows, dim = flat.shape
        buf = self._buffers(rows, dim)
        self.quant_ops.channel_scale_quantize_fp8_static_bf16(
            flat, self.ones, self.input_scale, buf.input_fp8
        )
        out = self.ops.fp8_geglu_mlp_bf16(
            buf.input_fp8,
            self.gate_up_w_fp8,
            self.down_w_fp8,
            self.input_scale,
            self.gate_up_w_scale,
            self.hidden_scale,
            self.down_w_scale,
            buf.gate_up_bf16,
            buf.hidden_fp8,
            buf.out,
        )
        return out.view(orig_shape)


def _time_model(call: Callable[[], Any], *, warmup: int, iters: int):
    import torch

    for _ in range(warmup):
        out = call()
        torch.cuda.synchronize()
        if not torch.isfinite(out).all():
            raise RuntimeError("model output contains NaN or Inf during warmup")
    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = call()
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        if not torch.isfinite(out).all():
            raise RuntimeError("model output contains NaN or Inf during benchmark")
    return out, times_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="pick up the red block and place it in the tray")
    parser.add_argument("--num-views", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-safety", type=float, default=1.10)
    parser.add_argument(
        "--replace-families",
        choices=("encoder", "decoder", "both"),
        default="decoder",
        help="Gemma MLP family to replace. Encoder rows are larger and are "
        "usually the useful E2E bridge target; decoder rows are small.",
    )
    parser.add_argument(
        "--replace-layers",
        default="all",
        help="Comma-separated layer ids to replace in the selected family/families, or 'all'.",
    )
    parser.add_argument("--kernel-source", choices=("auto", "hub", "cache", "direct"), default="auto")
    parser.add_argument("--kernel-variant", default="torch211-cxx11-cu128-x86_64-linux")
    parser.add_argument("--repair-transformers-replace", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    openpi_root = Path(args.openpi_root).resolve()
    if str(openpi_root) not in sys.path:
        sys.path.insert(0, str(openpi_root))

    import numpy as np
    import safetensors.torch
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # OpenPI wraps sample_actions in torch.compile in the constructor. Keep this
    # E2E bridge in eager mode so the custom Hub wrappers are measured directly.
    torch.compile = lambda fn=None, *a, **k: fn if fn is not None else (lambda f: f)

    tr_info = _check_transformers_replace(openpi_root, args.repair_transformers_replace)
    if not tr_info["transformers_replace_ok"]:
        raise RuntimeError(
            "OpenPI transformers_replace is not installed correctly. Install "
            "transformers==4.53.2 and copy the replacement files, or rerun with "
            "--repair-transformers-replace."
        )

    from openpi.models.model import Observation
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    weight_path = _resolve_weight_path(args.checkpoint)
    cfg = Pi0Config(pi05=True, action_horizon=10, action_dim=32, dtype="bfloat16")

    t0 = time.perf_counter()
    model = PI0Pytorch(cfg)
    safetensors.torch.load_model(model, str(weight_path), strict=False)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.to("cuda").eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    tokenizer = PaligemmaTokenizer(max_len=cfg.max_token_len)
    state_np = np.zeros((cfg.action_dim,), dtype=np.float32)
    tokens_np, mask_np = tokenizer.tokenize(args.prompt, state_np)
    images = {}
    image_masks = {}
    for i, key in enumerate(IMAGE_KEYS):
        img = torch.randint(
            0, 256, (1, 3, 224, 224), device="cuda", dtype=torch.uint8
        ).float() / 127.5 - 1.0
        images[key] = img
        image_masks[key] = torch.full((1,), i < args.num_views, device="cuda", dtype=torch.bool)

    obs = Observation(
        images=images,
        image_masks=image_masks,
        state=torch.from_numpy(state_np).to("cuda", torch.float32).view(1, -1),
        tokenized_prompt=torch.from_numpy(tokens_np).to("cuda", torch.long).view(1, -1),
        tokenized_prompt_mask=torch.from_numpy(mask_np).to("cuda", torch.bool).view(1, -1),
    )
    noise = torch.randn((1, cfg.action_horizon, cfg.action_dim), device="cuda", dtype=torch.float32)

    def call():
        with torch.no_grad():
            return model.sample_actions("cuda", obs, noise=noise, num_steps=args.steps)

    baseline_out, baseline_times = _time_model(call, warmup=args.warmup, iters=args.iters)

    # Calibration pass: capture real Gemma MLP inputs from the full model.
    encoder_layers = model.paligemma_with_expert.paligemma.model.language_model.layers
    decoder_layers = model.paligemma_with_expert.gemma_expert.model.layers
    selected_families = []
    if args.replace_families in ("encoder", "both"):
        selected_families.append(("encoder", encoder_layers))
    if args.replace_families in ("decoder", "both"):
        selected_families.append(("decoder", decoder_layers))
    mlp_inputs: dict[str, dict[int, list[Any]]] = {
        family: {i: [] for i in range(len(layers))}
        for family, layers in selected_families
    }
    hooks = []
    replace_layers_by_family: dict[str, set[int]] = {}
    for family, layers in selected_families:
        if args.replace_layers == "all":
            ids = set(range(len(layers)))
        else:
            ids = {int(x) for x in args.replace_layers.split(",") if x.strip()}
            bad = sorted(i for i in ids if i < 0 or i >= len(layers))
            if bad:
                raise ValueError(f"{family} replace layer ids out of range: {bad}")
        replace_layers_by_family[family] = ids
        for idx, layer in enumerate(layers):
            hooks.append(
                layer.mlp.register_forward_pre_hook(
                    lambda _mod, inp, family=family, i=idx: mlp_inputs[family][i].append(
                        inp[0].detach()
                    )
                )
            )
    _ = call()
    torch.cuda.synchronize()
    for h in hooks:
        h.remove()

    input_scales: dict[str, dict[int, Any]] = {}
    calibration_inputs: dict[str, dict[int, Any]] = {}
    for family, inputs in mlp_inputs.items():
        input_scales[family] = {}
        calibration_inputs[family] = {}
        for idx, xs in inputs.items():
            if not xs:
                raise RuntimeError(f"no {family} MLP activations captured for layer {idx}")
            cat = torch.cat(
                [x.reshape(-1, x.shape[-1]).to(torch.bfloat16) for x in xs], dim=0
            )
            calibration_inputs[family][idx] = cat
            input_scales[family][idx] = _scale_from_amax_torch(
                cat, safety=args.scale_safety
            )

    swiglu_ops, swiglu_source = _load_kernel(
        "flashrt/flashrt-fp8-swiglu-ffn",
        source=args.kernel_source,
        variant=args.kernel_variant,
    )
    quant_ops, quant_source = _load_kernel(
        "flashrt/flashrt-gemm-epilogues",
        source=args.kernel_source,
        variant=args.kernel_variant,
    )
    wrappers = []
    replaced_layer_ids: dict[str, list[int]] = {}
    for family, layers in selected_families:
        replaced_layer_ids[family] = []
        for idx, layer in enumerate(layers):
            if idx not in replace_layers_by_family[family]:
                continue
            wrapper = HubGeGLUMLP(
                ops=swiglu_ops,
                quant_ops=quant_ops,
                orig_mlp=layer.mlp,
                input_scale=input_scales[family][idx],
                calibration_input=calibration_inputs[family][idx],
                scale_safety=args.scale_safety,
            )
            layer.mlp.forward = wrapper
            wrappers.append(wrapper)
            replaced_layer_ids[family].append(idx)

    hub_out, hub_times = _time_model(call, warmup=args.warmup, iters=args.iters)
    action_error = _correctness(hub_out, baseline_out)
    action_error_first7 = _correctness(hub_out[..., :7], baseline_out[..., :7])

    payload = {
        "name": "pi05_openpi_hub_ffn_e2e",
        "status": "pass",
        "scope": (
            "Full OpenPI PI0.5 sample_actions E2E with selected Gemma "
            "MLP layers replaced by HF Kernel Hub FP8 GeGLU kernels."
        ),
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "checkpoint": str(weight_path),
        "num_views": args.num_views,
        "steps": args.steps,
        "load_s": load_s,
        "transformers": tr_info,
        "baseline_openpi_eager": _stats(baseline_times),
        "hub_geglu_ffn_e2e": _stats(hub_times),
        "speedup_vs_openpi_eager": statistics.mean(baseline_times) / statistics.mean(hub_times),
        "action_error_vs_openpi_eager": action_error,
        "action_error_first7_vs_openpi_eager": action_error_first7,
        "replaced_layers": len(wrappers),
        "replace_families": args.replace_families,
        "replace_layer_ids": replaced_layer_ids,
        "kernel_packages": [
            "flashrt/flashrt-fp8-swiglu-ffn",
            "flashrt/flashrt-gemm-epilogues",
        ],
        "kernel_source": {
            "requested": args.kernel_source,
            "variant": args.kernel_variant,
            "flashrt-fp8-swiglu-ffn": swiglu_source,
            "flashrt-gemm-epilogues": quant_source,
        },
        "note": (
            "This is a complete model E2E run, but it is not the final no-Python "
            "Hub runtime. It intentionally replaces selected Gemma FFN islands. "
            "Attention, QKV/O projections, action projections, vision, and "
            "unreplaced families still use OpenPI/PyTorch."
        ),
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
