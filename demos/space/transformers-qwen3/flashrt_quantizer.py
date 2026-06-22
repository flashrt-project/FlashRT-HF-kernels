"""FlashRT as a transformers quantization backend.

Registers a custom quantizer so FlashRT FP8 / NVFP4 GEMMs load through the
official transformers quantization API — the same entry point as torchao /
bitsandbytes / compressed-tensors:

    from transformers import AutoModelForCausalLM
    from flashrt_quantizer import FlashRTConfig
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", quantization_config=FlashRTConfig(mode="nvfp4"))

The quantizer swaps each decoder layer's MLP (gate/up/down) and attention
(q/k/v/o) ``nn.Linear`` for a FlashRT linear whose forward calls the Hub kernels
(`flashrt/fp4-gemm` for NVFP4, `flashrt/flashrt-fp8-ffn` for FP8). Weights are
quantized lazily on first use; NVFP4 / FP8 activations are quantized dynamically,
so no calibration pass is needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from kernels import get_kernel
from transformers.quantizers import HfQuantizer
from transformers.quantizers.auto import register_quantization_config, register_quantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
_TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj")


def _get(repo):
    try:
        return get_kernel(repo, version=1, trust_remote_code=True)
    except TypeError:
        return get_kernel(repo, version=1)


class FlashRTQuantLinear(nn.Module):
    """nn.Linear drop-in: holds bf16 weight (loaded normally), quantized lazily on
    first forward to FlashRT NVFP4 / FP8, then runs the FlashRT GEMM kernel."""

    def __init__(self, in_features, out_features, bias, mode, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode  # "nvfp4" | "fp8"
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype or torch.bfloat16),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype or torch.bfloat16),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)
        self._ops = None
        self._ready = False
        # FP8 static activation scale (compile/decode-friendly); fit by calibrate_fp8.
        self.act_scale = None
        self._calibrating = False
        self._amax = None

    @torch.no_grad()
    def _quantize(self):
        ops = self._ops
        if self.mode == "nvfp4":
            w = self.weight.detach().to(torch.float16).contiguous()
            packed, sfb = ops.quantize_fp4_sfa_fp16(w, is_sfb=True)
            self.register_buffer("w_packed", packed)
            self.register_buffer("w_sfb", sfb)
        else:  # fp8
            w = self.weight.detach().to(torch.bfloat16)
            wscale = (w.abs().amax() / FP8_MAX).clamp_min(1e-6).reshape(1).float()
            self.register_buffer("w_fp8", (w / wscale).clamp(-FP8_MAX, FP8_MAX).to(FP8).contiguous())
            self.register_buffer("w_scale", wscale)
        self.weight = None  # free bf16
        self._ready = True

    def forward(self, x):
        if self.weight is not None and self.weight.device.type != "cuda":
            return torch.nn.functional.linear(x, self.weight, self.bias)  # pre-GPU safety
        if not self._ready:
            self._quantize()
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if self.mode == "nvfp4":
            ap, sfa = self._ops.quantize_fp4_sfa_fp16(x2.to(torch.float16).contiguous())
            out = self._ops.fp4_w4a16_linear_bf16(ap, self.w_packed, sfa, self.w_sfb, variant=2)
        else:
            xb = x2.to(torch.bfloat16).contiguous()
            if self.act_scale is not None:
                # static scale: single elementwise quant, no per-call reduction
                # -> fuses under torch.compile and has no M=1 decode overhead.
                xscale = self.act_scale
            else:
                xscale = (xb.abs().amax() / FP8_MAX).clamp_min(1e-6).reshape(1).float()
                if self._calibrating:
                    self._amax = xb.abs().amax() if self._amax is None \
                        else torch.maximum(self._amax, xb.abs().amax())
            x_fp8 = (xb / xscale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
            out = self._ops.fp8_gemm_bf16(x_fp8, self.w_fp8, xscale, self.w_scale)
        if self.bias is not None:
            out = out + self.bias
        return out.view(*shape[:-1], self.out_features).to(torch.bfloat16)


@torch.no_grad()
def calibrate_fp8(model, run) -> int:
    """Fit static per-tensor FP8 activation scales from representative forwards.

    Switches FP8 ``FlashRTQuantLinear`` layers from dynamic per-call scaling to a
    static scale, which removes the per-forward amax reduction so the FP8 path
    fuses under ``torch.compile`` and has no M=1 decode overhead. ``run`` should do
    a few forwards on representative inputs. No-op for NVFP4 (dynamic per-block).
    """

    layers = [m for m in model.modules()
              if isinstance(m, FlashRTQuantLinear) and m.mode == "fp8"]
    for m in layers:
        m.act_scale = None
        m._amax = None
        m._calibrating = True
    run()
    n = 0
    for m in layers:
        m._calibrating = False
        if m._amax is not None:
            m.act_scale = (m._amax / FP8_MAX).clamp_min(1e-6).reshape(1).float()
            m._amax = None
            n += 1
    return n


if "flashrt" not in __import__("transformers").quantizers.auto.AUTO_QUANTIZATION_CONFIG_MAPPING:

    @register_quantization_config("flashrt")
    class FlashRTConfig(QuantizationConfigMixin):
        def __init__(self, mode: str = "nvfp4", modules_to_not_convert=None, **kwargs):
            self.quant_method = "flashrt"
            self.mode = mode
            self.modules_to_not_convert = modules_to_not_convert or []

        def to_dict(self):
            return {"quant_method": "flashrt", "mode": self.mode,
                    "modules_to_not_convert": self.modules_to_not_convert}

    @register_quantizer("flashrt")
    class FlashRTHfQuantizer(HfQuantizer):
        requires_calibration = False

        def validate_environment(self, *args, **kwargs):
            if not torch.cuda.is_available():
                raise RuntimeError("FlashRT quantization requires a CUDA GPU (Blackwell SM120).")

        def update_dtype(self, dtype):
            return torch.bfloat16

        def _process_model_before_weight_loading(self, model, **kwargs):
            mode = self.quantization_config.mode
            skip = set(self.quantization_config.modules_to_not_convert) | {"lm_head"}

            def convert(module, prefix=""):
                for name, child in list(module.named_children()):
                    full = f"{prefix}.{name}" if prefix else name
                    if (isinstance(child, nn.Linear)
                            and name in _TARGET_SUFFIXES
                            and not any(s in full for s in skip)):
                        new = FlashRTQuantLinear(
                            child.in_features, child.out_features, child.bias is not None,
                            mode, device=child.weight.device, dtype=child.weight.dtype)
                        setattr(module, name, new)
                    else:
                        convert(child, full)

            convert(model)

        def _process_model_after_weight_loading(self, model, **kwargs):
            mode = self.quantization_config.mode
            ops = _get("flashrt/fp4-gemm") if mode == "nvfp4" else _get("flashrt/flashrt-fp8-ffn")
            for m in model.modules():
                if isinstance(m, FlashRTQuantLinear):
                    m._ops = ops
                    # Eager-quantize now (weights already on GPU) so the module is
                    # static and torch.compile / CUDA graph can capture it. Falls
                    # back to lazy first-forward quant if weights aren't on CUDA yet.
                    if m.weight is not None and m.weight.device.type == "cuda":
                        m._quantize()
            return model

        def param_needs_quantization(self, model, param_name, **kwargs):
            return False

        @property
        def is_serializable(self):
            return False

        @property
        def is_trainable(self):
            return False
else:  # already registered (module re-imported) — reuse the existing config class
    FlashRTConfig = __import__("transformers").quantizers.auto.AUTO_QUANTIZATION_CONFIG_MAPPING["flashrt"]
