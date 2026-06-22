"""FlashRT as a diffusers quantization backend.

Mirrors the transformers quantizer: FlashRT FP8 / NVFP4 GEMMs load through the
official diffusers quantization API, the same entry point as torchao / bnb /
modelopt:

    from diffusers import WanTransformer3DModel
    from flashrt_diffusers_quantizer import FlashRTDiffusersConfig
    transformer = WanTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer",
        quantization_config=FlashRTDiffusersConfig(mode="nvfp4"), torch_dtype="bfloat16")

diffusers 0.38 has no public ``register_quantizer`` (unlike transformers), so we
register by inserting into the backend mapping dicts — the official
``DiffusersQuantizer`` base + ``quantization_config`` flow is used unchanged.

Targets every ``nn.Linear`` inside the DiT blocks (FFN up/down + attention
q/k/v/o). The VAE is Conv3d (not Linear) and is handled separately.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from kernels import get_kernel
from diffusers.quantizers.base import DiffusersQuantizer
from diffusers.quantizers import auto as _dq_auto
from diffusers.quantizers.quantization_config import QuantizationConfigMixin

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def _get(repo):
    try:
        return get_kernel(repo, version=1, trust_remote_code=True)
    except TypeError:
        return get_kernel(repo, version=1)


class FlashRTQuantLinear(nn.Module):
    """nn.Linear drop-in: bf16 weight loads normally, quantized lazily on first
    forward to FlashRT NVFP4 / FP8, then runs the FlashRT GEMM kernel."""

    def __init__(self, in_features, out_features, bias, mode, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype or torch.bfloat16),
            requires_grad=False)
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype or torch.bfloat16),
                requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self._ops = None
        self._ready = False
        self.act_scale = None       # FP8 static activation scale (calibrate_fp8)
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
        else:
            w = self.weight.detach().to(torch.bfloat16)
            wscale = (w.abs().amax() / FP8_MAX).clamp_min(1e-6).reshape(1).float()
            self.register_buffer("w_fp8", (w / wscale).clamp(-FP8_MAX, FP8_MAX).to(FP8).contiguous())
            self.register_buffer("w_scale", wscale)
        self.weight = None
        self._ready = True

    def forward(self, x):
        if self.weight is not None and self.weight.device.type != "cuda":
            return torch.nn.functional.linear(x, self.weight, self.bias)
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
                xscale = self.act_scale  # static: compile-friendly, no reduction
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
    """Fit static per-tensor FP8 activation scales from representative forwards
    (removes the per-call amax so FP8 fuses under torch.compile). No-op for NVFP4."""

    layers = [m for m in model.modules()
              if isinstance(m, FlashRTQuantLinear) and m.mode == "fp8"]
    for m in layers:
        m.act_scale, m._amax, m._calibrating = None, None, True
    run()
    n = 0
    for m in layers:
        m._calibrating = False
        if m._amax is not None:
            m.act_scale = (m._amax / FP8_MAX).clamp_min(1e-6).reshape(1).float()
            m._amax = None
            n += 1
    return n


class _QuantMethod(str):
    """str that also exposes ``.value`` (diffusers expects an enum-like)."""

    @property
    def value(self):
        return str(self)


class FlashRTDiffusersConfig(QuantizationConfigMixin):
    def __init__(self, mode: str = "nvfp4", **kwargs):
        self.quant_method = _QuantMethod("flashrt")
        self.mode = mode

    def to_dict(self):
        return {"quant_method": "flashrt", "mode": self.mode}

    def to_diff_dict(self):
        return self.to_dict()


class FlashRTDiffusersQuantizer(DiffusersQuantizer):
    use_keep_in_fp32_modules = False
    requires_calibration = False
    requires_parameters_quantization = False

    def validate_environment(self, *args, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("FlashRT quantization requires a CUDA GPU (Blackwell SM120).")

    def update_torch_dtype(self, torch_dtype):
        return torch_dtype or torch.bfloat16

    def check_if_quantized_param(self, *args, **kwargs):
        return False  # weights load as bf16; we quantize lazily on first forward

    def _process_model_before_weight_loading(self, model, **kwargs):
        mode = self.quantization_config.mode

        def convert(module):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear) and child.in_features % 16 == 0 \
                        and child.out_features % 16 == 0:
                    setattr(module, name, FlashRTQuantLinear(
                        child.in_features, child.out_features, child.bias is not None,
                        mode, device=child.weight.device, dtype=child.weight.dtype))
                else:
                    convert(child)

        # only the DiT blocks (FFN + attention projections); leave embedders / proj_out
        blocks = getattr(model, "blocks", None)
        convert(blocks if blocks is not None else model)

    def _process_model_after_weight_loading(self, model, **kwargs):
        mode = self.quantization_config.mode
        ops = _get("flashrt/fp4-gemm") if mode == "nvfp4" else _get("flashrt/flashrt-fp8-ffn")
        for m in model.modules():
            if isinstance(m, FlashRTQuantLinear):
                m._ops = ops
                # Eager-quantize now so the module is static and torch.compile can
                # capture it; lazy first-forward quant is the fallback otherwise.
                if m.weight is not None and m.weight.device.type == "cuda":
                    m._quantize()
        return model

    @property
    def is_serializable(self):
        return False

    @property
    def is_trainable(self):
        return False


# diffusers 0.38 exposes no register_quantizer; insert into the backend mappings.
if "flashrt" not in _dq_auto.AUTO_QUANTIZER_MAPPING:
    _dq_auto.AUTO_QUANTIZER_MAPPING["flashrt"] = FlashRTDiffusersQuantizer
    _dq_auto.AUTO_QUANTIZATION_CONFIG_MAPPING["flashrt"] = FlashRTDiffusersConfig
