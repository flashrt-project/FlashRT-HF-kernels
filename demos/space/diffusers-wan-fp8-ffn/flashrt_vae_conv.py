"""FP8 replacement for Wan VAE causal 3D convs, via world-model-conv.

The Wan VAE's dominant conv is ``WanCausalConv3d(3,3,3)`` (52 of 72 conv modules).
``flashrt/world-model-conv``'s ``fp8_conv3d_v18_ncdhw_res_bf16out`` is exactly a
causal 3x3x3 conv with a 2-frame cache; per-tensor scaling folds into its single
``alpha`` (alpha = weight_scale * input_scale). Running the VAE convs in FP8
accelerates decode, not just the transformer.

Per-conv correctness vs the BF16 conv is ~0.999 cosine.
"""

from __future__ import annotations

import torch
from kernels import get_kernel
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

FP8_E4M3_MAX = 448.0
_WMC = None


def _wmc():
    global _WMC
    if _WMC is None:
        try:
            _WMC = get_kernel("flashrt/world-model-conv", version=1, trust_remote_code=True)
        except TypeError:
            _WMC = get_kernel("flashrt/world-model-conv", version=1)
    return _WMC


def _amax_scale(x):
    return (x.float().abs().amax() / FP8_E4M3_MAX).clamp(min=1e-8).reshape(()).float()


def _to_fp8(x, scale):
    return (x.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)


class FlashRTFP8WanCausalConv3d(torch.nn.Module):
    """FP8 drop-in for ``WanCausalConv3d`` with kernel (3,3,3), preserving its
    ``forward(x, cache_x=None)`` interface (the VAE's feature-cache contract)."""

    def __init__(self, conv: WanCausalConv3d, input_scale: torch.Tensor):
        super().__init__()
        # keep the original conv for the variable-cache cases v18 does not cover
        self._orig = conv
        w = conv.weight.detach().to(torch.bfloat16).permute(0, 2, 3, 4, 1).contiguous()
        w_scale = _amax_scale(w)
        self.register_buffer("weight_fp8", _to_fp8(w, w_scale))
        bias = conv.bias
        self.register_buffer(
            "bias",
            (bias.detach().to(torch.bfloat16) if bias is not None
             else torch.zeros(conv.out_channels, dtype=torch.bfloat16, device=w.device)).contiguous(),
        )
        self.register_buffer("input_scale", input_scale.reshape(()).float())
        self.alpha = float(w_scale * self.input_scale)
        self.out_channels = conv.out_channels

    def forward(self, x, cache_x=None):
        n, _, t, h, w = x.shape
        # v18 requires a 2-frame cache; fall back to BF16 for other cache sizes
        if cache_x is not None and cache_x.shape[2] != 2:
            return self._orig(x, cache_x)
        new_fp8 = _to_fp8(x.permute(0, 2, 3, 4, 1).contiguous(), self.input_scale)
        if cache_x is None:
            cache_fp8 = torch.zeros(n, 2, h, w, x.shape[1], device=x.device, dtype=torch.float8_e4m3fn)
        else:
            cache_fp8 = _to_fp8(cache_x.permute(0, 2, 3, 4, 1).contiguous(), self.input_scale)
        residual = torch.zeros(n, self.out_channels, t, h, w, device=x.device, dtype=torch.bfloat16)
        return _wmc().fp8_conv3d_v18_ncdhw_res_bf16out(
            cache_fp8, new_fp8, self.weight_fp8, self.bias, residual, self.alpha
        )


def _is_target(mod):
    # v18 supports 3x3x3 causal convs with Ci % 32 == 0 and Co % 8 == 0.
    return (
        isinstance(mod, WanCausalConv3d)
        and tuple(mod.kernel_size) == (3, 3, 3)
        and mod.in_channels % 32 == 0
        and mod.out_channels % 8 == 0
    )


@torch.no_grad()
def collect_vae_conv_inputs(vae, run_decode):
    """Capture inputs of each target VAE conv during one ``run_decode()`` call."""

    captured, handles = {}, []
    for name, mod in vae.named_modules():
        if _is_target(mod):
            def make_hook(key):
                def hook(_m, args, _o):
                    if key not in captured:
                        captured[key] = args[0].detach()
                return hook
            handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        run_decode()
    finally:
        for h in handles:
            h.remove()
    return captured


def patch_wan_vae_conv(vae, calibration_inputs):
    """Replace every ``WanCausalConv3d(3,3,3)`` in the VAE with the FP8 version."""

    replaced = 0
    for name, mod in list(vae.named_modules()):
        if not _is_target(mod):
            continue
        calib = calibration_inputs.get(name)
        if calib is None:
            continue
        parent = vae.get_submodule(name.rsplit(".", 1)[0]) if "." in name else vae
        attr = name.rsplit(".", 1)[-1]
        setattr(parent, attr, FlashRTFP8WanCausalConv3d(mod, _amax_scale(calib)))
        replaced += 1
    return replaced
