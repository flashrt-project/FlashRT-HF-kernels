"""FlashRT SageAttention2 processor for Wan attention (Blackwell / SM120).

Routes Wan's self- and cross-attention through the FlashRT
``flashrt/sageattention2-blackwell`` Kernel-Hub kernel — an INT8-QK / FP8-PV
quantized prefill attention core for head_dim 128 — via the official diffusers
attention-processor API. It mirrors the stock Wan processor (Q/K projections,
Q/K RMSNorm, RoPE, optional I2V image branch, output projection) and only swaps
the attention core.

Drop in with ``pipe.transformer.set_attn_processor(WanSageAttention2Processor())``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from kernels import get_kernel
from diffusers.models.transformers.transformer_wan import (
    _get_added_kv_projections,
    _get_qkv_projections,
)


_SAGE = None


def _get_sage():
    # Load the Hub kernel once per process — re-importing it would re-run its
    # register_fake and raise on the duplicate op registration.
    global _SAGE
    if _SAGE is None:
        _SAGE = get_kernel("flashrt/sageattention2-blackwell", revision="v1")
    return _SAGE


def _apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class WanSageAttention2Processor:
    """Wan attention processor backed by FlashRT SageAttention2 (head_dim 128)."""

    def __init__(self, sage=None, *, fp8_v: bool = True):
        self._sage = sage or _get_sage()
        self._kernel = (
            self._sage.sage2_prefill_fp8v_bf16_d128
            if fp8_v
            else self._sage.sage2_prefill_f16_bf16_d128
        )

    def _attn(self, q, k, v):
        # q, k, v: (batch, seqlen, heads, head_dim), non-causal self/cross attn.
        if q.shape[-1] == 128:
            return self._kernel(q, k, v, causal=False)
        # The kernel is head_dim 128 only; other dims fall back to SDPA.
        qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))
        return F.scaled_dot_product_attention(qh, kh, vh).transpose(1, 2)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
    ):
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))
            hidden_states_img = self._attn(query, key_img, value_img)
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        hidden_states = self._attn(query, key, value).flatten(2, 3).type_as(query)
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
