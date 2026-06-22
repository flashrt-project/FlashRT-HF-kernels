"""FlashRT SageAttention2 attention backend for transformers (Blackwell / SM120).

Registers an ``attn_implementation`` named ``"sage2_blackwell"`` that routes causal
prefill self-attention through the FlashRT ``flashrt/sageattention2-blackwell``
Kernel-Hub kernel (INT8-QK / FP8-PV, head_dim 128, native GQA). Decode steps
(q_len == 1), masked attention, and non-128 head dims fall back to PyTorch SDPA —
SageAttention2 is a prefill kernel, so the win is on long-context prompts.

Usage::

    import flashrt_sage_attn  # registers "sage2_blackwell"
    model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="sage2_blackwell")
"""

from __future__ import annotations

from kernels import get_kernel
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

_SAGE = None


def _get_sage():
    global _SAGE
    if _SAGE is None:
        _SAGE = get_kernel("flashrt/sageattention2-blackwell", revision="v1")
    return _SAGE


def sage2_blackwell_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
):
    # query: (B, Hq, Sq, D); key/value: (B, Hkv, Sk, D) — GQA passed natively.
    Sq, D = query.shape[2], query.shape[3]
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    use_sage = (
        D == 128
        and attention_mask is None        # pure causal prefill (no padding mask)
        and Sq > 1                         # prefill, not decode
        and key.shape[2] == Sq            # self-attention
        and query.is_cuda
    )
    if use_sage:
        q = query.transpose(1, 2).contiguous()  # (B, Sq, Hq, D)
        k = key.transpose(1, 2).contiguous()    # (B, Sk, Hkv, D)
        v = value.transpose(1, 2).contiguous()
        out = _get_sage().sage2_prefill_fp8v_bf16_d128(
            q, k, v, causal=bool(causal), softmax_scale=scaling
        )
        return out.contiguous(), None            # (B, Sq, Hq, D)
    return sdpa_attention_forward(
        module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs
    )


AttentionInterface.register("sage2_blackwell", sage2_blackwell_attention)
