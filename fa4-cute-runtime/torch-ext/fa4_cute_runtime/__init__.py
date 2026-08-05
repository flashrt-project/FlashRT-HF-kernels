"""Forward-only FlashAttention-4 CuTe runtime for SM100-family GPUs."""

from __future__ import annotations

import os

import cutlass


def _version_tuple(value: str) -> tuple[int, int]:
    fields = value.split(".")
    return tuple(int(field) for field in fields[:2])  # type: ignore[return-value]


_DSL_VERSION = _version_tuple(str(getattr(cutlass, "__version__", "0.0")))
if not ((4, 4) <= _DSL_VERSION < (4, 6)):
    raise RuntimeError(
        "fa4-cute-runtime requires nvidia-cutlass-dsl 4.4.x or 4.5.x; "
        f"found {getattr(cutlass, '__version__', 'unknown')}"
    )

os.environ.setdefault(
    "CUTE_DSL_ARCH", "sm_101a" if _DSL_VERSION >= (4, 5) else "sm_110a"
)
os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_100a")

from flashrt_fa4.cute import flash_attn_func, flash_attn_varlen_func  # noqa: E402
from flashrt_fa4.cute.interface_fwd_sm100 import _flash_attn_fwd  # noqa: E402


def forward_static(
    q,
    k,
    v,
    out,
    *,
    softmax_scale=None,
    causal=False,
    pack_gqa=None,
    seqused_k=None,
):
    """Run forward attention into a caller-owned output tensor."""
    result, _ = _flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        pack_gqa=pack_gqa,
        seqused_k=seqused_k,
        out=out,
    )
    return result


__all__ = ["flash_attn_func", "flash_attn_varlen_func", "forward_static"]
