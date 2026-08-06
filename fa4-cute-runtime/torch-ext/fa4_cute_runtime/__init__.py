"""Forward-only FlashAttention-4 CuTe runtime for SM100-family GPUs."""

from __future__ import annotations

import os
import sys

import cutlass
from cutlass import cute


def _version_tuple(value: str) -> tuple[int, int]:
    fields = value.split(".")
    return tuple(int(field) for field in fields[:2])  # type: ignore[return-value]


_DSL_VERSION = _version_tuple(str(getattr(cutlass, "__version__", "0.0")))
if not ((4, 4) <= _DSL_VERSION < (4, 7)):
    raise RuntimeError(
        "fa4-cute-runtime requires nvidia-cutlass-dsl 4.4.x, 4.5.x, or 4.6.x; "
        f"found {getattr(cutlass, '__version__', 'unknown')}"
    )

os.environ.setdefault(
    "CUTE_DSL_ARCH", "sm_101a" if _DSL_VERSION >= (4, 5) else "sm_110a"
)
os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_100a")

# CUTLASS DSL 4.6 promoted these public types out of ``cute.core``. The
# vendored FA4 sources use the 4.4/4.5 annotation paths; restoring the aliases
# keeps those annotations importable without changing generated kernels.
if _DSL_VERSION >= (4, 6):
    if not hasattr(cute.core, "ThrMma"):
        cute.core.ThrMma = cute.ThrMma
    if not hasattr(cute.core, "ThrCopy"):
        cute.core.ThrCopy = cute.ThrCopy
    if not hasattr(cute, "make_fragment"):
        cute.make_fragment = cute.make_rmem_tensor

# Kernel Hub imports a noarch variant under a content-derived module name after
# flattening this package into the variant root. Preserve the public package
# alias used by the vendored FA4 absolute imports.
sys.modules.setdefault("fa4_cute_runtime", sys.modules[__name__])

# kernel-builder copies only the directory matching ``general.name`` for a
# torch-noarch package. The private vendor and quack subset therefore live
# inside this module and use package-qualified imports.
from .flashrt_fa4.cute import flash_attn_func, flash_attn_varlen_func  # noqa: E402
from .flashrt_fa4.cute.interface_fwd_sm100 import _flash_attn_fwd  # noqa: E402


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
