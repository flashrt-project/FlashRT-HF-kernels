"""Build and load this package directly from source."""

from __future__ import annotations

import os
from pathlib import Path
import torch

PACKAGE = Path(__file__).resolve().parents[1]


class SourceOps:
    def __init__(self, namespace: str, compileable) -> None:
        self.ops = getattr(torch.ops, namespace)
        self.compileable = compileable

    def fp8_causal_gqa_attention_bf16(self, q, k, v, *, softmax_scale, out=None):
        if out is None:
            return self.compileable(q, k, v, float(softmax_scale))
        self.ops.fp8_causal_gqa_attention_bf16_out(q, k, v, float(softmax_scale), out)
        return out


def load_source_ops(registration_include: str | None = None) -> SourceOps:
    from torch.utils.cpp_extension import load

    include = registration_include or os.environ.get(
        "KERNEL_BUILDER_REGISTRATION_INCLUDE"
    )
    if not include:
        candidate = (
            PACKAGE.parent.parent
            / "kernels/kernel-builder/src/pyproject/templates/torch"
        )
        if candidate.is_dir():
            include = str(candidate)
    if not include or not (Path(include) / "registration.h").is_file():
        raise RuntimeError("registration.h not found")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    namespace = "fp8_prefill_attention_blackwell_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext/torch_binding.cpp"),
            str(PACKAGE / "csrc/fmha_fp8_causal_gqa_sm120.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), include],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "-DCUDA_KERNEL"],
        is_python_module=False,
        verbose=False,
    )
    raw = getattr(torch.ops, namespace).fp8_causal_gqa_attention_bf16_out
    wrapper_namespace = "fp8_prefill_attention_blackwell_source_wrapper"

    @torch.library.custom_op(
        f"{wrapper_namespace}::run",
        mutates_args=(),
        device_types="cuda",
    )
    def compileable(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        output = torch.empty_like(query, dtype=torch.bfloat16)
        raw(query, key, value, softmax_scale, output)
        return output

    @torch.library.register_fake(f"{wrapper_namespace}::run")
    def fake(query, key, value, softmax_scale):
        return torch.empty_like(query, dtype=torch.bfloat16)

    return SourceOps(namespace, compileable)
