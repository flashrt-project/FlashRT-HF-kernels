"""FlashRT delayed-codebook selection and embedding kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


def _check(logits, codebook, delay: int, boc: int, codes, embedding) -> None:
    if logits.dim() != 2:
        raise RuntimeError("logits must have shape (num_codebooks, codebook_vocab)")
    if codebook.dim() != 3 or codebook.shape[:2] != logits.shape:
        raise RuntimeError("codebook must have shape (num_codebooks, codebook_vocab, hidden)")
    if codes.shape != (logits.shape[0],):
        raise RuntimeError("codes must have shape (num_codebooks,)")
    if embedding.shape != (codebook.shape[2],):
        raise RuntimeError("embedding must have shape (hidden,)")
    if not 0 <= delay <= logits.shape[0]:
        raise RuntimeError("delay must be in [0, num_codebooks]")
    if not 0 <= boc < logits.shape[1]:
        raise RuntimeError("boc must be a valid codebook index")


@torch.library.register_fake(
    add_op_namespace_prefix("delayed_codebook_argmax_embed_bf16")
)
def _argmax_fake(logits, codebook, delay: int, boc: int, codes, embedding) -> None:
    _check(logits, codebook, delay, boc, codes, embedding)
    return None


@torch.library.register_fake(
    add_op_namespace_prefix("delayed_codebook_sample_embed_bf16")
)
def _sample_fake(
    logits, codebook, delay: int, boc: int, temperature: float,
    seed: int, step: int, codes, embedding
) -> None:
    _check(logits, codebook, delay, boc, codes, embedding)
    if temperature <= 0:
        raise RuntimeError("temperature must be strictly positive")
    if step < 0:
        raise RuntimeError("step must be non-negative")
    return None


def _outputs(logits, codebook, codes, embedding):
    if codes is None:
        codes = torch.empty(
            (logits.shape[0],), device=logits.device, dtype=torch.int64
        )
    if embedding is None:
        embedding = torch.empty(
            (codebook.shape[2],), device=codebook.device, dtype=torch.bfloat16
        )
    return codes, embedding


def delayed_codebook_argmax_embed_bf16(
    logits: torch.Tensor,
    codebook: torch.Tensor,
    *,
    delay: int,
    boc: int,
    codes: Optional[torch.Tensor] = None,
    embedding: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    codes, embedding = _outputs(logits, codebook, codes, embedding)
    ops.delayed_codebook_argmax_embed_bf16(
        logits, codebook, int(delay), int(boc), codes, embedding
    )
    return codes, embedding


def delayed_codebook_sample_embed_bf16(
    logits: torch.Tensor,
    codebook: torch.Tensor,
    *,
    delay: int,
    boc: int,
    temperature: float,
    seed: int,
    step: int,
    codes: Optional[torch.Tensor] = None,
    embedding: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    codes, embedding = _outputs(logits, codebook, codes, embedding)
    ops.delayed_codebook_sample_embed_bf16(
        logits, codebook, int(delay), int(boc), float(temperature),
        int(seed), int(step), codes, embedding
    )
    return codes, embedding


__all__ = [
    "delayed_codebook_argmax_embed_bf16",
    "delayed_codebook_sample_embed_bf16",
]
