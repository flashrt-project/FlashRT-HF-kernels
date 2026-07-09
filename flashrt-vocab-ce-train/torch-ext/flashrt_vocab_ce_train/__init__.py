"""FlashRT fused linear-CE for small-N, huge-vocab heads (training).

Hybrid forward: the logits GEMM stays on cuBLAS (measured faster than any
streaming variant at these shapes — the GEMM is bandwidth-optimal there
already), then one fused kernel pass over the materialized logits produces
per-row online-softmax partials, replacing the separate cross-entropy +
logsumexp passes. The merge and the scalar loss are tiny (N,)-sized torch
ops. Backward reconstructs dlogits from the saved logits/lse and runs the
two GEMMs through cuBLAS.

Semantics match the materialized-logits reference exactly:
``ignore_index`` positions contribute nothing, the loss is
``sum / n_valid`` (all-ignored stays 0.0 with no host sync), and the z-loss
term is ``z * mean(lse^2 over valid)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from ._ops import ops

    _HAS_OPS = hasattr(ops, "vocab_ce_fwd_stream") and hasattr(ops, "vocab_ce_stats")
except Exception:  # source-tree tests before kernel-builder creates _ops.py
    ops = None
    _HAS_OPS = False

_MAX_ROWS = 128
_STATS_SPLITS = 8


def _use_ops(namespace_ops) -> None:
    """Install a manually built extension (dev/testing path)."""
    global ops, _HAS_OPS
    ops = namespace_ops
    _HAS_OPS = hasattr(ops, "vocab_ce_fwd_stream") and hasattr(ops, "vocab_ce_stats")
    _register_fakes()


# ---------------------------------------------------------------------------
# reference (differentiable; also the fallback for unsupported shapes)
# ---------------------------------------------------------------------------


def reference_vocab_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    z_loss_weight: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    compute_dtype = (
        torch.float64
        if hidden.dtype == torch.float64 or weight.dtype == torch.float64
        else torch.float32
    )
    logits = hidden.to(compute_dtype) @ weight.to(compute_dtype).t()
    valid = labels != int(ignore_index)
    n_valid = valid.sum().clamp(min=1)
    loss = F.cross_entropy(logits, labels, ignore_index=int(ignore_index), reduction="sum") / n_valid
    if float(z_loss_weight) > 0:
        lse = torch.logsumexp(logits, dim=-1)
        loss = loss + float(z_loss_weight) * (lse.square() * valid.to(lse.dtype)).sum() / n_valid
    return loss


# ---------------------------------------------------------------------------
# streaming CUDA path
# ---------------------------------------------------------------------------


def _register_fakes() -> None:
    if not _HAS_OPS:
        return
    try:
        namespace = ops.vocab_ce_fwd_stream.default.name().split("::")[0]

        @torch.library.register_fake(f"{namespace}::vocab_ce_fwd_stream")
        def _(hidden, weight, labels):
            rows, v = hidden.shape[0], weight.shape[0]
            tiles = v // 32
            return (
                hidden.new_empty((rows, v)),
                hidden.new_empty((rows, tiles)),
                hidden.new_empty((rows, tiles)),
                hidden.new_empty((rows,)),
            )

        @torch.library.register_fake(f"{namespace}::vocab_ce_stats")
        def _(logits):
            rows = logits.shape[0]
            return (
                logits.new_empty((rows, _STATS_SPLITS)),
                logits.new_empty((rows, _STATS_SPLITS)),
            )
    except Exception:
        pass


_register_fakes()


class _VocabCEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, z_loss_weight, ignore_index):
        logits = torch.mm(hidden, weight.t())
        pmax, psum = ops.vocab_ce_stats(logits)
        label_logit = logits.gather(1, labels.clamp(min=0)[:, None]).squeeze(1)
        m = pmax.max(dim=1).values
        lse = m + torch.log((psum * torch.exp(pmax - m[:, None])).sum(dim=1))
        valid = (labels != ignore_index).to(lse.dtype)
        n_valid = valid.sum().clamp(min=1.0)
        loss = ((lse - label_logit) * valid).sum() / n_valid
        if z_loss_weight > 0:
            loss = loss + z_loss_weight * (lse.square() * valid).sum() / n_valid
        ctx.save_for_backward(hidden, weight, labels, logits, lse, valid, n_valid)
        ctx.z_loss_weight = z_loss_weight
        return loss

    @staticmethod
    def backward(ctx, dloss):
        hidden, weight, labels, logits, lse, valid, n_valid = ctx.saved_tensors
        z = ctx.z_loss_weight
        p = torch.exp(logits - lse[:, None])
        dlse = valid * (1.0 + 2.0 * z * lse) / n_valid
        dlogits = p * dlse[:, None]
        rows = torch.arange(labels.shape[0], device=labels.device)
        dlogits[rows, labels.clamp(min=0)] -= valid / n_valid
        dlogits *= dloss
        dhidden = dlogits @ weight
        dweight = dlogits.t() @ hidden
        return dhidden, dweight, None, None, None


def _kernel_supported(hidden: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        _HAS_OPS
        and hidden.is_cuda
        and hidden.dim() == 2
        and 1 <= hidden.shape[0] <= _MAX_ROWS
        and weight.dtype == torch.float32
        and weight.shape[0] % 32 == 0
        and weight.shape[1] % 128 == 0
    )


def vocab_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    z_loss_weight: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Linear-CE loss over a huge vocab head; streaming kernel for small N."""
    if not _kernel_supported(hidden, weight):
        return reference_vocab_ce(hidden, weight, labels, z_loss_weight, ignore_index)
    hidden32 = hidden.float().contiguous()  # exact: matches the reference upcast
    return _VocabCEFn.apply(
        hidden32, weight.contiguous(), labels.contiguous(), float(z_loss_weight), int(ignore_index)
    )


def backend_marker(x: torch.Tensor) -> torch.Tensor:
    if ops is None:
        return x
    return ops._flashrt_training_package_marker(x)


__all__ = ["vocab_ce", "reference_vocab_ce", "backend_marker"]
