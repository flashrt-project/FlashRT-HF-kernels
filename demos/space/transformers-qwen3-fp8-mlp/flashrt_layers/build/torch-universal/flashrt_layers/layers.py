"""FlashRT kernel-layers for the transformers/kernels ``kernelize`` mechanism.

Each class' ``forward`` runs with ``self`` bound to the original decorated layer
(e.g. a model's RMSNorm), so it reads that layer's own weight/eps and calls the
published FlashRT op. These layer classes are what ``register_kernel_mapping`` +
``kernelize`` swap in, the same way gpt-oss maps RMSNorm to Liger kernels.
"""

import torch
from kernels import get_kernel

_RMS = None


def _rms_ops():
    global _RMS
    if _RMS is None:
        try:
            _RMS = get_kernel(
                "flashrt/flashrt-residual-norm-quant", version=1, trust_remote_code=True
            )
        except TypeError:
            _RMS = get_kernel("flashrt/flashrt-residual-norm-quant", version=1)
    return _RMS


class RMSNorm(torch.nn.Module):
    """FlashRT RMSNorm; ``self`` is the original layer (reads ``weight``/``variance_epsilon``)."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        eps = float(getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6)))
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        out = _rms_ops().rms_norm_bf16(x, self.weight.to(torch.bfloat16), eps)
        return out.view(shape).to(hidden_states.dtype)
