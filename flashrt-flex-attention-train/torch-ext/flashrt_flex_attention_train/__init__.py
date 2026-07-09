"""FlashRT Flex-style block-sparse attention training API.

The public function implements the PI052 prefix/action mask pattern:

* prefix query rows use the original K/V tensors, so prefix losses keep normal
  gradients into prefix K/V;
* action query rows read detached prefix K/V plus normal action K/V by default,
  matching the current training semantics.

Unsupported shapes route to the SDPA reference path. Native CUDA kernels are
not exposed until a shape-specialized implementation beats SDPA on the target
A100/5090 validation matrix.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

try:
    from ._ops import ops

    _HAS_OPS = hasattr(ops, "_flashrt_training_package_marker")
except Exception:  # source-tree tests before kernel-builder creates _ops.py
    ops = None
    _HAS_OPS = False


MASK_VALUE_F32 = -2.3819763e38


def _use_ops(namespace_ops) -> None:
    """Install a manually built extension (dev/testing path)."""
    global ops, _HAS_OPS
    ops = namespace_ops
    _HAS_OPS = hasattr(ops, "_flashrt_training_package_marker")


def _check_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, and v must be shaped (B, H, S, D)")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v batch dimensions must match")
    if k.shape != v.shape:
        raise ValueError("k and v shapes must match")
    if q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
        raise ValueError("q, k, and v sequence/head_dim dimensions must match")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")


def _as_valid(mask: Optional[torch.Tensor], batch: int, length: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch, length), dtype=torch.bool, device=device)
    if mask.shape != (batch, length):
        raise ValueError(f"mask must be shaped {(batch, length)}, got {tuple(mask.shape)}")
    return mask.to(device=device, dtype=torch.bool)


def build_block_sparse_bool_masks(
    prefix_valid: Optional[torch.Tensor],
    prefix_att: Optional[torch.Tensor],
    *,
    batch: int,
    prefix_len: int,
    action_len: int,
    action_block_size: int,
    non_fast_prefix_len: Optional[int] = None,
    action_valid: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build boolean masks for the split FlexAttention SDPA calls.

    Returns ``(prefix_rows, action_rows)`` with shapes ``(B, P, S)`` and
    ``(B, A, S)``. Boolean True means the key/value position is visible.

    ``prefix_att`` follows Lerobot's cumulative-block convention: prefix key
    ``j`` is visible to prefix query ``i`` when ``cumsum(prefix_att)[j] <=
    cumsum(prefix_att)[i]`` and both rows are valid. When omitted, prefix rows
    attend to all valid prefix tokens.
    """
    if action_block_size <= 0:
        raise ValueError("action_block_size must be positive")
    if prefix_len < 0 or action_len < 0:
        raise ValueError("prefix_len and action_len must be non-negative")
    total_len = prefix_len + action_len
    dev = device
    if dev is None:
        for t in (prefix_valid, prefix_att, action_valid):
            if t is not None:
                dev = t.device
                break
    if dev is None:
        dev = torch.device("cpu")

    p_valid = _as_valid(prefix_valid, batch, prefix_len, dev)
    a_valid = _as_valid(action_valid, batch, action_len, dev)

    if prefix_att is None:
        prefix_rows = p_valid[:, :, None] & p_valid[:, None, :]
    else:
        if prefix_att.shape != (batch, prefix_len):
            raise ValueError(
                f"prefix_att must be shaped {(batch, prefix_len)}, got {tuple(prefix_att.shape)}"
            )
        cum = torch.cumsum(prefix_att.to(device=dev, dtype=torch.long), dim=1)
        prefix_rows = (cum[:, None, :] <= cum[:, :, None]) & p_valid[:, :, None] & p_valid[:, None, :]

    prefix_pad = torch.zeros((batch, prefix_len, action_len), dtype=torch.bool, device=dev)
    prefix_rows = torch.cat([prefix_rows, prefix_pad], dim=2)

    nf = prefix_len if non_fast_prefix_len is None else int(non_fast_prefix_len)
    nf = max(0, min(nf, prefix_len))
    action_to_prefix = torch.zeros((batch, action_len, prefix_len), dtype=torch.bool, device=dev)
    if nf > 0:
        action_to_prefix[:, :, :nf] = p_valid[:, None, :nf]
    action_to_prefix &= a_valid[:, :, None]

    q_block = torch.arange(action_len, device=dev) // int(action_block_size)
    kv_block = q_block
    action_block = q_block[:, None] == kv_block[None, :]
    action_block = action_block[None, :, :].expand(batch, -1, -1)
    action_block = action_block & a_valid[:, :, None] & a_valid[:, None, :]
    action_rows = torch.cat([action_to_prefix, action_block], dim=2)

    if prefix_rows.shape != (batch, prefix_len, total_len):
        raise AssertionError("internal prefix mask shape error")
    if action_rows.shape != (batch, action_len, total_len):
        raise AssertionError("internal action mask shape error")
    return prefix_rows, action_rows


def _bool_to_sdpa_mask(mask: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    value = MASK_VALUE_F32
    if q.dtype.is_floating_point:
        finfo = torch.finfo(q.dtype)
        value = max(MASK_VALUE_F32, finfo.min)
    return torch.where(
        mask[:, None, :, :],
        torch.zeros((), dtype=q.dtype, device=q.device),
        torch.full((), value, dtype=q.dtype, device=q.device),
    )


def _slice_attention_mask(
    attention_mask: torch.Tensor,
    start: int,
    end: int,
    q: torch.Tensor,
) -> torch.Tensor:
    if attention_mask.dim() == 3:
        mask = attention_mask[:, start:end, :]
        if mask.dtype == torch.bool:
            return mask[:, None, :, :]
        return mask[:, None, :, :].to(dtype=q.dtype)
    if attention_mask.dim() == 4:
        mask = attention_mask[:, :, start:end, :]
        return mask if mask.dtype == torch.bool else mask.to(dtype=q.dtype)
    raise ValueError("attention_mask must be (B, S, S) or (B, 1|H, S, S)")


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    scale: Optional[float],
    dropout_p: float,
    enable_gqa: bool,
) -> torch.Tensor:
    kwargs = {"attn_mask": mask, "dropout_p": float(dropout_p), "scale": scale}
    if enable_gqa:
        kwargs["enable_gqa"] = True
    try:
        return F.scaled_dot_product_attention(q, k, v, **kwargs)
    except TypeError:
        kwargs.pop("enable_gqa", None)
        return F.scaled_dot_product_attention(q, k, v, **kwargs)


def reference_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    prefix_len: int,
    action_block_size: int,
    attention_mask: Optional[torch.Tensor] = None,
    prefix_valid: Optional[torch.Tensor] = None,
    prefix_att: Optional[torch.Tensor] = None,
    non_fast_prefix_len: Optional[int] = None,
    action_valid: Optional[torch.Tensor] = None,
    detach_prefix_kv_for_action: bool = True,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    enable_gqa: Optional[bool] = None,
) -> torch.Tensor:
    """SDPA reference for the PI052 FlexAttention replacement shape.

    Args:
        q, k, v: ``(B, Hq/Hkv, S, D)`` tensors.
        prefix_len: number of prefix rows/columns at the start of sequence.
        action_block_size: size of each block-diagonal action segment.
        attention_mask: optional prebuilt additive or boolean full mask.
        prefix_valid: optional ``(B, P)`` valid prefix positions.
        prefix_att: optional ``(B, P)`` cumulative-block markers.
        non_fast_prefix_len: prefix columns visible to action rows.
        action_valid: optional ``(B, A)`` valid action positions.
        detach_prefix_kv_for_action: detach prefix K/V on the action-row path.
        scale: SDPA scale. Defaults to ``D ** -0.5``.
        dropout_p: SDPA dropout probability.
        enable_gqa: pass SDPA GQA mode when q heads and kv heads differ.
    """
    _check_qkv(q, k, v)
    batch, _, total_len, head_dim = q.shape
    if not (0 <= int(prefix_len) <= total_len):
        raise ValueError("prefix_len must be in [0, S]")
    prefix_len = int(prefix_len)
    action_len = total_len - prefix_len
    if scale is None:
        scale = head_dim**-0.5
    if enable_gqa is None:
        enable_gqa = q.shape[1] != k.shape[1]

    q_prefix = q[:, :, :prefix_len, :]
    q_action = q[:, :, prefix_len:, :]
    k_prefix = k[:, :, :prefix_len, :]
    k_action = k[:, :, prefix_len:, :]
    v_prefix = v[:, :, :prefix_len, :]
    v_action = v[:, :, prefix_len:, :]

    if attention_mask is None:
        prefix_bool, action_bool = build_block_sparse_bool_masks(
            prefix_valid,
            prefix_att,
            batch=batch,
            prefix_len=prefix_len,
            action_len=action_len,
            action_block_size=action_block_size,
            non_fast_prefix_len=non_fast_prefix_len,
            action_valid=action_valid,
            device=q.device,
        )
        prefix_mask = _bool_to_sdpa_mask(prefix_bool, q)
        action_mask = _bool_to_sdpa_mask(action_bool, q)
    else:
        prefix_mask = _slice_attention_mask(attention_mask, 0, prefix_len, q)
        action_mask = _slice_attention_mask(attention_mask, prefix_len, total_len, q)

    out_parts = []
    if prefix_len:
        out_parts.append(
            _sdpa(
                q_prefix,
                k,
                v,
                prefix_mask,
                scale=scale,
                dropout_p=dropout_p,
                enable_gqa=bool(enable_gqa),
            )
        )
    if action_len:
        prefix_k = k_prefix.detach() if detach_prefix_kv_for_action else k_prefix
        prefix_v = v_prefix.detach() if detach_prefix_kv_for_action else v_prefix
        k_for_action = torch.cat([prefix_k, k_action], dim=2)
        v_for_action = torch.cat([prefix_v, v_action], dim=2)
        out_parts.append(
            _sdpa(
                q_action,
                k_for_action,
                v_for_action,
                action_mask,
                scale=scale,
                dropout_p=dropout_p,
                enable_gqa=bool(enable_gqa),
            )
        )
    if not out_parts:
        return q.new_empty(q.shape)
    return torch.cat(out_parts, dim=2) if len(out_parts) == 2 else out_parts[0]


def flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    prefix_len: int,
    action_block_size: int,
    attention_mask: Optional[torch.Tensor] = None,
    prefix_valid: Optional[torch.Tensor] = None,
    prefix_att: Optional[torch.Tensor] = None,
    non_fast_prefix_len: Optional[int] = None,
    action_valid: Optional[torch.Tensor] = None,
    detach_prefix_kv_for_action: bool = True,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    enable_gqa: Optional[bool] = None,
    force_fallback: bool = False,
) -> torch.Tensor:
    """Flex-style block-sparse attention with automatic SDPA fallback."""
    _ = force_fallback
    return reference_flex_attention(
        q,
        k,
        v,
        prefix_len=prefix_len,
        action_block_size=action_block_size,
        attention_mask=attention_mask,
        prefix_valid=prefix_valid,
        prefix_att=prefix_att,
        non_fast_prefix_len=non_fast_prefix_len,
        action_valid=action_valid,
        detach_prefix_kv_for_action=detach_prefix_kv_for_action,
        scale=scale,
        dropout_p=dropout_p,
        enable_gqa=enable_gqa,
    )


def flex_attention_forward(*args, **kwargs) -> torch.Tensor:
    """Forward-only compatibility wrapper."""
    return flex_attention(*args, **kwargs)


def backend_marker(x: torch.Tensor) -> torch.Tensor:
    if ops is None:
        return x
    return ops._flashrt_training_package_marker(x)


__all__ = [
    "MASK_VALUE_F32",
    "backend_marker",
    "build_block_sparse_bool_masks",
    "flex_attention",
    "flex_attention_forward",
    "reference_flex_attention",
]
