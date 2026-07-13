"""Experimental native E0M3/INT4 tensor-core primitives for Blackwell SM12x."""

from __future__ import annotations

from importlib.resources import files
from typing import Literal

import torch

from ._ops import ops

OperandMode = Literal["e2m1", "a", "b", "ab"]

_CUBIN_NAMES = {
    "e2m1": "probe.cubin",
    "a": "probe_int4a.cubin",
    "b": "probe_int4b.cubin",
    "ab": "probe_int4.cubin",
}
_SUPPORTED_ARCHES = {(12, 0): "sm120", (12, 1): "sm121"}
_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def _architecture(device: int) -> str:
    capability = torch.cuda.get_device_capability(device)
    try:
        return _SUPPORTED_ARCHES[capability]
    except KeyError as error:
        supported = ", ".join(name.upper() for name in _SUPPORTED_ARCHES.values())
        raise RuntimeError(
            f"int4-blackwell supports {supported}; got SM{capability[0]}{capability[1]}"
        ) from error


def _cubin(mode: OperandMode, device: int) -> torch.Tensor:
    if mode not in _CUBIN_NAMES:
        raise ValueError(
            f"mode must be one of {tuple(_CUBIN_NAMES)}, got {mode!r}"
        )
    architecture = _architecture(device)
    key = (architecture, mode)
    if key not in _CACHE:
        data = (
            files(__package__)
            .joinpath("cubin", architecture, _CUBIN_NAMES[mode])
            .read_bytes()
        )
        _CACHE[key] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return _CACHE[key]


def _device_index(device: int | torch.device | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    parsed = torch.device(device)
    if parsed.type != "cuda":
        raise ValueError(f"device must be CUDA, got {parsed}")
    return torch.cuda.current_device() if parsed.index is None else parsed.index


def codebook_probe(
    mode: OperandMode = "ab", *, device: int | torch.device | None = None
) -> torch.Tensor:
    """Return the 16-value A-operand decode table measured by native MMA.

    ``mode`` selects standard E2M1 or the patched INT4 format independently
    for the A and B operands. The result is synchronized and returned on CPU.
    """
    dev = _device_index(device)
    tile = ops.run_codebook_probe(_cubin(mode, dev), dev)
    torch.cuda.synchronize(dev)
    if not torch.equal(tile, tile[:, :1].expand_as(tile)):
        raise RuntimeError("native MMA output tile is not uniform")
    return (tile[:, 0] / 64.0).cpu()


def mma_probe(
    mode: OperandMode = "ab",
    *,
    iterations: int = 8192,
    blocks: int | None = None,
    launches: int = 1,
    device: int | torch.device | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch the register-resident MMA throughput probe asynchronously."""
    dev = _device_index(device)
    if blocks is None:
        blocks = torch.cuda.get_device_properties(dev).multi_processor_count * 4
    if out is None:
        out = torch.empty((blocks, 256), device=f"cuda:{dev}", dtype=torch.float32)
    ops.run_mma_probe(_cubin(mode, dev), out, iterations, blocks, launches, dev)
    return out


__all__ = ["codebook_probe", "mma_probe"]
