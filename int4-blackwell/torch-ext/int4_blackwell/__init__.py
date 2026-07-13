"""Experimental native E0M3/INT4 tensor-core primitives for Blackwell."""

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


def _capability(device: int) -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(device))


def _is_tcgen05(device: int) -> bool:
    return _capability(device) in {(10, 0), (10, 3), (11, 0)}


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
    if _is_tcgen05(dev):
        if mode != "ab":
            raise ValueError(
                "the tcgen05 backend currently exposes the native INT4 x INT4 "
                "descriptor; use mode='ab'"
            )
        m = n = k = 128
        b_packed = torch.full(
            (n, k // 2), 0x11, device=f"cuda:{dev}", dtype=torch.uint8
        )
        # Constant-one UE4M3 storage is layout-invariant. Deliberately
        # overallocate the physical CUTLASS scale-factor tensors for this
        # instruction canary so no private layout helper leaks into the API.
        sfa = torch.full((m * k,), 0x38, device=f"cuda:{dev}", dtype=torch.uint8)
        sfb = torch.full((n * k,), 0x38, device=f"cuda:{dev}", dtype=torch.uint8)
        values = []
        for value in range(16):
            packed = value | (value << 4)
            a_packed = torch.full(
                (m, k // 2), packed, device=f"cuda:{dev}", dtype=torch.uint8
            )
            tile = ops.tcgen05_int4_gemm_bf16(a_packed, sfa, b_packed, sfb)
            first = tile[0, 0]
            if not torch.equal(tile, first.expand_as(tile)):
                raise RuntimeError("tcgen05 INT4 codebook output is not uniform")
            values.append(first.float() / k)
        return torch.stack(values).cpu()
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
    if _is_tcgen05(dev):
        raise RuntimeError(
            "mma_probe is the SM120/SM121 register-resident OMMA probe; "
            "benchmark tcgen05_int4_gemm_bf16 on SM100/SM103/SM110"
        )
    if blocks is None:
        blocks = torch.cuda.get_device_properties(dev).multi_processor_count * 4
    if out is None:
        out = torch.empty((blocks, 256), device=f"cuda:{dev}", dtype=torch.float32)
    ops.run_mma_probe(_cubin(mode, dev), out, iterations, blocks, launches, dev)
    return out


def tcgen05_int4_gemm_bf16(
    a_packed: torch.Tensor,
    sfa_physical: torch.Tensor,
    b_packed: torch.Tensor,
    sfb_physical: torch.Tensor,
) -> torch.Tensor:
    """Run native E0M3 x E0M3 block-scaled GEMM on SM100/SM103/SM110.

    ``a_packed`` and ``b_packed`` contain two sign-magnitude INT4 values per
    byte. Scale tensors are the physical CUTLASS UE4M3 block-16 layouts.
    """
    return ops.tcgen05_int4_gemm_bf16(
        a_packed, sfa_physical, b_packed, sfb_physical
    )


__all__ = ["codebook_probe", "mma_probe", "tcgen05_int4_gemm_bf16"]
