#!/usr/bin/env python3
"""Minimal Hub usage for flashrt/turboquant-kv."""

from __future__ import annotations

import torch
from kernels import get_kernel


def main() -> None:
    tq = get_kernel("flashrt/turboquant-kv")
    m = 1024
    k_idx = torch.randint(0, 256, (m, 128), device="cuda", dtype=torch.uint8)
    k_qjl = torch.randint(0, 256, (m, 32), device="cuda", dtype=torch.uint8)
    v_idx = torch.randint(0, 256, (m, 128), device="cuda", dtype=torch.uint8)
    cb_k = torch.randn((16,), device="cuda", dtype=torch.float32)
    cb_v = torch.randn((16,), device="cuda", dtype=torch.float32)
    y_k, qjl_bf, y_v = tq.unpack_packed_bf16(k_idx, k_qjl, v_idx, cb_k, cb_v, 3, 4)
    torch.cuda.synchronize()
    print(y_k.shape, qjl_bf.shape, y_v.shape)


if __name__ == "__main__":
    main()
