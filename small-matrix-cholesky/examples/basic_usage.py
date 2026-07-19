#!/usr/bin/env python3
"""Minimal Hugging Face Kernel Hub usage example."""

import torch
from kernels import get_kernel


ops = get_kernel(
    "flashrt/small-matrix-cholesky",
    version=1,
    trust_remote_code=True,
)

n = 64
x = torch.randn(32, n, n, device="cuda", dtype=torch.float32)
a = x @ x.transpose(-1, -2) + 0.5 * torch.eye(n, device="cuda")
l = ops.cholesky_small_fp32(a.contiguous())

torch.testing.assert_close(
    l @ l.transpose(-1, -2),
    a,
    rtol=2e-4,
    atol=2e-4,
)
print(l.shape)
