from __future__ import annotations

import torch
from kernels import get_kernel


def main() -> None:
    sage3 = get_kernel(
        "flashrt/sageattention3-blackwell", version=1, trust_remote_code=True
    )
    q = torch.randn((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Production callers compute centered Q/K and delta_s while preparing the
    # model attention inputs. This compact example uses global Q centering.
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    q_mean = qh.mean(dim=-2, keepdim=True)
    q = (qh - q_mean).transpose(1, 2).contiguous()
    k = (kh - kh.mean(dim=-2, keepdim=True)).transpose(1, 2).contiguous()
    delta_s = torch.matmul(q_mean, k.transpose(1, 2).transpose(-2, -1)).float()

    workspace = sage3.allocate_workspace(q, k, v)
    sage3.prepare_qkv_fp4_nhd(q, k, v, workspace)
    out = sage3.blockscaled_fp4_attention_static(
        workspace, delta_s.contiguous(), unpadded_k=k.shape[1],
        per_block_mean=False,
    )
    torch.cuda.synchronize()
    print(out.shape, out.dtype, sage3.capabilities())


if __name__ == "__main__":
    main()
