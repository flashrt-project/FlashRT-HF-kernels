"""Minimal example for flashrt/MiniMaxAI-msa-sm121.

By default this loads the uploaded Hub artifact. Use --source-tree for local
source validation before the native extension is built; source-tree mode can
exercise the Triton attention path but will not expose native ops.
"""

from __future__ import annotations

import argparse
import importlib

import torch
from kernels import get_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="flashrt/MiniMaxAI-msa-sm121")
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--source-tree", action="store_true")
    args = parser.parse_args()

    if args.source_tree:
        msa = importlib.import_module("minimaxai_msa_sm121")
    else:
        msa = get_kernel(args.repo, version=args.version, trust_remote_code=True)

    device = "cuda"
    dtype = torch.bfloat16
    batch, hq, hkv, d = 1, 64, 4, 128
    block, topk = 128, 16
    ctx = int(args.ctx)

    q = torch.randn(batch, hq, d, device=device, dtype=dtype)
    k_cache = torch.randn(ctx, hkv, d, device=device, dtype=dtype)
    v_cache = torch.randn(ctx, hkv, d, device=device, dtype=dtype)
    q_index = torch.randn(batch, 1, d, device=device, dtype=dtype)
    k_index = torch.randn(ctx, 1, d, device=device, dtype=dtype)
    req_to_token = torch.arange(ctx, device=device, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([ctx], device=device, dtype=torch.int32)
    slot_ids = torch.zeros(batch, device=device, dtype=torch.int64)

    _index_value, topk_idx, _real_seq_lens = msa.flash_decode_with_topk_idx(
        q_index,
        None,
        k_index,
        None,
        req_to_token,
        seq_lens,
        max_seqlen=ctx,
        slot_ids=slot_ids,
        block_size=block,
        topk=topk,
        init_blocks=0,
        local_blocks=1,
        disable_index_value=True,
    )

    out = msa.flash_decode_with_gqa_share_sparse(
        q,
        None,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens,
        slot_ids,
        block,
        topk_idx.expand(hkv, batch, topk).contiguous(),
    )
    print({
        "out_shape": tuple(out.shape),
        "out_dtype": str(out.dtype),
        "native_ops": bool(getattr(msa, "has_native_ops", lambda: False)()),
        "topk_shape": tuple(topk_idx.shape),
    })


if __name__ == "__main__":
    main()
