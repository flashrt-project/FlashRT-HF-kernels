import torch

from minimaxai_msa_sm121 import flash_decode_with_gqa_share_sparse


def main() -> None:
    device = "cuda"
    dtype = torch.bfloat16
    batch, hq, hkv, d = 1, 64, 4, 128
    ctx, block, topk = 2048, 128, 16
    max_slots = ctx

    q = torch.randn(batch, hq, d, device=device, dtype=dtype)
    k_cache = torch.randn(max_slots, hkv, d, device=device, dtype=dtype)
    v_cache = torch.randn(max_slots, hkv, d, device=device, dtype=dtype)
    req_to_token = torch.arange(ctx, device=device, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([ctx], device=device, dtype=torch.int32)
    slot_ids = torch.zeros(batch, device=device, dtype=torch.int64)
    topk_idx = torch.arange(topk, device=device, dtype=torch.int32).view(1, 1, topk)
    topk_idx = topk_idx.expand(hkv, batch, topk).contiguous()

    out = flash_decode_with_gqa_share_sparse(
        q,
        None,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens,
        slot_ids,
        block,
        topk_idx,
    )
    print(out.shape, out.dtype)


if __name__ == "__main__":
    main()
