import torch

from flashrt_smallm_gemm import nvfp4_w4a4_decode_matvec_bf16out


def swizzled_scale_bytes(rows: int, cols: int) -> int:
    n_blocks = cols // 16
    return ((rows + 127) // 128) * ((n_blocks + 3) // 4) * 512


def run_decode_matvec(K: int = 4096, N: int = 1024):
    a_packed = torch.full((K // 2,), 0x11, device="cuda", dtype=torch.uint8)
    b_packed = torch.full((N, K // 2), 0x11, device="cuda", dtype=torch.uint8)
    sfa = torch.full(
        (swizzled_scale_bytes(1, K),),
        0x38,
        device="cuda",
        dtype=torch.uint8,
    )
    sfb = torch.full(
        (swizzled_scale_bytes(N, K),),
        0x38,
        device="cuda",
        dtype=torch.uint8,
    )
    return nvfp4_w4a4_decode_matvec_bf16out(a_packed, b_packed, sfa, sfb, alpha=0.5)


if __name__ == "__main__":
    out = run_decode_matvec()
    torch.cuda.synchronize()
    print("out", tuple(out.shape), out.dtype)
