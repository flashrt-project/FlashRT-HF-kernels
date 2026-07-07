from kernels import get_kernel
import torch


def main() -> None:
    ops = get_kernel("flashrt/speculative-draft-primitives", version=1, trust_remote_code=True)
    logits = torch.randn((16, 32000), device="cuda", dtype=torch.bfloat16)
    argmax = ops.argmax_bf16(logits)
    drafts = argmax[:15].clone()
    _, accept_n = ops.accept_greedy_bf16(logits, drafts, spec_k=15)
    torch.cuda.synchronize()
    print({"argmax_shape": tuple(argmax.shape), "accept_n": int(accept_n.cpu()[0])})


if __name__ == "__main__":
    main()
