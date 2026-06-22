"""Minimal Qwen3-8B serving example — FlashRT via the official transformers APIs.

Everything loads through official entry points — no model surgery:
  * FP8 / NVFP4 GEMMs : the transformers quantization API
                        (`quantization_config=FlashRTConfig(...)`, like torchao/bnb)
  * RMSNorm           : the official `kernelize()` mechanism (the gpt-oss path)
  * attention         : the `attn_implementation` backend — FlashRT SageAttention2
                        (`flashrt/sageattention2-blackwell`) for long-context prefill,
                        SDPA for decode

    python serving_example.py [bf16|fp8|nvfp4]   # default nvfp4

Measured (RTX 5090, 1024-token prefill + greedy decode):
    bf16   prefill 100 ms   decode 56 tok/s
    fp8    prefill  62 ms   decode 64 tok/s   (static-scale, quality-preserving)
    nvfp4  prefill  36 ms   decode 69 tok/s   <- prefill 2.8x, decode 1.22x
"""

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashrt_quantizer import FlashRTConfig, calibrate_fp8   # registers the quantizer
from flashrt_kernelize import kernelize_model
import flashrt_sage_attn  # noqa: F401  registers attn_implementation="sage2_blackwell"

MODEL_ID = "Qwen/Qwen3-8B"


def load(mode: str):
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if mode != "bf16":
        kw["quantization_config"] = FlashRTConfig(mode=mode)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, attn_implementation="sage2_blackwell", **kw)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    model.eval()
    if mode != "bf16":
        kernelize_model(model)
    return model


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "nvfp4"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load(mode)
    ids = tok("Explain why optimized GPU kernels matter for serving LLMs.",
              return_tensors="pt").input_ids.to("cuda")

    # prefill (1024 tokens)
    pf = ids.repeat(1, (1024 // ids.shape[1]) + 1)[:, :1024]
    if mode == "fp8":  # static FP8 act-scale -> compile-friendly + fast decode
        with torch.inference_mode():
            calibrate_fp8(model, lambda: model(pf))
    with torch.inference_mode():
        for _ in range(3):
            model(pf)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(6):
            model(pf)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t) / 6 * 1e3

        # decode throughput
        for _ in range(2):
            model.generate(ids, max_new_tokens=32, do_sample=False)
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = model.generate(ids, max_new_tokens=128, do_sample=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
    n_new = out.shape[1] - ids.shape[1]
    print(f"[{mode}] prefill(1024) {prefill_ms:.0f} ms | decode {n_new / dt:.1f} tok/s")


if __name__ == "__main__":
    main()
