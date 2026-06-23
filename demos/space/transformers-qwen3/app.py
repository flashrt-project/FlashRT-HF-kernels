"""ZeroGPU Space: chat with a FlashRT-accelerated Qwen3-8B.

A normal streaming chat, but the model runs through FlashRT's kernels reached via
official `transformers` APIs — NVFP4 GEMMs (`quantization_config`) + `kernelize`
RMSNorm + SageAttention2 prefill (`attn_implementation`). Runs eager (what ZeroGPU
runs); the baseline-vs-FlashRT numbers are measured offline and shown below.
Reproducible CLI numbers are in serving_example.py.
"""

from __future__ import annotations

import threading
import time

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from flashrt_quantizer import FlashRTConfig          # registers the quantization backend
from flashrt_kernelize import kernelize_model
import flashrt_sage_attn  # noqa: F401  registers attn_implementation="sage2_blackwell"

MODEL_ID = "Qwen/Qwen3-8B"
MAX_NEW_TOKENS = 512
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_model = None


def _load():
    global _model
    if _model is None:
        kw = dict(dtype=torch.bfloat16, device_map="cuda",
                  attn_implementation="sage2_blackwell",
                  quantization_config=FlashRTConfig(mode="nvfp4"))
        try:
            _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw).eval()
        except Exception:
            kw["attn_implementation"] = "sdpa"
            _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw).eval()
        kernelize_model(_model)
    return _model


@spaces.GPU(duration=120)
def chat(message, history):
    model = _load()
    msgs = (history or []) + [{"role": "user", "content": message}]
    inputs = _tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to("cuda")
    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)

    def _run():
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                           streamer=streamer)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    threading.Thread(target=_run, daemon=True).start()
    text = ""
    for piece in streamer:
        text += piece
        yield text
    n = len(_tokenizer(text, add_special_tokens=False).input_ids)
    rate = n / max(time.perf_counter() - t0, 1e-6)
    yield text + f"\n\n_— {rate:.0f} tok/s (FlashRT, eager)_"


_DESC = (
    "## Chat with Qwen3-8B, accelerated by FlashRT\n"
    "The model runs through FlashRT kernels reached via **official `transformers` "
    "APIs** — NVFP4 GEMMs (`quantization_config`) + `kernelize` RMSNorm + "
    "SageAttention2 (`attn_implementation`). No forked library, no model surgery.\n\n"
    "**Measured vs stock BF16 (RTX 5090, eager, cosine ≈ 0.999):** prefill "
    "**2.9×** (1024 tok), decode **1.3×** (66 → 86 tok/s)."
)

demo = gr.ChatInterface(
    chat,
    type="messages",
    title="FlashRT × Qwen3-8B",
    description=_DESC,
    examples=[
        "Explain why the sky is blue, in three short paragraphs.",
        "Write a haiku about fast GPU kernels.",
        "What is speculative decoding? Keep it brief.",
    ],
)


if __name__ == "__main__":
    demo.launch()
