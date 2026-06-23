"""ZeroGPU Space: a BF16-vs-FlashRT generation race for Qwen3-8B.

Same prompt, same greedy decode, run back-to-back — a stock BF16 model (SDPA) and
the FlashRT-accelerated one (NVFP4 GEMMs via the quantization API + kernelize
RMSNorm + SageAttention2 prefill, all through official transformers APIs). Each
model gets the GPU to itself (so the tokens/sec is a fair number) and streams
live, so the kernel speedup is something you watch rather than read off a table.

Everything runs eager (what ZeroGPU runs). Reproducible CLI numbers are in
serving_example.py.
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
MAX_NEW_TOKENS = 256

_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_models: dict[str, object] = {}


def _ids(prompt: str):
    return _tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to("cuda")


def _load(mode: str):
    """mode: 'bf16' (stock baseline, SDPA) | 'nvfp4' (FlashRT full stack)."""
    m = _models.get(mode)
    if m is not None:
        return m
    attn = "sdpa" if mode == "bf16" else "sage2_blackwell"
    kw = dict(dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn)
    if mode != "bf16":
        kw["quantization_config"] = FlashRTConfig(mode=mode)
    try:
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw).eval()
    except Exception:  # fall back if the attention backend is unavailable
        kw["attn_implementation"] = "sdpa"
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw).eval()
    if mode != "bf16":
        kernelize_model(m)
    with torch.no_grad():  # warm once so the first race reports steady-state tok/s
        m.generate(**_ids("hello"), max_new_tokens=8, do_sample=False)
    _models[mode] = m
    return m


def _stream(model, prompt: str, store: dict):
    """Greedy-generate in a thread; yield as text streams into `store`."""
    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = _ids(prompt)

    def _run():
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                           streamer=streamer)

    torch.cuda.synchronize()
    store["t0"] = time.perf_counter()
    threading.Thread(target=_run, daemon=True).start()
    for piece in streamer:
        store["text"] += piece
        store["dt"] = time.perf_counter() - store["t0"]
        yield
    store["done"] = True
    yield


def _toks(text: str) -> int:
    return len(_tokenizer(text, add_special_tokens=False).input_ids)


def _panel(store: dict, idle: str = "") -> str:
    if store.get("t0") is None:
        return idle
    n, dt = _toks(store["text"]), max(store.get("dt", 0.0), 1e-6)
    rate = n / dt
    tag = "✅ done" if store.get("done") else "⏳ generating…"
    return f"**{rate:5.1f} tok/s**  ·  {n} tokens  ·  {tag}\n\n{store['text']}"


def _rate(store: dict) -> float:
    return _toks(store["text"]) / max(store.get("dt", 1e-6), 1e-6)


@spaces.GPU(duration=120)
def race(prompt: str):
    if not prompt.strip():
        yield "Enter a prompt to start the race.", "", ""
        return
    bf16, frt = _load("bf16"), _load("nvfp4")
    sb = {"text": "", "dt": 0.0, "done": False, "t0": None}
    sf = {"text": "", "dt": 0.0, "done": False, "t0": None}
    idle = "_waiting…_"

    # Round 1 — stock BF16 (GPU to itself)
    for _ in _stream(bf16, prompt, sb):
        yield "### Round 1 — BF16 baseline…", _panel(sb), _panel(sf, idle)
    # Round 2 — FlashRT (GPU to itself)
    for _ in _stream(frt, prompt, sf):
        yield "### Round 2 — FlashRT…", _panel(sb), _panel(sf)

    rb, rf = _rate(sb), _rate(sf)
    speedup = rf / rb if rb else 0.0
    yield (f"### FlashRT **{rf:.1f} tok/s** vs BF16 **{rb:.1f} tok/s** — "
           f"**{speedup:.2f}× faster**, same greedy output."), _panel(sb), _panel(sf)


with gr.Blocks(title="FlashRT vs BF16 — Qwen3-8B race") as demo:
    gr.Markdown(
        "# FlashRT vs BF16 — Qwen3-8B generation race\n"
        "Same prompt, same greedy decode, run back-to-back (each model gets the GPU "
        "to itself, for a fair tokens/sec). **Left:** stock BF16. **Right:** FlashRT "
        "through the official `transformers` APIs — NVFP4 GEMMs (`quantization_config`) "
        "+ `kernelize` RMSNorm + SageAttention2 (`attn_implementation`). Runs eager."
    )
    prompt = gr.Textbox(
        label="Prompt",
        value="Explain why the sky is blue, in three short paragraphs.",
    )
    go = gr.Button("Race", variant="primary")
    banner = gr.Markdown()
    with gr.Row():
        left = gr.Markdown(label="BF16")
        right = gr.Markdown(label="FlashRT (NVFP4 + kernelize + SageAttention2)")
    go.click(race, [prompt], [banner, left, right])


if __name__ == "__main__":
    demo.launch()
