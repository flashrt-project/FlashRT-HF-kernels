"""ZeroGPU Space: FlashRT FP8 / NVFP4 transformer blocks for a transformers LM.

Loads a standard ``transformers`` model and swaps the SwiGLU MLP and the
attention Q/K/V/O projections for FlashRT FP8 Hub kernels, with static scales
calibrated on real prompt activations. It reports prefill (compute-bound)
latency and a perplexity guardrail for the BF16 baseline and the FlashRT path.

An optional NVFP4 (4-bit) mode runs the same projections in 4-bit for more
prefill speed; an LLM has no residual-denoise self-correction, so 4-bit costs
some perplexity (reported), and FP8 stays the quality-preserving default.

Scope note: this is an integration + quality demo. FP8/NVFP4 weights help most in
the compute-bound / large-batch regime (prefill, throughput). Peak single-stream
decode is bandwidth- and launch-bound and is the job of a fully-fused runtime
with CUDA-graph capture, not a per-module patch on top of ``generate``.
"""

from __future__ import annotations

import time

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashrt_kernelize import kernelize_model
from flashrt_quantizer import FlashRTConfig  # registers the FlashRT transformers quantizer
import flashrt_sage_attn  # noqa: F401  registers attn_implementation="sage2_blackwell"

MODEL_ID = "Qwen/Qwen3-8B"
EVAL_TEXT = (
    "Optimized compute kernels reduce latency and memory traffic without changing "
    "the model's outputs in any way that matters for quality."
)
PREFILL_TOKENS = 1024

_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_state: dict[str, object] = {}


def _load(mode: str):
    """mode: 'baseline' | 'fp8' | 'nvfp4'.

    Everything goes through official transformers APIs: the FlashRT FP8/NVFP4
    GEMMs load via ``from_pretrained(quantization_config=...)`` (the same entry
    point as torchao / bitsandbytes), and RMSNorm is swapped via the official
    ``kernelize`` mechanism.
    """

    model = _state.get(mode)
    if model is not None:
        return model
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if mode != "baseline":
        kw["quantization_config"] = FlashRTConfig(mode=mode)
    try:  # FlashRT SageAttention2 prefill backend; fall back to the default
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, attn_implementation="sage2_blackwell", **kw)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    model.eval()
    if mode != "baseline":
        kernelize_model(model)  # RMSNorm -> FlashRT via the official kernelize path
    _state[mode] = model
    return model


@torch.inference_mode()
def _prefill_ms(model, iters=5):
    ids = _tokenizer(EVAL_TEXT, return_tensors="pt").input_ids.to("cuda")
    ids = ids.repeat(1, (PREFILL_TOKENS // ids.shape[1]) + 1)[:, :PREFILL_TOKENS]
    for _ in range(2):
        model(ids)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        model(ids)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


@torch.inference_mode()
def _perplexity(model):
    ids = _tokenizer(EVAL_TEXT, return_tensors="pt").input_ids.to("cuda")
    return float(torch.exp(model(ids, labels=ids).loss))


@spaces.GPU(duration=240)
def compare(precision: str):
    mode = "nvfp4" if precision.startswith("NVFP4") else "fp8"
    label = "FlashRT NVFP4 (4-bit)" if mode == "nvfp4" else "FlashRT FP8"
    base = _load("baseline")
    base_ms, base_ppl = _prefill_ms(base), _perplexity(base)
    fr = _load(mode)
    fr_ms, fr_ppl = _prefill_ms(fr), _perplexity(fr)
    speedup = base_ms / fr_ms if fr_ms > 0 else 0.0
    return (
        f"| path | prefill {PREFILL_TOKENS}tok (ms) | perplexity |\n"
        f"|---|---:|---:|\n"
        f"| BF16 baseline | {base_ms:.1f} | {base_ppl:.3f} |\n"
        f"| {label} | {fr_ms:.1f} | {fr_ppl:.3f} |\n"
        f"\n**Prefill speedup: {speedup:.2f}x.** Perplexity shown above on the "
        f"same passage (FP8 is the quality-preserving default; NVFP4 is FlashRT's "
        f"production 4-bit path)."
    )


with gr.Blocks(title="FlashRT FP8 / NVFP4 transformer blocks") as demo:
    gr.Markdown(
        "# FlashRT FP8 / NVFP4 on transformers — through the official APIs\n"
        "FlashRT plugs in as a **transformers quantization backend**: the FP8 / NVFP4 "
        "GEMMs load via `from_pretrained(quantization_config=FlashRTConfig(...))` (the "
        "same entry point as torchao / bitsandbytes), and RMSNorm is swapped via the "
        "official `kernelize` mechanism — no hand-patching of the model. FP8 preserves "
        "quality; NVFP4 is FlashRT's production 4-bit path for more prefill speed. "
        "Peak decode is the fully-fused runtime's job, not a patch on `generate`."
    )
    precision = gr.Radio(
        ["FP8 (quality preserved)", "NVFP4 (4-bit, faster)"],
        value="FP8 (quality preserved)",
        label="FlashRT precision",
    )
    run = gr.Button("Compare BF16 vs FlashRT", variant="primary")
    summary = gr.Markdown()
    run.click(compare, [precision], [summary])


if __name__ == "__main__":
    demo.launch()
