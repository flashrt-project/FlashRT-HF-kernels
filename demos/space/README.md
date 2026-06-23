# Space Demos

Deployable Hugging Face Space demos that show FlashRT Hub kernels accelerating
standard `transformers` and `diffusers` models. Each subfolder is a
self-contained Gradio Space (`app.py`, `requirements.txt`, `README.md`) that
loads kernels through `kernels.get_kernel` and runs on a Blackwell (SM120) GPU,
including the ZeroGPU RTX PRO 6000 pool.

Each demo accelerates specific hot blocks of a model and reports both the patched
path and the stock baseline on the same model and inputs. These are block-level
acceleration demos, not full-runtime replacements.

## Demos

| Folder | Library | Model | Integration |
|---|---|---|---|
| `transformers-qwen3` | transformers | Qwen3-8B | official APIs: quantization backend (FP8/NVFP4) + `kernelize` (RMSNorm) + `attn_implementation` (SageAttention2) |
| `diffusers-wan2.2` | diffusers | Wan2.2 TI2V-5B | official APIs: diffusers quantization backend (FP8/NVFP4) + `set_attn_processor` (SageAttention2) + FP8 VAE |

Both demos go **entirely through each library's official APIs** — FlashRT
registers as a quantization backend (`quantization_config`, like torchao /
bitsandbytes) so the FP8/NVFP4 GEMMs load via `from_pretrained`. Numbers below are
RTX 5090 (SM120). The Wan Space AoT-compiles its transformer at startup (aokit) —
ZeroGPU has no `torch.compile` (JIT would burn the quota); the Qwen chat is eager.

- **transformers (Qwen3-8B):** quantization backend (NVFP4 GEMMs) + `kernelize()`
  RMSNorm (the gpt-oss path) + `attn_implementation` FlashRT SageAttention2
  (`flashrt/sageattention2-blackwell`). Prefill 4096: bf16 379 → **114 ms (3.32x)**
  with `torch.compile`; decode 66.4 → **86.7 tok/s (1.31x)**.
- **diffusers (Wan2.2):** the DiT quantized via the diffusers quantization API +
  FlashRT SageAttention2 (`set_attn_processor`) + FP8 VAE convs. Per denoise step
  (480×480×33): bf16 161 → **42.5 ms/step (3.79x)** with `torch.compile`, cosine ~0.999.

## Conventions

- Demos load kernels from the Hub with `get_kernel`, never inside the hot loop.
- Wrappers preserve the replaced module's signature so the model runs unchanged.
- Baselines use the unmodified model on the same prompts; no random-input numbers
  are presented as model speed claims.
