# FlashRT Kernel Usage Examples

These examples show how an application should call the published FlashRT
Kernel Hub packages. They are intentionally separate from benchmark scripts.

For the package map and model-integration rules, read `docs/usage.md` first.

Run these examples in a Python environment with CUDA PyTorch and the Hugging
Face `kernels` package installed:

```bash
pip install kernels
```

## 1. Minimal Kernel Call

Run one Hub-loaded FP8 GELU MLP kernel:

```bash
python examples/minimal_fp8_ffn.py
```

The important part is:

```python
from kernels import get_kernel

ops = get_kernel("flashrt/flashrt-fp8-ffn", version=1, trust_remote_code=True)
y = ops.fp8_gelu_mlp_bf16(
    x_fp8,
    up_w_fp8,
    up_bias,
    down_w_fp8,
    down_bias,
    x_scale,
    up_w_scale,
    hidden_scale,
    down_w_scale,
)
```

## 2. Replacing A Model FFN

`replace_torch_ffn.py` shows the integration pattern for a normal PyTorch
module shaped like:

```text
Linear -> GELU(tanh) -> Linear
```

Run:

```bash
python examples/replace_torch_ffn.py
```

This example is a clean model-integration skeleton. It uses static scales and
quantized weights, but it still starts from a BF16 activation tensor because it
is demonstrating drop-in replacement. For the fastest production path, connect
FlashRT FP8 producers directly to FlashRT FP8 consumers so the model does not
re-quantize BF16 activations at every layer.

## 3. LeRobot pi05 (partial integration)

`lerobot_pi05_fp8_mlp.py` replaces the Gemma GeGLU MLP blocks (action expert and
prefix language model) of a [LeRobot](https://github.com/huggingface/lerobot)
pi05 policy with the fused `fp8_geglu_mlp_bf16` kernel. The policy keeps its own
`torch.compile`, which recompiles around the FP8 modules.

```bash
pip install "lerobot[pi,dataset]"
huggingface-cli login        # the PaliGemma tokenizer is gated
python examples/lerobot_pi05_fp8_mlp.py
```

On RTX 5090 (LIBERO finetune, 10 denoise steps) this runs at about `1.17x`
end-to-end with action cosine similarity `~0.998` versus the BF16 baseline.

The example calibrates static FP8 scales on a real observation from the target
dataset, run through the policy's own preprocessor, and captures activation
statistics in eager mode. Only the MLP blocks are replaced; attention
projections stay in BF16.

## Notes

- `flashrt-fp8-ffn` expects FP8 E4M3 input and weights plus CUDA float32 scalar
  scales.
- `flashrt-gemm-epilogues` includes helper kernels for static BF16-to-FP8
  activation quantization.
- Model-level speedups require replacing a meaningful continuous block, not
  sprinkling many tiny Python-level wrappers through a model.
