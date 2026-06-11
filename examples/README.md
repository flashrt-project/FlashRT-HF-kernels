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

`lerobot_pi05_fp8_mlp.py` hot-plugs the fused `fp8_geglu_mlp_bf16` kernel into
the Gemma GeGLU MLP blocks of a real [LeRobot](https://github.com/huggingface/lerobot)
pi05 policy (action expert + prefix language model). The model keeps its own
`torch.compile`, so it simply recompiles around the FP8 modules.

```bash
pip install "lerobot[pi,dataset]"
huggingface-cli login        # the PaliGemma tokenizer is gated
python examples/lerobot_pi05_fp8_mlp.py
```

Measured on RTX 5090 (LIBERO finetune, 10 denoise steps): about `1.17x`
end-to-end with action cosine similarity `~0.998` versus the BF16 baseline.

Two lessons it encodes:

- **Calibrate FP8 scales on a real observation.** pi05's prefix mixes image and
  text tokens with very wide activation magnitudes; random-input calibration
  produces broken per-tensor scales. The example pulls a real frame from the
  dataset the checkpoint targets and runs the policy's own preprocessor.
- **Calibrate in eager mode.** A compiled graph does not fire Python forward
  hooks, so the example temporarily drops the compiled methods while capturing
  activation statistics.

It is *partial* on purpose: only the MLP blocks are swapped. The attention
projections are small per-token GEMMs, and a naive per-projection FP8 swap
re-quantizes activations four times per layer and loses to cuBLAS at those
sizes -- that path needs quantize fusion, not a drop-in.

## Notes

- `flashrt-fp8-ffn` expects FP8 E4M3 input and weights plus CUDA float32 scalar
  scales.
- `flashrt-gemm-epilogues` includes helper kernels for static BF16-to-FP8
  activation quantization.
- Model-level speedups require replacing a meaningful continuous block, not
  sprinkling many tiny Python-level wrappers through a model.
