# Diffusion Step Ops

FlashRT CUDA kernels for small but frequent diffusion/runtime step operations.

These kernels target static-buffer and CUDA Graph friendly pipelines where
PyTorch eager glue can become visible in the hot path.

## Available Functions

- `add_bf16(a, b)`: BF16 elementwise add.
- `euler_step_bf16(latent, velocity, dt)`: BF16 Euler update.
- `cfg_combine_into_residual_bf16(residual, v_cond, v_uncond, beta)`: in-place classifier-free guidance residual combine.
- `cfg_combine_into_residual_fp16(residual, v_cond, v_uncond, beta)`: FP16 variant.
- `teacher_force_first_frame_bf16(video_latent, cond_latent)`: copy conditioning frame into `video_latent[:, :, 0]`.
- `motus_decode_postprocess_bf16_to_fp32(decoded)`: drop first frame and map `[-1, 1]` to `[0, 1]`.
- `cast_bf16_to_fp32(src)`: BF16 to FP32 cast.
- `pack_tail_bf16(tail, flat_dim)`: zero-pad a BF16 tail into a flat vector.
- `add_bias_zero_tail_bf16(input, bias, valid_cols)`: add bias and zero padded columns.
- `extract_tail_f32_to_bf16(flat, tail_numel)`: extract and cast an action tail.
- `add_bias_pair_bf16(input, bias_a, bias_b)`: preserve two BF16 add-rounding stages.
- `unipc_step_f32_bf16(...)`: fused UniPC corrector/predictor update.

## Usage

```python
from kernels import get_kernel

ops = get_kernel("flashrt/diffusion-step-ops")

latent = ops.euler_step_bf16(latent, velocity, dt=-0.125)
ops.cfg_combine_into_residual_bf16(residual, v_cond, v_uncond, beta=4.5)
ops.teacher_force_first_frame_bf16(video_latent, cond_latent)

next_sample, current_m, current_last = ops.unipc_step_f32_bf16(
    sample, velocity, prev_m1, prev_m2, prev_last,
    sigma, corrector_order, predictor_order,
    corrector_coefficients, predictor_coefficients,
)
```

All APIs require CUDA contiguous tensors. Unsupported shapes fail at the
wrapper boundary.
