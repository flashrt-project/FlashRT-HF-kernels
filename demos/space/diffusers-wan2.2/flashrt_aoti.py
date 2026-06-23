"""Ahead-of-time (AoT) compilation for the Wan transformer on ZeroGPU.

ZeroGPU does not run `torch.compile` — a JIT recompile on every GPU allocation
would burn the ZeroGPU quota (the failure mode HF flags). Instead the transformer
is compiled *ahead of time* once and replayed. This uses Hugging Face's
`aokit` (https://github.com/huggingface/aokit), falling back to the built-in
`spaces` AoTI helpers, then to eager.

The FlashRT Hub kernels (fp4-gemm, sageattention2-blackwell, world-model-conv) are
`torch.library` ops with fake/meta impls, so `torch.export` traces through them
and the AoT-compiled graph calls them via the dispatcher at runtime (verified:
the full Wan transformer exports to ~9k nodes with the FlashRT ops intact).

`aot_compile_transformer` is best-effort: if no AoT toolkit is present (off
ZeroGPU) or export/compile fails, it returns False and the caller keeps the eager
pipeline (still NVFP4-accelerated) — the demo never breaks.
"""

from __future__ import annotations

import torch


def _capture_call(pipe, *, example_prompt, height, width, num_frames):
    """Capture the transformer's real (args, kwargs) from one 1-step pipe call."""
    cap = {}
    tr = pipe.transformer
    orig = tr.forward

    def grab(*a, **k):
        cap.setdefault("args", a)
        cap.setdefault("kwargs", k)
        return orig(*a, **k)

    tr.forward = grab
    try:
        with torch.no_grad():
            pipe(prompt=example_prompt, height=height, width=width,
                 num_frames=num_frames, num_inference_steps=1)
    finally:
        tr.forward = orig
    return cap.get("args", ()), cap.get("kwargs", {})


def aot_compile_transformer(pipe, *, example_prompt, height, width, num_frames,
                            package_dir="flashrt_wan_aot") -> bool:
    """AoT-compile and hot-swap pipe.transformer in place. Returns True on success."""
    try:
        args, kwargs = _capture_call(
            pipe, example_prompt=example_prompt, height=height, width=width,
            num_frames=num_frames)
        exported = torch.export.export(pipe.transformer, args=args, kwargs=kwargs)
    except Exception as e:
        print(f"[flashrt_aoti] export failed, staying eager: {type(e).__name__}: {e}")
        return False

    # Preferred: Hugging Face aokit (persists a reusable package).
    try:
        import aokit
        aokit.compile_and_save(package_dir=package_dir, exported_program=exported)
        aokit.load_from_package_dir(pipe.transformer, package_dir)
        return True
    except Exception as e:
        print(f"[flashrt_aoti] aokit unavailable ({type(e).__name__}); trying spaces AoTI")

    # Fallback: the built-in `spaces` AoTI helpers.
    try:
        import spaces
        compiled = spaces.aoti_compile(exported)
        spaces.aoti_apply(compiled, pipe.transformer)
        return True
    except Exception as e:
        print(f"[flashrt_aoti] AoTI unavailable, staying eager: {type(e).__name__}: {e}")
        return False
