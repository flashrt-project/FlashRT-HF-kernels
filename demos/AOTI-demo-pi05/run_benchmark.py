#!/usr/bin/env python3
"""Benchmark the pi05-fast optimization ladder against the BF16 eager baseline.

Reports end-to-end latency (full pi05 hot path: SigLIP vision + projector +
PaliGemma prefix + 10-step denoise + action head) and action cosine similarity
versus the eager baseline, for each rung of the stack.

    python run_benchmark.py                 # full ladder
    python run_benchmark.py --compile-mode export-aoti
    python run_benchmark.py --no-fp8        # compile-only rung

Requirements:
    pip install "lerobot[pi,dataset]"
    huggingface-cli login        # the PaliGemma tokenizer is gated
"""

from __future__ import annotations

import argparse
import time

import torch

import pi05_fast

CHECKPOINT = "lerobot/pi05_libero_finetuned_v044"
CALIB_DATASET = "physical-intelligence/libero"
SEED = 0


def build_policy():
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    return PI05Policy.from_pretrained(CHECKPOINT).eval().cuda()


def build_observation(policy) -> dict:
    """One real observation from the target dataset, via the policy's preprocessor."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    dataset = LeRobotDataset(CALIB_DATASET)
    frame = dataset[0]
    preprocessor, _ = make_pi05_pre_post_processors(policy.config, dataset.meta.stats)
    observation = {k: v for k, v in frame.items() if k.startswith("observation.")}
    observation["task"] = frame["task"]
    batch = preprocessor(observation)
    device = next(policy.parameters()).device
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_once(policy, batch) -> torch.Tensor:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with torch.inference_mode():
        return policy.predict_action_chunk({k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()})


def median_ms(fn, warmup: int = 4, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.time()
        fn()
        torch.cuda.synchronize()
        samples.append((time.time() - start) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def measure(name: str, cfg: dict, baseline_action, observation_builder) -> tuple[str, float, float]:
    import gc

    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    policy = build_policy()
    batch = observation_builder(policy)
    if cfg is not None:
        pi05_fast.optimize(policy, batch, **cfg)
    action = run_once(policy, batch).float().clone()
    torch.cuda.synchronize()
    latency = median_ms(lambda: run_once(policy, batch))
    cos = 1.0 if baseline_action is None else cosine(action, baseline_action)
    del policy
    torch.cuda.empty_cache()
    return name, latency, cos, action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-mode", choices=["disabled", "compile", "export-aoti"], default="compile")
    parser.add_argument("--no-fp8", action="store_true")
    parser.add_argument("--vision-fp8", action="store_true", help="also swap the SigLIP vision MLP to FP8")
    parser.add_argument("--no-inductor-flags", action="store_true")
    parser.add_argument("--safety", type=float, default=1.0)
    parser.add_argument("--single", action="store_true", help="run only the configured rung, not the full ladder")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    if args.single:
        rungs = [(
            f"fp8={'off' if args.no_fp8 else 'on'} vision={'on' if args.vision_fp8 else 'off'} compile={args.compile_mode}",
            dict(fp8=not args.no_fp8, vision_fp8=args.vision_fp8, inductor_flags=not args.no_inductor_flags,
                 compile_mode=args.compile_mode, safety=args.safety),
        )]
    else:
        rungs = [
            ("bf16 eager (baseline)", dict(fp8=False, inductor_flags=False, compile_mode="disabled")),
            ("+ torch.compile", dict(fp8=False, inductor_flags=True, compile_mode="compile")),
            ("+ FlashRT fp8 MLP", dict(fp8=True, inductor_flags=True, compile_mode="compile", safety=args.safety)),
            ("+ vision fp8 (full stack)", dict(fp8=True, vision_fp8=True, inductor_flags=True, compile_mode="compile", safety=args.safety)),
        ]

    results = []
    baseline_action = None
    for name, cfg in rungs:
        name, latency, cos, action = measure(name, cfg, baseline_action, build_observation)
        if baseline_action is None:
            baseline_action = action
        results.append((name, latency, cos))
        print(f"  {name:34s} {latency:7.2f} ms   cos={cos:.5f}", flush=True)

    base = results[0][1]
    print("\n=== pi05-fast ladder (RTX, full hot path, vs bf16 eager) ===")
    for name, latency, cos in results:
        print(f"  {name:34s} {latency:7.2f} ms   {base / latency:5.2f}x   cos={cos:.5f}")


if __name__ == "__main__":
    main()
