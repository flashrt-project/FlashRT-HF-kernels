import argparse

import torch

import int4_blackwell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    blocks = props.multi_processor_count * 4
    warps = blocks * 8
    flops = warps * 4 * args.iterations * 2 * 16 * 8 * 64
    for mode in ("e2m1", "a", "b", "ab"):
        int4_blackwell.mma_probe(mode, iterations=args.iterations, blocks=blocks)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        best_ms = float("inf")
        for _ in range(args.repeats):
            start.record()
            int4_blackwell.mma_probe(mode, iterations=args.iterations, blocks=blocks)
            end.record()
            end.synchronize()
            best_ms = min(best_ms, start.elapsed_time(end))
        tflops = flops / (best_ms * 1e-3) / 1e12
        print(f"{mode:5s} {best_ms * 1e3:9.3f} us {tflops:8.1f} TFLOPS")


if __name__ == "__main__":
    main()
