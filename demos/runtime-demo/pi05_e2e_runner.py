#!/usr/bin/env python3
"""PI0.5 E2E benchmark orchestrator for FlashRT HF runtime work.

This script keeps three concepts separate:

1. official/OpenPI PyTorch baseline: the ecosystem-facing baseline;
2. official FlashRT pipeline sanity: proves the checkpoint/runtime is healthy;
3. HF Kernel Hub runtime hot path: measures the public Hub-kernel path.

The script writes a single JSON/Markdown report and marks unavailable baselines
as skipped instead of silently substituting FlashRT for PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PI_ROOT = ROOT.parent
OFFICIAL_FLASHRT = PI_ROOT / "official" / "FlashRT"
DEFAULT_CKPT = PI_ROOT / "checkpoints" / "pi05_libero_pytorch"
DEFAULT_HUB_PY = PI_ROOT / ".flashrt-hub-smoke-torch211" / "bin" / "python"
DEFAULT_OPENPI_ROOT = PI_ROOT / "openpi_src" / "src"
DEFAULT_CONTAINER_REPO = "/workspace/PI/FlashRT-HF-kernels"
DEFAULT_CONTAINER_OPENPI_ROOT = "/workspace/PI/openpi_src/src"
DEFAULT_CONTAINER_CKPT = "/workspace/PI/checkpoints/pi05_libero_pytorch"
DEFAULT_CONTAINER_FLASHRT_ROOT = "/workspace/PI/official/FlashRT"
DEFAULT_CONTAINER_OPENPI_PY = (
    "/workspace/PI/FlashRT-HF-kernels/internal-tests/envs/"
    "openpi-baseline/bin/python"
)


@dataclass
class StepResult:
    name: str
    status: str
    command: list[str]
    cwd: str
    returncode: int | None
    metrics: dict[str, Any]
    note: str
    stdout_tail: str = ""
    stderr_tail: str = ""


def _tail(s: str, n: int = 4000) -> str:
    return s[-n:] if len(s) > n else s


def _run(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int,
) -> StepResult:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=merged_env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return StepResult(
        name=name,
        status="pass" if proc.returncode == 0 else "fail",
        command=command,
        cwd=str(cwd),
        returncode=proc.returncode,
        metrics={},
        note="",
        stdout_tail=_tail(proc.stdout),
        stderr_tail=_tail(proc.stderr),
    )


def _skip(name: str, note: str, *, cwd: Path, command: list[str] | None = None) -> StepResult:
    return StepResult(
        name=name,
        status="skip",
        command=command or [],
        cwd=str(cwd),
        returncode=None,
        metrics={},
        note=note,
    )


def _flashrt_env() -> dict[str, str]:
    env = {
        "PYTHONPATH": ".",
        # The local conda libstdc++ is older than the freshly built extension.
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
    }
    return env


def run_openpi_probe(args: argparse.Namespace) -> StepResult:
    if not Path(args.openpi_root).exists():
        return _skip(
            "openpi_pytorch_baseline",
            f"openpi source root missing: {args.openpi_root}",
            cwd=OFFICIAL_FLASHRT,
        )
    code = r"""
import importlib.util, json, sys
mods = ["openpi", "jax", "transformers", "safetensors", "torch"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print(json.dumps({"missing": missing}))
    raise SystemExit(3)
print(json.dumps({"missing": []}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.openpi_root)
    command = [args.baseline_python, "-c", code]
    proc = subprocess.run(
        command,
        cwd=str(OFFICIAL_FLASHRT),
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    if proc.returncode == 0:
        res = StepResult(
            name="openpi_pytorch_baseline",
            status="ready",
            command=command,
            cwd=str(OFFICIAL_FLASHRT),
            returncode=proc.returncode,
            metrics={},
            note=(
                "Dependencies are importable. Wire a concrete OpenPI latency "
                "command here before publishing full E2E numbers."
            ),
            stdout_tail=_tail(proc.stdout),
            stderr_tail=_tail(proc.stderr),
        )
        return res
    missing = "unknown"
    try:
        missing = ",".join(json.loads(proc.stdout.strip().splitlines()[-1])["missing"])
    except Exception:
        pass
    return _skip(
        "openpi_pytorch_baseline",
        (
            "official PyTorch/OpenPI baseline not run in this environment; "
            f"missing imports: {missing}. Do not substitute FlashRT numbers "
            "for this baseline."
        ),
        cwd=OFFICIAL_FLASHRT,
        command=command,
    )


def _container_path(host_path: Path, *, container_repo: str) -> str:
    rel = host_path.resolve().relative_to(ROOT.resolve())
    return str(Path(container_repo) / rel)


def _openpi_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    latency = payload.get("latency", {})
    return {
        "mode": payload.get("mode"),
        "openpi_p50_ms": latency.get("p50_ms"),
        "openpi_p95_ms": latency.get("p95_ms"),
        "openpi_mean_ms": latency.get("mean_ms"),
        "first_ms": payload.get("first_ms"),
        "load_s": payload.get("load_s"),
    }


def run_openpi_baseline(args: argparse.Namespace) -> StepResult:
    if args.openpi_baseline_mode == "probe":
        return run_openpi_probe(args)

    out_json = args.output.parent / "pi05_openpi_baseline_tmp.json"
    if args.openpi_baseline_mode == "local":
        command = [
            args.baseline_python,
            "demos/runtime-demo/pi05_openpi_baseline.py",
            "--openpi-root",
            args.openpi_root,
            "--checkpoint",
            args.checkpoint,
            "--num-views",
            str(args.openpi_num_views),
            "--steps",
            str(args.steps),
            "--warmup",
            str(args.openpi_warmup),
            "--iters",
            str(args.openpi_iters),
            "--compile",
            args.openpi_compile,
            "--output",
            str(out_json),
        ]
        res = _run(
            "openpi_pytorch_baseline",
            command,
            cwd=ROOT,
            timeout=args.timeout,
        )
    else:
        container_out = _container_path(out_json, container_repo=args.container_repo)
        inner = (
            f"cd {shlex.quote(args.container_repo)} && "
            f"PYTHONPATH={shlex.quote(args.container_openpi_root)} "
            f"{shlex.quote(args.container_python)} "
            "demos/runtime-demo/pi05_openpi_baseline.py "
            f"--openpi-root {shlex.quote(args.container_openpi_root)} "
            f"--checkpoint {shlex.quote(args.container_checkpoint)} "
            f"--num-views {args.openpi_num_views} "
            f"--steps {args.steps} "
            f"--warmup {args.openpi_warmup} "
            f"--iters {args.openpi_iters} "
            f"--compile {shlex.quote(args.openpi_compile)} "
            f"--output {shlex.quote(container_out)}"
        )
        command = ["docker", "exec", args.container, "bash", "-lc", inner]
        res = _run(
            "openpi_pytorch_baseline",
            command,
            cwd=ROOT,
            timeout=args.timeout,
        )

    if out_json.exists():
        payload = json.loads(out_json.read_text())
        res.metrics.update(_openpi_metrics(payload))
        res.note = payload.get("note", "")
    elif res.status == "pass":
        res.note = "OpenPI baseline command passed but did not write output JSON."
    return res


def run_flashrt_fp16(args: argparse.Namespace) -> StepResult:
    if args.flashrt_mode == "docker":
        inner = (
            f"cd {shlex.quote(args.container_flashrt_root)} && "
            f"PYTHONPATH=. {shlex.quote(args.container_flashrt_python)} "
            "examples/blackwell/bench_pi05_fp16.py "
            f"--checkpoint {shlex.quote(args.container_checkpoint)} "
            f"--num-views {args.num_views} "
            f"--steps {args.steps} "
            f"--warmup {args.flashrt_warmup} "
            f"--iters {args.flashrt_iters} "
            f"--hardware {shlex.quote(args.hardware)}"
        )
        command = ["docker", "exec", args.container, "bash", "-lc", inner]
        res = _run(
            "flashrt_full_fp16",
            command,
            cwd=ROOT,
            timeout=args.timeout,
        )
    else:
        command = [
            args.flashrt_python,
            "examples/blackwell/bench_pi05_fp16.py",
            "--checkpoint",
            args.checkpoint,
            "--num-views",
            str(args.num_views),
            "--steps",
            str(args.steps),
            "--warmup",
            str(args.flashrt_warmup),
            "--iters",
            str(args.flashrt_iters),
            "--hardware",
            args.hardware,
        ]
        res = _run(
            "flashrt_full_fp16",
            command,
            cwd=OFFICIAL_FLASHRT,
            env=_flashrt_env(),
            timeout=args.timeout,
        )
    m = re.search(
        r"RESULT wall_ms .*?p50=([0-9.]+).*?p95=([0-9.]+).*?mean=([0-9.]+)",
        res.stdout_tail,
        flags=re.S,
    )
    if m:
        res.metrics.update(
            {
                "wall_p50_ms": float(m.group(1)),
                "wall_p95_ms": float(m.group(2)),
                "wall_mean_ms": float(m.group(3)),
            }
        )
    res.note = (
        "Checkpoint-backed FlashRT full-model FP16 E2E. This is the optimized "
        "FlashRT full pipeline, not the public HF Kernel Hub runtime."
    )
    return res


def run_flashrt_fp8(args: argparse.Namespace) -> StepResult:
    if not args.run_flashrt_fp8:
        return _skip(
            "flashrt_full_fp8",
            "disabled; pass --run-flashrt-fp8 to run the full FP8 model",
            cwd=ROOT,
        )
    if args.flashrt_mode == "docker":
        inner = (
            f"cd {shlex.quote(args.container_flashrt_root)} && "
            f"PYTHONPATH=. {shlex.quote(args.container_flashrt_python)} "
            "examples/quickstart.py "
            f"--checkpoint {shlex.quote(args.container_checkpoint)} "
            "--framework torch --config pi05 "
            f"--num_views {args.num_views} "
            f"--hardware {shlex.quote(args.hardware)} "
            f"--benchmark {args.flashrt_fp8_iters} "
            f"--warmup {args.flashrt_fp8_warmup} "
            "--autotune 3"
        )
        command = ["docker", "exec", args.container, "bash", "-lc", inner]
        res = _run(
            "flashrt_full_fp8",
            command,
            cwd=ROOT,
            timeout=args.timeout,
        )
    else:
        command = [
            args.flashrt_python,
            "examples/quickstart.py",
            "--checkpoint",
            args.checkpoint,
            "--framework",
            "torch",
            "--config",
            "pi05",
            "--num_views",
            str(args.num_views),
            "--hardware",
            args.hardware,
            "--benchmark",
            str(args.flashrt_fp8_iters),
            "--warmup",
            str(args.flashrt_fp8_warmup),
            "--autotune",
            "3",
        ]
        res = _run(
            "flashrt_full_fp8",
            command,
            cwd=OFFICIAL_FLASHRT,
            env=_flashrt_env(),
            timeout=args.timeout,
        )
    m = re.search(
        r"P50:\s*([0-9.]+)\s*ms.*?min:\s*([0-9.]+),\s*mean:\s*([0-9.]+),\s*max:\s*([0-9.]+)\s*ms",
        res.stdout_tail,
        flags=re.S,
    )
    if m:
        res.metrics.update(
            {
                "wall_p50_ms": float(m.group(1)),
                "wall_min_ms": float(m.group(2)),
                "wall_mean_ms": float(m.group(3)),
                "wall_max_ms": float(m.group(4)),
            }
        )
    res.note = (
        "Checkpoint-backed FlashRT full-model FP8 E2E in the validated "
        "container runtime."
    )
    return res


def run_hub_runtime(args: argparse.Namespace) -> StepResult:
    out_json = args.output.parent / "pi05_hub_runtime_tmp.json"
    command = [
        args.hub_python,
        "demos/runtime-demo/pi05_runtime_demo.py",
        "--profile",
        args.hub_profile,
        "--layers",
        str(args.hub_layers),
        "--warmup",
        str(args.hub_warmup),
        "--iters",
        str(args.hub_iters),
        "--output",
        str(out_json),
    ]
    if args.cuda_graph:
        command.append("--cuda-graph")
    res = _run(
        "hf_kernel_hub_runtime_hotpath",
        command,
        cwd=ROOT,
        timeout=args.timeout,
    )
    if out_json.exists():
        res.metrics.update(json.loads(out_json.read_text()))
    res.note = (
        "Public HF Kernel Hub hot-path runtime with synthetic PI0.5-shaped "
        "inputs. This is not yet checkpoint-backed full policy E2E."
    )
    return res


def write_reports(path: Path, results: list[StepResult]) -> None:
    payload = {
        "scope": (
            "PI0.5 E2E staging report. Use OpenPI/PyTorch as the future "
            "published baseline; FlashRT sanity and HF Hub hotpath are separate."
        ),
        "results": [asdict(r) for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")

    md = path.with_suffix(".md")
    lines = [
        "# PI0.5 E2E Staging Report",
        "",
        "This report deliberately separates official PyTorch/OpenPI baseline, "
        "FlashRT pipeline sanity, and HF Kernel Hub runtime hot path.",
        "",
        "| Step | Status | Key metrics | Note |",
        "| --- | --- | --- | --- |",
    ]
    for r in results:
        metrics = ", ".join(f"{k}={v}" for k, v in r.metrics.items() if k.endswith(("ms", "us", "vs_eager")))
        if not metrics and r.metrics:
            metrics = ", ".join(f"{k}={v}" for k, v in list(r.metrics.items())[:6])
        lines.append(f"| `{r.name}` | `{r.status}` | {metrics or 'n/a'} | {r.note} |")
    lines.extend(["", "## Commands", ""])
    for r in results:
        if r.command:
            lines.append(f"### {r.name}")
            lines.append("")
            lines.append("```bash")
            lines.append(" ".join(shlex.quote(x) for x in r.command))
            lines.append("```")
            lines.append("")
    md.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument(
        "--openpi-baseline-mode",
        choices=("probe", "local", "docker"),
        default="probe",
        help="Use docker to run the real OpenPI/PyTorch baseline in pi0-stablehlo-test.",
    )
    p.add_argument("--baseline-python", default=sys.executable)
    p.add_argument("--openpi-root", default=str(DEFAULT_OPENPI_ROOT))
    p.add_argument("--openpi-num-views", type=int, default=2, choices=(1, 2, 3))
    p.add_argument("--openpi-warmup", type=int, default=5)
    p.add_argument("--openpi-iters", type=int, default=20)
    p.add_argument("--openpi-compile", choices=("off", "on"), default="off")
    p.add_argument("--container", default="pi0-stablehlo-test")
    p.add_argument("--container-python", default=DEFAULT_CONTAINER_OPENPI_PY)
    p.add_argument("--container-repo", default=DEFAULT_CONTAINER_REPO)
    p.add_argument("--container-openpi-root", default=DEFAULT_CONTAINER_OPENPI_ROOT)
    p.add_argument("--container-checkpoint", default=DEFAULT_CONTAINER_CKPT)
    p.add_argument("--container-flashrt-root", default=DEFAULT_CONTAINER_FLASHRT_ROOT)
    p.add_argument("--container-flashrt-python", default="python3")
    p.add_argument("--flashrt-mode", choices=("local", "docker"), default="docker")
    p.add_argument("--flashrt-python", default=sys.executable)
    p.add_argument("--hub-python", default=str(DEFAULT_HUB_PY))
    p.add_argument("--hardware", default="rtx_sm120")
    p.add_argument("--num-views", type=int, default=2)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--flashrt-warmup", type=int, default=10)
    p.add_argument("--flashrt-iters", type=int, default=50)
    p.add_argument("--run-flashrt-fp8", action="store_true")
    p.add_argument("--flashrt-fp8-warmup", type=int, default=50)
    p.add_argument("--flashrt-fp8-iters", type=int, default=50)
    p.add_argument("--hub-profile", default="pi05_hotpath")
    p.add_argument("--hub-layers", type=int, default=4)
    p.add_argument("--hub-warmup", type=int, default=10)
    p.add_argument("--hub-iters", type=int, default=50)
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--output", type=Path, default=ROOT / "internal-tests/runtime-demo/pi05-e2e-staging.json")
    args = p.parse_args()
    if not args.output.is_absolute():
        args.output = (Path.cwd() / args.output).resolve()

    results = [
        run_openpi_baseline(args),
        run_flashrt_fp16(args),
        run_flashrt_fp8(args),
        run_hub_runtime(args),
    ]
    write_reports(args.output, results)
    print(json.dumps({"results": [asdict(r) for r in results]}, indent=2))
    failed = [r.name for r in results if r.status == "fail"]
    if failed:
        raise SystemExit(f"failed steps: {failed}")


if __name__ == "__main__":
    main()
