#!/usr/bin/env python3
"""Promote published FlashRT kernel artifacts to the default `main` refs.

Kernel Builder publishes real artifacts on version branches such as `v1`.
That is correct for versioned usage:

    get_kernel("flashrt/<package>", revision="v1")

But many integration points use the default ref. If `main` only contains the
Hub card, `get_kernel("flashrt/<package>")` can fail or silently fall back.
This script makes the legacy model `main` ref a usable alias:

* for normal compiled packages, copy the latest Kernel Hub `vN` branch to the
  legacy model `main` branch;
* if the legacy model `main` already contains a usable `build/**` tree (for
  example a pure Python `torch-universal` package), use that as the source and
  leave it untouched instead of overwriting it with an older `vN`.

Note: current `huggingface_hub.upload_folder()` does not support
`repo_type="kernel"` uploads, so Kernel Hub `main` cannot be updated by this
script. Versioned Kernel Hub refs (`vN`) remain the canonical compiled
artifacts; legacy model `main` is the compatibility alias for default
`get_kernel("flashrt/<package>")` calls in older clients and downstream stacks.

The legacy model `vN` branches are still handled by
`mirror_kernel_legacy_model.py`.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


VERSION_BRANCH_RE = re.compile(r"^v[0-9]+$")

DELETE_PATTERNS = [
    "build/**",
    "README.md",
    "CARD.md",
    "VALIDATION.md",
    "benchmarks/**",
    "csrc/**",
    "examples/**",
    "tests/**",
    "torch-ext/**",
    "build.toml",
    "flake.lock",
    "flake.nix",
]


def package_dirs(root: Path) -> list[str]:
    return sorted(path.parent.name for path in root.glob("*/build.toml"))


def latest_version_branch(api: HfApi, repo_id: str, token: str | None) -> str:
    refs = api.list_repo_refs(repo_id, repo_type="kernel", token=token)
    versions = sorted(
        [branch.name for branch in refs.branches if VERSION_BRANCH_RE.match(branch.name)],
        key=lambda item: int(item[1:]),
    )
    if not versions:
        raise RuntimeError(f"{repo_id} has no Kernel Hub vN branch")
    return versions[-1]


def repo_files(api: HfApi, repo_id: str, repo_type: str, revision: str, token: str | None) -> list[str]:
    try:
        return api.list_repo_files(repo_id, repo_type=repo_type, revision=revision, token=token)
    except Exception:
        return []


def has_usable_build(files: list[str]) -> bool:
    return any(path.startswith("build/") for path in files)


def source_ref(api: HfApi, repo_id: str, token: str | None) -> tuple[str, str, str]:
    model_main_files = repo_files(api, repo_id, "model", "main", token)
    if has_usable_build(model_main_files):
        return "model", "main", "legacy model main"
    branch = latest_version_branch(api, repo_id, token)
    return "kernel", branch, f"kernel {branch}"


def download_source(repo_id: str, repo_type: str, revision: str, token: str | None, local_dir: Path) -> None:
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=local_dir,
        token=token,
    )


def upload_model_main_alias(
    api: HfApi,
    repo_id: str,
    folder: Path,
    token: str | None,
    dry_run: bool,
) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
    if dry_run:
        print(f"DRY-RUN upload model:{repo_id}@main from {folder}")
        return
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        folder_path=folder,
        path_in_repo="",
        delete_patterns=DELETE_PATTERNS,
        commit_message="Promote latest kernel artifacts to main",
        token=token,
    )


def promote_package(
    api: HfApi,
    namespace: str,
    package: str,
    token: str | None,
    dry_run: bool,
) -> None:
    repo_id = f"{namespace}/{package}"
    src_type, src_rev, src_label = source_ref(api, repo_id, token)
    with tempfile.TemporaryDirectory(prefix=f"flashrt-main-alias-{package}-") as tmp:
        local_dir = Path(tmp)
        download_source(repo_id, src_type, src_rev, token, local_dir)
        build_files = [str(path.relative_to(local_dir)) for path in local_dir.glob("build/**") if path.is_file()]
        if not build_files:
            raise RuntimeError(f"{repo_id}@{src_label} has no build/** files")

        if src_type == "model" and src_rev == "main":
            print(f"{repo_id}: model main already usable from {src_label} ({len(build_files)} build files)")
            return
        print(f"{repo_id}: promote {src_label} ({len(build_files)} build files) to legacy model main")
        upload_model_main_alias(api, repo_id, local_dir, token, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="flashrt")
    parser.add_argument("--package", default="all")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    root = Path(args.repo_root).resolve()
    packages = package_dirs(root)
    if args.package != "all":
        if args.package not in packages:
            raise RuntimeError(f"Unknown package {args.package!r}; known packages: {', '.join(packages)}")
        packages = [args.package]

    api = HfApi(token=token)
    failures: list[tuple[str, str]] = []
    for package in packages:
        try:
            promote_package(api, args.namespace, package, token, args.dry_run)
            print(f"OK {args.namespace}/{package}")
        except Exception as exc:  # noqa: BLE001 - continue and summarize all failures.
            failures.append((package, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL {args.namespace}/{package}: {type(exc).__name__}: {exc}")

    if failures:
        detail = "\n".join(f"- {package}: {reason}" for package, reason in failures)
        raise SystemExit(f"Main alias promotion failed for {len(failures)} package(s):\n{detail}")


if __name__ == "__main__":
    main()
