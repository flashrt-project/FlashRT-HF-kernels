#!/usr/bin/env python3
"""Mirror Kernel Hub repositories to legacy model repos.

Older `kernels` clients used by some downstream stacks call Hugging Face Hub
without `repo_type="kernel"`. They therefore resolve versions from model repo
refs and download artifacts from model repo snapshots. This script mirrors the
canonical Kernel Hub `vN` branches into same-name model repositories so both
old and new clients can load the same published artifacts.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


VERSION_BRANCH_RE = re.compile(r"^v[0-9]+$")


def package_dirs(root: Path) -> list[str]:
    return sorted(path.parent.name for path in root.glob("*/build.toml"))


def write_legacy_readme(path: Path, repo_id: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"# {repo_id}",
                "",
                "This repository is a compatibility mirror for older `kernels` clients",
                "that resolve repositories through the default Hugging Face model repo API.",
                "",
                f"Canonical Kernel Hub repo: https://huggingface.co/kernels/{repo_id}",
                "",
                "Do not edit this mirror by hand. It is generated from the Kernel Hub",
                "`vN` branches and contains the same `build/**` artifacts.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def get_version_branches(api: HfApi, repo_id: str, token: str) -> list[str]:
    refs = api.list_repo_refs(repo_id, repo_type="kernel", token=token)
    return sorted(
        [branch.name for branch in refs.branches if VERSION_BRANCH_RE.match(branch.name)],
        key=lambda item: int(item[1:]),
    )


def ensure_model_repo(api: HfApi, repo_id: str, token: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
    with tempfile.TemporaryDirectory(prefix="flashrt-legacy-main-") as tmp:
        tmp_path = Path(tmp)
        write_legacy_readme(tmp_path / "README.md", repo_id)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            revision="main",
            folder_path=tmp_path,
            path_in_repo="",
            commit_message="Initialize legacy kernels compatibility mirror",
            token=token,
        )


def mirror_branch(api: HfApi, repo_id: str, branch: str, token: str) -> None:
    with tempfile.TemporaryDirectory(prefix=f"flashrt-kernel-{repo_id.replace('/', '-')}-{branch}-") as tmp:
        local_dir = Path(tmp)
        snapshot_download(
            repo_id=repo_id,
            repo_type="kernel",
            revision=branch,
            local_dir=local_dir,
            allow_patterns=["build/**", "README.md"],
            token=token,
        )

        api.create_branch(
            repo_id=repo_id,
            repo_type="model",
            branch=branch,
            revision="main",
            exist_ok=True,
            token=token,
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            revision=branch,
            folder_path=local_dir,
            path_in_repo="",
            allow_patterns=["build/**", "README.md"],
            delete_patterns=["build/**", "README.md"],
            commit_message=f"Mirror Kernel Hub artifacts for {branch}",
            token=token,
        )
        try:
            api.delete_tag(repo_id=repo_id, repo_type="model", tag=branch, token=token)
        except Exception:
            pass
        api.create_tag(
            repo_id=repo_id,
            repo_type="model",
            tag=branch,
            revision=branch,
            exist_ok=True,
            token=token,
        )


def mirror_package(api: HfApi, namespace: str, package: str, token: str) -> tuple[str, list[str]]:
    repo_id = f"{namespace}/{package}"
    versions = get_version_branches(api, repo_id, token)
    if not versions:
        raise RuntimeError(f"{repo_id} has no Kernel Hub vN branches to mirror")

    ensure_model_repo(api, repo_id, token)
    for branch in versions:
        print(f"Mirroring {repo_id}@{branch} to legacy model repo")
        mirror_branch(api, repo_id, branch, token)
    return repo_id, versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="flashrt")
    parser.add_argument("--package", default="all")
    parser.add_argument("--repo-root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required")

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
            repo_id, versions = mirror_package(api, args.namespace, package, token)
            print(f"OK {repo_id}: mirrored {', '.join(versions)}")
        except Exception as exc:  # noqa: BLE001 - CI summary should include every package failure.
            failures.append((package, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL {args.namespace}/{package}: {type(exc).__name__}: {exc}")

    if failures:
        detail = "\n".join(f"- {package}: {reason}" for package, reason in failures)
        raise SystemExit(f"Legacy mirror failed for {len(failures)} package(s):\n{detail}")


if __name__ == "__main__":
    main()
