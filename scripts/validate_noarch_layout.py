#!/usr/bin/env python3
"""Validate the final import layout produced for a torch-noarch package."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tomllib


IMPORT_PROBE = r"""
import importlib.util
from pathlib import Path
import sys

variant = Path(sys.argv[1]).resolve()
module_name = sys.argv[2]
entry = variant / "__init__.py"
if not entry.is_file():
    raise SystemExit(f"missing final-layout entry point: {entry}")

spec = importlib.util.spec_from_file_location(
    module_name,
    entry,
    submodule_search_locations=[str(variant)],
)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot create import spec for {entry}")
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
print(f"final-layout import passed: {module_name}")
"""


def _load_config(package: Path) -> tuple[str, Path]:
    config_path = package / "build.toml"
    if not config_path.is_file():
        raise ValueError(f"missing {config_path}")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if "torch-noarch" not in config:
        raise ValueError(f"{package} is not a [torch-noarch] package")

    name = config.get("general", {}).get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"missing general.name in {config_path}")
    module_name = name.replace("-", "_")
    source = package / "torch-ext" / module_name
    if not (source / "__init__.py").is_file():
        raise ValueError(
            f"builder source module is missing: {source}/__init__.py"
        )
    return module_name, source


def _copy_builder_layout(source: Path, variant: Path) -> None:
    variant.mkdir(parents=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        destination = variant / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def validate(package: Path, backend: str) -> None:
    package = package.resolve()
    module_name, source = _load_config(package)
    with tempfile.TemporaryDirectory(prefix="flashrt-noarch-layout-") as tmp:
        variant = Path(tmp) / f"torch-{backend}"
        _copy_builder_layout(source, variant)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, "-c", IMPORT_PROBE, str(variant), module_name],
            cwd=tmp,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stdout:
            print(completed.stdout.strip())
    print(
        f"PASS: {package.name} matches the torch-noarch final-layout import "
        f"contract for torch-{backend}"
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="flashrt-noarch-selftest-") as tmp:
        root = Path(tmp)
        good = root / "good-package"
        bad = root / "bad-package"
        for package, name in ((good, "good-package"), (bad, "bad-package")):
            package.mkdir()
            (package / "build.toml").write_text(
                textwrap.dedent(
                    f"""
                    [general]
                    name = "{name}"
                    version = 1

                    [torch-noarch]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

        good_module = good / "torch-ext" / "good_package"
        good_module.mkdir(parents=True)
        (good_module / "__init__.py").write_text(
            "from .helper import VALUE\n", encoding="utf-8"
        )
        (good_module / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

        bad_module = bad / "torch-ext" / "bad_package"
        sibling = bad / "torch-ext" / "missing_sibling"
        bad_module.mkdir(parents=True)
        sibling.mkdir(parents=True)
        (bad_module / "__init__.py").write_text(
            "from missing_sibling import VALUE\n", encoding="utf-8"
        )
        (sibling / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")

        validate(good, "cuda")
        try:
            validate(bad, "cuda")
        except subprocess.CalledProcessError:
            print("PASS: self-test rejected a sibling package omitted by the builder")
        else:
            raise AssertionError("self-test failed to reject an omitted sibling package")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", nargs="?", type=Path)
    parser.add_argument("--backend", choices=("cuda", "rocm"), default="cuda")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.package is None:
        parser.error("package is required unless --self-test is used")
    validate(args.package, args.backend)


if __name__ == "__main__":
    main()
