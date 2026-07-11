#!/usr/bin/env python3
"""Ensure the runtime dependencies required by the pipeline are installed."""

from __future__ import annotations

import argparse
import json
import importlib
from importlib import metadata
import subprocess
import sys
from pathlib import Path
from typing import Sequence

try:
    from packaging.requirements import Requirement
except Exception:  # pragma: no cover - packaging is normally available
    Requirement = None

DEFAULT_REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "python-multipart",
    "pydantic",
    "pypdf",
    "neo4j",
    "numpy",
    "requests",
    "Pillow>=10.2.0,<11.0.0",
    "pydantic-settings",
    "httpx",
    "torch>=2.4.0,<2.6.0",
]

OPTIONAL_PACKAGES = [
    "qdrant-client>=1.14.3,<2.0.0",
    "graphrag>=0.1.0",
    "loguru>=0.7.0",
    "node2vec>=0.4.7",
    "sentence-transformers>=3.0.0",
    "pyarrow>=15.0.0,<16.0.0",
    "bertopic>=0.15.0",
    "hdbscan>=0.8.0",
    "umap-learn>=0.5.4",
    "networkx>=3.0",
    "langgraph>=0.1.0",
    "timesfm>=1.0.0",
    "pytorch-lightning>=2.0.0",
]


def _package_to_module_name(package_spec: str) -> str:
    normalized = package_spec.lower()
    mapping = {
        "qdrant-client": "qdrant_client",
        "sentence-transformers": "sentence_transformers",
        "pydantic-settings": "pydantic_settings",
        "python-multipart": "multipart",
        "pillow": "PIL",
    }
    for prefix, module_name in mapping.items():
        if normalized.startswith(prefix):
            return module_name

    if normalized.startswith("pyarrow"):
        return "pyarrow"
    return normalized.split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].split("<", 1)[0].replace("-", "_")


def _is_package_satisfied(package_spec: str) -> bool:
    module_name = _package_to_module_name(package_spec)

    try:
        if Requirement is None:
            importlib.import_module(module_name)
            return True

        requirement = Requirement(package_spec)
        importlib.import_module(module_name)

        if not requirement.specifier:
            return True

        try:
            installed_version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            module = importlib.import_module(module_name)
            installed_version = getattr(module, "__version__", None)

        if not installed_version:
            return False

        return requirement.specifier.contains(installed_version, prereleases=True)
    except Exception:
        return False


def find_missing_dependencies(required_packages: Sequence[str] | None = None, *, include_optional: bool = False) -> list[str]:
    """Return the importable module names that are not available in the current environment."""
    packages = list(required_packages or DEFAULT_REQUIRED_PACKAGES)
    if include_optional:
        packages.extend(OPTIONAL_PACKAGES)
    missing: list[str] = []

    for package in packages:
        if not _is_package_satisfied(package):
            missing.append(package)

    return missing


def ensure_dependencies(
    requirements_path: str | Path,
    python_executable: str | None = None,
    required_packages: Sequence[str] | None = None,
    *,
    include_optional: bool = False,
) -> list[str]:
    """Install only the required dependencies that are still missing."""
    requirements_file = Path(requirements_path)
    if not requirements_file.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_file}")

    python_bin = python_executable or sys.executable
    missing_packages = find_missing_dependencies(required_packages, include_optional=include_optional)
    if not missing_packages:
        return []

    install_targets = list(dict.fromkeys(missing_packages))
    print(f"Installing missing dependencies: {', '.join(install_targets)}")
    subprocess.check_call(
        [
            python_bin,
            "-m",
            "pip",
            "install",
            "--prefer-binary",
            "--no-cache-dir",
            *install_targets,
        ]
    )

    probe = subprocess.run(
        [
            python_bin,
            "-c",
            (
                "import json, sys; "
                "from scripts.ensure_dependencies import find_missing_dependencies; "
                "required = json.loads(sys.argv[1]); "
                "include_optional = sys.argv[2] == '1'; "
                "missing = find_missing_dependencies(required, include_optional=include_optional); "
                "print('\\n'.join(missing)); "
                "raise SystemExit(1 if missing else 0)"
            ),
            json.dumps(list(required_packages) if required_packages is not None else None),
            "1" if include_optional else "0",
        ],
        capture_output=True,
        text=True,
    )
    missing_after_install = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if probe.returncode != 0 and not missing_after_install:
        raise RuntimeError(
            f"Dependency verification failed: {probe.stderr.strip() or 'unknown error'}"
        )
    if missing_after_install:
        unresolved_required = find_missing_dependencies(required_packages, include_optional=False)
        if unresolved_required:
            raise RuntimeError(
                f"Dependency installation did not resolve the following packages: {', '.join(unresolved_required)}"
            )

        print(
            f"⚠ Optional dependencies still missing after install: {', '.join(missing_after_install)}"
        )

    return missing_after_install


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", default="requirements.txt")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--include-optional", action="store_true", help="Install optional full-stack dependencies too")
    args = parser.parse_args()

    missing_packages = find_missing_dependencies(include_optional=args.include_optional)
    if not missing_packages:
        print("✓ Required runtime dependencies are already available")
        return 0

    print(f"Detected missing runtime dependencies: {', '.join(missing_packages)}")
    ensure_dependencies(args.requirements, python_executable=args.python, include_optional=args.include_optional)
    print("✓ Runtime dependencies installed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
