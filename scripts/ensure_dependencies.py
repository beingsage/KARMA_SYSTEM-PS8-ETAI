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
    "torch",
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


def _normalize_package_spec(package_spec: str) -> str:
    normalized = package_spec.strip()
    if normalized.lower().startswith("-e "):
        normalized = normalized[3:].strip()

    if normalized.startswith("git+"):
        if "#egg=" in normalized:
            return normalized.split("#egg=", 1)[1]
        return normalized

    return normalized


def _package_to_module_name(package_spec: str) -> str:
    normalized = _normalize_package_spec(package_spec).lower()
    mapping = {
        "qdrant-client": "qdrant_client",
        "sentence-transformers": "sentence_transformers",
        "pydantic-settings": "pydantic_settings",
        "python-multipart": "multipart",
        "pillow": "PIL",
        "langchain-community": "langchain_community",
        "pytorch-lightning": "pytorch_lightning",
        "scikit-learn": "sklearn",
        "pyarrow": "pyarrow",
        "segment-anything": "segment_anything",
        "doclayout_yolo": "doclayout_yolo",
    }
    for prefix, module_name in mapping.items():
        if normalized.startswith(prefix):
            return module_name

    if normalized.startswith("git+") and "#egg=" in normalized:
        return normalized.split("#egg=", 1)[1]

    if normalized.startswith("pyarrow"):
        return "pyarrow"

    return normalized.split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].split("<", 1)[0].replace("-", "_")


def _is_package_satisfied(package_spec: str) -> bool:
    module_name = _package_to_module_name(package_spec)

    try:
        if Requirement is None:
            importlib.import_module(module_name)
            return True

        normalized_spec = _normalize_package_spec(package_spec)
        try:
            requirement = Requirement(normalized_spec)
        except Exception:
            importlib.import_module(module_name)
            return True

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


def parse_requirements_file(requirements_path: str | Path) -> list[str]:
    requirements_file = Path(requirements_path)
    if not requirements_file.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_file}")

    packages: list[str] = []
    for raw_line in requirements_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r", "--requirement", "-c", "--constraint")):
            continue
        packages.append(line)
    return packages


def find_missing_dependencies(required_packages: Sequence[str] | None = None, *, include_optional: bool = False) -> list[str]:
    """Return the package specs whose importable modules are not available in the current environment."""
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

    requirements = parse_requirements_file(requirements_file)

    install_targets = list(required_packages or requirements)

    if include_optional:
        install_targets.extend(
            pkg for pkg in OPTIONAL_PACKAGES
            if pkg not in install_targets
        )

    missing_packages = find_missing_dependencies(
        install_targets,
        include_optional=False,
    )

    import os

    # Kaggle already provides CUDA-enabled PyTorch.
    if os.path.isdir("/kaggle"):
        missing_packages = [
            pkg
            for pkg in missing_packages
            if not _normalize_package_spec(pkg).lower().startswith(
                ("torch", "torchvision", "torchaudio")
            )
        ]

    if not missing_packages:
        return []

    print(f"Installing missing dependencies: {', '.join(missing_packages)}")

    failed_packages: list[str] = []
    for package in missing_packages:
        normalized_package = _normalize_package_spec(package).lower()
        if os.path.isdir("/kaggle") and "groundingdino" in normalized_package:
            print(f"  SKIP: skipping {package} on Kaggle; GroundingDINO will remain unavailable")
            continue

        install_cmd = [
            python_bin,
            "-m",
            "pip",
            "install",
            "--prefer-binary",
            "--no-cache-dir",
            package,
        ]
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            failed_packages.append(package)
            print(f"  WARNING: failed to install {package}")
            if result.stderr.strip():
                print(result.stderr.strip())
            continue

    # Verify directly instead of importing scripts.ensure_dependencies
    remaining = find_missing_dependencies(
        install_targets,
        include_optional=False,
    )

    if remaining:
        print(
            "  WARNING: some dependencies remain unavailable: "
            + ", ".join(remaining)
        )
        return failed_packages

    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", default="requirements.txt")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--include-optional", action="store_true", help="Install optional full-stack dependencies too")
    args = parser.parse_args()

    requirements = parse_requirements_file(args.requirements)
    missing_packages = find_missing_dependencies(requirements, include_optional=args.include_optional)
    if not missing_packages:
        print("✓ Required runtime dependencies are already available")
        return 0

    print(f"Detected missing runtime dependencies: {', '.join(missing_packages)}")
    failed_packages = ensure_dependencies(
        args.requirements,
        python_executable=args.python,
        include_optional=args.include_optional,
    )
    if failed_packages:
        print(
            "⚠ Dependency install completed with failures: "
            + ", ".join(failed_packages)
        )
        return 1

    print("✓ Runtime dependencies installed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
