#!/usr/bin/env python3
"""Ensure the runtime dependencies required by the pipeline are installed."""

from __future__ import annotations

import argparse
import json
import importlib
import os
from importlib import metadata
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.pipeline.compat import ensure_pyarrow_compat
from app.pipeline.runtime import cuda_is_usable

ensure_pyarrow_compat()

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
    "numpy>=2.0,<3.0",
    "requests>=2.32.3,<3.0.0",
    "Pillow>=11.0.0,<13.0.0",
    "pydantic-settings>=2.14.2,<3.0.0",
    "httpx>=0.28.1,<1.0.0",
    "pyarrow>=21.0.0,<26.0.0",
    "torch",
]

OPTIONAL_PACKAGES = [
    "qdrant-client>=1.18.0,<2.0.0",
    "graphrag>=3.1.0",
    "loguru>=0.7.0",
    "node2vec>=0.5.0",
    "sentence-transformers>=5.6.0",
    "bertopic>=0.17.4",
    "hdbscan>=0.8.44",
    "umap-learn>=0.5.12",
    "networkx>=3.0",
    "langgraph>=0.2.76",
    "langchain>=0.3.0",
    "langchain-community>=0.4.2",
    "llama-index-core>=0.12.0,<0.15.0",
    "timesfm>=2.0.2",
    "pytorch-lightning>=2.2.0",
]

PYTORCH_PACKAGE_VERSIONS = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "torchaudio": "2.10.0",
}

def _normalize_package_spec(package_spec: str) -> str:
    normalized = package_spec.strip()
    if normalized.lower().startswith("-e "):
        normalized = normalized[3:].strip()

    if normalized.startswith("git+"):
        if "#egg=" in normalized:
            return normalized.split("#egg=", 1)[1]
        return normalized

    return normalized


def _is_pytorch_package(package_spec: str) -> bool:
    normalized = _normalize_package_spec(package_spec).lower()
    return normalized.startswith("torch") or normalized.startswith("torchvision") or normalized.startswith("torchaudio")


def _is_vcs_package(package_spec: str) -> bool:
    normalized = package_spec.strip().lower()
    return normalized.startswith("git+") or " @ git+" in normalized


def _cuda_tag_for_environment() -> str:
    if os.environ.get("PYTORCH_CUDA_TAG"):
        return os.environ["PYTORCH_CUDA_TAG"]
    if shutil.which("nvidia-smi"):
        return "cu126"
    return "cpu"


def _pytorch_install_cmd(package: str) -> list[str]:
    base = _normalize_package_spec(package)
    package_name = base.split("[", 1)[0].split("==", 1)[0].strip()
    version = base.split("==", 1)[1] if "==" in base else PYTORCH_PACKAGE_VERSIONS.get(package_name, "")
    target_spec = base if "==" in base else f"{package_name}=={version}" if version else package_name
    tag = _cuda_tag_for_environment()
    if tag and (base.startswith("torch") or base.startswith("torchvision") or base.startswith("torchaudio")):
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--upgrade",
            "--prefer-binary",
            "--only-binary=:all:",
            "--no-cache-dir",
            "--index-url",
            f"https://download.pytorch.org/whl/{tag}",
            target_spec,
        ]
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--prefer-binary",
        "--only-binary=:all:",
        "--no-cache-dir",
        package,
    ]


def _package_to_module_name(package_spec: str) -> str:
    normalized = _normalize_package_spec(package_spec).lower()
    mapping = {
        "groundingdino-py": "groundingdino",
        "groundingdino": "groundingdino",
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
        "surya-ocr": "surya",
        "umap-learn": "umap",
        "nougat-ocr": "nougat",
        "pytesseract": "pytesseract",
        "seqeval": "seqeval",
        "blink": "blink",
        "langgraph": "langgraph",
        "node2vec": "node2vec",
        "llama-index-core": "llama_index.core",
    }
    for prefix, module_name in mapping.items():
        if normalized.startswith(prefix):
            return module_name

    if normalized.startswith("git+") and "#egg=" in normalized:
        return normalized.split("#egg=", 1)[1]

    if normalized.startswith("pyarrow"):
        return "pyarrow"

    return normalized.split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].split("<", 1)[0].replace("-", "_")


OPTIONAL_PACKAGE_MODULES = {_package_to_module_name(pkg) for pkg in OPTIONAL_PACKAGES}


def _is_optional_package(package_spec: str) -> bool:
    return _package_to_module_name(package_spec) in OPTIONAL_PACKAGE_MODULES


def _is_kaggle_environment() -> bool:
    return bool(
        os.path.isdir("/kaggle")
        or os.environ.get("KAGGLE_URL_BASE", "").strip()
        or os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").strip()
    )


def _should_skip_on_kaggle(package_spec: str) -> bool:
    return False


def _is_package_satisfied(package_spec: str) -> bool:
    module_name = _package_to_module_name(package_spec)

    if _should_skip_on_kaggle(package_spec):
        return True

    try:
        if package_spec.lower().startswith("pyarrow"):
            if ensure_pyarrow_compat():
                return True
            import pyarrow
            return hasattr(pyarrow, "PyExtensionType")

        if _is_pytorch_package(package_spec):
            try:
                import torch

                try:
                    gpu_visible = bool(torch.cuda.is_available())
                except Exception:
                    gpu_visible = False

                if gpu_visible and not cuda_is_usable():
                    return False
            except Exception:
                return False

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
            return True

        return requirement.specifier.contains(installed_version, prereleases=True)
    except Exception:
        return False


def parse_requirements_file(requirements_path: str | Path) -> list[str]:
    requirements_file = Path(requirements_path).resolve()
    if not requirements_file.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_file}")

    packages: list[str] = []
    seen_requirements: set[Path] = set()
    seen_packages: set[str] = set()

    def _parse_file(file_path: Path) -> None:
        resolved_path = file_path.resolve()
        if resolved_path in seen_requirements:
            return
        seen_requirements.add(resolved_path)

        for raw_line in resolved_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith(("-r ", "--requirement ")):
                include_path = line.split(maxsplit=1)[1].strip()
                nested = (resolved_path.parent / include_path).resolve()
                _parse_file(nested)
                continue

            if line.startswith(("-c ", "--constraint ")):
                continue

            if line not in seen_packages:
                seen_packages.add(line)
                packages.append(line)

    _parse_file(requirements_file)
    return packages


def find_missing_dependencies(required_packages: Sequence[str] | None = None, *, include_optional: bool = False) -> list[str]:
    """Return the package specs whose importable modules are not available in the current environment."""
    packages = list(DEFAULT_REQUIRED_PACKAGES if required_packages is None else required_packages)
    if include_optional:
        packages.extend(OPTIONAL_PACKAGES)
    missing: list[str] = []
    seen: set[str] = set()

    for package in packages:
        if package in seen:
            continue
        seen.add(package)

        if not include_optional and _is_optional_package(package):
            continue

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

    install_targets = list(requirements if required_packages is None else required_packages)

    if include_optional:
        install_targets.extend(
            pkg for pkg in OPTIONAL_PACKAGES
            if pkg not in install_targets
        )

    missing_packages = find_missing_dependencies(
        install_targets,
        include_optional=include_optional,
    )

    if not missing_packages:
        return []

    print(f"Installing missing dependencies: {', '.join(missing_packages)}")

    failed_packages: list[str] = []
    for package in missing_packages:
        normalized_package = _normalize_package_spec(package).lower()
        if _should_skip_on_kaggle(package):
            print(f"  SKIP: skipping {package} on Kaggle; fallback behavior will be used")
            failed_packages.append(package)
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
        if normalized_package.startswith("pyarrow"):
            install_cmd = [
                python_bin,
                "-m",
                "pip",
                "install",
                "--prefer-binary",
                "--no-cache-dir",
                "pyarrow>=21.0.0,<26.0.0",
            ]
        elif _is_pytorch_package(package):
            install_cmd = _pytorch_install_cmd(package)
        elif _is_vcs_package(package):
            vcs_args = [
                python_bin,
                "-m",
                "pip",
                "install",
                "--prefer-binary",
                "--no-cache-dir",
            ]
            if "groundingdino" in normalized_package:
                vcs_args.append("--no-build-isolation")
            install_cmd = [
                package,
            ]
            install_cmd = vcs_args + install_cmd
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            failed_packages.append(package)
            print(f"  WARNING: failed to install {package}")
            stderr = result.stderr or ""
            if stderr.strip():
                print(stderr.strip())
            continue

    # Verify directly instead of importing scripts.ensure_dependencies
    remaining = find_missing_dependencies(
        install_targets,
        include_optional=include_optional,
    )

    if remaining:
        print(
            "  WARNING: some dependencies remain unavailable: "
            + ", ".join(remaining)
        )
        for package in remaining:
            if package not in failed_packages:
                failed_packages.append(package)
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
