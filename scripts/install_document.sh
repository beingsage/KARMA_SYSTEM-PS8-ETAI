#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${1:-python}"

printf "Installing document layer dependencies from requirements/document.txt\n"
"$PYTHON_BIN" -m pip install --prefer-binary --no-cache-dir -r "$ROOT_DIR/requirements/document.txt"
