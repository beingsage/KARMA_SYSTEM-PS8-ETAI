#!/usr/bin/env bash
# Recreate a clean Python virtualenv and install pinned ML dependencies.
# The layered `requirements/*.txt` files are the source of truth for the stack.

set -euo pipefail

VENV_DIR=".venv.clean"
PYTHON=${PYTHON:-python3}

echo "Creating virtualenv in ${VENV_DIR}..."
${PYTHON} -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate
pip install --upgrade pip setuptools wheel

echo "Installing PyTorch trio (adjust CUDA tag if needed)..."
# Adjust the index URL if you need a different CUDA build.
pip install --force-reinstall --upgrade --prefer-binary --only-binary=:all: --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

echo "Installing remaining pinned requirements..."
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  pip install -r requirements.full.txt
fi

echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
