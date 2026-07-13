#!/usr/bin/env bash
# Recreate a clean Python virtualenv and install pinned ML dependencies.
# Edit `requirements.lock.example` to match your CUDA and deployment specifics before running.

set -euo pipefail

VENV_DIR=".venv.clean"
PYTHON=${PYTHON:-python3}

echo "Creating virtualenv in ${VENV_DIR}..."
${PYTHON} -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate
pip install --upgrade pip setuptools wheel

echo "Installing PyTorch trio (adjust CUDA tag if needed)..."
# Adjust the extra-index for the correct CUDA build; change cu128 to your CUDA (cu118/cu128/etc)
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0

echo "Installing remaining pinned requirements..."
if [ -f "requirements.lock" ]; then
  pip install -r requirements.lock
else
  pip install -r requirements.lock.example
fi

echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
