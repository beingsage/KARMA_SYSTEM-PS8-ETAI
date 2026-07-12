#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "[kaggle_setup] Starting Kaggle-friendly environment setup"

# Configure caches to /kaggle/working for persistence in Kaggle
export HF_HOME="/kaggle/working/hf_cache"
export TRANSFORMERS_CACHE="/kaggle/working/hf_cache"
export TORCH_HOME="/kaggle/working/torch_cache"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME"
echo "[kaggle_setup] HF_HOME=$HF_HOME"

echo "[kaggle_setup] Upgrading pip, setuptools, wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[kaggle_setup] Installing compatible pyarrow and safetensors"
pip install --prefer-binary --only-binary=:all: pyarrow==15.0.2 safetensors || true

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[kaggle_setup] GPU detected (nvidia-smi available). Trying CUDA-enabled PyTorch (cu118)"
  if pip install --prefer-binary --extra-index-url https://download.pytorch.org/whl/cu118 "torch" "torchvision" "torchaudio"; then
    echo "[kaggle_setup] Installed CUDA PyTorch (cu118) from PyTorch index"
  else
    echo "[kaggle_setup] cu118 wheel install failed, falling back to CPU-only wheels"
    pip install --prefer-binary --index-url https://download.pytorch.org/whl/cpu "torch" "torchvision" "torchaudio" || true
  fi
else
  echo "[kaggle_setup] No GPU detected; installing CPU-only PyTorch wheels"
  pip install --prefer-binary --index-url https://download.pytorch.org/whl/cpu "torch" "torchvision" "torchaudio" || true
fi

echo "[kaggle_setup] Installing common extras"
pip install --prefer-binary -r "$ROOT_DIR/requirements.txt" || echo "[kaggle_setup] requirements install reported errors; continue"

echo "[kaggle_setup] Kaggle environment setup complete"
exit 0
