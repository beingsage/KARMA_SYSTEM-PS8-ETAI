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

echo "[kaggle_setup] Ensuring safetensors is available"
pip install --prefer-binary --only-binary=:all: safetensors || true

check_torch_install() {
  echo "[kaggle_setup] Inspecting installed torch package..."
  python - <<'PY'
import sys
try:
    import torch
    print(f"[kaggle_setup] torch.__version__={torch.__version__}")
    print(f"[kaggle_setup] torch.version.cuda={torch.version.cuda}")
    print(f"[kaggle_setup] torch.cuda.is_available()={torch.cuda.is_available()}")
except Exception as exc:
    print(f"[kaggle_setup] torch import failed: {exc}")
    sys.exit(1)
PY
}

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[kaggle_setup] GPU detected (nvidia-smi available). Installing CUDA-enabled PyTorch (cu126)"
  if pip install --force-reinstall --upgrade --prefer-binary --only-binary=:all: --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
      torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0; then
    echo "[kaggle_setup] Installed CUDA PyTorch (cu126) from PyTorch index"
  else
    echo "[kaggle_setup] cu126 wheel install failed, falling back to CPU-only wheels"
    pip install --force-reinstall --upgrade --prefer-binary --only-binary=:all: --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 || true
  fi
  if ! check_torch_install >/dev/null 2>&1; then
    echo "[kaggle_setup] Warning: torch installation succeeded but import or GPU verification failed. Retaining best-effort install."
  fi
else
  echo "[kaggle_setup] No GPU detected; installing CPU-only PyTorch wheels"
  pip install --force-reinstall --upgrade --prefer-binary --only-binary=:all: --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 || true
fi

echo "[kaggle_setup] Bootstrapping repo dependencies through the managed installer"
python "$ROOT_DIR/scripts/ensure_dependencies.py" --requirements "$ROOT_DIR/requirements.txt" --python "$(command -v python)" || echo "[kaggle_setup] dependency bootstrap reported errors; continue"

echo "[kaggle_setup] Kaggle environment setup complete"
exit 0
