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
  echo "[kaggle_setup] GPU detected (nvidia-smi available). Trying CUDA-enabled PyTorch (cu118)"
  if pip install --prefer-binary --extra-index-url https://download.pytorch.org/whl/cu118 "torch" "torchvision" "torchaudio"; then
    echo "[kaggle_setup] Installed CUDA PyTorch (cu118) from PyTorch index"
  else
    echo "[kaggle_setup] cu118 wheel install failed, falling back to CPU-only wheels"
    pip install --prefer-binary --index-url https://download.pytorch.org/whl/cpu "torch" "torchvision" "torchaudio" || true
  fi
  if ! check_torch_install >/dev/null 2>&1; then
    echo "[kaggle_setup] Warning: torch installation succeeded but import or GPU verification failed. Retaining best-effort install."
  fi
else
  echo "[kaggle_setup] No GPU detected; installing CPU-only PyTorch wheels"
  pip install --prefer-binary --index-url https://download.pytorch.org/whl/cpu "torch" "torchvision" "torchaudio" || true
fi

echo "[kaggle_setup] Installing known Kaggle package dependencies that often fail in mixed installs"
pip install --prefer-binary groundingdino-py==0.4.0 docling==2.108.0 seqeval==1.2.2 blink==0.2.0 || echo "[kaggle_setup] Warning: explicit package install failed; continuing with best-effort environment"

echo "[kaggle_setup] Bootstrapping repo dependencies through the managed installer"
python "$ROOT_DIR/scripts/ensure_dependencies.py" --requirements "$ROOT_DIR/requirements.txt" --python "$(command -v python)" || echo "[kaggle_setup] dependency bootstrap reported errors; continue"

echo "[kaggle_setup] Kaggle environment setup complete"
exit 0
