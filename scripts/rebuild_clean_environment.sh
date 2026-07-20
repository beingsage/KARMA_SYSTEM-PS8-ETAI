#!/usr/bin/env bash
# Rebuild the Python environment from scratch with locked, compatible dependencies.
# This removes all conflicting ML packages and installs pinned versions.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

echo "=================================================================================="
echo "Rebuilding clean ML environment"
echo "=================================================================================="
echo ""

# Backup existing venv if it exists
if [ -d "$VENV_DIR" ]; then
  BACKUP_DIR="${VENV_DIR}.backup.$(date +%s)"
  echo "Backing up existing .venv to $BACKUP_DIR"
  mv "$VENV_DIR" "$BACKUP_DIR"
fi

# Create fresh venv
echo "Creating fresh virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip/setuptools/wheel first
echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Detect CUDA/CPU environment
echo "Detecting compute environment..."
CUDA_TAG="cpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_TAG="cu126"
  echo "GPU detected: using CUDA build"
else
  echo "No GPU detected: using CPU build"
fi

# Install PyTorch trio with matching ABI
echo ""
echo "Installing PyTorch trio (torch, torchvision, torchaudio)..."
if [ "$CUDA_TAG" = "cpu" ]; then
  pip install --force-reinstall --upgrade --prefer-binary --only-binary=:all: --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0"
else
  pip install --force-reinstall --upgrade --prefer-binary --only-binary=:all: --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
    "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0"
fi

# Verify torch ABI
echo ""
echo "Verifying PyTorch ABI match..."
python3 - <<'PYVERIFY'
import torch, torchvision, torchaudio
from app.pipeline.runtime import cuda_is_usable

print(f"torch:       {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"torchaudio:  {torchaudio.__version__}")

if torch.cuda.is_available():
  print(f"cuda_usable: {cuda_is_usable()}")
  if not cuda_is_usable():
    print("ERROR: CUDA is visible but unusable with the installed PyTorch wheel")
    raise SystemExit(1)
  print("✓ CUDA runtime OK")
else:
  print("✓ CPU runtime OK")
PYVERIFY

# Install core and ML dependencies from lock file
echo ""
echo "Installing locked dependencies..."
if [ -f "$ROOT_DIR/requirements.txt" ]; then
  echo "Using requirements.txt"
  pip install --prefer-binary --no-cache-dir -r "$ROOT_DIR/requirements.txt"
else
  echo "Using requirements.full.txt"
  pip install --prefer-binary --no-cache-dir -r "$ROOT_DIR/requirements.full.txt"
fi

# Verify key imports
echo ""
echo "Verifying key imports..."
python3 - <<'PYVERIFY'
import sys
packages = {
  'transformers': 'transformers',
  'sentence-transformers': 'sentence_transformers',
  'docling': 'docling',
  'gliner': 'gliner',
  'glirel': 'glirel',
  'pyarrow': 'pyarrow',
  'neo4j': 'neo4j',
  'torch': 'torch',
}

failed = []
for pkg_name, module_name in packages.items():
  try:
    __import__(module_name)
    print(f"✓ {pkg_name}")
  except Exception as e:
    print(f"✗ {pkg_name}: {type(e).__name__}: {e}")
    failed.append(pkg_name)

if failed:
  print(f"\nFailed to import: {', '.join(failed)}")
  sys.exit(1)

print("\n✓ All key imports successful")
PYVERIFY

echo ""
echo "=================================================================================="
echo "✓ Environment rebuild complete!"
echo "=================================================================================="
echo ""
echo "Activate with: source $VENV_DIR/bin/activate"
echo ""
