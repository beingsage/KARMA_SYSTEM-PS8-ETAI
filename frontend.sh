#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

find_python_with_gradio() {
  local candidates=("$ROOT_DIR/.venv/bin/python" "python3" "python")
  for candidate in "${candidates[@]}"; do
    local interpreter=""
    if [ -x "$candidate" ]; then
      interpreter="$candidate"
    else
      interpreter="$(command -v "$candidate" 2>/dev/null || true)"
    fi
    if [ -n "$interpreter" ] && [ -x "$interpreter" ]; then
      if "$interpreter" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('gradio') else 1)" >/dev/null 2>&1; then
        echo "$interpreter"
        return 0
      fi
    fi
  done
  return 1
}

install_frontend_requirements() {
  local frontend_reqs="$ROOT_DIR/requirements.frontend.txt"
  local venv_python="$ROOT_DIR/.venv/bin/python"
  if [ -x "$venv_python" ] && [ -f "$frontend_reqs" ]; then
    echo "Installing frontend dependencies into .venv..."
    "$venv_python" "$ROOT_DIR/scripts/ensure_dependencies.py" --requirements "$frontend_reqs" --python "$venv_python"
    return $?
  fi
  return 1
}

PYTHON_BIN="$(find_python_with_gradio || true)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "$PYTHON_BIN" ]; then
  if install_frontend_requirements && PYTHON_BIN="$(find_python_with_gradio || true)"; then
    :
  fi
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "Error: No Python interpreter with gradio installed was found."
  if [ -f "$ROOT_DIR/requirements.frontend.txt" ]; then
    echo "Install it in the repo venv with:"
    echo "  python3 -m venv .venv"
    echo "  . .venv/bin/activate"
    echo "  python -m pip install -r requirements.frontend.txt"
  else
    echo "Install gradio manually:"
    echo "  python3 -m pip install gradio"
  fi
  echo "Then rerun ./frontend.sh"
  exit 1
fi

DEMO_SCRIPT="$ROOT_DIR/app/frontend/demo-ui/app.py"
if [ ! -f "$DEMO_SCRIPT" ]; then
  echo "Error: Demo script not found at $DEMO_SCRIPT"
  exit 1
fi

echo "Starting Industrial Pipeline demo UI using $PYTHON_BIN"
exec "$PYTHON_BIN" "$DEMO_SCRIPT"
