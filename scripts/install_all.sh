#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${1:-python}"

printf "Installing full layered runtime stack\n"
"$ROOT_DIR/scripts/install_base.sh" "$PYTHON_BIN"
"$ROOT_DIR/scripts/install_vision.sh" "$PYTHON_BIN"
"$ROOT_DIR/scripts/install_document.sh" "$PYTHON_BIN"
"$ROOT_DIR/scripts/install_nlp.sh" "$PYTHON_BIN"
"$ROOT_DIR/scripts/install_graph.sh" "$PYTHON_BIN"
"$ROOT_DIR/scripts/install_analytics.sh" "$PYTHON_BIN"
"$ROOT_DIR/scripts/install_agents.sh" "$PYTHON_BIN"
