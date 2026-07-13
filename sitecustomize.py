"""Initialize workspace-level compatibility shims before third-party imports run."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.pipeline.compat import ensure_pyarrow_compat

ensure_pyarrow_compat()
