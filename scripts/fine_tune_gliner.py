#!/usr/bin/env python3
"""Fine-tune GLiNER for industrial domain entity extraction.

This script is a starter workflow for preparing and storing a fine-tuned GLiNER model.
It currently loads a GLiNER base model, validates dataset format, and writes the output
model directory for later fine-tuning or training loop implementation.

Usage:
  python scripts/fine_tune_gliner.py --data data/gliner_training.jsonl --output models/gliner-industrial-v1

Expected dataset format (JSONL):
  {"text": "Pump pressure surged", "entities": [{"start": 0, "end": 4, "label": "equipment"}]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.pipeline.compat import allow_trusted_torch_pickle, ensure_pyarrow_compat

ensure_pyarrow_compat()

try:
    from gliner import GLiNER
except ImportError as exc:
    raise SystemExit("GLiNER is required to run this script. Install it in your environment.") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def validate_dataset(examples: list[dict[str, Any]]) -> None:
    if not examples:
        raise ValueError("Dataset is empty. Provide at least one JSONL example.")

    for idx, example in enumerate(examples[:5], start=1):
        if "text" not in example or "entities" not in example:
            raise ValueError(
                f"Example {idx} must contain 'text' and 'entities' fields."
            )
        if not isinstance(example["entities"], list):
            raise ValueError(f"Example {idx}: 'entities' must be a list.")


def write_preview(examples: list[dict[str, Any]], path: Path) -> None:
    preview = examples[:5]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as preview_file:
        preview_file.write(json.dumps(preview, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and save a GLiNER model directory for industrial fine-tuning."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training data in JSONL format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/gliner-industrial-v1"),
        help="Directory where the fine-tuned GLiNER model will be saved.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="urchade/gliner_medium-v2.1",
        help="Base GLiNER model to start from.",
    )
    parser.add_argument(
        "--preview-output",
        type=Path,
        default=Path("data/gliner_training_preview.json"),
        help="Optional preview file with the first examples.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Training data file not found: {args.data}")

    examples = load_jsonl(args.data)
    validate_dataset(examples)
    write_preview(examples, args.preview_output)

    print(f"Loaded {len(examples)} training examples from {args.data}")
    print(f"Saved dataset preview to {args.preview_output}")

    with allow_trusted_torch_pickle():
        model = GLiNER.from_pretrained(args.base_model)
    print(f"Loaded base GLiNER model: {args.base_model}")

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Saving base GLiNER model directory for later fine-tuning...")
    model.save_pretrained(str(args.output_dir))
    print(f"Model artifacts saved to: {args.output_dir}")
    print(
        "\nNOTE: This script does not yet run a full training loop. "
        "Use the prepared model directory as the starting point for industrial fine-tuning."
    )


if __name__ == "__main__":
    main()
