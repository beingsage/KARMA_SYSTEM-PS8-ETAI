#!/bin/bash

# Lightweight pipeline runner
# Toggle stages with environment variables:
#   RUN_STAGE_1=1 RUN_STAGE_2=0 ./run_pipeline.sh
#   RUN_STAGE_1=1 RUN_STAGE_2=1 ./run_pipeline.sh

set -euo pipefail

echo "=========================================="
echo "Pipeline Runner"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PDF_PATH="$SCRIPT_DIR/1_Hydro MPC-1-10 (1).pdf"
STAGE1_DIR="$SCRIPT_DIR/data/pipeline/1.docling_surya_ocr"
STAGE2_DIR="$SCRIPT_DIR/data/pipeline/2.doclayout_yolo_analysis"
STAGE3_DIR="$SCRIPT_DIR/data/pipeline/3.surya_layout_understanding"
STAGE4_DIR="$SCRIPT_DIR/data/pipeline/4.table_structure_analysis"
STAGE5_DIR="$SCRIPT_DIR/data/pipeline/5.groundingdino_detection"
STAGE6_DIR="$SCRIPT_DIR/data/pipeline/6.groundingdino_detection"
STAGE7_DIR="$SCRIPT_DIR/data/pipeline/7.sam2_segmentation"
STAGE1_OUTPUT="$STAGE1_DIR/ocr_output.json"
STAGE2_OUTPUT="$STAGE2_DIR/doclayout_output.json"
STAGE3_OUTPUT="$STAGE3_DIR/layout_output.json"
STAGE4_OUTPUT="$STAGE4_DIR/table_output.json"
STAGE5_OUTPUT="$STAGE5_DIR/grounding_output.json"
STAGE6_OUTPUT="$STAGE6_DIR/grounding_output.json"
STAGE7_OUTPUT="$STAGE7_DIR/sam_output.json"

RUN_STAGE_1="${RUN_STAGE_1:-1}"
RUN_STAGE_2="${RUN_STAGE_2:-0}"
RUN_STAGE_3="${RUN_STAGE_3:-0}"
RUN_STAGE_4="${RUN_STAGE_4:-0}"
RUN_STAGE_5="${RUN_STAGE_5:-0}"
RUN_STAGE_6="${RUN_STAGE_6:-0}"
RUN_STAGE_7="${RUN_STAGE_7:-0}"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

echo ""
echo "📄 PDF: $PDF_PATH"
echo "📁 Stage 1 output: $STAGE1_OUTPUT"
echo "📁 Stage 2 output: $STAGE2_OUTPUT"
echo "📁 Stage 3 output: $STAGE3_OUTPUT"
echo "📁 Stage 4 output: $STAGE4_OUTPUT"
echo "📁 Stage 5 output: $STAGE5_OUTPUT"
echo "📁 Stage 6 output: $STAGE6_OUTPUT"
echo "📁 Stage 7 output: $STAGE7_OUTPUT"
echo ""

echo "RUN_STAGE_1=$RUN_STAGE_1"
echo "RUN_STAGE_2=$RUN_STAGE_2"
echo "RUN_STAGE_3=$RUN_STAGE_3"
echo "RUN_STAGE_4=$RUN_STAGE_4"
echo "RUN_STAGE_5=$RUN_STAGE_5"
echo "RUN_STAGE_6=$RUN_STAGE_6"
echo "RUN_STAGE_7=$RUN_STAGE_7"
echo ""

if [ "$RUN_STAGE_1" = "1" ]; then
  echo "Running Stage 1: docling_surya_ocr"
  PDF_PATH="$PDF_PATH" STAGE1_DIR="$STAGE1_DIR" "$PYTHON_BIN" - <<'PY'
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from app.pipeline.engine_v2 import get_pipeline


async def main():
    pdf_path = Path(os.environ["PDF_PATH"])
    stage1_dir = Path(os.environ["STAGE1_DIR"])

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pipeline = get_pipeline()
    ocr_result = await pipeline._run_stage(
        "docling_surya_ocr",
        pipeline._process_ocr,
        required=True,
        pdf_bytes=pdf_bytes,
        _progress_context={"stage_index": 1, "total_stages": 7},
    )

    payload = {
        "timestamp": datetime.now().isoformat(),
        "stage": "docling_surya_ocr",
        "status": ocr_result.get("status", "unknown"),
        "text_length": len(ocr_result.get("text", "")),
        "page_count": ocr_result.get("page_count", 0),
        "text_preview": ocr_result.get("text", "")[:1000],
        "full_output": ocr_result,
    }

    output_file = stage1_dir / "ocr_output.json"
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"Stage 1 complete: {output_file}")
    print(f"Text length: {payload['text_length']}")


asyncio.run(main())
PY
else
  echo "Skipping Stage 1"
fi

if [ "$RUN_STAGE_2" = "1" ]; then
  if [ ! -f "$STAGE1_OUTPUT" ]; then
    echo "Stage 1 output not found: $STAGE1_OUTPUT"
    exit 1
  fi

  echo "Running Stage 2: doclayout_yolo_analysis"
  PDF_PATH="$PDF_PATH" STAGE1_OUTPUT="$STAGE1_OUTPUT" STAGE2_DIR="$STAGE2_DIR" "$PYTHON_BIN" - <<'PY'
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from app.pipeline.engine_v2 import get_pipeline


async def main():
    pdf_path = Path(os.environ["PDF_PATH"])
    stage1_output = Path(os.environ["STAGE1_OUTPUT"])
    stage2_dir = Path(os.environ["STAGE2_DIR"])

    with open(stage1_output, "r") as fh:
        stage1_data = json.load(fh)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pipeline = get_pipeline()
    ocr_result = stage1_data.get("full_output", {})
    doclayout_result = await pipeline._run_stage(
        "doclayout_yolo_analysis",
        pipeline._analyze_doclayout_yolo,
        required=False,
        ocr_result=ocr_result,
        pdf_bytes=pdf_bytes,
        _progress_context={"stage_index": 2, "total_stages": 7},
    )

    payload = {
        "timestamp": datetime.now().isoformat(),
        "stage": "doclayout_yolo_analysis",
        "status": doclayout_result.get("status", "unknown"),
        "stage_1_input_text_length": stage1_data.get("text_length", 0),
        "full_output": doclayout_result,
    }

    output_file = stage2_dir / "doclayout_output.json"
    with open(output_file, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    print(f"Stage 2 complete: {output_file}")


asyncio.run(main())
PY
else
  echo "Skipping Stage 2"
fi

if [ "$RUN_STAGE_3" = "1" ]; then
  echo "Running Stage 3: surya_layout_understanding"
  PDF_PATH="$PDF_PATH" STAGE1_OUTPUT="$STAGE1_OUTPUT" STAGE3_DIR="$STAGE3_DIR" "$PYTHON_BIN" - <<'PY'
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from app.pipeline.engine_v2 import get_pipeline


async def main():
    pdf_path = Path(os.environ["PDF_PATH"])
    stage1_output = Path(os.environ["STAGE1_OUTPUT"])
    stage3_dir = Path(os.environ["STAGE3_DIR"])

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pipeline = get_pipeline()
    ocr_result = {}
    if stage1_output.exists():
        with open(stage1_output, "r") as fh:
            stage1_data = json.load(fh)
        ocr_result = stage1_data.get("full_output", {})

    if not ocr_result.get("text"):
        ocr_result = await pipeline._run_stage(
            "docling_surya_ocr",
            pipeline._process_ocr,
            required=True,
            pdf_bytes=pdf_bytes,
        )

    stage3_result = await pipeline._run_stage(
        "surya_layout_understanding",
        pipeline._build_structural_stage3_summary,
        required=False,
        ocr_result=ocr_result,
        pdf_bytes=pdf_bytes,
        _progress_context={"stage_index": 3, "total_stages": 7},
    )

    payload = {
        "timestamp": datetime.now().isoformat(),
        "stage": "surya_layout_understanding",
        "status": stage3_result.get("status", "unknown"),
        "text_length": len(ocr_result.get("text", "") or ""),
        "full_output": stage3_result,
    }

    output_file = stage3_dir / "layout_output.json"
    with open(output_file, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    print(f"Stage 3 complete: {output_file}")


asyncio.run(main())
PY
else
  echo "Skipping Stage 3"
fi

if [ "$RUN_STAGE_4" = "1" ]; then
  if [ ! -f "$STAGE1_OUTPUT" ] || [ ! -f "$STAGE3_OUTPUT" ]; then
    echo "Stage 1 or Stage 3 output missing for Stage 4"
    exit 1
  fi

  echo "Running Stage 4: table_structure_analysis"
  PDF_PATH="$PDF_PATH" STAGE1_OUTPUT="$STAGE1_OUTPUT" STAGE3_OUTPUT="$STAGE3_OUTPUT" STAGE4_DIR="$STAGE4_DIR" "$PYTHON_BIN" - <<'PY'
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from app.pipeline.engine_v2 import get_pipeline


async def main():
    pdf_path = Path(os.environ["PDF_PATH"])
    stage1_output = Path(os.environ["STAGE1_OUTPUT"])
    stage3_output = Path(os.environ["STAGE3_OUTPUT"])
    stage4_dir = Path(os.environ["STAGE4_DIR"])

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    with open(stage1_output, "r") as fh:
        stage1_data = json.load(fh)
    with open(stage3_output, "r") as fh:
        stage3_data = json.load(fh)

    pipeline = get_pipeline()
    ocr_result = stage1_data.get("full_output", {})
    stage3_result = stage3_data.get("full_output", {})
    stage4_result = await pipeline._run_stage(
        "table_structure_analysis",
        pipeline._build_structural_stage4_summary,
        required=False,
        ocr_result=ocr_result,
        pdf_bytes=pdf_bytes,
        stage3_result=stage3_result,
        _progress_context={"stage_index": 4, "total_stages": 7},
    )

    payload = {
        "timestamp": datetime.now().isoformat(),
        "stage": "table_structure_analysis",
        "status": stage4_result.get("status", "unknown"),
        "text_length": len(ocr_result.get("text", "") or ""),
        "full_output": stage4_result,
    }

    output_file = stage4_dir / "table_output.json"
    with open(output_file, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    print(f"Stage 4 complete: {output_file}")


asyncio.run(main())
PY
else
  echo "Skipping Stage 4"
fi

if [ "$RUN_STAGE_5" = "1" ]; then
  if [ ! -f "$STAGE1_OUTPUT" ] || [ ! -f "$STAGE4_OUTPUT" ]; then
    echo "Stage 1 or Stage 4 output missing for Stage 5"
    exit 1
  fi

  echo "Running Stage 5: groundingdino_detection"
  PDF_PATH="$PDF_PATH" STAGE1_OUTPUT="$STAGE1_OUTPUT" STAGE4_OUTPUT="$STAGE4_OUTPUT" STAGE5_DIR="$STAGE5_DIR" "$PYTHON_BIN" - <<'PY'
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from app.pipeline.engine_v2 import get_pipeline


async def main():
    pdf_path = Path(os.environ["PDF_PATH"])
    stage1_output = Path(os.environ["STAGE1_OUTPUT"])
    stage4_output = Path(os.environ["STAGE4_OUTPUT"])
    stage5_dir = Path(os.environ["STAGE5_DIR"])

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    with open(stage1_output, "r") as fh:
        stage1_data = json.load(fh)
    with open(stage4_output, "r") as fh:
        stage4_data = json.load(fh)

    pipeline = get_pipeline()
    ocr_result = stage1_data.get("full_output", {})
    stage4_result = stage4_data.get("full_output", {})
    stage5_result = await pipeline._run_stage(
        "groundingdino_detection",
        pipeline._build_structural_stage5_summary,
        required=False,
        ocr_result=ocr_result,
        pdf_bytes=pdf_bytes,
        stage4_result=stage4_result,
        _progress_context={"stage_index": 5, "total_stages": 7},
    )

    payload = {
        "timestamp": datetime.now().isoformat(),
        "stage": "groundingdino_detection",
        "status": stage5_result.get("status", "unknown"),
        "text_length": len(ocr_result.get("text", "") or ""),
        "full_output": stage5_result,
    }

    output_file = stage5_dir / "grounding_output.json"
    with open(output_file, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    print(f"Stage 5 complete: {output_file}")


asyncio.run(main())
PY
else
  echo "Skipping Stage 5"
fi

if [ "$RUN_STAGE_6" = "1" ]; then
  if [ ! -f "$STAGE1_OUTPUT" ]; then
    echo "Stage 1 output missing for Stage 6"
    exit 1
  fi

  echo "Running Stage 6: groundingdino_detection"
  PDF_PATH="$PDF_PATH" STAGE1_OUTPUT="$STAGE1_OUTPUT" STAGE6_DIR="$STAGE6_DIR" "$PYTHON_BIN" - <<'PY'
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from app.pipeline.engine_v2 import get_pipeline


async def main():
    pdf_path = Path(os.environ["PDF_PATH"])
    stage1_output = Path(os.environ["STAGE1_OUTPUT"])
    stage6_dir = Path(os.environ["STAGE6_DIR"])

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    with open(stage1_output, "r") as fh:
        stage1_data = json.load(fh)

    pipeline = get_pipeline()
    ocr_result = stage1_data.get("full_output", {})
    stage6_result = await pipeline._run_stage(
        "groundingdino_detection",
        pipeline._detect_groundingdino_objects,
        required=False,
        pdf_bytes=pdf_bytes,
        text=ocr_result.get("text", "") or "",
        _progress_context={"stage_index": 6, "total_stages": 7},
    )

    payload = {
        "timestamp": datetime.now().isoformat(),
        "stage": "groundingdino_detection",
        "status": stage6_result.get("status", "unknown"),
        "text_length": len(ocr_result.get("text", "") or ""),
        "full_output": stage6_result,
    }

    output_file = stage6_dir / "grounding_output.json"
    with open(output_file, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    print(f"Stage 6 complete: {output_file}")


asyncio.run(main())
PY
else
  echo "Skipping Stage 6"
fi

if [ "$RUN_STAGE_7" = "1" ]; then
  if [ ! -f "$STAGE1_OUTPUT" ] || [ ! -f "$STAGE6_OUTPUT" ]; then
    echo "Stage 1 or Stage 6 output missing for Stage 7"
    exit 1
  fi

  echo "Running Stage 7: sam2_segmentation"
  PDF_PATH="$PDF_PATH" STAGE6_OUTPUT="$STAGE6_OUTPUT" STAGE7_DIR="$STAGE7_DIR" "$PYTHON_BIN" - <<'PY'
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from app.pipeline.engine_v2 import get_pipeline


async def main():
    pdf_path = Path(os.environ["PDF_PATH"])
    stage6_output = Path(os.environ["STAGE6_OUTPUT"])
    stage7_dir = Path(os.environ["STAGE7_DIR"])

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    with open(stage6_output, "r") as fh:
        stage6_data = json.load(fh)

    pipeline = get_pipeline()
    stage6_result = stage6_data.get("full_output", {})
    stage7_result = await pipeline._run_stage(
        "sam2_segmentation",
        pipeline._segment_with_sam,
        required=False,
        pdf_bytes=pdf_bytes,
        context=stage6_result,
        _progress_context={"stage_index": 7, "total_stages": 7},
    )

    payload = {
        "timestamp": datetime.now().isoformat(),
        "stage": "sam2_segmentation",
        "status": stage7_result.get("status", "unknown"),
        "full_output": stage7_result,
    }

    output_file = stage7_dir / "sam_output.json"
    with open(output_file, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    print(f"Stage 7 complete: {output_file}")


asyncio.run(main())
PY
else
  echo "Skipping Stage 7"
fi

echo ""
echo "✓ Pipeline run complete"
