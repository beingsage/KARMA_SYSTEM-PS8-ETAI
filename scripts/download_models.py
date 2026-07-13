#!/usr/bin/env python3
"""
Download all models required for the industrial PDF-to-graph pipeline.
This script ensures all models are available before starting the application.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.model_helpers import (
    PIDSymbolDetector,
    GroundingDinoDetector,
    SamSegmenter,
    BgeEmbedder,
    BgeReranker,
    GLiRELRelationExtractor,
    BlinkEntityLinker,
    download_model_checkpoint,
    MODELS_DIR,
)
from app.pipeline.entity_extractor import GlinerEntityExtractor


def download_all_models():
    """Download and initialize all models for the pipeline."""
    
    print("=" * 80)
    print("Industrial PDF-to-Graph Pipeline - Model Download Script")
    print("=" * 80)
    print()

    checkpoint_files = [
        ("yolov8n.pt", "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"),
        ("groundingdino_swint_ogc.pth", "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"),
        ("sam_vit_b_01ec64.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
    ]
    for filename, url in checkpoint_files:
        try:
            download_model_checkpoint(url, MODELS_DIR / filename)
        except Exception as exc:
            print(f"  ⚠ Failed to pre-download {filename}: {exc}")
    
    models_to_download = [
        ("PID Symbol Detector (YOLOv12)", PIDSymbolDetector),
        ("GroundingDINO Zero-shot Detector", GroundingDinoDetector),
        ("SAM2 Segmenter", SamSegmenter),
        ("BGE-M3 Embedder", BgeEmbedder),
        ("BGE-Reranker-v2", BgeReranker),
        ("GLiREL Relation Extractor", GLiRELRelationExtractor),
        ("BLINK Entity Linker", BlinkEntityLinker),
        ("GLiNER Entity Extractor", GlinerEntityExtractor),
    ]
    
    print("Initializing models...")
    print()
    
    successful = 0
    failed = 0
    
    for model_name, model_class in models_to_download:
        print(f"Loading: {model_name}...")
        try:
            model = model_class()
            ready = True
            if hasattr(model, "is_ready"):
                ready = bool(getattr(model, "is_ready", False))
            elif hasattr(model, "model"):
                ready = getattr(model, "model", None) is not None
            elif hasattr(model, "segmenter"):
                ready = getattr(model, "segmenter", None) is not None
            elif hasattr(model, "mask_generator"):
                ready = getattr(model, "mask_generator", None) is not None

            backend = getattr(model, "backend", None)

            if ready:
                if backend == "heuristic":
                    print(f"  ⚠ {model_name} initialized with heuristic fallback")
                elif backend == "fallback":
                    print(f"  ⚠ {model_name} initialized with fallback")
                else:
                    print(f"  ✓ {model_name} initialized successfully")
                successful += 1
            else:
                print(f"  ⚠ {model_name} initialized but is not ready")
                failed += 1
        except Exception as e:
            print(f"  ✗ {model_name} failed: {e}")
            failed += 1
        print()
    
    print("=" * 80)
    print(f"Results: {successful} successful, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("✓ All models downloaded and initialized successfully!")
        return 0
    else:
        print(f"✗ {failed} model(s) failed to initialize")
        return 1


if __name__ == "__main__":
    exit_code = download_all_models()
    sys.exit(exit_code)
