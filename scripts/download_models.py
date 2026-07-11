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
)


def download_all_models():
    """Download and initialize all models for the pipeline."""
    
    print("=" * 80)
    print("Industrial PDF-to-Graph Pipeline - Model Download Script")
    print("=" * 80)
    print()
    
    models_to_download = [
        ("PID Symbol Detector (YOLOv12)", PIDSymbolDetector),
        ("GroundingDINO Zero-shot Detector", GroundingDinoDetector),
        ("SAM2 Segmenter", SamSegmenter),
        ("BGE-M3 Embedder", BgeEmbedder),
        ("BGE-Reranker-v2", BgeReranker),
        ("GLiREL Relation Extractor", GLiRELRelationExtractor),
        ("BLINK Entity Linker", BlinkEntityLinker),
    ]
    
    print("Initializing models...")
    print()
    
    successful = 0
    failed = 0
    
    for model_name, model_class in models_to_download:
        print(f"Loading: {model_name}...")
        try:
            if model_class == BgeEmbedder:
                model = model_class()
            elif model_class == BgeReranker:
                model = model_class()
            elif model_class == GLiRELRelationExtractor:
                model = model_class()
            elif model_class == BlinkEntityLinker:
                model = model_class()
            else:
                model = model_class()
            
            print(f"  ✓ {model_name} initialized successfully")
            successful += 1
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
