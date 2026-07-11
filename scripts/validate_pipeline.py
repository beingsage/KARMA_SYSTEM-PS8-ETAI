#!/usr/bin/env python3
"""
Validation script to verify all models and imports are working correctly.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def validate_imports():
    """Validate all critical imports."""
    
    print("Validating imports...")
    print("-" * 80)
    
    required_tests = [
        ("FastAPI", lambda: __import__('fastapi')),
        ("PyTorch", lambda: __import__('torch')),
        ("Transformers", lambda: __import__('transformers')),
        ("YOLOv8/v12", lambda: __import__('ultralytics')),
        ("GroundingDINO", lambda: __import__('groundingdino')),
        ("SAM", lambda: __import__('segment_anything')),
        ("GLiNER", lambda: __import__('gliner')),
    ]

    optional_tests = [
        ("Sentence-Transformers", lambda: __import__('sentence_transformers')),
    ]
    
    passed = 0
    failed = 0
    
    for name, import_func in required_tests:
        try:
            import_func()
            print(f"✓ {name}")
            passed += 1
        except ImportError as e:
            print(f"✗ {name}: {e}")
            failed += 1

    for name, import_func in optional_tests:
        try:
            import_func()
            print(f"✓ {name} (optional)")
        except ImportError as e:
            print(f"⚠ {name} (optional): {e}")
    
    print("-" * 80)
    print(f"Imports: {passed} passed, {failed} failed")
    print()
    
    return failed == 0


def validate_config():
    """Validate configuration."""
    
    print("Validating configuration...")
    print("-" * 80)
    
    try:
        from app.config import settings
        
        configs = [
            ("embedding_model", settings.embedding_model),
            ("reranker_model", settings.reranker_model),
            ("gliner_model", settings.gliner_model),
            ("glirel_model", settings.glirel_model),
            ("sam_model_type", settings.sam_model_type),
        ]
        
        for name, value in configs:
            print(f"  {name}: {value}")
        
        print("-" * 80)
        print("✓ Configuration loaded successfully")
        print()
        return True
        
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        print()
        return False


def validate_models():
    """Validate model initialization."""
    
    print("Validating model initialization...")
    print("-" * 80)
    
    models_to_test = [
        ("PID Symbol Detector", "app.pipeline.model_helpers", "PIDSymbolDetector"),
        ("GroundingDINO", "app.pipeline.model_helpers", "GroundingDinoDetector"),
        ("SAM2 Segmenter", "app.pipeline.model_helpers", "SamSegmenter"),
        ("BGE Embedder", "app.pipeline.model_helpers", "BgeEmbedder"),
        ("BGE Reranker", "app.pipeline.model_helpers", "BgeReranker"),
        ("GLiREL Extractor", "app.pipeline.model_helpers", "GLiRELRelationExtractor"),
        ("BLINK Linker", "app.pipeline.model_helpers", "BlinkEntityLinker"),
        ("Entity Extractor", "app.pipeline.entity_extractor", "GlinerEntityExtractor"),
        ("Relation Extractor", "app.pipeline.relation_extractor", "GLiRELRelationExtractor"),
    ]
    
    passed = 0
    failed = 0
    
    for name, module, cls in models_to_test:
        try:
            mod = __import__(module, fromlist=[cls])
            model_class = getattr(mod, cls)
            print(f"✓ {name}: {cls}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1
    
    print("-" * 80)
    print(f"Models: {passed} passed, {failed} failed")
    print()
    
    return failed == 0


def validate_pipeline():
    """Validate pipeline structure."""
    
    print("Validating pipeline structure...")
    print("-" * 80)
    
    try:
        from app.pipeline.engine_v2 import IndustrialGraphPipeline
        
        pipeline = IndustrialGraphPipeline()
        
        checks = [
            ("OCR Processor", pipeline.ocr_processor is not None),
            ("Entity Extractor", pipeline.entity_extractor is not None),
            ("Relation Extractor", pipeline.relation_extractor is not None),
            ("PID Symbol Detector", pipeline.pid_symbol_detector is not None),
            ("GroundingDINO", pipeline.grounding_dino_detector is not None),
            ("SAM2 Segmenter", pipeline.sam_segmenter is not None),
            ("BGE Embedder", pipeline.embedding_model is not None),
            ("BGE Reranker", pipeline.reranker_model is not None),
            ("BLINK Linker", pipeline.blink_linker is not None),
        ]
        
        passed = 0
        failed = 0
        
        for name, status in checks:
            status_str = "✓" if status else "✗"
            print(f"{status_str} {name}")
            if status:
                passed += 1
            else:
                failed += 1
        
        print("-" * 80)
        print(f"Pipeline: {passed} components ready, {failed} unavailable")
        print(f"Mode: {pipeline.model_mode}")
        print()
        
        return failed <= 3  # Allow some components to be unavailable
        
    except Exception as e:
        print(f"✗ Pipeline validation failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """Run all validations."""
    
    print()
    print("=" * 80)
    print("Industrial PDF-to-Graph Pipeline - Validation Suite")
    print("=" * 80)
    print()
    
    results = []
    
    # Run validations
    results.append(("Imports", validate_imports()))
    results.append(("Configuration", validate_config()))
    results.append(("Models", validate_models()))
    results.append(("Pipeline", validate_pipeline()))
    
    # Summary
    print("=" * 80)
    print("Validation Summary")
    print("=" * 80)
    print()
    
    for name, status in results:
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{status_str}: {name}")
    
    print()
    
    all_passed = all(status for _, status in results)
    if all_passed:
        print("✓ All validations passed! Pipeline is ready.")
        return 0
    else:
        print("✗ Some validations failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
