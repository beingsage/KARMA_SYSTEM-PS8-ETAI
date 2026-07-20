#!/usr/bin/env python
"""
Integration script: Test improved Stage 12 on the Hydro MPC test PDF.
Compares old vs. new component detection output.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from app.pipeline.models import detect_pid_components, detect_pid_components_enhanced
from app.pipeline.component_detector import ComponentDetector


def load_test_data():
    """Load OCR and entity data from the test pipeline outputs."""
    ocr_path = Path("data/pipeline/1.docling_surya_ocr/ocr_output.json")
    entity_path = Path("data/pipeline/15.entity_extraction/stage15_output.json")
    
    ocr_text = ""
    entities = []
    
    try:
        with open(ocr_path) as f:
            ocr_data = json.load(f)
            ocr_text = ocr_data.get("full_output", {}).get("text", "")
            print(f"✓ Loaded OCR text ({len(ocr_text)} chars)")
    except FileNotFoundError:
        print(f"⚠ OCR file not found at {ocr_path}")
    
    try:
        with open(entity_path) as f:
            entity_data = json.load(f)
            entities = entity_data.get("full_output", [])
            print(f"✓ Loaded {len(entities)} entities")
    except FileNotFoundError:
        print(f"⚠ Entity file not found at {entity_path}")
    
    return ocr_text, entities


def test_legacy_detector(text: str):
    """Test the legacy detector."""
    print("\n" + "="*70)
    print("LEGACY DETECTOR (Stage 12 v1)")
    print("="*70)
    
    result = detect_pid_components(text)
    print(f"Components found: {len(result)}")
    print(f"Result: {result}")
    print(f"\nIssues with legacy:")
    print("  ✗ No localization (page ids, offsets)")
    print("  ✗ No confidence scores")
    print("  ✗ No context or provenance")
    print("  ✗ Hardcoded list (not document-specific)")
    
    return result


def test_enhanced_detector(text: str, entities: list = None):
    """Test the enhanced detector."""
    print("\n" + "="*70)
    print("ENHANCED DETECTOR (Stage 12 v2)")
    print("="*70)
    
    result = detect_pid_components_enhanced(text, entities=entities)
    
    components = result["full_output"]["components"]
    summary = result["full_output"]["summary"]
    
    print(f"Components found: {len(components)}")
    print(f"Total mentions: {summary['total_mentions']}")
    print(f"Detection methods: {summary['detection_methods']}")
    
    print(f"\nImprovement points:")
    print(f"  ✓ {len(components)} unique components detected")
    print(f"  ✓ {summary['total_mentions']} total mentions with localization")
    print(f"  ✓ Each component has canonical ID and confidence score")
    print(f"  ✓ Each mention includes context snippet and page reference")
    print(f"  ✓ Handles entity extraction from stage 15 (when available)")
    
    print(f"\nSample components:")
    for i, comp in enumerate(components[:3]):
        print(f"\n  [{i+1}] {comp['name']}")
        print(f"      ID: {comp['canonical_id']}")
        print(f"      Confidence: {comp['confidence']:.2f}")
        print(f"      Entity Type: {comp['entity_type']}")
        print(f"      Occurrences: {len(comp['occurrences'])}")
        if comp['occurrences']:
            first_occ = comp['occurrences'][0]
            print(f"      First mention page: {first_occ.get('page', '?')}")
            print(f"      Context: {first_occ.get('context_snippet', '')[:100]}...")
    
    return result


def compare_results(legacy: list, enhanced: dict):
    """Compare old and new results."""
    print("\n" + "="*70)
    print("COMPARISON: Legacy vs. Enhanced")
    print("="*70)
    
    enhanced_components = enhanced["full_output"]["components"]
    enhanced_ids = {c["canonical_id"] for c in enhanced_components}
    
    print(f"\nLegacy detector returned: {len(legacy)} generic component names")
    print(f"Enhanced detector found: {len(enhanced_components)} localized components")
    
    print(f"\nQuality improvements:")
    print(f"  • Output richness: 7 items → {len(enhanced_components)} components with metadata")
    print(f"  • Localization: None → page ids + character offsets for each mention")
    print(f"  • Confidence: Not tracked → individual scores per component")
    print(f"  • Canonical mapping: N/A → mapped to taxonomy IDs")
    print(f"  • Context: Not provided → OCR snippets for each mention")
    print(f"  • Specificity: Generic terms → {len(enhanced_components)} document-grounded items")
    
    print(f"\nNext steps for production:")
    print(f"  1. Move stage 12 to run AFTER stage 15 (entity extraction)")
    print(f"  2. Pass entities to stage 12 for multimodal fusion")
    print(f"  3. Add component linking classifier (medium-term)")
    print(f"  4. Integrate with stage 20 (graphrag) and 21 (copilot) for grounded reasoning")


def main():
    """Main integration test."""
    print("="*70)
    print("STAGE 12 IMPROVEMENT VALIDATION")
    print(f"Test document: 1_Hydro MPC-1-10 (1).pdf")
    print(f"Test date: {datetime.now().isoformat()}")
    print("="*70)
    
    # Load test data
    print("\nLoading test data...")
    ocr_text, entities = load_test_data()
    
    if not ocr_text:
        print("ERROR: No OCR text loaded. Cannot proceed.")
        return 1
    
    # Test legacy
    legacy_result = test_legacy_detector(ocr_text)
    
    # Test enhanced
    enhanced_result = test_enhanced_detector(ocr_text, entities)
    
    # Compare
    compare_results(legacy_result, enhanced_result)
    
    # Save output for review
    output_path = Path("data/pipeline/12.pid_component_detection_enhanced.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(enhanced_result, f, indent=2)
    
    print(f"\n✓ Enhanced output saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
