#!/usr/bin/env python3
"""Simple direct test of stages 15-17."""

import sys
import json
from pathlib import Path
from datetime import datetime

try:
    from app.pipeline.engine_v2 import get_pipeline
    
    root = Path.cwd()
    pdf_path = root / '1_Hydro MPC-1-10 (1).pdf'
    
    print("Loading pipeline...")
    pipeline = get_pipeline()
    print("✓ Pipeline loaded")
    
    # Read PDF
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    print(f"✓ PDF loaded: {len(pdf_bytes)} bytes")
    
    # Get OCR
    print("\nRunning OCR...")
    ocr_result = pipeline._process_ocr(pdf_bytes)
    text = ocr_result.get('text', '')
    print(f"✓ OCR complete: {len(text)} chars")
    
    # Stage 15
    print("\nStage 15: Entity extraction...")
    entities = pipeline._extract_entities(text)
    print(f"✓ Stage 15: {len(entities)} entities")
    print(f"  Type: {type(entities)}")
    if entities and len(entities) > 0:
        print(f"  Sample: {entities[0]}")
    
    # Persist stage 15
    out_dir = root / 'data' / 'pipeline' / '15.entity_extraction'
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'timestamp': datetime.now().isoformat(),
        'stage': 'entity_extraction',
        'status': 'completed',
        'full_output': entities,
    }
    out_file = out_dir / 'stage15_output.json'
    with open(out_file, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"✓ Saved: {out_file.stat().st_size / 1024:.1f}K")
    
    # Stage 16
    print("\nStage 16: Relation extraction...")
    relations = pipeline._extract_relations(text, entities)
    print(f"✓ Stage 16: {len(relations)} relations")
    print(f"  Type: {type(relations)}")
    
    # Persist stage 16
    out_dir = root / 'data' / 'pipeline' / '16.relation_extraction'
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'timestamp': datetime.now().isoformat(),
        'stage': 'relation_extraction',
        'status': 'completed',
        'full_output': relations,
    }
    out_file = out_dir / 'stage16_output.json'
    with open(out_file, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"✓ Saved: {out_file.stat().st_size / 1024:.1f}K")
    
    # Stage 17
    print("\nStage 17: Entity linking...")
    linked_entities = pipeline._link_entities(entities)
    print(f"✓ Stage 17: {len(linked_entities)} linked entities")
    print(f"  Type: {type(linked_entities)}")
    
    # Persist stage 17
    out_dir = root / 'data' / 'pipeline' / '17.entity_linking'
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'timestamp': datetime.now().isoformat(),
        'stage': 'entity_linking',
        'status': 'completed',
        'full_output': linked_entities,
    }
    out_file = out_dir / 'stage17_output.json'
    with open(out_file, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"✓ Saved: {out_file.stat().st_size / 1024:.1f}K")
    
    print("\n✓ All stages complete")
    sys.exit(0)
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
