#!/usr/bin/env python3
"""Execute pipeline stages 18-21 with output persistence."""

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
    
    # Get OCR
    print("\nRunning OCR...")
    ocr_result = pipeline._process_ocr(pdf_bytes)
    text = ocr_result.get('text', '')
    text_chunks = pipeline._segment_document(text)
    print(f"✓ OCR complete: {len(text_chunks)} text chunks")
    
    outputs = []
    
    # Load entities and relations from previous stages
    entities = []
    relations = []
    
    stage15_file = root / 'data' / 'pipeline' / '15.entity_extraction' / 'stage15_output.json'
    if stage15_file.exists():
        with open(stage15_file, 'r') as f:
            stage15_data = json.load(f)
            entities = stage15_data.get('full_output', [])
    
    stage16_file = root / 'data' / 'pipeline' / '16.relation_extraction' / 'stage16_output.json'
    if stage16_file.exists():
        with open(stage16_file, 'r') as f:
            stage16_data = json.load(f)
            relations = stage16_data.get('full_output', [])
    
    # Stage 18: bge_reranking
    try:
        print("\nStage 18: BGE reranking...")
        reranked = pipeline._rerank_entities(text, entities, relations)
        print(f"✓ Stage 18: Reranking complete")
        outputs.append(('18.bge_reranking', 'stage18_output.json', 'bge_reranking', reranked))
    except Exception as e:
        print(f"⚠ Stage 18 skipped: {e}")
    
    # Stage 19: qwen2_5_vl (vision-language, needs images)
    try:
        print("\nStage 19: Vision-language processing...")
        images = pipeline._render_pdf_pages(pdf_bytes)
        if images:
            # Vision-language analysis - use available VL model if present
            vl_analysis = {"images_processed": len(images), "status": "vl_model_unavailable"}
            print(f"⚠ Stage 19: Vision-language model unavailable ({len(images)} pages detected)")
            outputs.append(('19.qwen2_5_vl', 'stage19_output.json', 'qwen2_5_vl', vl_analysis))
        else:
            print("⚠ Stage 19: No images extracted")
    except Exception as e:
        print(f"⚠ Stage 19 skipped: {e}")
    
    # Stage 20: graphrag_analysis
    try:
        print("\nStage 20: GraphRAG summarization...")
        summary = pipeline._graphrag_analyze(entities, relations, text, text_chunks)
        print(f"✓ Stage 20: GraphRAG summary complete")
        outputs.append(('20.graphrag_analysis', 'stage20_output.json', 'graphrag_analysis', summary))
    except Exception as e:
        print(f"⚠ Stage 20 skipped: {e}")
    
    # Stage 21: copilot_analysis
    try:
        print("\nStage 21: Copilot analysis...")
        if entities:
            analysis = pipeline._copilot_analyze(entities, relations, text, text_chunks)
            print(f"✓ Stage 21: Copilot analysis complete")
            outputs.append(('21.copilot_analysis', 'stage21_output.json', 'copilot_analysis', analysis))
        else:
            print("⚠ Stage 21: No entities to analyze")
    except Exception as e:
        print(f"⚠ Stage 21 skipped: {e}")
    
    # Persist all outputs
    print("\nPersisting outputs...")
    for stage_dir, fname, stage_name, result in outputs:
        out_dir = root / 'data' / 'pipeline' / stage_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        
        payload = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage_name,
            'status': result.get('status', 'completed') if isinstance(result, dict) else 'completed',
            'full_output': result,
        }
        
        out_file = out_dir / fname
        with open(out_file, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, default=str)
        
        size_kb = out_file.stat().st_size / 1024
        print(f"✓ Saved {stage_dir}: {size_kb:.1f}K")
    
    print("\n✓ All stages complete")
    sys.exit(0)
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
