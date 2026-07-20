#!/usr/bin/env python3
"""Execute pipeline stages 15-17 with output persistence."""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

async def main():
    """Run stages 15-17 and persist outputs."""
    try:
        from app.pipeline.engine_v2 import get_pipeline
        
        root = Path.cwd()
        pdf_path = root / '1_Hydro MPC-1-10 (1).pdf'
        
        if not pdf_path.exists():
            print(f"ERROR: PDF not found at {pdf_path}")
            return 1
        
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        print("Initializing pipeline...")
        pipeline = get_pipeline()
        print("✓ Pipeline initialized")
        
        # Get OCR text
        ocr_result = await pipeline._run_stage(
            'docling_surya_ocr', 
            pipeline._process_ocr, 
            required=True, 
            pdf_bytes=pdf_bytes
        )
        text = ocr_result.get('text', '')
        print(f"✓ OCR complete: {len(text)} chars")
        
        outputs = []
        
        # Stage 15: entity_extraction
        try:
            print("\nStage 15: Entity extraction...")
            entities = await pipeline._run_stage(
                'entity_extraction',
                pipeline._extract_entities,
                required=True,
                text=text
            )
            print(f"✓ Stage 15: {len(entities)} entities extracted")
            outputs.append(('15.entity_extraction', 'stage15_output.json', 'entity_extraction', entities))
        except Exception as e:
            print(f"✗ Stage 15 failed: {e}")
            return 1
        
        # Stage 16: relation_extraction
        try:
            print("\nStage 16: Relation extraction...")
            relations = await pipeline._run_stage(
                'relation_extraction',
                pipeline._extract_relations,
                required=False,
                text=text,
                entities=entities
            )
            print(f"✓ Stage 16: {len(relations)} relations extracted")
            outputs.append(('16.relation_extraction', 'stage16_output.json', 'relation_extraction', relations))
        except Exception as e:
            print(f"⚠ Stage 16 skipped: {e}")
        
        # Stage 17: entity_linking
        try:
            print("\nStage 17: Entity linking...")
            resolved_entities = await pipeline._run_stage(
                'entity_linking',
                pipeline._link_entities,
                required=False,
                entities=entities
            )
            print(f"✓ Stage 17: {len(resolved_entities)} entities linked")
            outputs.append(('17.entity_linking', 'stage17_output.json', 'entity_linking', resolved_entities))
        except Exception as e:
            print(f"⚠ Stage 17 skipped: {e}")
        
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
        return 0
        
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
