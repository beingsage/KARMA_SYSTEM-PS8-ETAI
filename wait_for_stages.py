#!/usr/bin/env python3
"""Wait for stages 15-17 to complete with simple polling."""

import asyncio
import json
from pathlib import Path
from datetime import datetime
import time

async def main():
    """Poll for output files."""
    root = Path.cwd()
    
    # Wait for output files to appear
    expected_files = [
        ('15.entity_extraction', 'data/pipeline/15.entity_extraction/stage15_output.json'),
        ('16.relation_extraction', 'data/pipeline/16.relation_extraction/stage16_output.json'),
        ('17.entity_linking', 'data/pipeline/17.entity_linking/stage17_output.json'),
    ]
    
    print("Waiting for stage output files...")
    max_wait = 2400  # 40 minutes
    start_time = time.time()
    
    found = set()
    while len(found) < len(expected_files) and (time.time() - start_time) < max_wait:
        for stage_name, file_path in expected_files:
            full_path = root / file_path
            if stage_name not in found and full_path.exists():
                size_kb = full_path.stat().st_size / 1024
                print(f"✓ {stage_name}: {size_kb:.1f}K")
                found.add(stage_name)
        
        if len(found) < len(expected_files):
            await asyncio.sleep(5)
    
    if len(found) == len(expected_files):
        print(f"\n✓ All stage outputs complete")
        return 0
    else:
        print(f"\n✗ Timeout waiting for outputs. Found {len(found)}/{len(expected_files)}")
        for stage_name, file_path in expected_files:
            full_path = root / file_path
            print(f"  {stage_name}: {'✓' if full_path.exists() else '✗'}")
        return 1

if __name__ == '__main__':
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
