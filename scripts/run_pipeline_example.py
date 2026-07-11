"""
Industrial PDF-to-Graph Pipeline - Integration Guide

This module demonstrates the complete end-to-end pipeline with all models integrated:
1. P&ID Symbol Detection (YOLOv12)
2. Zero-shot Object Detection (GroundingDINO)
3. Segmentation (SAM2)
4. Entity Extraction (GLiNER)
5. Relation Extraction (GLiREL + heuristic fallback)
6. Entity Linking (BLINK)
7. Embeddings (BGE-M3)
8. Reranking (BGE-Reranker-v2)
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

# Example imports
from app.pipeline.engine_v2 import IndustrialGraphPipeline


async def run_full_pipeline(pdf_path: str) -> Dict[str, Any]:
    """
    Run the complete industrial PDF-to-graph pipeline.
    
    Args:
        pdf_path: Path to the PDF file to process
        
    Returns:
        Dictionary containing pipeline results with all stages
    """
    
    print("=" * 80)
    print("Industrial PDF-to-Graph Pipeline - End-to-End Execution")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    print("Initializing pipeline with all models...")
    pipeline = IndustrialGraphPipeline()
    
    print(f"Pipeline mode: {pipeline.model_mode}")
    print(f"Status: {pipeline.stage_status if pipeline.stage_status else 'Ready'}")
    print()
    
    # Read PDF
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    # Run pipeline
    print(f"Processing PDF: {pdf_path.name}")
    print(f"File size: {len(pdf_bytes) / 1024 / 1024:.2f} MB")
    print()
    
    result = await pipeline.run(pdf_path.name, pdf_bytes)
    
    print()
    print("=" * 80)
    print("Pipeline Execution Summary")
    print("=" * 80)
    print()
    
    # Display results
    print(f"Job ID: {result.get('job_id', 'N/A')}")
    print(f"Status: {result.get('status', 'N/A')}")
    print(f"Message: {result.get('message', 'N/A')}")
    print()
    
    # Model outputs summary
    model_outputs = result.get("model_outputs", {})
    if model_outputs:
        print("Model Outputs:")
        for key, value in model_outputs.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, (list, dict)):
                        if isinstance(v, list):
                            print(f"    {k}: {len(v)} items")
                        else:
                            print(f"    {k}: {len(v)} keys")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    print()
    
    # Graph components
    print("Knowledge Graph Components:")
    print(f"  Entities: {len(model_outputs.get('entities', []))} extracted")
    print(f"  Relations: {len(model_outputs.get('relations', []))} identified")
    print(f"  Linked entities: {len(model_outputs.get('resolved_entities', []))} linked")
    print()
    
    # Vision outputs
    pid_symbols = model_outputs.get("pid_symbol_insights", {})
    if pid_symbols and pid_symbols.get("count", 0) > 0:
        print(f"  P&ID Symbols: {pid_symbols.get('count', 0)} detected ({pid_symbols.get('source', 'N/A')})")
    
    groundingdino = model_outputs.get("groundingdino_info", {})
    if groundingdino and groundingdino.get("count", 0) > 0:
        print(f"  GroundingDINO detections: {groundingdino.get('count', 0)}")
    
    sam_segments = model_outputs.get("sam_segmentation_info", {})
    if sam_segments and sam_segments.get("count", 0) > 0:
        print(f"  SAM2 segments: {sam_segments.get('count', 0)}")
    print()
    
    # Semantic search
    bge_ranking = model_outputs.get("bge_ranking", {})
    if bge_ranking and bge_ranking.get("ranked"):
        print(f"  BGE-Reranked candidates: {len(bge_ranking.get('ranked', []))}")
    print()
    
    # Semantic indexing
    semantic_index = model_outputs.get("semantic_index", {})
    if semantic_index:
        print(f"  Semantic index: {semantic_index.get('indexed_chunks', 0)} chunks indexed")
    print()
    
    return result


def display_entities(result: Dict[str, Any]):
    """Display extracted entities."""
    entities = result.get("model_outputs", {}).get("entities", [])
    
    if not entities:
        print("No entities extracted")
        return
    
    print("Extracted Entities:")
    print("-" * 80)
    
    for entity in entities[:10]:  # Show first 10
        print(f"  Name: {entity.get('name', 'N/A')}")
        print(f"    Type: {entity.get('entity_type', 'N/A')}")
        print(f"    Confidence: {entity.get('confidence', 0):.2f}")
        print(f"    Canonical: {entity.get('canonical_name', 'N/A')}")
        print()


def display_relations(result: Dict[str, Any]):
    """Display extracted relations."""
    relations = result.get("model_outputs", {}).get("relations", [])
    
    if not relations:
        print("No relations extracted")
        return
    
    print("Extracted Relations:")
    print("-" * 80)
    
    for relation in relations[:10]:  # Show first 10
        source = relation.get('source', 'N/A')
        target = relation.get('target', 'N/A')
        rel_type = relation.get('relation_type', 'N/A')
        confidence = relation.get('confidence', 0)
        
        print(f"  {source} -[{rel_type}]-> {target}")
        print(f"    Confidence: {confidence:.2f} (via {relation.get('source_method', 'N/A')})")
        print()


def save_results_to_file(result: Dict[str, Any], output_path: str = "pipeline_results.json"):
    """Save pipeline results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")


async def main():
    """Main entry point for testing the pipeline."""
    
    # Example usage with a test PDF
    test_pdf_path = "data/sample.pdf"
    
    try:
        result = await run_full_pipeline(test_pdf_path)
        
        # Display detailed results
        print()
        display_entities(result)
        print()
        display_relations(result)
        
        # Save results
        save_results_to_file(result)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please provide a valid PDF file path")
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
