# Industrial PDF-to-Graph Pipeline - Model Integration Guide

This document describes the complete end-to-end implementation of all AI models in the industrial PDF-to-graph pipeline.

## Architecture Overview

```
PDF Input
    │
    ├─► OCR & Document Processing (Docling, Surya)
    │
    ├─► Vision Models
    │   ├─► YOLOv12 (P&ID Symbol Detection)
    │   ├─► GroundingDINO (Zero-shot Object Detection)
    │   └─► SAM2 (Segmentation)
    │
    ├─► Text Processing
    │   ├─► Document Segmentation
    │   └─► Semantic Indexing (BGE-M3)
    │
    ├─► Entity & Relation Extraction
    │   ├─► GLiNER (Entity Extraction)
    │   ├─► GLiREL/REBEL (Relation Extraction)
    │   └─► BLINK (Entity Linking)
    │
    ├─► Semantic Search & Ranking
    │   ├─► BGE-M3 (Embeddings)
    │   └─► BGE-Reranker-v2 (Relevance Ranking)
    │
    └─► Knowledge Graph Generation
        └─► Neo4j Storage
```

## Model Integration Details

### 1. P&ID Symbol Detection (YOLOv12)
**Location:** `app/pipeline/model_helpers.py::PIDSymbolDetector`  
**Pipeline Stage:** `pid_symbol_detection`  
**Config:** `app/config.py`

**Configuration:**
```python
pid_yolo_weights: str  # Path to custom YOLOv12 weights (optional)
pid_yolo_model_name: str = "yolov8n.pt"  # Default model name
```

**Usage:**
```python
detector = PIDSymbolDetector()
symbols = detector.detect([image1, image2, ...])
# Returns: List of detected symbols with bboxes and confidence scores
```

**Integration in Pipeline:**
- Called in `_detect_pid_symbols()` method
- Input: PDF bytes converted to images
- Output: List of detected industrial symbols with coordinates
- Used for identifying P&ID components in engineering drawings

---

### 2. Zero-shot Object Detection (GroundingDINO)
**Location:** `app/pipeline/model_helpers.py::GroundingDinoDetector`  
**Pipeline Stage:** `groundingdino_detection`  
**Model:** GroundingDINO SwinT OGC

**Features:**
- Zero-shot detection (doesn't require training data)
- Natural language prompts for object detection
- Default prompt for industrial equipment

**Usage:**
```python
detector = GroundingDinoDetector()
detections = detector.detect(
    images=[...],
    prompt="industrial equipment, component, valve, pump, sensor"
)
# Returns: List of detections with labels, bboxes, and confidence
```

**Integration in Pipeline:**
- Called in `_detect_groundingdino_objects()` method
- Complements YOLO for more generic object detection
- Useful for detecting various industrial components

---

### 3. Segmentation (SAM2)
**Location:** `app/pipeline/model_helpers.py::SamSegmenter`  
**Pipeline Stage:** `sam2_segmentation`  
**Model:** SAM ViT-B

**Features:**
- Automatic mask generation for all objects in image
- Provides instance segmentation masks
- Calculates mask quality scores

**Usage:**
```python
segmenter = SamSegmenter()
masks = segmenter.segment(image)
# Returns: List of masks with bbox, area, and quality scores
```

**Integration in Pipeline:**
- Called in `_segment_with_sam()` method
- Provides pixel-level understanding of image regions
- Used after GroundingDINO detection for detailed segmentation

---

### 4. Entity Extraction (GLiNER)
**Location:** `app/pipeline/entity_extractor.py::GlinerEntityExtractor`  
**Pipeline Stage:** `entity_extraction`  
**Models:** 
- Fine-tuned: `models/gliner-industrial-v1` (if available)
- Base: `urchade/gliner_medium-v2.1`

**Features:**
- Named entity recognition optimized for industrial domain
- Pre-defined entity types: equipment, process, parameter, material, control_system, location, failure_mode, maintenance
- Fallback to heuristic extraction if model unavailable

**Industrial Entity Types:**
```
- equipment: Pumps, motors, compressors, valves, sensors
- process: Distillation, compression, heating, cooling
- parameter: Pressure, temperature, flow rate, viscosity
- material: Oil, water, gas, chemical compound
- control_system: PLC, SCADA, HMI, DCS
- location: Reactor, vessel, pipeline, heat exchanger
- failure_mode: Cavitation, corrosion, fouling
- maintenance: Inspection, repair, replacement, calibration
```

**Usage:**
```python
extractor = GlinerEntityExtractor()
entities = extractor.extract(text, threshold=0.3)
# Returns: List of entities with name, type, confidence, canonical_name
```

**Integration in Pipeline:**
- Called in `_extract_entities()` method
- Processes normalized text chunks (350 chars max)
- Output: List of entities for downstream processing

---

### 5. Relation Extraction (GLiREL + REBEL Fallback)
**Location:** `app/pipeline/relation_extractor.py::GLiRELRelationExtractor`  
**Pipeline Stage:** `relation_extraction`
**Models:**
- Primary: `jackboyla/glirel-base` (GLiREL)
- Fallback: `Babelscape/rebel-large` (REBEL)

**Features:**
- Extracts relationships between entities
- Industrial relation types: connected_to, controls, measures, receives_input_from, sends_output_to, is_part_of, operates_at, related_to
- Fallback to heuristic extraction (entity co-occurrence)

**Usage:**
```python
extractor = GLiRELRelationExtractor()
relations = extractor.extract(text, entities, schema=['connected_to', 'controls', ...])
# Returns: List of relations with source, target, relation_type, confidence
```

**Integration in Pipeline:**
- Called in `_extract_relations()` method
- Uses extracted entities to find relationships
- Supports both GLiREL (primary) and REBEL (fallback)

---

### 6. Entity Linking (BLINK)
**Location:** `app/pipeline/model_helpers.py::BlinkEntityLinker`  
**Entity Linker:** `app/pipeline/entity_linker.py::BlinkEntityLinker`  
**Pipeline Stage:** `entity_linking`
**Model:** facebook/blink-base-uncased (fallback to embeddings)

**Features:**
- Links entities to knowledge base identifiers
- Disambiguates entity names
- Fallback to sentence-transformers embeddings

**Usage:**
```python
linker = BlinkEntityLinker()
linked_entities = linker.link_entities(entities)
# Returns: Entities with linked_id, link_confidence, link_source
```

**Integration in Pipeline:**
- Called in `_link_entities()` method
- Creates canonical entity IDs for knowledge graph
- Enables entity resolution across documents

---

### 7. Embedding Model (BGE-M3)
**Location:** `app/pipeline/model_helpers.py::BgeEmbedder`  
**Pipeline Stage:** `semantic_indexing`
**Model:** `BAAI/bge-m3`

**Features:**
- Dense embedding generation
- Supports multiple backends (sentence-transformers, transformers)
- Batch encoding support
- Multilingual support

**Usage:**
```python
embedder = BgeEmbedder()
embedding = embedder.encode("text chunk")
embeddings_batch = embedder.encode_batch(["text1", "text2", ...])
# Returns: List of float vectors (1024 dimensions for BGE-M3)
```

**Integration in Pipeline:**
- Called in `_index_text_chunks()` method
- Creates semantic index for all text chunks
- Enables semantic similarity search

**Vector Dimensions:** 1024 (BGE-M3)

---

### 8. Reranker Model (BGE-Reranker-v2)
**Location:** `app/pipeline/model_helpers.py::BgeReranker`  
**Pipeline Stage:** `bge_reranking`
**Model:** `BAAI/bge-reranker-v2`

**Features:**
- Relevance scoring for query-document pairs
- Ranking functionality
- Cross-encoder architecture

**Usage:**
```python
reranker = BgeReranker()
score = reranker.score_pair(query, candidate)
ranked_results = reranker.rank_candidates(query, [candidate1, candidate2, ...])
# Returns: Sorted list of candidates by relevance score
```

**Integration in Pipeline:**
- Called in `_rerank_entities()` method
- Ranks entities and relations by relevance
- Scores top entities/relations against document text

---

## Configuration

### Environment Variables

```bash
# P&ID Detection
export PID_YOLO_WEIGHTS=""  # Path to custom weights
export PID_YOLO_MODEL="yolov8n.pt"  # Model name/path

# Segmentation
export SAM_MODEL_TYPE="vit_b"  # vit_b, vit_l, vit_h
export SAM_MODEL_NAME="sam_vit_b_01ec64.pth"

# Entity Extraction
export GLINER_MODEL="urchade/gliner_medium-v2.1"
export GLINER_FINETUNED="models/gliner-industrial-v1"

# Relation Extraction
export GLIREL_MODEL="jackboyla/glirel-base"

# Entity Linking
export BLINK_MODEL="facebook/blink-base-uncased"

# Embeddings & Reranking
export EMBEDDING_MODEL="BAAI/bge-m3"
export RERANKER_MODEL="BAAI/bge-reranker-v2"
```

### Python Configuration

Edit `app/config.py`:
```python
from app.config import settings

# All settings are accessible via settings object
print(settings.embedding_model)  # "BAAI/bge-m3"
print(settings.pid_yolo_model_name)  # "yolov8n.pt"
```

## Model Initialization Flow

1. **Pipeline Initialization** (`engine_v2.py::IndustrialGraphPipeline.__init__`)
   - Initializes all models in `_initialize_all_models()`
   - Prints status for each component
   - Gracefully handles missing models

2. **Model Order:**
   - OCR Processor
   - Entity Extractor (GLiNER)
   - Relation Extractor (GLiREL/REBEL)
   - PID Symbol Detector (YOLOv12)
   - GroundingDINO Detector
   - SAM2 Segmenter
   - BGE Embedder
   - BGE Reranker
   - BLINK Entity Linker
   - Graph Store (Neo4j)
   - RAG Summarizer
   - Copilot Agent

## Pipeline Execution Flow

```
PDF Input
    ↓
[OCR & Layout Analysis]
    ↓
[Document Segmentation] → [Semantic Indexing with BGE-M3]
    ↓
[Entity Extraction with GLiNER]
    ↓
[Relation Extraction with GLiREL/REBEL]
    ↓
[Entity Linking with BLINK]
    ↓
[Image Processing] → [YOLOv12 + GroundingDINO + SAM2]
    ↓
[BGE Reranking]
    ↓
[Knowledge Graph Generation]
    ↓
[Neo4j Storage]
    ↓
Results
```

## Model Downloads

All models are automatically downloaded on first use. To pre-download all models:

```bash
python scripts/download_models.py
```

This script:
- Downloads all model checkpoints
- Initializes all components
- Reports success/failure for each model
- Creates necessary directories

## Performance Considerations

### Memory Requirements
- YOLOv8/v12: ~100MB
- GroundingDINO: ~500MB
- SAM2: ~400MB
- GLiNER: ~600MB
- BGE-M3: ~450MB
- BLINK: ~500MB
- **Total: ~2.5GB GPU memory recommended**

### Inference Times (Approximate)
- YOLOv12: 50-100ms per image
- GroundingDINO: 200-400ms per image
- SAM2: 300-600ms per image
- GLiNER: 10-50ms per text chunk
- GLiREL: 50-200ms per relation pair
- BGE-M3 embedding: 5-20ms per chunk
- BGE Reranking: 10-30ms per pair

## Troubleshooting

### Out of Memory
- Use smaller SAM model: `export SAM_MODEL_TYPE="vit_b"`
- Reduce batch sizes in configuration
- Run on GPU with more memory

### Slow Inference
- Check GPU utilization: `nvidia-smi`
- Models use CPU if CUDA not available
- Install PyTorch with CUDA support

### Model Download Failures
- Check internet connection
- Verify disk space in `models/` directory
- Manually download and place in `models/` folder

## References

- **YOLOv12:** https://github.com/ultralytics/ultralytics
- **GroundingDINO:** https://github.com/IDEA-Research/GroundingDINO
- **SAM2:** https://github.com/facebookresearch/segment-anything
- **GLiNER:** https://github.com/urchade/GLiNER
- **GLiREL:** https://github.com/jackboyla/glirel
- **BLINK:** https://github.com/facebookresearch/BLINK
- **BGE-M3:** https://huggingface.co/BAAI/bge-m3
- **BGE-Reranker-v2:** https://huggingface.co/BAAI/bge-reranker-v2
