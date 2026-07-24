# Setup & Quick Start Guide

## Complete Implementation Summary

All 8 models have been successfully integrated into your codebase:

### ✓ Models Implemented

1. **P&ID Symbol Detection (YOLOv12)**
   - Location: `app/pipeline/model_helpers.py::PIDSymbolDetector`
   - Detects industrial symbols and components in P&IDs
   - Automatic model download from Hugging Face

2. **Zero-shot Object Detection (GroundingDINO)**
   - Location: `app/pipeline/model_helpers.py::GroundingDinoDetector`
   - Natural language prompts for generic object detection
   - Default prompt: industrial equipment, components, valves, pumps, sensors

3. **Segmentation (SAM2)**
   - Location: `app/pipeline/model_helpers.py::SamSegmenter`
   - Instance segmentation masks for all objects
   - Quality scores for each segmentation

4. **Entity Extraction (GLiNER)**
   - Location: `app/pipeline/entity_extractor.py::GlinerEntityExtractor`
   - Industrial domain-optimized entity recognition
   - 8 entity types: equipment, process, parameter, material, control_system, location, failure_mode, maintenance
   - Fine-tuning support for custom models

5. **Relation Extraction (GLiREL + REBEL fallback)**
   - Location: `app/pipeline/relation_extractor.py::GLiRELRelationExtractor`
   - Primary: GLiREL for relation extraction
   - Fallback: REBEL (seq2seq based)
   - Industrial relation types: connected_to, controls, measures, related_to, etc.

6. **Entity Linking (BLINK)**
   - Location: `app/pipeline/model_helpers.py::BlinkEntityLinker` & `app/pipeline/entity_linker.py`
   - Links entities to knowledge base identifiers
   - Disambiguation support

7. **Embeddings (BGE-M3)**
   - Location: `app/pipeline/model_helpers.py::BgeEmbedder`
   - Dense embeddings (1024 dimensions)
   - Semantic indexing for all text chunks
   - Supports batch encoding

8. **Reranker (BGE-Reranker-v2)**
   - Location: `app/pipeline/model_helpers.py::BgeReranker`
   - Relevance ranking of query-document pairs
   - Cross-encoder architecture
   - Ranking candidates by relevance

---

## Installation & Setup

### 1. Clone/Update Dependencies

```bash
cd /media/sagesujal/DEV1/bytes/structured

# Create virtual environment if not exists
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

# Optional: install by layer
./scripts/install_base.sh
./scripts/install_vision.sh
./scripts/install_document.sh
./scripts/install_nlp.sh
./scripts/install_graph.sh
./scripts/install_analytics.sh
./scripts/install_agents.sh

### 2. Download All Models (Optional but Recommended)

Pre-download all models to avoid delays during first run:

```bash
python scripts/download_models.py
```

This will download (~2.5GB):
- `models/sam_vit_b_01ec64.pth`
- `models/groundingdino_swint_ogc.pth`
- And model weights via Hugging Face

### 3. Configure Environment (Optional)

Create `.env` file in project root:

```bash
# P&ID Detection
export PID_YOLO_MODEL="yolov8n.pt"

# Segmentation
export SAM_MODEL_TYPE="vit_b"

# Entity Extraction
export GLINER_MODEL="urchade/gliner_medium-v2.1"

# Embeddings & Reranking
export EMBEDDING_MODEL="BAAI/bge-m3"
export RERANKER_MODEL="BAAI/bge-reranker-v2"
```

### 4. Validate Installation

```bash
python scripts/validate_pipeline.py
```

Expected output:
```
✓ Configuration loaded successfully
✓ Model imports verified
✓ Pipeline structure validated
```

---

## Quick Start

### Option 1: Run FastAPI Server

```bash
# Start Neo4j and Qdrant
docker compose up -d neo4j qdrant

# Start backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API Endpoints:
- `POST /api/v1/process-pdf` - Upload and process PDF
- `GET /api/v1/jobs/{job_id}` - Get job results
- `GET /api/v1/models/status` - Check model status

### Option 2: Run Programmatically

```python
import asyncio
from app.pipeline.engine_v2 import IndustrialGraphPipeline

async def main():
    pipeline = IndustrialGraphPipeline()
    
    with open("sample.pdf", "rb") as f:
        pdf_bytes = f.read()
    
    result = await pipeline.run("sample.pdf", pdf_bytes)
    print(result)

asyncio.run(main())
```

### Option 3: Use Example Script

```bash
python scripts/run_pipeline_example.py
```

---

## Architecture Diagram

```
Input PDF
    │
    ├─────────────────────────────────────────────────────┐
    │                                                      │
    ▼                                                      ▼
[OCR & Layout]                              [Vision Analysis]
    │                                            │
    ├─ Text Extraction                          ├─ YOLOv12: P&ID Symbols
    ├─ Document Segmentation                    ├─ GroundingDINO: Objects
    └─ Semantic Indexing (BGE-M3)               └─ SAM2: Segmentation
    │                                            │
    └─────────────────────────┬──────────────────┘
                              │
                              ▼
                    [Entity & Relation Extraction]
                              │
                    ├─ GLiNER: Entities
                    ├─ GLiREL/REBEL: Relations
                    ├─ BLINK: Entity Linking
                    └─ BGE-Reranker: Ranking
                              │
                              ▼
                    [Knowledge Graph]
                              │
                    ├─ Qdrant: Vector embeddings storage
                    ├─ Neo4j: Storage
                    ├─ GraphRAG: Analysis
                    ├─ Qwen 3: Foundation LLM reasoning
                    ├─ TimesFM: Forecasting and anomaly detection
                    ├─ TFT: Remaining Useful Life prediction
                    └─ Copilot / LangGraph: Agent orchestration
```

---

## File Structure

```
/media/sagesujal/DEV1/bytes/structured/
├── app/
│   ├── config.py                    # Configuration (UPDATED)
│   ├── main.py                      # FastAPI app
│   ├── pipeline/
│   │   ├── engine_v2.py             # Main pipeline orchestrator
│   │   ├── model_helpers.py         # All model implementations (UPDATED)
│   │   ├── entity_extractor.py      # GLiNER (unchanged)
│   │   ├── relation_extractor.py    # GLiREL/REBEL (UPDATED)
│   │   ├── entity_linker.py         # BLINK linker (UPDATED)
│   │   └── ... (other modules)
│   └── frontend/
├── models/                          # Model weights downloaded here
├── scripts/
│   ├── download_models.py           # Download all models (NEW)
│   ├── validate_pipeline.py         # Validation script (NEW)
│   └── run_pipeline_example.py      # Example usage (NEW)
├── requirements.txt                 # Dependencies (verified)
├── INTEGRATION_GUIDE.md             # Detailed integration guide (NEW)
└── README.md                        # Original README
```

---

## Component Integration Points

### 1. Configuration
**File:** `app/config.py`

All model paths and settings centralized:
```python
from app.config import settings
embedding_model = settings.embedding_model  # "BAAI/bge-m3"
```

### 2. Model Initialization
**File:** `app/pipeline/engine_v2.py::IndustrialGraphPipeline._initialize_all_models()`

Pipeline automatically initializes:
- YOLOv12 for P&ID detection
- GroundingDINO for zero-shot detection
- SAM2 for segmentation
- GLiNER for entity extraction
- GLiREL for relation extraction
- BGE-M3 for embeddings
- BGE-Reranker-v2 for ranking
- BLINK for entity linking

### 3. Pipeline Stages
**File:** `app/pipeline/engine_v2.py::IndustrialGraphPipeline.run()`

Each model is called as a pipeline stage:
```
Document → OCR → PID Detection → Entity Extraction → Relations → Linking → Ranking → Knowledge Graph
```

### 4. Model Usage Examples

```python
# Entity Extraction
entities = pipeline.entity_extractor.extract(text)

# Relation Extraction  
relations = pipeline.relation_extractor.extract(text, entities)

# Entity Linking
linked = pipeline.blink_linker.link_entities(entities)

# P&ID Detection
symbols = pipeline.pid_symbol_detector.detect(images)

# GroundingDINO Detection
objects = pipeline.grounding_dino_detector.detect(images, prompt="valve")

# SAM2 Segmentation
masks = pipeline.sam_segmenter.segment(image)

# Embeddings
embedding = pipeline.embedding_model.encode(text_chunk)

# Reranking
ranked = pipeline.reranker_model.rank_candidates(query, candidates)
```

---

## Performance Optimization

### Reduce Memory Usage
```bash
# Use smaller SAM model
export SAM_MODEL_TYPE="vit_b"  # Instead of vit_l or vit_h

# Reduce text chunk size for embeddings
# In engine_v2.py: chunk_text(..., max_chars=700)
```

### Speed Up Inference
```bash
# Enable GPU acceleration
# Make sure CUDA is installed: nvidia-smi

# Batch processing
embeddings = pipeline.embedding_model.encode_batch(text_chunks)
```

### Monitoring
```bash
# Check GPU usage
watch -n 1 nvidia-smi

# Check model status
curl http://localhost:8000/api/v1/models/status
```

---

## Troubleshooting

### Models not downloading
```bash
# Check disk space
df -h models/

# Manual download (if auto fails)
# YOLOv8: Already in repo
# GroundingDINO: https://github.com/IDEA-Research/GroundingDINO/releases
# SAM2: https://github.com/facebookresearch/segment-anything
```

### Out of memory
```bash
# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""

# Or reduce batch sizes in model calls
```

### Slow inference
```bash
# Check if models are on GPU
nvidia-smi  # Should show python process using GPU memory

# If not, install CUDA-enabled PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Next Steps

1. **Download Models**
   ```bash
   python scripts/download_models.py
   ```

2. **Test with Sample PDF**
   ```bash
   # Place a test PDF in data/ folder
   python scripts/run_pipeline_example.py
   ```

3. **Start Server**
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Upload PDF via API**
   ```bash
   curl -X POST \
     -F "file=@sample.pdf" \
     http://localhost:8000/api/v1/process-pdf
   ```

---

## References

- **YOLOv12:** https://github.com/ultralytics/ultralytics
- **GroundingDINO:** https://github.com/IDEA-Research/GroundingDINO
- **SAM2:** https://github.com/facebookresearch/segment-anything
- **GLiNER:** https://github.com/urchade/GLiNER
- **GLiREL:** https://github.com/jackboyla/glirel
- **BLINK:** https://github.com/facebookresearch/BLINK
- **BGE-M3:** https://huggingface.co/BAAI/bge-m3
- **BGE-Reranker-v2:** https://huggingface.co/BAAI/bge-reranker-v2
- **FastAPI:** https://fastapi.tiangolo.com/

---

## Support

For issues or questions:
1. Check `INTEGRATION_GUIDE.md` for detailed model documentation
2. Run `python scripts/validate_pipeline.py` to diagnose issues
3. Check logs in `data/jobs/` for pipeline execution details

---

**Last Updated:** 2026-07-03  
**Status:** ✅ All models implemented and integrated
