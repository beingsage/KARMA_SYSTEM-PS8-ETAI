# Industrial PDF Graph Backend

This project provides a production-grade backend for an industrial PDF-to-knowledge-graph pipeline with integrated AI models for P&ID analysis and structured data extraction.

## Quick Start (One Command!)

```bash
# Setup virtual environment, download all models, and start the server
bash run_all.sh
```

That's it! The script will:
✓ Create virtual environment if needed  
✓ Install all dependencies  
✓ Create necessary directories  
✓ Download all AI models (~2.5GB)  
✓ Start Neo4j database  
✓ Launch the backend server  

Then visit: **http://localhost:8001/docs** to explore the API

---

## What it Does

- Accepts PDF uploads through a REST API
- Extracts text and visual information using OCR
- Detects P&ID symbols using YOLOv12
- Performs zero-shot object detection using GroundingDINO
- Generates segmentation masks using SAM2
- Extracts entities using GLiNER (industrial-optimized)
- Identifies relations using GLiREL with heuristic fallback
- Links entities to knowledge base using BLINK
- Creates semantic embeddings using BGE-M3
- Indexes vectors into Qdrant for semantic search
- Stores graph in Neo4j and applies GraphRAG reasoning
- Performs advanced document understanding using Qwen 3
- Runs TimesFM for time-series forecasting and anomaly detection
- Predicts Remaining Useful Life with Temporal Fusion Transformer
- Coordinates reasoning workflows through LangGraph agent orchestration

---

## API Endpoints

```
POST   /api/v1/process-pdf        - Upload and process a PDF
GET    /api/v1/jobs               - List all processing jobs
GET    /api/v1/jobs/{job_id}      - Get results of a specific job
GET    /api/v1/models/status      - Check which models are loaded
GET    /health                    - Health check
```

### Example: Process a PDF

```bash
curl -X POST \
  -F "file=@sample.pdf" \
  http://localhost:8001/api/v1/process-pdf
```

---

## Advanced API Endpoints

```
GET    /api/v1/advanced/models/status           - Advanced models status
POST   /api/v1/advanced/vector-search           - Search embeddings (Qdrant)
POST   /api/v1/advanced/graph-reasoning         - GraphRAG reasoning
POST   /api/v1/advanced/llm-analysis            - Qwen 3 LLM analysis
POST   /api/v1/advanced/anomaly-detection       - TimesFM anomaly detection
POST   /api/v1/advanced/rul-prediction          - TFT RUL prediction
POST   /api/v1/advanced/root-cause-analysis     - RCA with full analysis
POST   /api/v1/advanced/failure-prediction      - Predict equipment failure
GET    /api/v1/advanced/pipeline-stages         - List all advanced stages
```

---

## AI Models Included

| Model | Purpose | Size |
|-------|---------|------|
| YOLOv12 | P&ID Symbol Detection | ~100MB |
| GroundingDINO | Zero-shot Object Detection | ~500MB |
| SAM2 | Instance Segmentation | ~400MB |
| GLiNER | Industrial Entity Extraction | ~600MB |
| GLiREL | Relation Extraction | ~400MB |
| BLINK | Entity Linking | ~500MB |
| BGE-M3 | Semantic Embeddings | ~450MB |
| BGE-Reranker-v2 | Relevance Ranking | ~300MB |

**Total:** ~2.5GB (models downloaded on first run)

---

## Manual Setup (if you prefer step-by-step)

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models
python scripts/download_models.py

# 4. Start Neo4j and Qdrant
docker compose up -d neo4j qdrant

# 5. Start backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Docker Troubleshooting

If Neo4j connection fails, you may need to configure host networking (Linux):

```bash
# Stop Neo4j
docker compose down

# Edit docker-compose.yml and add `network_mode: host` to neo4j service
# Then restart
docker compose up -d neo4j

# Check logs if still having issues
docker compose logs --tail=200 neo4j
```

---

## Additional Resources

- **Full Integration Guide:** See `INTEGRATION_GUIDE.md` for detailed model documentation
- **Setup & Configuration:** See `SETUP.md` for comprehensive setup instructions
- **Implementation Details:** See `IMPLEMENTATION_SUMMARY.md` for technical overview

---

## Fine-tuning Models

Fine-tune GLiNER with your own industrial data:

```bash
python scripts/fine_tune_gliner.py \
  --data data/gliner_training.jsonl \
  --output models/gliner-industrial-v1
```

---

## Architecture

```
Input PDF
    ↓
[OCR & Layout Analysis]
    ↓
┌─────────────────────────────────────┐
│ Vision Models                       │
├─────────────────────────────────────┤
│ • YOLOv12: P&ID Symbols             │
│ • GroundingDINO: Objects            │
│ • SAM2: Segmentation                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ NLP Models                          │
├─────────────────────────────────────┤
│ • GLiNER: Entity Extraction         │
│ • GLiREL: Relation Extraction       │
│ • BLINK: Entity Linking             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Semantic Models                     │
├─────────────────────────────────────┤
│ • BGE-M3: Embeddings                │
│ • BGE-Reranker: Ranking             │
└─────────────────────────────────────┘
    ↓
[Knowledge Graph Output]
    ↓
Neo4j + JSON Storage
```

---

## System Requirements

- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- 3GB disk space for models
- Docker & Docker Compose (for Neo4j)
- GPU recommended for faster inference (CUDA 11.8+)

---

## Performance

Typical inference times (on GPU):
- P&ID Detection: 50-100ms per image
- Object Detection: 200-400ms per image
- Entity Extraction: 10-50ms per chunk
- Relation Extraction: 50-200ms
- Embeddings: 5-20ms per chunk
- Ranking: 10-30ms per pair

---

## Support & Documentation

1. **Validation:** `python scripts/validate_pipeline.py`
2. **Example Usage:** `python scripts/run_pipeline_example.py`
3. **Integration Guide:** `INTEGRATION_GUIDE.md`
4. **Setup Guide:** `SETUP.md`
5. **Implementation Summary:** `IMPLEMENTATION_SUMMARY.md`

---

## License & Attribution

Uses pre-trained models from:
- Ultralytics (YOLOv12)
- IDEA-Research (GroundingDINO)
- Meta (SAM2)
- Urchade (GLiNER)
- Jack Boyla (GLiREL)
- Facebook Research (BLINK)
- BAAI (BGE models)

---

**Last Updated:** 2026-07-03  
**Status:** ✅ Production Ready







