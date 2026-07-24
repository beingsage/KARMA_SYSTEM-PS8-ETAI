# Architecture Overview

This repository implements an industrial document intelligence platform that ingests PDFs or text, runs a multi-stage analysis pipeline, persists graph and vector artifacts, and exposes both a REST API and a frontend-friendly workflow bundle.

## 1. System Shape

The runtime is centered on these layers:

1. API layer
   - [app/main.py](app/main.py)
   - FastAPI application, health endpoints, background job submission, workflow endpoints, advanced analysis endpoints, and Copilot/voice routes.

2. Job orchestration layer
   - [app/storage.py](app/storage.py)
   - Enforces durable JSON job records keyed by `job_id`.
   - Tracks status, source metadata, and pipeline outputs.

3. Pipeline engine
   - [app/pipeline/engine_v2.py](app/pipeline/engine_v2.py)
   - Main orchestration engine that runs stages sequentially, records stage progress, and exposes a rich final payload.

4. Model and utility layer
   - [app/pipeline/model_helpers.py](app/pipeline/model_helpers.py)
   - [app/pipeline/ocr_processor.py](app/pipeline/ocr_processor.py)
   - [app/pipeline/entity_extractor.py](app/pipeline/entity_extractor.py)
   - [app/pipeline/relation_extractor.py](app/pipeline/relation_extractor.py)
   - [app/pipeline/entity_linker.py](app/pipeline/entity_linker.py)
   - [app/pipeline/advanced_models.py](app/pipeline/advanced_models.py)
   - Houses OCR, vision, NLP, embedding, reranking, GraphRAG, and forecasting components.

5. Knowledge and retrieval layer
   - [app/pipeline/neo4j_store.py](app/pipeline/neo4j_store.py)
   - [app/pipeline/advanced_pipeline.py](app/pipeline/advanced_pipeline.py)
   - Neo4j graph persistence, ontology enrichment, semantic indexing, vector search, and advanced reasoning.

## 2. Runtime Execution Path

A typical request follows this path:

```text
HTTP POST /api/v1/workflows/analyze
    -> create job record in storage
    -> enqueue background async task
    -> IndustrialGraphPipeline.run() executes stage sequence
    -> job JSON updated incrementally
    -> final payload returned via /api/v1/jobs/{job_id}
    -> frontend bundle via /api/v1/workflows/{job_id}/bundle
```

## 3. Core Pipeline Stages

The production runtime in [app/pipeline/engine_v2.py](app/pipeline/engine_v2.py) proceeds through a broad stage chain:

1. OCR and text extraction
2. Layout and table analysis
3. Visual detection (GroundingDINO, YOLO, SAM2)
4. Formula and reading-order extraction
5. PID component detection
6. Document segmentation and chunking
7. Embedding generation
8. Entity extraction
9. Relation extraction
10. Entity linking
11. Ontology enrichment and schema evolution
12. Reranking
13. Vision-language understanding
14. GraphRAG analysis
15. Copilot analysis
16. Neo4j persistence
17. Advanced analytics stages (semantic indexing, graph reasoning, LLM analysis, anomaly detection, RUL, RCA, failure prediction, clustering, lessons learned)

## 4. Key Architectural Traits

### Hybrid fallback architecture
The pipeline is intentionally resilient:

- model-backed stages are preferred
- if a model fails during initialization or inference, the code falls back to heuristic or legacy logic
- final outputs remain structured so frontend and downstream systems can continue operating

### Background job model
The API does not block on the whole pipeline. Instead:

- a job object is created immediately
- job status is persisted
- the pipeline runs in a background thread via `asyncio.to_thread`
- the frontend can poll for progress and stage status

### Schema-first outputs
The result contract is represented by [app/schemas.py](app/schemas.py) using Pydantic models. A `JobResult` is the main API envelope and includes text, entities, relations, layout, tables, formulas, ontology metadata, and `pipeline_metadata`.

## 5. Important Modules

### API surface
- [app/main.py](app/main.py)
  - health endpoints
  - upload endpoints
  - workflow endpoints
  - job progress endpoints
  - advanced analytics endpoints
  - Copilot response endpoints
  - voice-assistant routes

### Pipeline runtime
- [app/pipeline/engine_v2.py](app/pipeline/engine_v2.py)
  - orchestrates all stages
  - logs per-stage status
  - manages fallback and summarization

### Output normalization
- [app/pipeline/workflow_bundle.py](app/pipeline/workflow_bundle.py)
  - converts raw job payloads into a more frontend-friendly view
  - groups timeline, summaries, metrics, and entity/relationship samples

### Configuration and environment
- [app/config.py](app/config.py)
  - all service settings, model paths, and environment-dependent options

## 6. Storage Model

The persistent data model is simple and durable:

- each job becomes a JSON document under the configured jobs directory
- the JSON file stores:
  - `job_id`
  - `status`
  - `uploaded_filename`
  - `message`
  - raw extracted text
  - entities and relations
  - graph/ontology metadata
  - `pipeline_metadata` with stage state and outputs

## 7. Architectural Summary

This project is best described as a production-oriented, document-to-knowledge-graph platform with a rich AI stack and a graceful degradation strategy. The primary design choice is to keep the service operational even when advanced ML components are partially unavailable.
