# API Reference Summary

This repository exposes a FastAPI backend centered on document ingestion, graph extraction, workflow health, and advanced reasoning endpoints.

## 1. Core Endpoints

### Health
- `GET /health`
  - basic service health
- `GET /api/v1/health`
  - structured readiness and component status

### PDF and document ingestion
- `POST /api/v1/process-pdf`
  - upload a PDF and run the full pipeline
- `POST /api/v1/process-doc`
  - upload or submit text and run the text pipeline
- `POST /api/v1/workflows/analyze`
  - frontend-facing unified analysis endpoint
- `POST /api/v1/sources/ingest`
  - ingest local or parsed sources for downstream analysis

### Job lifecycle
- `GET /api/v1/jobs/{job_id}/progress`
  - stage-by-stage execution and progress snapshot
- `GET /api/v1/jobs/{job_id}`
  - retrieve the full stored job payload
- `GET /api/v1/jobs`
  - list all persisted jobs

### Workflow bundle and catalog
- `GET /api/v1/workflows/{job_id}/bundle`
  - return a frontend-friendly normalized bundle
- `GET /api/v1/workflows/catalog`
  - expose the active API and stage catalog

## 2. Model and Ontology Endpoints

- `GET /api/v1/models/status`
  - returns model and component readiness information
- `POST /api/v1/ontology/backfill`
  - backfill ontology metadata for existing graph records
- `POST /api/v1/admin/neo4j/migrate-ontology`
  - one-time ontology migration request for Neo4j content

## 3. Advanced Analytics Endpoints

The advanced API area includes endpoints for:

- vector search
- graph reasoning
- document query
- direct LLM analysis
- anomaly detection
- RUL prediction
- root cause analysis
- failure prediction
- lessons learned mining
- clustering
- graph embeddings
- pipeline-stage introspection

These are exposed in the advanced route group under `/api/v1/advanced/...`.

## 4. Copilot and Voice Routes

Additional conversational routes include:

- `GET /api/v1/copilot/rca/{job_id}`
- `GET /api/v1/copilot/maintenance/{job_id}`
- `GET /api/v1/copilot/compliance/{job_id}`
- `GET /api/v1/copilot/risk/{job_id}`
- `GET /api/v1/copilot/analyze/{job_id}`
- `POST /api/v1/copilot/chat`
- `POST /api/v1/voice/assistant`

## 5. Response Pattern

The system typically returns:

- a `JobResult` on submission
- a persisted job JSON when polling a `job_id`
- a normalized workflow bundle for UI presentation

The stored result includes both raw extracted artifacts and operational metadata so the result can be consumed by multiple clients without needing a separate data contract.
