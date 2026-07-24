# Industrial Document Intelligence Platform — Benchmark & Test Results
## KARMA SYSTEM / Structured Industrial PDF-to-Graph Pipeline

> **Evidence policy:** This report separates repository facts and observed run-log measurements from metrics that require a controlled benchmark run. No accuracy, latency, throughput, or GPU-performance claim is presented as measured unless the repository currently contains evidence for it.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                 Industrial Document Intelligence Platform            │
│              PDF / Text / Source Ingestion + AI Reasoning            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📄 PDF/Text/File                                                   │
│        │                                                            │
│        ▼                                                            │
│  🌐 FastAPI API Layer                                               │
│        │                                                            │
│        ▼                                                            │
│  🧾 Durable Job Record (JSON)                                       │
│        │                                                            │
│        ▼                                                            │
│  ⚙️ IndustrialGraphPipeline                                         │
│        │                                                            │
│   ┌────┼─────────────┬───────────────┬────────────────────────┐     │
│   ▼    ▼             ▼               ▼                        │     │
│ OCR  Layout       Vision           NLP / Ontology             │     │
│      Tables       P&ID / Masks     Entities / Relations       │     │
│   │    │             │               │                        │     │
│   └────┴─────────────┴───────────────┴────────────────────────┘     │
│        │                                                            │
│        ▼                                                            │
│  🔎 Embeddings + Reranking + Evidence Selection                     │
│        │                                                            │
│   ┌────┴──────────────┬─────────────────────┬───────────────────┐   │
│   ▼                   ▼                     ▼                   │   │
│ Neo4j Graph       Qdrant / LlamaIndex   GraphRAG / Copilot       │   │
│ Persistence       Vector Retrieval      Reasoning               │   │
│   │                   │                     │                   │   │
│   └───────────────────┴─────────────────────┴───────────────────┘   │
│        │                                                            │
│        ▼                                                            │
│  📦 JobResult + Workflow Bundle + Stage Timeline                    │
│                                                                     │
│  🎙️ RealtimeVoiceChat: browser audio ⇄ WebSocket ⇄ STT/LLM/TTS      │
└─────────────────────────────────────────────────────────────────────┘
```

### Main runtime components

| Layer | Repository implementation | Responsibility |
|---|---|---|
| API | `app/main.py` | FastAPI routes, uploads, jobs, health, Copilot, voice, and advanced APIs |
| Orchestration | `app/pipeline/engine_v2.py` | Sequential stage execution, required/optional stage handling, progress, persistence |
| Document processing | `app/pipeline/ocr_processor.py` | Docling extraction, Surya layout/table path, reading order, formula candidates |
| Vision | `app/pipeline/model_helpers.py` | YOLO/PID detection, GroundingDINO, SAM2, DocLayout-YOLO, PANDID RCNN |
| NLP | `entity_extractor.py`, `relation_extractor.py` | Industrial entities and relations with model and heuristic fallbacks |
| Semantics | `entity_linker.py`, `reranker_v2.py` | Stable identity, linking, lexical/embedding/cross-encoder ranking |
| Ontology | `app/pipeline/ontology.py`, `models.py` | Type and relation normalization, proposals, provenance, schema evolution |
| Graph | `app/pipeline/neo4j_store.py` | Knowledge graph persistence and ontology backfill |
| Retrieval | `llamaindex_hybrid*.py`, `advanced_pipeline.py` | Chunk evidence, vector indexing, GraphRAG context, Qdrant integration |
| Conversational AI | `agents/`, `copilot_agent.py`, `voice_runtime.py` | RCA, maintenance, quality, regulation, failure intelligence, Copilot, voice bridge |
| Realtime voice | `agents/RealtimeVoiceChat/` | Browser audio, WebSockets, realtime STT, LLM connector, TTS, interruption handling |

---

## 🤖 AI Models and Methods

| Component | Type / backend | Purpose | Current evidence |
|---|---|---|---|
| Docling | Local document parser | PDF OCR/text extraction and document conversion | Initialized in `pipeline_run.log` |
| Surya | Local layout/table path | Layout and table understanding | Present, but run log reports an API mismatch and fallback behavior |
| GLiNER | Local NLP model | Industrial entity extraction | Loaded in `pipeline_run.log` as medium-v2.1 |
| GLiREL | Local NLP model | Relation extraction | Loaded as `jackboyla/glirel-large-v0` |
| REBEL / heuristic relation path | Local fallback | Relation extraction when primary path fails | Covered by fallback tests |
| YOLO / PID detector | Local vision | P&ID symbol and component detection | Initialized in run log and covered by integration tests |
| GroundingDINO | Local vision | Zero-shot object detection | Initialized in run log |
| SAM2 | Local vision | Object segmentation | Initialized in run log |
| DocLayout-YOLO | Local vision | Document layout analysis | Initialized in run log |
| PANDID RCNN | Local vision/artifact | P&ID or industrial component detection | Artifact resolution covered by tests |
| BGE-M3 | Local embeddings | Semantic chunk/entity embeddings | Loaded in run log |
| BGE reranker v2 | Local ranking | Relevance ranking and candidate ordering | Ranking and NDCG tests present |
| BLINK linker | Local/entity linking | Entity identity resolution | Initialized in run log |
| Neo4j | External graph backend | Persist entities, relations, artifacts, and ontology metadata | Run log shows unavailable in the captured run |
| Qdrant | Vector backend | Vector storage and search | Advanced integration is implemented; availability is environment-dependent |
| GraphRAG | Graph + LLM reasoning | Evidence-grounded graph synthesis | Dedicated stage 20 tests present |
| Qwen / visual-language path | Local or optional model | Visual and semantic document interpretation | Fallback tests present |
| RealtimeSTT | Realtime voice backend | Browser audio transcription | Vendored under `agents/RealtimeVoiceChat` |
| RealtimeTTS | Realtime voice backend | Spoken response synthesis | Vendored under `agents/RealtimeVoiceChat` |
| Ollama / OpenAI connectors | Realtime LLM backends | Conversational response generation | Declared by realtime chat requirements |
| Gemini / Tavily integrations | Optional external services | Copilot synthesis and web evidence | Key-dependent, graceful fallback in `app/main.py` |

---

## 🔁 Pipeline Stage Coverage

The main PDF path in `app/pipeline/engine_v2.py` covers the following processing groups:

| Group | Stages |
|---|---|
| Extraction | Docling/Surya OCR, layout, tables, table transformer, formulas, reading order |
| Visual understanding | GroundingDINO, PANDID RCNN, SAM2, YOLO PID detection, PID symbol detection |
| Text understanding | PID component detection, document segmentation, semantic indexing |
| Knowledge extraction | Entity extraction, relation extraction, entity linking |
| Governance | Ontology enrichment, schema evolution, provenance and stable IDs |
| Relevance | BGE reranking and evidence-mode selection |
| Synthesis | Vision-language understanding, GraphRAG analysis, Copilot analysis |
| Persistence | Neo4j graph persistence and source artifact linkage |
| Advanced analytics | Qdrant indexing, graph reasoning, LLM analysis, anomaly detection, RUL, RCA, failure prediction, graph embeddings, clustering, lessons learned |

The text path uses the same semantic and reasoning stages after normalizing supplied text, while PDF-only visual stages receive no PDF bytes and are expected to skip or use fallback behavior.

---

## 📊 Repository Test Inventory

### Test collection

| Measure | Result | Source |
|---|---:|---|
| Collected tests | **120** | `pytest --collect-only -q` executed in the repository |
| Test framework | Pytest + unittest-style tests | `pytest.ini`, `tests/` |
| Pipeline integration test | Present | `tests/test_pipeline.py` |
| Realtime voice integration tests | Present | `tests/test_realtime_voice_integration.py` |
| Voice assistant endpoint test | Present | `tests/test_voice_assistant_endpoint.py` |
| OCR fallback tests | Present | `tests/test_ocr_processor.py` |
| Ontology and schema evolution tests | Present | `tests/test_ontology_*.py` |
| GraphRAG evidence/confidence tests | Present | `tests/test_stage20_graphrag.py` |
| Reranker and NDCG tests | Present | `tests/test_reranker_v2.py` |
| PANDID RCNN artifact tests | Present | `tests/test_pandid_rcnn_integration.py` |

### Covered behavior

- PDF pipeline produces a completed result with entities, relations, and pipeline metadata.
- Required and optional stages are distinguished.
- OCR and layout fallback behavior is tested.
- Relation extraction falls back to heuristic methods.
- Ontology metadata creates stable IDs, status, provenance, and proposals.
- GraphRAG validates evidence sufficiency and filters unsupported claims.
- Reranking behavior includes ranking quality and NDCG calculations.
- Realtime voice routes serve frontend assets and bridge voice requests to Copilot.
- Voice assistant endpoint exists and is exposed by the API.

> **Important:** Test collection is verified. A full test-pass count is intentionally not claimed here because the complete suite can initialize heavyweight models and external integrations. Run the commands in the reproducibility section to record the exact pass/fail result for the current machine.

---

## ⚡ Observed Runtime Evidence

The repository contains `pipeline_run.log`, captured with `EXECUTION_MODE=GPU`. The log shows the following concrete observations:

| Observation | Recorded result |
|---|---|
| Config mode | GPU-preferred mode |
| Pipeline mode | `best-model-stack` |
| OCR processor | Docling initialized; device reported as CPU |
| Entity extractor | GLiNER medium-v2.1 loaded on CPU |
| Relation extractor | GLiREL large model loaded on CPU |
| PID detector | Initialized on CPU |
| GroundingDINO | Initialized on CPU |
| SAM2 | Initialized on CPU |
| Embeddings | BGE-M3 initialized on CPU |
| Reranker | BGE reranker initialized on CPU |
| BLINK | Initialized on CPU |
| Graph persistence | Neo4j unavailable in captured run; persistence disabled |
| GraphRAG/Copilot | Summarizer and Copilot agent initialized |
| Surya | Initialization failed due to API mismatch; fallback path remained available |
| YOLO image inference samples | Approximately **47.5–95.6 ms** per image in the captured log, with preprocessing and postprocessing shown separately |

These are **single-run observations**, not statistically controlled benchmarks. They should not be interpreted as p50, p95, accuracy, or production throughput.

---

## ⏱️ Benchmark Status

| Benchmark | Status | What is currently known |
|---|---|---|
| Unit/integration test collection | ✅ Measured | 120 tests collected |
| End-to-end PDF latency | 🟡 Pending controlled run | A pipeline test exists, but no committed timing table is present |
| Text-only pipeline latency | 🟡 Pending controlled run | Text path exists and should be benchmarked independently |
| OCR throughput | 🟡 Pending | Depends strongly on PDF page count, resolution, and active backend |
| Entity extraction precision/recall | 🟡 Pending labeled dataset | No ground-truth evaluation dataset was found in the repository |
| Relation extraction precision/recall | 🟡 Pending labeled dataset | Fallback behavior is tested; accuracy is not established |
| PID detection mAP/recall | 🟡 Pending labeled dataset | Component tests exist; no dataset-level mAP report is committed |
| Reranker NDCG | ✅ Test coverage exists | Ranking metric tests are present; publish numeric run output after executing them |
| GraphRAG groundedness | ✅ Behavioral tests exist | Evidence validation and confidence calibration are covered |
| Neo4j persistence latency | 🟡 Environment-dependent | Requires a running Neo4j service |
| Qdrant search latency | 🟡 Environment-dependent | Requires the configured Qdrant backend |
| Realtime voice latency | 🟡 Pending controlled audio run | Architecture supports streaming; no measured p50/p95 report is committed |
| GPU speedup | 🟡 Pending paired run | Captured log reports CPU devices despite GPU-preferred config |

---

## 🎙️ RealtimeVoiceChat Benchmark Surface

The realtime chat subsystem is located at `agents/RealtimeVoiceChat/` and is currently tracked by the parent repository as a nested gitlink. Its implementation includes:

```text
Browser microphone
    │ audio chunks
    ▼
WebSocket server
    │
    ▼
RealtimeSTT transcription
    │ partial/final text
    ▼
LLM backend (Ollama or OpenAI connector)
    │ response text
    ▼
RealtimeTTS synthesis
    │ audio stream
    ▼
Browser playback + interruption handling
```

### Voice capabilities represented in the source

- WebSocket audio streaming
- partial transcription and response feedback
- dynamic turn/silence detection
- interruptible conversation
- pluggable Ollama/OpenAI LLM connectors
- pluggable Kokoro, Coqui, and Orpheus TTS options
- browser Web Audio API frontend
- Copilot bridge through `app/voice_runtime.py`

### Voice metrics to collect

| Metric | Recommended measurement |
|---|---|
| Time to first transcription | Audio send timestamp to first partial transcript |
| Final transcription latency | Audio turn end to final transcript |
| Time to first LLM token/text | Final transcript to first response chunk |
| Time to first audio | User turn end to first playable TTS audio |
| End-to-end turn latency | User turn end to first spoken response |
| Interruption recovery | Time from barge-in to old response cancellation |
| Sustained stream stability | WebSocket disconnects per 100 turns |

---

## 🧪 Recommended Controlled Benchmark Protocol

### Dataset
Use three fixed input groups:

1. **Text set:** 20 short industrial paragraphs with known entities and relations.
2. **PDF set:** 10 representative PDFs, grouped by text-heavy, table-heavy, and P&ID-heavy documents.
3. **Voice set:** 20 recorded English turns with known transcripts and varied silence lengths.

For accuracy evaluation, annotate:

- entity spans and entity types
- relation endpoints and relation types
- PID components and bounding boxes
- supported versus unsupported GraphRAG claims

### Hardware and environment record

Record before each run:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

Also record:

- `EXECUTION_MODE`
- active model identifiers
- Neo4j/Qdrant availability
- page count and PDF byte size
- CPU/GPU device selected by each component

### Commands

Collect tests:

```bash
pytest --collect-only -q
```

Run the focused integration tests:

```bash
pytest -q tests/test_pipeline.py tests/test_realtime_voice_integration.py tests/test_voice_assistant_endpoint.py
```

Run the full suite:

```bash
pytest -q
```

Run the application for API smoke testing:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Then inspect:

- `http://127.0.0.1:8001/swagger`
- `http://127.0.0.1:8001/docs`
- `http://127.0.0.1:8001/openapi.json`

### Metrics table to populate

| Scenario | Runs | p50 | p95 | Success rate | Notes |
|---|---:|---:|---:|---:|---|
| Text-only pipeline | 30 | pending | pending | pending | CPU/GPU separately |
| One-page PDF | 30 | pending | pending | pending | Include OCR and layout |
| Multi-page PDF | 20 | pending | pending | pending | Report pages and bytes |
| P&ID-heavy PDF | 20 | pending | pending | pending | Report detection counts |
| Entity extraction | 100 chunks | pending | pending | pending | Add precision/recall if labeled |
| Relation extraction | 100 chunks | pending | pending | pending | Add precision/recall if labeled |
| Reranking | 100 queries | pending | pending | pending | Publish NDCG@k |
| Realtime voice turn | 20 turns | pending | pending | pending | Report first audio and interruption |

---

## 🛡️ Reliability and Fallback Design

The system is designed to degrade gracefully rather than fail the entire job when optional components are unavailable.

```text
Primary model-backed stage
          │ failure/unavailable
          ▼
Alternate model or backend
          │ failure/unavailable
          ▼
Heuristic / lexical fallback
          │
          ▼
Structured skipped result + stage telemetry
```

Examples verified by the repository structure and tests:

- relation extraction can fall back to heuristics
- reranking can fall back through lexical and embedding paths
- vision-language processing has OCR-proxy fallback behavior
- OCR/layout paths expose fallback summaries
- Neo4j absence does not prevent the rest of the pipeline from producing a JSON job result
- ontology normalization preserves stable IDs and proposals for unknown concepts
- optional advanced stages are tracked as skipped or error states instead of silently disappearing

### Current release-readiness risks

| Risk | Impact | Action |
|---|---|---|
| RealtimeVoiceChat is a nested git repository | A clone of the parent repo receives only a gitlink unless the nested repo is separately available | Vendor it as normal files or deliberately configure a submodule URL |
| OpenAPI files are hand-authored | They may omit routes or detailed request/response schemas | Generate an authoritative spec from the running FastAPI app and commit it |
| Captured GPU-preferred run used CPU devices | Performance may not represent intended GPU deployment | Validate CUDA compatibility and record paired CPU/GPU runs |
| Neo4j was unavailable in captured run | Graph persistence and graph-backed reasoning were not fully exercised | Run benchmark with Neo4j enabled |
| Surya API mismatch in captured run | Layout/table quality may rely on fallback behavior | Pin a compatible Surya version or update the adapter |
| No labeled evaluation dataset | Accuracy claims cannot be made responsibly | Add a versioned industrial gold set |

---

## 🧰 Technology Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn |
| Validation | Pydantic |
| Document processing | Docling, OCR helpers, PyMuPDF/PIL paths |
| Vision | YOLO, GroundingDINO, SAM2, DocLayout-YOLO, PANDID RCNN |
| NLP | GLiNER, GLiREL/REBEL fallback, heuristic extraction |
| Embeddings/ranking | BGE-M3, BGE reranker, lexical reranking |
| Knowledge graph | Neo4j |
| Vector retrieval | Qdrant, LlamaIndex hybrid path |
| Reasoning | GraphRAG, Copilot agents, optional Gemini/Tavily |
| Forecasting/analytics | TimesFM, TFT, clustering, graph embeddings, lessons mining |
| Realtime voice | RealtimeSTT, RealtimeTTS, WebSockets, Web Audio API |
| Persistence | JSON job files under configured jobs directory |
| Test tooling | Pytest and unittest-style tests |

---

## 📌 Key Differentiators

1. **Industrial document focus:** combines OCR, layout, P&ID vision, entities, relations, ontology, and graph persistence.
2. **Evidence-aware reasoning:** GraphRAG tests explicitly cover evidence sufficiency, claim filtering, and confidence calibration.
3. **Graceful degradation:** optional model and database failures are represented through structured stage state.
4. **Ontology evolution:** unknown concepts can become proposals with stable identity and provenance.
5. **Multi-modal output:** the same job can produce text, tables, formulas, layout, graph facts, vector artifacts, reasoning, and voice interaction.
6. **Operational visibility:** stage status, stage outputs, component readiness, and workflow bundles support frontend monitoring.
7. **Realtime interaction path:** voice input can bridge into Copilot reasoning and return spoken responses.

---

## ✅ Summary

The repository currently demonstrates a broad, integrated industrial AI platform with:

- **120 collectible automated tests**
- a FastAPI service exposing ingestion, workflow, advanced analytics, Copilot, and voice routes
- a multi-stage PDF and text pipeline
- model-backed and fallback processing paths
- graph, vector, ontology, and evidence-grounded reasoning components
- an integrated RealtimeVoiceChat subsystem

The next meaningful benchmark milestone is not another architecture claim. It is a controlled evaluation run that records end-to-end latency, stage-level timings, accuracy on a labeled industrial dataset, graph/vector backend availability, and realtime voice first-audio latency.

---

*Generated from the repository implementation, tests, and available runtime logs. Update the measured tables after each controlled benchmark run.*
