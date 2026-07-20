import asyncio

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from pathlib import Path
from typing import Dict, List, Any, Optional

from app.config import settings
import warnings
from app.pipeline.compat import ensure_pyarrow_compat, install_safe_torch_load_default
from app.pipeline.runtime import cuda_is_usable
from app.pipeline.neo4j_store import Neo4jGraphStore

# Suppress expected FutureWarnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.meshgrid.*")
warnings.filterwarnings("ignore", message=".*num_batches_tracked.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*Failed to load custom C\+\+ ops.*")
# Provide a safe default for torch.load where supported to prefer weights-only loading
try:
    install_safe_torch_load_default()
    ensure_pyarrow_compat()

    import torch

    def _print_runtime_info():
        try:
            tv = getattr(torch, "__version__", "unknown")
            print(f"[runtime] torch=={tv}, cuda_usable={cuda_is_usable()}")
        except Exception:
            print("[runtime] torch installed but version info unavailable")

    _print_runtime_info()
    if getattr(torch, "_structured_safe_torch_load_installed", False):
        print("[runtime] Applied safe default: torch.load(..., weights_only=True)")
    else:
        print("[runtime] torch.load signature does not support weights_only or inspect failed; skipping monkeypatch")
except Exception:
    # torch not installed yet in environment
    pass
from app.pipeline.engine_v2 import get_pipeline
from app.pipeline.advanced_models import (
    QdrantVectorStore,
    GraphRAGEngine,
    Qwen3LLM,
    TimesFMForecaster,
    TemporalFusionTransformer,
    RootCauseAnalysisAgent,
    initialize_advanced_models
)
from app.pipeline.advanced_pipeline import AdvancedPipelineStages
from app.pipeline.workflow_bundle import build_api_catalog, build_workflow_bundle
from app.schemas import JobResult, OntologyMigrationRequest
from app.storage import list_jobs, load_job, create_job, update_job

app = FastAPI(title=settings.app_name)

@app.get("/swagger", include_in_schema=False)
def custom_swagger_ui_html() -> Any:
    """Serve the Swagger UI for the FastAPI app."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{settings.app_name} - Swagger UI",
    )

@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
def swagger_ui_redirect() -> Any:
    return get_swagger_ui_oauth2_redirect_html()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    pipeline = get_pipeline()
    health_status = pipeline.get_health_status()
    return {
        "status": health_status["status"],
        "service": health_status["service"],
        "runtime_mode": health_status["runtime_mode"],
        "model_counts": health_status["model_counts"],
        "components": health_status["components"],
        "backend_integrations": health_status["backend_integrations"],
        "stage_status": health_status["stage_status"],
        "stages": health_status["stages"],
    }


@app.get("/api/v1/health")
def api_health() -> dict[str, object]:
    """Structured health endpoint for pipeline readiness and component availability."""
    return health()


@app.on_event("startup")
async def warm_pipeline() -> None:
    """Preload the main pipeline so the first request doesn't pay bootstrap cost."""
    # Run environment validation checks first (fail-fast on critical issues)
    try:
        from scripts.check_environment import run_all_checks

        ok = await asyncio.to_thread(run_all_checks)
        if not ok:
            print("[startup] WARNING: Environment validation completed with warnings. Continuing in degraded mode.")
    except Exception as exc:
        print(f"[startup] WARNING: Environment validation error: {exc}. Continuing in degraded mode.")

    # Warm pipeline and advanced models after environment checked
    await asyncio.to_thread(get_pipeline)
    await asyncio.to_thread(initialize_advanced_models)


def _run_pipeline_sync(uploaded_filename: str, pdf_bytes: bytes, job_id: str) -> dict[str, object]:
    """Run the pipeline in a dedicated thread with its own event loop."""
    from app.pipeline.engine_v2 import run_pipeline as engine_run_pipeline

    return asyncio.run(engine_run_pipeline(uploaded_filename, pdf_bytes, job_id=job_id))


def _run_doc_sync(uploaded_filename: str, text: str, job_id: str) -> dict[str, object]:
    """Run the generic text pipeline in a dedicated thread with its own event loop."""
    from app.pipeline.engine_v2 import get_pipeline

    return asyncio.run(get_pipeline().run_from_text(uploaded_filename, text, job_id=job_id))


def _extract_text_from_document(content: bytes, file_name: Optional[str], mime_type: Optional[str]) -> str:
    try:
        from app.pipeline.llamaindex_hybrid import LlamaIndexHybrid

        loader = LlamaIndexHybrid()
        extracted_text = loader.extract_text(file_bytes=content, file_name=file_name, mime_type=mime_type)
        if extracted_text:
            return extracted_text
    except Exception:
        pass

    try:
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return ""


async def _submit_pdf_workflow(uploaded_filename: str, pdf_bytes: bytes) -> JobResult:
    job = create_job(uploaded_filename)
    job_id = job["job_id"]
    update_job(job_id, {"status": "processing", "message": "Pipeline started."})
    asyncio.create_task(_process_pdf_background(job_id, uploaded_filename, pdf_bytes))
    return JobResult(**load_job(job_id))


async def _submit_text_workflow(uploaded_filename: str, text: str) -> JobResult:
    job = create_job(uploaded_filename)
    job_id = job["job_id"]
    update_job(job_id, {"status": "processing", "message": "Pipeline started."})
    asyncio.create_task(_process_doc_background(job_id, uploaded_filename, text))
    return JobResult(**load_job(job_id))


async def _process_pdf_background(job_id: str, uploaded_filename: str, pdf_bytes: bytes) -> None:
    try:
        result = await asyncio.to_thread(_run_pipeline_sync, uploaded_filename, pdf_bytes, job_id)
        update_job(job_id, result)
    except Exception as exc:
        update_job(
            job_id,
            {
                "job_id": job_id,
                "status": "failed",
                "message": str(exc),
                "error": type(exc).__name__,
            },
        )


async def _process_doc_background(job_id: str, uploaded_filename: str, text: str) -> None:
    try:
        result = await asyncio.to_thread(_run_doc_sync, uploaded_filename, text, job_id)
        update_job(job_id, result)
    except Exception as exc:
        update_job(
            job_id,
            {
                "job_id": job_id,
                "status": "failed",
                "message": str(exc),
                "error": type(exc).__name__,
            },
        )


@app.post("/api/v1/process-pdf", response_model=JobResult)
async def process_pdf(file: UploadFile = File(...), file_name: str | None = Form(None)) -> JobResult:
    """Process PDF through full industrial pipeline"""
    
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty")

    uploaded_filename = file_name or file.filename
    return await _submit_pdf_workflow(uploaded_filename, payload)


@app.post("/api/v1/process-doc", response_model=JobResult)
async def process_document(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    file_name: str | None = Form(None),
    mime_type: str | None = Form(None),
) -> JobResult:
    """Process a generic document or text through the pipeline."""

    if file is None and text is None:
        raise HTTPException(status_code=400, detail="Provide either a file upload or raw text")

    if file is not None:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded document is empty")
        uploaded_filename = file_name or file.filename or "document"
        text = _extract_text_from_document(payload, uploaded_filename, mime_type or file.content_type)
    else:
        uploaded_filename = file_name or "text_document"

    if not text:
        raise HTTPException(status_code=400, detail="Document could not be parsed into text")

    return await _submit_text_workflow(uploaded_filename, text)


@app.post("/api/v1/workflows/analyze", response_model=JobResult)
async def analyze_workflow(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    file_name: str | None = Form(None),
    mime_type: str | None = Form(None),
) -> JobResult:
    """Single frontend-facing entrypoint for end-to-end document analysis."""

    if file is None and text is None:
        raise HTTPException(status_code=400, detail="Provide either a file upload or raw text")

    if file is not None:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        uploaded_filename = file_name or file.filename or "document"
        inferred_mime = (mime_type or file.content_type or "").lower()
        if uploaded_filename.lower().endswith(".pdf") or "pdf" in inferred_mime:
            return await _submit_pdf_workflow(uploaded_filename, payload)
        text = _extract_text_from_document(payload, uploaded_filename, mime_type or file.content_type)
        if not text:
            raise HTTPException(status_code=400, detail="Document could not be parsed into text")
        return await _submit_text_workflow(uploaded_filename, text)

    uploaded_filename = file_name or "text_document"
    if not text:
        raise HTTPException(status_code=400, detail="Provide raw text for workflow analysis")
    return await _submit_text_workflow(uploaded_filename, text)


@app.get("/api/v1/jobs/{job_id}", response_model=JobResult)
def get_job(job_id: str) -> JobResult:
    """Get job results"""
    
    try:
        payload = load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return JobResult(**payload)


@app.get("/api/v1/jobs")
def list_job_summaries() -> list[dict[str, object]]:
    """List all jobs"""
    
    return list_jobs()


@app.get("/api/v1/workflows/{job_id}/bundle")
def get_workflow_bundle(job_id: str, include_raw: bool = False) -> dict[str, object]:
    """Return a normalized, frontend-friendly bundle for a finished or in-flight job."""

    try:
        payload = load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    return build_workflow_bundle(payload, include_raw=include_raw)


@app.get("/api/v1/workflows/catalog")
def workflow_catalog() -> dict[str, object]:
    """Expose the limited frontend-facing API surface and backend capability groups."""

    pipeline = get_pipeline()
    health = pipeline.get_health_status()
    return {
        "status": health.get("status"),
        "runtime_mode": health.get("runtime_mode"),
        "api_catalog": build_api_catalog(),
        "stage_catalog": health.get("stages", []),
        "backend_integrations": health.get("backend_integrations", {}),
        "model_counts": health.get("model_counts", {}),
    }


@app.post("/api/v1/ontology/backfill")
def ontology_backfill(dry_run: bool = False, limit: int | None = None) -> dict[str, object]:
    """Backfill stable ontology metadata for existing Neo4j entities and relations."""
    store = Neo4jGraphStore()
    try:
        return store.backfill_ontology_metadata(dry_run=dry_run, limit=limit)
    finally:
        store.close()


@app.get("/api/v1/models/status")
def model_status() -> dict[str, object]:
    """Get pipeline model status and capabilities"""
    
    pipeline = get_pipeline()
    
    return {
        "pipeline": "industrial-pdf-model-stack",
        "version": "2.0-full-implementation",
        "runtime_mode": pipeline.model_mode,
        "ready": True,
        "stages": [
            "docling_surya_ocr",
            "doclayout_yolo_analysis",
            "surya_layout_understanding",
            "table_extraction",
            "table_transformer_extraction",
            "nougat_formula_recognition",
            "docling_reading_order",
            "yolo_pid_detector",
            "pid_component_detection",
            "document_segmentation",
            "semantic_indexing",
            "entity_extraction",
            "relation_extraction",
            "entity_linking",
            "ontology_enrichment",
            "qwen2_5_vl",
            "neo4j_persistence",
            "graphrag_analysis",
            "copilot_analysis",
        ],
        "loaded_components": {
            "ocr": pipeline.ocr_processor is not None,
            "entity_extractor": pipeline.entity_extractor is not None,
            "relation_extractor": pipeline.relation_extractor is not None,
            "graph_store": pipeline.graph_store is not None,
            "rag_summarizer": pipeline.rag_summarizer is not None,
            "copilot_agent": pipeline.copilot_agent is not None,
            "ontology_enricher": getattr(pipeline, "ontology_enricher", None) is not None,
            "yolo": pipeline.yolo_model is not None,
            "embeddings": pipeline.embedding_model is not None,
        },
    }


@app.post("/api/v1/admin/neo4j/migrate-ontology")
def migrate_neo4j_ontology(request: OntologyMigrationRequest) -> dict[str, object]:
    """Run the one-time ontology backfill/migration against existing Neo4j data."""

    pipeline = get_pipeline()
    graph_store = getattr(pipeline, "graph_store", None)
    if not graph_store or not getattr(graph_store, "connected", False):
        raise HTTPException(status_code=503, detail="Neo4j graph store not available")

    return graph_store.backfill_ontology_metadata(
        dry_run=request.dry_run,
        limit=request.limit,
        include_all_nodes=request.include_all_nodes,
        create_typed_labels=request.create_typed_labels,
    )


@app.get("/api/v1/copilot/rca/{job_id}")
def get_rca_analysis(job_id: str) -> dict[str, object]:
    """Get Root Cause Analysis for a job"""
    
    try:
        payload = load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    
    pipeline = get_pipeline()
    if not pipeline.copilot_agent:
        raise HTTPException(status_code=503, detail="Copilot agent not available")
    
    entities = payload.get("entities", [])
    relations = payload.get("relations", [])
    
    rca = pipeline.copilot_agent.root_cause_analysis(entities, relations, "")
    
    return {
        "job_id": job_id,
        "analysis": "root_cause_analysis",
        "results": rca,
    }


@app.get("/api/v1/copilot/maintenance/{job_id}")
def get_maintenance_plan(job_id: str) -> dict[str, object]:
    """Get maintenance plan for a job"""
    
    try:
        payload = load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    
    pipeline = get_pipeline()
    if not pipeline.copilot_agent:
        raise HTTPException(status_code=503, detail="Copilot agent not available")
    
    entities = payload.get("entities", [])
    relations = payload.get("relations", [])
    
    plan = pipeline.copilot_agent.get_maintenance_plan(entities, relations)
    
    return {
        "job_id": job_id,
        "analysis": "maintenance_plan",
        "results": plan,
    }


@app.get("/api/v1/copilot/compliance/{job_id}")
def get_compliance_status(job_id: str) -> dict[str, object]:
    """Get compliance check for a job"""
    
    try:
        payload = load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    
    pipeline = get_pipeline()
    if not pipeline.copilot_agent:
        raise HTTPException(status_code=503, detail="Copilot agent not available")
    
    entities = payload.get("entities", [])
    compliance = pipeline.copilot_agent.compliance_check(entities)
    
    return {
        "job_id": job_id,
        "analysis": "compliance_check",
        "results": compliance,
    }


@app.get("/api/v1/copilot/risk/{job_id}")
def get_risk_assessment(job_id: str) -> dict[str, object]:
    """Get risk assessment for a job"""
    
    try:
        payload = load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    
    pipeline = get_pipeline()
    if not pipeline.copilot_agent:
        raise HTTPException(status_code=503, detail="Copilot agent not available")
    
    entities = payload.get("entities", [])
    relations = payload.get("relations", [])
    
    risk = pipeline.copilot_agent.risk_assessment(entities, relations)
    
    return {
        "job_id": job_id,
        "analysis": "risk_assessment",
        "results": risk,
    }


@app.get("/api/v1/copilot/analyze/{job_id}")
def get_full_analysis(job_id: str) -> dict[str, object]:
    """Get full copilot analysis for a job"""
    
    try:
        payload = load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    
    pipeline = get_pipeline()
    if not pipeline.copilot_agent:
        raise HTTPException(status_code=503, detail="Copilot agent not available")
    
    entities = payload.get("entities", [])
    relations = payload.get("relations", [])
    text = payload.get("text", "")
    
    full_analysis = pipeline.copilot_agent.reason(entities, relations, text)
    
    return {
        "job_id": job_id,
        "analysis": "full_copilot_analysis",
        "results": full_analysis,
    }


# ============================================================================
# NEW: ADVANCED MODELS API ENDPOINTS
# ============================================================================

# Initialize advanced models
_advanced_models = None

def get_advanced_models():
    """Get or initialize advanced models."""
    global _advanced_models
    if _advanced_models is None:
        _advanced_models = initialize_advanced_models()
    return _advanced_models


@app.get("/api/v1/advanced/models/status")
def get_advanced_models_status() -> dict[str, Any]:
    """Get status of all advanced models."""
    models = get_advanced_models()
    
    return {
        "advanced_models": {
            "qdrant_vector_db": models.get("qdrant") is not None,
            "graphrag": models.get("graphrag") is not None,
            "qwen3_llm": models.get("qwen3") is not None,
            "timesfm_forecaster": models.get("timesfm") is not None,
            "tft_rul_predictor": models.get("tft") is not None,
            "bertopic_lessons_miner": models.get("bertopic") is not None,
            "hdbscan_clusterer": models.get("hdbscan") is not None,
            "node2vec_graph_embedder": models.get("node2vec") is not None,
            "langgraph_agent": models.get("agent") is not None,
            "rca_agent": models.get("rca") is not None,
        },
        "models_initialized": len([m for m in models.values() if m is not None]),
        "total_models": len(models),
        "features": {
            "vector_search": "Enabled" if models.get("qdrant") else "Disabled",
            "graph_reasoning": "Enabled" if models.get("graphrag") else "Disabled",
            "llm_analysis": "Enabled" if models.get("qwen3") else "Disabled",
            "time_series_forecasting": "Enabled" if models.get("timesfm") else "Disabled",
            "rul_prediction": "Enabled" if models.get("tft") else "Disabled",
            "lessons_learned": "Enabled" if models.get("bertopic") else "Disabled",
            "clustering": "Enabled" if models.get("hdbscan") else "Disabled",
            "graph_embeddings": "Enabled" if models.get("node2vec") else "Disabled",
            "root_cause_analysis": "Enabled" if models.get("rca") else "Disabled",
        }
    }


@app.post("/api/v1/advanced/vector-search")
def vector_search(query_embedding: List[float] = Body(...), 
                 top_k: int = 5) -> dict[str, Any]:
    """Search for similar vectors in Qdrant."""
    models = get_advanced_models()
    qdrant = models.get("qdrant")
    
    if not qdrant:
        raise HTTPException(status_code=503, detail="Vector database not available")
    
    results = qdrant.search(query_embedding, top_k=top_k)
    
    return {
        "query_result": {
            "results": results,
            "count": len(results),
            "top_k": top_k
        }
    }


@app.post("/api/v1/advanced/graph-reasoning")
def graph_reasoning(query: str = Body(...)) -> dict[str, Any]:
    """Query knowledge graph with GraphRAG reasoning."""
    models = get_advanced_models()
    graphrag = models.get("graphrag")
    
    if not graphrag:
        raise HTTPException(status_code=503, detail="GraphRAG not available")
    
    result = graphrag.query_graph(query)
    
    return {
        "reasoning_result": result
    }


@app.post("/api/v1/advanced/doc-query")
@app.post("/api/v1/advanced/llama-query")
def llama_query(
    query: str = Body(...),
    job_id: str | None = Body(None),
    text: str | None = Body(None),
    top_k: int = Body(5),
) -> dict[str, Any]:
    """Query document content using LlamaIndex retrieval."""
    if job_id is None and text is None:
        raise HTTPException(status_code=400, detail="Provide either a job_id or raw text for query")

    if job_id is not None:
        try:
            payload = load_job(job_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Job not found")
        if text is None:
            text = payload.get("text", "")
        text_chunks = payload.get("document_segments") or []
    else:
        text_chunks = []

    if not text and not text_chunks:
        raise HTTPException(status_code=400, detail="No document text or chunks available for query")

    try:
        from app.pipeline.llamaindex_hybrid import LlamaIndexHybrid

        llm = LlamaIndexHybrid(embedder=get_pipeline().embedding_model)
        if text_chunks:
            llm.build_index(text_chunks=text_chunks, text_chunks_metadata=[{"chunk_id": i} for i in range(len(text_chunks))])
        else:
            llm.build_index_from_text(text=text)

        evidence = llm.retrieve([query], entities=[])
        return {
            "query": query,
            "top_k": top_k,
            "results": evidence.contexts[:top_k],
            "coverage_score": evidence.coverage_score,
            "combined_text": evidence.combined_text,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"LlamaIndex query unavailable: {exc}")


@app.post("/api/v1/advanced/llm-analysis")
def llm_analysis(prompt: str = Body(...), 
                max_tokens: int = 512) -> dict[str, Any]:
    """Analyze content with Qwen 3 LLM."""
    models = get_advanced_models()
    qwen3 = models.get("qwen3")
    
    if not qwen3:
        raise HTTPException(status_code=503, detail="Qwen 3 LLM not available")
    
    response = qwen3.generate(prompt, max_tokens=max_tokens)
    
    return {
        "llm_response": {
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "response": response,
            "model": "Qwen3"
        }
    }


@app.post("/api/v1/advanced/anomaly-detection")
def detect_anomalies(time_series: List[float] = Body(...),
                    threshold: float = 2.0) -> dict[str, Any]:
    """Detect anomalies in time-series data."""
    models = get_advanced_models()
    timesfm = models.get("timesfm")
    
    if not timesfm:
        raise HTTPException(status_code=503, detail="TimesFM not available")
    
    anomalies = timesfm.detect_anomalies(time_series, threshold=threshold)
    forecast = timesfm.forecast(time_series, steps_ahead=50)
    
    return {
        "anomaly_detection": {
            "anomalies": anomalies,
            "forecast": forecast,
            "alert": "Anomalies detected!" if anomalies.get("detected_count", 0) > 0 else "No anomalies"
        }
    }


@app.post("/api/v1/advanced/rul-prediction")
def predict_rul(machine_id: str = Body(...),
               sensor_data: Dict[str, List[float]] = Body(...)) -> dict[str, Any]:
    """Predict Remaining Useful Life (RUL) for equipment."""
    models = get_advanced_models()
    tft = models.get("tft")
    
    if not tft:
        raise HTTPException(status_code=503, detail="TFT RUL Predictor not available")
    
    rul_prediction = tft.predict_rul(sensor_data, machine_id)
    maintenance = tft.maintenance_recommendation(rul_prediction["estimated_rul_days"])
    
    return {
        "rul_prediction": {
            "prediction": rul_prediction,
            "maintenance_recommendation": maintenance
        }
    }


@app.post("/api/v1/advanced/root-cause-analysis")
def analyze_root_cause(incident: str = Body(...),
                      logs: List[str] = Body(...),
                      sensor_data: Dict[str, List[float]] = Body(...)) -> dict[str, Any]:
    """Perform root cause analysis on an incident."""
    models = get_advanced_models()
    rca = models.get("rca")
    
    if not rca:
        raise HTTPException(status_code=503, detail="RCA Agent not available")
    
    result = rca.analyze_incident(incident, logs, sensor_data)
    
    return {
        "root_cause_analysis": result
    }


@app.post("/api/v1/advanced/failure-prediction")
def predict_failure(machine_id: str = Body(...),
                   sensor_data: Dict[str, List[float]] = Body(...)) -> dict[str, Any]:
    """Predict potential equipment failure."""
    models = get_advanced_models()
    rca = models.get("rca")
    
    if not rca:
        raise HTTPException(status_code=503, detail="RCA Agent not available")
    
    result = rca.predict_failure(machine_id, sensor_data)
    
    return {
        "failure_prediction": result
    }


@app.post("/api/v1/advanced/lessons-learned")
def lessons_learned(documents: List[str] = Body(...),
                    top_n: int = Body(10)) -> dict[str, Any]:
    """Mine lessons learned using BERTopic."""
    models = get_advanced_models()
    bertopic = models.get("bertopic")

    if not bertopic:
        raise HTTPException(status_code=503, detail="BERTopic lessons miner not available")

    lessons = bertopic.mine_lessons(documents, top_n=top_n)

    return {
        "lessons_learned": lessons
    }


@app.post("/api/v1/advanced/clustering")
def clustering(embeddings: List[List[float]] = Body(...),
               min_cluster_size: int = Body(10)) -> dict[str, Any]:
    """Perform clustering over embeddings using HDBSCAN."""
    models = get_advanced_models()
    hdbscan = models.get("hdbscan")

    if not hdbscan:
        raise HTTPException(status_code=503, detail="HDBSCAN clusterer not available")

    clusters = hdbscan.cluster(embeddings, min_cluster_size=min_cluster_size)

    return {
        "clustering": clusters
    }


@app.post("/api/v1/advanced/graph-embeddings")
def graph_embeddings(node_walk_length: int = Body(80),
                     dimensions: int = Body(128),
                     num_walks: int = Body(30)) -> dict[str, Any]:
    """Generate graph embeddings using Node2Vec."""
    models = get_advanced_models()
    node2vec = models.get("node2vec")

    if not node2vec:
        raise HTTPException(status_code=503, detail="Node2Vec graph embedder not available")

    embeddings = node2vec.generate_embeddings(dimensions=dimensions, walk_length=node_walk_length, num_walks=num_walks)

    return {
        "graph_embeddings": embeddings
    }


@app.get("/api/v1/advanced/pipeline-stages")
def get_pipeline_stages() -> dict[str, Any]:
    """Get all advanced pipeline stages."""
    advanced_pipeline = AdvancedPipelineStages()
    
    return {
        "advanced_stages": {
            "stages": advanced_pipeline.get_stage_names(),
            "descriptions": advanced_pipeline.get_stage_descriptions(),
            "total_stages": len(advanced_pipeline.get_stage_names())
        }
    }


# Mount static files at the end, AFTER all API routes, so routes take priority.
# Skip the mount if the frontend bundle is absent so API startup still succeeds.
frontend_dir = Path(__file__).resolve().parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
