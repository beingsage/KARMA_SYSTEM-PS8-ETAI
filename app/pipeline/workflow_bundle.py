"""Helpers for building frontend-friendly workflow bundles.

The backend persists a rich job payload, but the frontend needs a normalized
view that groups related outputs into sections, timelines, and action catalogs.
This module derives that view without changing the underlying job contract.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence


def _titleize(value: str) -> str:
    text = str(value or "").strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.title() if text else "Unknown"


def _preview_text(value: Any, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:max_chars]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _compact_value(value: Any, *, depth: int = 0, max_items: int = 3) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return {
            "kind": "text",
            "length": len(value),
            "preview": _preview_text(value, 240),
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "count": len(value),
            "sample": [_compact_value(item, depth=depth + 1, max_items=max_items) for item in value[:max_items]],
        }
    if isinstance(value, dict):
        summary: Dict[str, Any] = {
            "kind": "dict",
            "keys": list(value.keys())[:24],
        }
        for key in (
            "status",
            "message",
            "reason",
            "summary",
            "confidence",
            "score",
            "count",
            "total",
            "entity_count",
            "relation_count",
            "indexed_chunks",
            "vectors_indexed",
            "anomaly_count",
            "risk_level",
        ):
            if key in value and value.get(key) is not None:
                summary[key] = value.get(key)

        if depth < 1:
            for key in (
                "summary",
                "reasoning",
                "executive_summary",
                "analysis",
                "graph_query",
                "results",
                "recommendations",
                "insights",
                "key_insights",
                "alerts",
                "alert",
                "ontology_report",
                "ontology_proposals",
            ):
                if key in value:
                    summary[key] = _compact_value(value[key], depth=depth + 1, max_items=max_items)

        return summary
    return {"kind": type(value).__name__, "value": _preview_text(value, 240)}


def _entity_count(entities: Sequence[Dict[str, Any]]) -> int:
    return len(list(entities or []))


def _relation_count(relations: Sequence[Dict[str, Any]]) -> int:
    return len(list(relations or []))


def _stage_order_key(stage: Dict[str, Any]) -> tuple[int, int, str]:
    status_priority = {"failed": 0, "running": 1, "completed": 2, "skipped": 3}
    return (
        status_priority.get(str(stage.get("status") or "").lower(), 4),
        _safe_int(stage.get("elapsed_seconds"), 0),
        str(stage.get("stage") or "").lower(),
    )


def build_stage_timeline(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata = job.get("pipeline_metadata") if isinstance(job.get("pipeline_metadata"), dict) else {}
    stage_status = list(metadata.get("stage_status") or [])
    stage_outputs = list(metadata.get("stage_outputs") or [])

    output_by_stage = {
        str(item.get("stage") or ""): item
        for item in stage_outputs
        if isinstance(item, dict) and item.get("stage")
    }

    stages: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for entry in stage_status:
        if not isinstance(entry, dict):
            continue
        stage_name = str(entry.get("stage") or "").strip()
        if not stage_name:
            continue
        seen.add(stage_name)
        merged = dict(entry)
        merged.setdefault("label", _titleize(stage_name))
        output = output_by_stage.get(stage_name)
        if output:
            merged.setdefault("elapsed_seconds", output.get("elapsed_seconds"))
            merged.setdefault("output", output.get("output"))
            merged.setdefault("output_keys", output.get("output_keys"))
            merged.setdefault("output_count", output.get("output_count"))
            merged.setdefault("substeps", output.get("substeps"))
            merged.setdefault("output_type", output.get("output_type"))
        stages.append(merged)

    for entry in stage_outputs:
        if not isinstance(entry, dict):
            continue
        stage_name = str(entry.get("stage") or "").strip()
        if not stage_name or stage_name in seen:
            continue
        merged = dict(entry)
        merged.setdefault("label", _titleize(stage_name))
        stages.append(merged)

    stages.sort(key=lambda item: (item.get("timestamp") or "", item.get("stage") or ""))
    return stages


def _sample_entities(entities: Sequence[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for entity in list(entities or [])[:limit]:
        items.append(
            {
                "name": entity.get("name"),
                "entity_type": entity.get("entity_type"),
                "confidence": _safe_float(entity.get("confidence"), 0.0),
                "canonical_name": entity.get("canonical_name"),
                "stable_id": entity.get("stable_id"),
                "ontology_label": entity.get("ontology_label"),
                "ontology_type_id": entity.get("ontology_type_id"),
                "status": entity.get("status"),
                "unknown_candidate": entity.get("unknown_candidate"),
                "evidence_span": entity.get("evidence_span"),
            }
        )
    return items


def _sample_relations(relations: Sequence[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for relation in list(relations or [])[:limit]:
        items.append(
            {
                "source": relation.get("source"),
                "target": relation.get("target"),
                "relation_type": relation.get("relation_type"),
                "confidence": _safe_float(relation.get("confidence"), 0.0),
                "stable_id": relation.get("stable_id"),
                "ontology_label": relation.get("ontology_label"),
                "ontology_relation_id": relation.get("ontology_relation_id"),
                "status": relation.get("status"),
                "unknown_candidate": relation.get("unknown_candidate"),
                "evidence_span": relation.get("evidence_span"),
            }
        )
    return items


def build_api_catalog() -> Dict[str, Any]:
    return {
        "trigger": [
            {"method": "POST", "path": "/api/v1/workflows/analyze", "purpose": "Run the end-to-end document pipeline"},
            {"method": "GET", "path": "/api/v1/workflows/{job_id}/bundle", "purpose": "Fetch the rich dashboard bundle"},
            {"method": "GET", "path": "/api/v1/jobs/{job_id}", "purpose": "Fetch the raw stored job payload"},
        ],
        "copilot": [
            {"method": "GET", "path": "/api/v1/copilot/analyze/{job_id}", "purpose": "Full copilot reasoning"},
            {"method": "GET", "path": "/api/v1/copilot/rca/{job_id}", "purpose": "Root cause analysis"},
            {"method": "GET", "path": "/api/v1/copilot/maintenance/{job_id}", "purpose": "Maintenance plan"},
            {"method": "GET", "path": "/api/v1/copilot/compliance/{job_id}", "purpose": "Compliance summary"},
            {"method": "GET", "path": "/api/v1/copilot/risk/{job_id}", "purpose": "Risk assessment"},
        ],
        "advanced": [
            {"method": "GET", "path": "/api/v1/advanced/models/status", "purpose": "Advanced model availability"},
            {"method": "GET", "path": "/api/v1/advanced/pipeline-stages", "purpose": "Advanced stage catalog"},
            {"method": "POST", "path": "/api/v1/advanced/graph-reasoning", "purpose": "Graph reasoning"},
            {"method": "POST", "path": "/api/v1/advanced/doc-query", "purpose": "Document query with LlamaIndex"},
            {"method": "POST", "path": "/api/v1/advanced/llm-analysis", "purpose": "Direct LLM analysis"},
            {"method": "POST", "path": "/api/v1/advanced/anomaly-detection", "purpose": "Time-series anomaly detection"},
            {"method": "POST", "path": "/api/v1/advanced/rul-prediction", "purpose": "Remaining useful life prediction"},
            {"method": "POST", "path": "/api/v1/advanced/root-cause-analysis", "purpose": "Advanced RCA"},
            {"method": "POST", "path": "/api/v1/advanced/failure-prediction", "purpose": "Failure prediction"},
            {"method": "POST", "path": "/api/v1/advanced/lessons-learned", "purpose": "Lessons learned mining"},
            {"method": "POST", "path": "/api/v1/advanced/clustering", "purpose": "Embedding clustering"},
            {"method": "POST", "path": "/api/v1/advanced/graph-embeddings", "purpose": "Graph embeddings"},
            {"method": "POST", "path": "/api/v1/advanced/vector-search", "purpose": "Vector search"},
        ],
        "admin": [
            {"method": "POST", "path": "/api/v1/ontology/backfill", "purpose": "Legacy ontology backfill"},
            {"method": "POST", "path": "/api/v1/admin/neo4j/migrate-ontology", "purpose": "One-time ontology migration"},
        ],
    }


def build_workflow_bundle(job: Dict[str, Any], *, include_raw: bool = False) -> Dict[str, Any]:
    metadata = job.get("pipeline_metadata") if isinstance(job.get("pipeline_metadata"), dict) else {}
    entities = list(job.get("entities") or [])
    relations = list(job.get("relations") or [])
    timeline = build_stage_timeline(job)
    stage_counts = Counter(str(stage.get("status") or "").lower() for stage in timeline)

    ontology_report = job.get("ontology_report") if isinstance(job.get("ontology_report"), dict) else {}
    ontology_enrichment = job.get("ontology_enrichment") if isinstance(job.get("ontology_enrichment"), dict) else {}
    rag_analysis = job.get("rag_analysis") if isinstance(job.get("rag_analysis"), dict) else {}
    copilot_analysis = job.get("copilot_analysis") if isinstance(job.get("copilot_analysis"), dict) else {}

    overview_cards = [
        {"label": "Status", "value": job.get("status", "unknown"), "kind": "status"},
        {"label": "Entities", "value": _entity_count(entities), "kind": "metric"},
        {"label": "Relations", "value": _relation_count(relations), "kind": "metric"},
        {"label": "Stages", "value": len(timeline), "kind": "metric"},
        {"label": "Completed", "value": stage_counts.get("completed", 0), "kind": "metric"},
        {"label": "Failed", "value": stage_counts.get("failed", 0), "kind": "metric"},
    ]

    entity_types = Counter(str(entity.get("entity_type") or "unknown") for entity in entities)
    relation_types = Counter(str(relation.get("relation_type") or "related_to") for relation in relations)

    sections = [
        {
            "id": "overview",
            "title": "Overview",
            "kind": "metrics",
            "summary": _preview_text(job.get("message") or "Workflow completed"),
            "cards": overview_cards,
        },
        {
            "id": "timeline",
            "title": "Pipeline Timeline",
            "kind": "timeline",
            "summary": "Stage trace with summaries, nested substages, and execution timing.",
            "items": timeline,
        },
        {
            "id": "entities",
            "title": "Entities",
            "kind": "table",
            "summary": f"{len(entities)} extracted entities with ontology metadata.",
            "cards": [
                {"label": "Unique Types", "value": len(entity_types), "kind": "metric"},
                {"label": "Top Type", "value": entity_types.most_common(1)[0][0] if entity_types else "n/a", "kind": "metric"},
            ],
            "items": _sample_entities(entities, limit=16),
        },
        {
            "id": "relations",
            "title": "Relations",
            "kind": "table",
            "summary": f"{len(relations)} extracted relations and graph links.",
            "cards": [
                {"label": "Unique Types", "value": len(relation_types), "kind": "metric"},
                {"label": "Top Type", "value": relation_types.most_common(1)[0][0] if relation_types else "n/a", "kind": "metric"},
            ],
            "items": _sample_relations(relations, limit=16),
        },
        {
            "id": "ontology",
            "title": "Ontology",
            "kind": "analysis",
            "summary": _preview_text((ontology_report or {}).get("reason") or "Ontology enrichment and schema proposals"),
            "cards": [
                {"label": "Coverage", "value": _safe_float(ontology_report.get("coverage"), 0.0), "kind": "percent"},
                {"label": "Proposed Entities", "value": _safe_int(ontology_report.get("proposed_entities"), 0), "kind": "metric"},
                {"label": "Proposed Relations", "value": _safe_int(ontology_report.get("proposed_relations"), 0), "kind": "metric"},
            ],
            "items": {
                "report": ontology_report,
                "enrichment": ontology_enrichment,
                "proposals": job.get("ontology_proposals") or {},
                "schema_proposals": job.get("schema_proposals") or [],
            },
        },
        {
            "id": "reasoning",
            "title": "Reasoning",
            "kind": "analysis",
            "summary": "GraphRAG, copilot, and evidence-grounded reasoning outputs.",
            "items": {
                "rag_analysis": rag_analysis,
                "copilot_analysis": copilot_analysis,
                "advanced_graph": job.get("advanced_graph") or {},
                "advanced_llm": job.get("advanced_llm") or {},
                "advanced_semantic": job.get("advanced_semantic") or {},
            },
        },
        {
            "id": "analytics",
            "title": "Analytics",
            "kind": "analysis",
            "summary": "Predictive and exploratory outputs derived from the document graph.",
            "items": {
                "anomaly_detection": job.get("anomaly_detection") or {},
                "rul_prediction": job.get("rul_prediction") or {},
                "root_cause_analysis": job.get("root_cause_analysis") or {},
                "failure_prediction": job.get("failure_prediction") or {},
                "advanced_clustering": job.get("advanced_clustering") or {},
                "advanced_lessons_learned": job.get("advanced_lessons_learned") or {},
                "advanced_graph_embeddings": job.get("advanced_graph_embeddings") or {},
            },
        },
        {
            "id": "artifacts",
            "title": "Artifacts",
            "kind": "artifacts",
            "summary": "OCR, layout, tables, segmentation, and visual evidence artifacts.",
            "items": {
                "document_segments": job.get("document_segments") or [],
                "layout": job.get("layout") or [],
                "tables": job.get("tables") or [],
                "formulas": job.get("formulas") or [],
                "reading_order": job.get("reading_order") or [],
                "doclayout_yolo": job.get("doclayout_yolo") or {},
                "table_transformer": job.get("table_transformer") or {},
                "groundingdino": job.get("groundingdino") or {},
                "sam_segments": job.get("sam_segments") or {},
                "yolo_pid_insights": job.get("yolo_pid_insights") or {},
                "pid_symbol_insights": job.get("pid_symbol_insights") or {},
                "pid_components": job.get("pid_components") or [],
                "semantic_indexing": job.get("semantic_indexing") or {},
                "bge_ranking": job.get("bge_ranking") or {},
            },
        },
        {
            "id": "apis",
            "title": "API Catalog",
            "kind": "catalog",
            "summary": "Limited frontend-facing triggers and the supporting domain APIs.",
            "items": build_api_catalog(),
        },
    ]

    summary = {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "uploaded_filename": job.get("uploaded_filename"),
        "message": job.get("message"),
        "timestamp": job.get("timestamp"),
        "entity_count": len(entities),
        "relation_count": len(relations),
        "stage_count": len(timeline),
        "completed_stages": stage_counts.get("completed", 0),
        "failed_stages": stage_counts.get("failed", 0),
        "skipped_stages": stage_counts.get("skipped", 0),
        "proposed_entities": _safe_int(ontology_report.get("proposed_entities"), 0),
        "proposed_relations": _safe_int(ontology_report.get("proposed_relations"), 0),
        "ontology_coverage": _safe_float(ontology_report.get("coverage"), 0.0),
        "neo4j_status": job.get("neo4j_status"),
    }

    bundle: Dict[str, Any] = {
        "summary": summary,
        "timeline": timeline,
        "sections": sections,
        "capabilities": {
            "pipeline_metadata": metadata,
            "api_catalog": build_api_catalog(),
        },
        "highlights": {
            "entities": _sample_entities(entities, limit=12),
            "relations": _sample_relations(relations, limit=12),
            "stage_outputs": list(metadata.get("stage_outputs") or [])[:12],
        },
    }

    if include_raw:
        bundle["raw"] = job

    return bundle
