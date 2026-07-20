import re
from typing import Any, Dict, List, Optional
from app.pipeline.component_detector import detect_pid_components_v2

MODEL_STAGE_SEQUENCE = [
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
]

DEFAULT_COMPONENT_TERMS = [
    "pump",
    "valve",
    "motor",
    "sensor",
    "line",
    "control",
    "compressor",
    "tank",
    "scada",
    "plc",
]


def get_model_stage_manifest() -> List[Dict[str, Any]]:
    return [{"name": name, "mode": "heuristic-fallback"} for name in MODEL_STAGE_SEQUENCE]


def detect_pid_components(text: str) -> List[str]:
    """
    Legacy function for backward compatibility.
    Returns simple list of component names found in text.
    """
    lowered = text.lower()
    hits = []
    for term in DEFAULT_COMPONENT_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            hits.append(term)
    return hits


def detect_pid_components_enhanced(
    text: str, 
    entities: Optional[List[Dict[str, Any]]] = None,
    page_map: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Improved PID component detection with localization and canonical mapping.
    
    Args:
        text: Full OCR text
        entities: Optional entities from stage 15
        page_map: Optional page-to-text mapping for localization
    
    Returns:
        Full pipeline output dict with enhanced component metadata
    """
    return detect_pid_components_v2(text, entities=entities, page_map=page_map)


def canonicalize_entity_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def normalize_entity_payload(
    entity: Dict[str, Any],
    *,
    registry: Any = None,
    default_status: str = "proposed",
) -> Dict[str, Any]:
    """Normalize an entity payload into the ontology-aware runtime contract."""
    from app.pipeline.ontology import load_ontology_registry

    resolved_registry = registry or load_ontology_registry()

    name = str(entity.get("name") or entity.get("canonical_name") or entity.get("stable_id") or "").strip()
    entity_type = str(entity.get("entity_type") or entity.get("type") or "").strip()
    canonical_name = str(entity.get("canonical_name") or canonicalize_entity_name(name) or entity_type).strip()
    stable_id = str(entity.get("stable_id") or entity.get("canonical_id") or canonical_name or canonicalize_entity_name(name)).strip()
    incoming_unknown_candidate = entity.get("unknown_candidate")

    match = resolved_registry.resolve_entity(name or canonical_name, entity_type=entity_type or None, context=str(entity.get("context") or entity.get("evidence") or ""))
    proposal = None
    if match is not None:
        ontology_type_id = match.type_id
        ontology_label = match.label
        ontology_parent_type_id = match.parent_type_id
        ontology_status = match.status or "active"
        ontology_confidence = float(match.score)
        ontology_path = list(match.path)
        ontology = {
            "type_id": match.type_id,
            "label": match.label,
            "parent_type_id": match.parent_type_id,
            "status": ontology_status,
            "confidence": ontology_confidence,
            "path": list(match.path),
            "source": match.source,
            "reason": match.reason,
        }
        if isinstance(incoming_unknown_candidate, dict):
            unknown_candidate = {**incoming_unknown_candidate}
            unknown_candidate.setdefault("candidate_label", name or canonical_name or ontology_label)
            unknown_candidate.setdefault("candidate_type", entity_type or ontology_label or "unknown")
            unknown_candidate.setdefault("parent_type_id", ontology_parent_type_id or unknown_candidate.get("parent_type_id") or "entity")
            unknown_candidate.setdefault("reason", unknown_candidate.get("reason") or "prompt-proposed candidate")
        elif incoming_unknown_candidate:
            unknown_candidate = {
                "candidate_label": name or canonical_name or ontology_label,
                "candidate_type": entity_type or ontology_label or "unknown",
                "parent_type_id": ontology_parent_type_id or "entity",
                "reason": "prompt-proposed candidate",
            }
        else:
            unknown_candidate = None
    else:
        proposal = resolved_registry.propose_entity(name or canonical_name, entity_type=entity_type or None, context=str(entity.get("context") or entity.get("evidence") or ""), confidence=float(entity.get("confidence", 0.0) or 0.0))
        ontology_type_id = proposal.candidate_id
        ontology_label = proposal.label
        ontology_parent_type_id = proposal.parent_type_id
        ontology_status = proposal.status or default_status
        ontology_confidence = float(proposal.confidence)
        ontology_path = [proposal.parent_type_id] if proposal.parent_type_id else []
        ontology = {
            "type_id": proposal.candidate_id,
            "label": proposal.label,
            "parent_type_id": proposal.parent_type_id,
            "status": ontology_status,
            "confidence": ontology_confidence,
            "path": ontology_path,
            "source": proposal.source,
            "reason": "proposed",
        }
        if isinstance(incoming_unknown_candidate, dict):
            unknown_candidate = {**incoming_unknown_candidate}
        else:
            unknown_candidate = {}
        unknown_candidate.setdefault("candidate_label", unknown_candidate.get("candidate_label") or proposal.label)
        unknown_candidate.setdefault("candidate_type", unknown_candidate.get("candidate_type") or entity_type or "unknown")
        unknown_candidate.setdefault("parent_type_id", unknown_candidate.get("parent_type_id") or proposal.parent_type_id or "entity")
        unknown_candidate.setdefault("reason", unknown_candidate.get("reason") or "proposed from zero-shot extraction")
        unknown_candidate.setdefault("confidence", float(unknown_candidate.get("confidence") or proposal.confidence))

    return {
        **entity,
        "name": name or canonical_name,
        "canonical_name": canonical_name or name,
        "stable_id": stable_id,
        "entity_type": entity_type or "unknown",
        "confidence": float(entity.get("confidence", 0.0) or 0.0),
        "ontology_type_id": ontology_type_id,
        "ontology_label": ontology_label,
        "ontology_parent_type_id": ontology_parent_type_id,
        "ontology_status": ontology_status,
        "ontology_confidence": ontology_confidence,
        "ontology_source": match.source if match is not None else proposal.source,
        "ontology_reason": match.reason if match is not None else "proposed",
        "ontology_path": ontology_path,
        "ontology": ontology,
        "unknown_candidate": unknown_candidate,
        "schema_version": str(entity.get("schema_version") or "1.0.0"),
        "status": entity.get("status") or ontology_status,
        "type_id": entity.get("type_id") or ontology_type_id,
        "parent_type_id": entity.get("parent_type_id") or ontology_parent_type_id,
        "provenance": entity.get("provenance") or {
            "source_document": entity.get("source_document"),
            "source_method": entity.get("source") or "heuristic",
            "evidence": entity.get("evidence") or entity.get("context") or "",
        },
    }


def normalize_relation_payload(
    relation: Dict[str, Any],
    *,
    registry: Any = None,
    default_status: str = "proposed",
) -> Dict[str, Any]:
    """Normalize a relation payload into the ontology-aware runtime contract."""
    from app.pipeline.ontology import load_ontology_registry

    resolved_registry = registry or load_ontology_registry()

    source = str(relation.get("source") or "").strip()
    target = str(relation.get("target") or "").strip()
    relation_type = str(relation.get("relation_type") or relation.get("type") or "related_to").strip() or "related_to"
    source_stable_id = str(relation.get("source_stable_id") or relation.get("source_id") or canonicalize_entity_name(source)).strip()
    target_stable_id = str(relation.get("target_stable_id") or relation.get("target_id") or canonicalize_entity_name(target)).strip()
    stable_id = str(relation.get("stable_id") or relation.get("relation_id") or f"{source_stable_id}__{canonicalize_entity_name(relation_type)}__{target_stable_id}").strip()
    incoming_unknown_candidate = relation.get("unknown_candidate")

    match = resolved_registry.resolve_relation(relation_type, context=str(relation.get("context") or relation.get("evidence") or ""))
    if match is not None:
        ontology_relation_id = match.type_id
        ontology_label = match.label
        ontology_status = match.status or "active"
        ontology_confidence = float(match.score)
        ontology = {
            "relation_id": match.type_id,
            "label": match.label,
            "status": ontology_status,
            "confidence": ontology_confidence,
            "source": match.source,
            "reason": match.reason,
        }
        if isinstance(incoming_unknown_candidate, dict):
            unknown_candidate = {**incoming_unknown_candidate}
            unknown_candidate.setdefault("candidate_label", relation_type or ontology_label)
            unknown_candidate.setdefault("candidate_type", relation_type or ontology_label or "unknown")
            unknown_candidate.setdefault("parent_type_id", "related_to")
            unknown_candidate.setdefault("reason", unknown_candidate.get("reason") or "prompt-proposed candidate")
        elif incoming_unknown_candidate:
            unknown_candidate = {
                "candidate_label": relation_type or ontology_label,
                "candidate_type": relation_type or ontology_label or "unknown",
                "parent_type_id": "related_to",
                "reason": "prompt-proposed candidate",
            }
        else:
            unknown_candidate = None
    else:
        proposal = resolved_registry.propose_relation(relation_type, context=str(relation.get("context") or relation.get("evidence") or ""), confidence=float(relation.get("confidence", 0.0) or 0.0))
        ontology_relation_id = proposal.candidate_id
        ontology_label = proposal.label
        ontology_status = proposal.status or default_status
        ontology_confidence = float(proposal.confidence)
        ontology = {
            "relation_id": proposal.candidate_id,
            "label": proposal.label,
            "status": ontology_status,
            "confidence": ontology_confidence,
            "source": proposal.source,
            "reason": "proposed",
        }
        if isinstance(incoming_unknown_candidate, dict):
            unknown_candidate = {**incoming_unknown_candidate}
        else:
            unknown_candidate = {}
        unknown_candidate.setdefault("candidate_label", unknown_candidate.get("candidate_label") or proposal.label)
        unknown_candidate.setdefault("candidate_type", unknown_candidate.get("candidate_type") or relation_type or "unknown")
        unknown_candidate.setdefault("parent_type_id", unknown_candidate.get("parent_type_id") or "related_to")
        unknown_candidate.setdefault("reason", unknown_candidate.get("reason") or "proposed from zero-shot extraction")
        unknown_candidate.setdefault("confidence", float(unknown_candidate.get("confidence") or proposal.confidence))

    return {
        **relation,
        "source": source,
        "target": target,
        "source_stable_id": source_stable_id,
        "target_stable_id": target_stable_id,
        "stable_id": stable_id,
        "relation_type": relation_type,
        "confidence": float(relation.get("confidence", 0.0) or 0.0),
        "ontology_relation_id": ontology_relation_id,
        "ontology_label": ontology_label,
        "ontology_status": ontology_status,
        "ontology_confidence": ontology_confidence,
        "ontology_source": match.source if match is not None else proposal.source,
        "ontology_reason": match.reason if match is not None else "proposed",
        "ontology": ontology,
        "unknown_candidate": unknown_candidate,
        "schema_version": str(relation.get("schema_version") or "1.0.0"),
        "status": relation.get("status") or ontology_status,
        "type_id": relation.get("type_id") or ontology_relation_id,
        "provenance": relation.get("provenance") or {
            "source_document": relation.get("source_document"),
            "source_method": relation.get("source_method") or relation.get("source") or "heuristic",
            "evidence": relation.get("evidence") or relation.get("context") or "",
        },
    }
