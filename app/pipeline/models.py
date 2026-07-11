import re
from typing import Any, Dict, List

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
    lowered = text.lower()
    hits = []
    for term in DEFAULT_COMPONENT_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            hits.append(term)
    return hits


def canonicalize_entity_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
