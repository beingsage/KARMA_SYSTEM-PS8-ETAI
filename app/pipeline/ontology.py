"""Ontology registry, enrichment, and evolution utilities.

This module turns the markdown ontology plan into a versioned runtime contract:
- a stable core registry for entity and relation types
- zero-shot classification against the registry
- few-shot retrieval from seeded and observed examples
- proposal handling for unknown concepts
- persistent schema evolution state that never mutates existing IDs in place
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.config import settings
from app.pipeline.models import canonicalize_entity_name

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY_STATE_PATH = settings.data_dir / "ontology" / "ontology_state.json"
DEFAULT_MARKDOWN_PACK_ORDER = {"zero": 0, "first": 1, "second": 2, "third": 3}


def _resolve_markdown_pack_dirs() -> Tuple[Path, ...]:
    configured = [
        part.strip()
        for part in (settings.ontology_pack_dirs or "").split(os.pathsep)
        if part.strip()
    ]
    if not configured:
        configured = ["app/data.models"]

    resolved: List[Path] = []
    for entry in configured:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        resolved.append(candidate)
    return tuple(dict.fromkeys(resolved))


def discover_markdown_ontology_pack_files(base_dirs: Sequence[Path] | None = None) -> Tuple[Path, ...]:
    """Discover markdown ontology pack files without relying on a curated list."""

    search_dirs = tuple(base_dirs or _resolve_markdown_pack_dirs())
    discovered: List[Path] = []
    for directory in search_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            if path.is_file():
                discovered.append(path.resolve())

    def _sort_key(path: Path) -> Tuple[int, str, str]:
        stem = path.stem.lower()
        return (
            DEFAULT_MARKDOWN_PACK_ORDER.get(stem, 1000),
            stem,
            str(path),
        )

    unique = list(dict.fromkeys(discovered))
    unique.sort(key=_sort_key)
    return tuple(unique)


ONTOLOGY_SOURCE_PATHS = discover_markdown_ontology_pack_files()
ONTOLOGY_SOURCE_DOCS = tuple(str(path) for path in ONTOLOGY_SOURCE_PATHS)
ONTOLOGY_SOURCE_DOC_MAP = {path.stem.lower(): str(path) for path in ONTOLOGY_SOURCE_PATHS}


def _source_doc_for_stem(stem: str, fallback_index: int = 0) -> str:
    if stem.lower() in ONTOLOGY_SOURCE_DOC_MAP:
        return ONTOLOGY_SOURCE_DOC_MAP[stem.lower()]
    if 0 <= fallback_index < len(ONTOLOGY_SOURCE_DOCS):
        return ONTOLOGY_SOURCE_DOCS[fallback_index]
    return str((REPO_ROOT / "app" / "data.models" / f"{stem}.md").resolve())


def _now() -> str:
    return datetime.now().isoformat()


def _normalize(text: str) -> str:
    return canonicalize_entity_name(text or "")


def _tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if token]


def _stable_hash(text: str) -> str:
    return hashlib.md5((text or "").strip().lower().encode("utf-8")).hexdigest()


def _build_example_text(label: str, examples: Sequence[str], description: str = "") -> str:
    parts = [label]
    if description:
        parts.append(description)
    parts.extend(examples)
    return " | ".join(part for part in parts if part)


@dataclass(frozen=True)
class OntologyTypeDefinition:
    type_id: str
    label: str
    parent_type_id: Optional[str] = None
    aliases: Tuple[str, ...] = ()
    keywords: Tuple[str, ...] = ()
    description: str = ""
    layer: str = ""
    pack: str = "core"
    status: str = "active"
    source_docs: Tuple[str, ...] = ()
    examples: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OntologyRelationDefinition:
    relation_id: str
    label: str
    aliases: Tuple[str, ...] = ()
    keywords: Tuple[str, ...] = ()
    source_type_ids: Tuple[str, ...] = ()
    target_type_ids: Tuple[str, ...] = ()
    description: str = ""
    pack: str = "core"
    status: str = "active"
    source_docs: Tuple[str, ...] = ()
    examples: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OntologyMatch:
    kind: str
    type_id: str
    label: str
    score: float
    status: str
    parent_type_id: Optional[str] = None
    source: str = "registry"
    reason: str = ""
    path: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    examples: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OntologyProposal:
    kind: str
    candidate_id: str
    label: str
    parent_type_id: Optional[str] = None
    status: str = "proposed"
    confidence: float = 0.0
    observed_count: int = 0
    source: str = "zero_shot"
    evidence: str = ""
    aliases: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    source_docs: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OntologyMarkdownPack:
    pack_id: str
    title: str
    path: str
    summary: str
    section_count: int = 0
    bullet_count: int = 0
    content_hash: str = ""
    excerpt: str = ""
    keywords: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _type(
    type_id: str,
    label: str,
    *,
    parent_type_id: Optional[str] = None,
    aliases: Sequence[str] = (),
    keywords: Sequence[str] = (),
    description: str = "",
    layer: str = "",
    pack: str = "core",
    status: str = "active",
    source_docs: Sequence[str] = (),
    examples: Sequence[str] = (),
) -> OntologyTypeDefinition:
    return OntologyTypeDefinition(
        type_id=type_id,
        label=label,
        parent_type_id=parent_type_id,
        aliases=tuple(dict.fromkeys([label, *aliases])),
        keywords=tuple(dict.fromkeys([*keywords, *_tokenize(label)])),
        description=description,
        layer=layer,
        pack=pack,
        status=status,
        source_docs=tuple(source_docs),
        examples=tuple(examples),
    )


def _relation(
    relation_id: str,
    label: str,
    *,
    aliases: Sequence[str] = (),
    keywords: Sequence[str] = (),
    source_type_ids: Sequence[str] = (),
    target_type_ids: Sequence[str] = (),
    description: str = "",
    pack: str = "core",
    status: str = "active",
    source_docs: Sequence[str] = (),
    examples: Sequence[str] = (),
) -> OntologyRelationDefinition:
    return OntologyRelationDefinition(
        relation_id=relation_id,
        label=label,
        aliases=tuple(dict.fromkeys([label, *aliases])),
        keywords=tuple(dict.fromkeys([*keywords, *_tokenize(label)])),
        source_type_ids=tuple(source_type_ids),
        target_type_ids=tuple(target_type_ids),
        description=description,
        pack=pack,
        status=status,
        source_docs=tuple(source_docs),
        examples=tuple(examples),
    )


def _load_markdown_pack_manifest(path: Path) -> OntologyMarkdownPack:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        content = ""

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    title = lines[0] if lines else path.stem.replace("_", " ").title()
    non_meta_lines = [line for line in lines if line not in {"---", "```"}]
    summary = " ".join(non_meta_lines[:5]).strip()
    summary = summary[:320] if summary else title
    section_count = len(re.findall(r"^(?:#|\d+\.)\s+", content, flags=re.MULTILINE))
    bullet_count = len(re.findall(r"^\s*[-*]\s+", content, flags=re.MULTILINE))
    keywords = _tokenize(f"{path.stem} {title} {summary}")
    return OntologyMarkdownPack(
        pack_id=path.stem.lower(),
        title=title,
        path=str(path),
        summary=summary,
        section_count=section_count,
        bullet_count=bullet_count,
        content_hash=_stable_hash(content),
        excerpt=content.strip().replace("\n", " ")[:240],
        keywords=tuple(dict.fromkeys(keywords[:12])),
    )


def _group_types(
    *,
    prefix: str,
    label: str,
    parent_type_id: Optional[str],
    entries: Sequence[Dict[str, Any]],
    layer: str,
    source_docs: Sequence[str],
    pack: Optional[str] = None,
) -> List[OntologyTypeDefinition]:
    resolved_pack = pack or ("core" if layer == "kernel" else f"{re.sub(r'[^a-z0-9]+', '_', layer.lower()).strip('_')}_pack")
    definitions = [
        _type(
            prefix,
            label,
            parent_type_id=parent_type_id,
            aliases=[prefix.replace("_", " ")],
            keywords=[prefix],
            layer=layer,
            pack=resolved_pack,
            source_docs=source_docs,
            examples=[label],
        )
    ]
    for entry in entries:
        suffix = entry["suffix"]
        definitions.append(
            _type(
                f"{prefix}.{suffix}",
                entry["label"],
                parent_type_id=prefix,
                aliases=entry.get("aliases", ()),
                keywords=entry.get("keywords", ()),
                description=entry.get("description", ""),
                layer=layer,
                pack=resolved_pack,
                examples=entry.get("examples", ()),
                source_docs=source_docs,
            )
        )
    return definitions


def _build_core_type_definitions() -> List[OntologyTypeDefinition]:
    types: List[OntologyTypeDefinition] = [
        _type("entity", "Entity", layer="kernel", source_docs=ONTOLOGY_SOURCE_DOCS, examples=["Any node in the knowledge graph"]),
        _type("metadata", "Metadata", parent_type_id="entity", layer="kernel", source_docs=(ONTOLOGY_SOURCE_DOCS[2],), examples=["Version, status, owner, confidence"]),
        _type("provenance", "Provenance", parent_type_id="entity", layer="kernel", source_docs=(ONTOLOGY_SOURCE_DOCS[2],), examples=["Source document, page, paragraph, citation"]),
        _type("spatial", "Spatial", parent_type_id="entity", layer="kernel", source_docs=(ONTOLOGY_SOURCE_DOCS[1],), examples=["Building, floor, room, coordinates"]),
        _type("event", "Event", parent_type_id="entity", layer="kernel", source_docs=(ONTOLOGY_SOURCE_DOCS[1],), examples=["Startup, shutdown, failure, alarm"]),
        _type("workflow", "Workflow", parent_type_id="entity", layer="kernel", source_docs=(ONTOLOGY_SOURCE_DOCS[1],), examples=["Task, approval, verification"]),
        _type("iot", "IoT", parent_type_id="entity", layer="kernel", source_docs=(ONTOLOGY_SOURCE_DOCS[1],), examples=["Sensor, telemetry, anomaly"]),
        _type("knowledge_quality", "Knowledge Quality", parent_type_id="entity", layer="kernel", source_docs=(ONTOLOGY_SOURCE_DOCS[2],), examples=["Completeness, freshness, citation density"]),
    ]

    types.extend(
        _group_types(
            prefix="organization",
            label="Organization",
            parent_type_id="entity",
            layer="organization",
            source_docs=(ONTOLOGY_SOURCE_DOCS[1],),
            entries=[
                {"suffix": "business_unit", "label": "Business Unit", "aliases": ["BU"], "examples": ["Maintenance business unit"]},
                {"suffix": "plant", "label": "Plant", "examples": ["Plant A", "Plant 3"]},
                {"suffix": "site", "label": "Site", "examples": ["Refinery site"]},
                {"suffix": "department", "label": "Department", "examples": ["Maintenance department"]},
                {"suffix": "division", "label": "Division"},
                {"suffix": "workshop", "label": "Workshop"},
                {"suffix": "production_line", "label": "Production Line", "aliases": ["line"]},
                {"suffix": "area", "label": "Area"},
                {"suffix": "zone", "label": "Zone"},
                {"suffix": "building", "label": "Building"},
                {"suffix": "floor", "label": "Floor"},
                {"suffix": "room", "label": "Room"},
                {"suffix": "vendor", "label": "Vendor", "examples": ["Vendor A"]},
                {"suffix": "customer", "label": "Customer", "examples": ["Customer site"]},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="human",
            label="Human",
            parent_type_id="entity",
            layer="human_knowledge",
            source_docs=(ONTOLOGY_SOURCE_DOCS[1],),
            entries=[
                {"suffix": "employee", "label": "Employee"},
                {"suffix": "operator", "label": "Operator", "examples": ["Operator A"]},
                {"suffix": "engineer", "label": "Engineer"},
                {"suffix": "maintenance_engineer", "label": "Maintenance Engineer"},
                {"suffix": "technician", "label": "Technician"},
                {"suffix": "supervisor", "label": "Supervisor"},
                {"suffix": "manager", "label": "Manager"},
                {"suffix": "safety_officer", "label": "Safety Officer"},
                {"suffix": "inspector", "label": "Inspector"},
                {"suffix": "auditor", "label": "Auditor"},
                {"suffix": "contractor", "label": "Contractor"},
                {"suffix": "expert", "label": "Expert"},
                {"suffix": "trainer", "label": "Trainer"},
                {"suffix": "shift", "label": "Shift"},
                {"suffix": "role", "label": "Role"},
                {"suffix": "skill", "label": "Skill"},
                {"suffix": "certification", "label": "Certification"},
                {"suffix": "training_record", "label": "Training Record"},
                {"suffix": "experience", "label": "Experience"},
                {"suffix": "knowledge_note", "label": "Knowledge Note"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="asset",
            label="Asset",
            parent_type_id="entity",
            layer="asset",
            source_docs=(ONTOLOGY_SOURCE_DOCS[1],),
            entries=[
                {"suffix": "equipment", "label": "Equipment", "examples": ["Pump-7 equipment"]},
                {"suffix": "machine", "label": "Machine"},
                {"suffix": "pump", "label": "Pump", "aliases": ["centrifugal pump", "booster pump"], "examples": ["CR 120", "Pump-7"]},
                {"suffix": "motor", "label": "Motor", "examples": ["Motor B"]},
                {"suffix": "valve", "label": "Valve", "examples": ["Valve-12"]},
                {"suffix": "compressor", "label": "Compressor"},
                {"suffix": "heat_exchanger", "label": "Heat Exchanger"},
                {"suffix": "tank", "label": "Tank"},
                {"suffix": "boiler", "label": "Boiler"},
                {"suffix": "generator", "label": "Generator"},
                {"suffix": "transformer", "label": "Transformer"},
                {"suffix": "bearing", "label": "Bearing"},
                {"suffix": "seal", "label": "Seal"},
                {"suffix": "gearbox", "label": "Gearbox"},
                {"suffix": "conveyor", "label": "Conveyor"},
                {"suffix": "sensor", "label": "Sensor", "examples": ["Pressure sensor", "Temperature sensor"]},
                {"suffix": "actuator", "label": "Actuator"},
                {"suffix": "plc", "label": "PLC", "aliases": ["programmable logic controller", "controller"]},
                {"suffix": "dcs", "label": "DCS"},
                {"suffix": "scada_node", "label": "SCADA Node", "aliases": ["scada"]},
                {"suffix": "panel", "label": "Panel"},
                {"suffix": "pipe", "label": "Pipe"},
                {"suffix": "pipeline", "label": "Pipeline"},
                {"suffix": "instrument", "label": "Instrument"},
                {"suffix": "cable", "label": "Cable"},
                {"suffix": "foundation", "label": "Foundation"},
                {"suffix": "spare_part", "label": "Spare Part"},
                {"suffix": "assembly", "label": "Assembly"},
                {"suffix": "subassembly", "label": "Subassembly"},
                {"suffix": "control_system", "label": "Control System", "aliases": ["automation system"]},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="document",
            label="Document",
            parent_type_id="entity",
            layer="document",
            source_docs=(ONTOLOGY_SOURCE_DOCS[1],),
            entries=[
                {"suffix": "pdf", "label": "PDF"},
                {"suffix": "sop", "label": "SOP", "examples": ["Startup SOP"]},
                {"suffix": "manual", "label": "Manual"},
                {"suffix": "oem_manual", "label": "OEM Manual"},
                {"suffix": "drawing", "label": "Drawing"},
                {"suffix": "pid", "label": "P&ID", "aliases": ["PID", "P&ID"]},
                {"suffix": "isometric_drawing", "label": "Isometric Drawing"},
                {"suffix": "maintenance_record", "label": "Maintenance Record"},
                {"suffix": "inspection_report", "label": "Inspection Report"},
                {"suffix": "permit", "label": "Permit"},
                {"suffix": "checklist", "label": "Checklist"},
                {"suffix": "calibration_report", "label": "Calibration Report"},
                {"suffix": "work_order", "label": "Work Order"},
                {"suffix": "purchase_order", "label": "Purchase Order"},
                {"suffix": "invoice", "label": "Invoice"},
                {"suffix": "email", "label": "Email"},
                {"suffix": "memo", "label": "Memo"},
                {"suffix": "audit_report", "label": "Audit Report"},
                {"suffix": "incident_report", "label": "Incident Report"},
                {"suffix": "near_miss", "label": "Near Miss"},
                {"suffix": "root_cause_analysis", "label": "Root Cause Analysis"},
                {"suffix": "risk_assessment", "label": "Risk Assessment"},
                {"suffix": "standard", "label": "Standard"},
                {"suffix": "specification", "label": "Specification"},
                {"suffix": "msds", "label": "MSDS", "aliases": ["sds"]},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="operation",
            label="Operation",
            parent_type_id="entity",
            layer="operations",
            source_docs=(ONTOLOGY_SOURCE_DOCS[1],),
            entries=[
                {"suffix": "startup", "label": "Startup"},
                {"suffix": "shutdown", "label": "Shutdown"},
                {"suffix": "batch", "label": "Batch"},
                {"suffix": "production_run", "label": "Production Run"},
                {"suffix": "recipe", "label": "Recipe"},
                {"suffix": "shift_log", "label": "Shift Log"},
                {"suffix": "alarm", "label": "Alarm"},
                {"suffix": "event", "label": "Event"},
                {"suffix": "trip", "label": "Trip"},
                {"suffix": "fault", "label": "Fault"},
                {"suffix": "maintenance_event", "label": "Maintenance"},
                {"suffix": "breakdown", "label": "Breakdown"},
                {"suffix": "inspection_event", "label": "Inspection"},
                {"suffix": "calibration_event", "label": "Calibration"},
                {"suffix": "cleaning", "label": "Cleaning"},
                {"suffix": "repair", "label": "Repair"},
                {"suffix": "replacement", "label": "Replacement"},
                {"suffix": "testing", "label": "Testing"},
                {"suffix": "commissioning", "label": "Commissioning"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="process",
            label="Process",
            parent_type_id="entity",
            layer="process_engineering",
            source_docs=(ONTOLOGY_SOURCE_DOCS[0], ONTOLOGY_SOURCE_DOCS[1]),
            entries=[
                {"suffix": "unit_operation", "label": "Unit Operation"},
                {"suffix": "flow", "label": "Flow"},
                {"suffix": "pressure", "label": "Pressure"},
                {"suffix": "temperature", "label": "Temperature"},
                {"suffix": "level", "label": "Level"},
                {"suffix": "density", "label": "Density"},
                {"suffix": "viscosity", "label": "Viscosity"},
                {"suffix": "flow_rate", "label": "Flow Rate"},
                {"suffix": "setpoint", "label": "Setpoint"},
                {"suffix": "operating_window", "label": "Operating Window"},
                {"suffix": "control_loop", "label": "Control Loop"},
                {"suffix": "pid_controller", "label": "PID Controller"},
                {"suffix": "feed", "label": "Feed"},
                {"suffix": "product", "label": "Product"},
                {"suffix": "intermediate", "label": "Intermediate"},
                {"suffix": "raw_material", "label": "Raw Material"},
                {"suffix": "chemical", "label": "Chemical"},
                {"suffix": "catalyst", "label": "Catalyst"},
                {"suffix": "reaction", "label": "Reaction"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="quality",
            label="Quality",
            parent_type_id="entity",
            layer="quality",
            source_docs=(ONTOLOGY_SOURCE_DOCS[1],),
            entries=[
                {"suffix": "quality_check", "label": "Quality Check"},
                {"suffix": "inspection", "label": "Inspection"},
                {"suffix": "defect", "label": "Defect"},
                {"suffix": "deviation", "label": "Deviation"},
                {"suffix": "capa", "label": "CAPA"},
                {"suffix": "ncr", "label": "NCR"},
                {"suffix": "sample", "label": "Sample"},
                {"suffix": "result", "label": "Result"},
                {"suffix": "measurement", "label": "Measurement"},
                {"suffix": "test", "label": "Test"},
                {"suffix": "certificate", "label": "Certificate"},
                {"suffix": "acceptance_criteria", "label": "Acceptance Criteria"},
                {"suffix": "rejection", "label": "Rejection"},
                {"suffix": "trend", "label": "Trend"},
                {"suffix": "complaint", "label": "Complaint"},
                {"suffix": "audit_finding", "label": "Audit Finding"},
                {"suffix": "observation", "label": "Observation"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="safety",
            label="Safety",
            parent_type_id="entity",
            layer="safety",
            source_docs=(ONTOLOGY_SOURCE_DOCS[1],),
            entries=[
                {"suffix": "hazard", "label": "Hazard"},
                {"suffix": "risk", "label": "Risk"},
                {"suffix": "incident", "label": "Incident"},
                {"suffix": "near_miss", "label": "Near Miss"},
                {"suffix": "loto", "label": "LOTO"},
                {"suffix": "ppe", "label": "PPE"},
                {"suffix": "permit", "label": "Permit"},
                {"suffix": "fire", "label": "Fire"},
                {"suffix": "explosion", "label": "Explosion"},
                {"suffix": "leak", "label": "Leak"},
                {"suffix": "gas_release", "label": "Gas Release"},
                {"suffix": "confined_space", "label": "Confined Space"},
                {"suffix": "hot_work", "label": "Hot Work"},
                {"suffix": "emergency", "label": "Emergency"},
                {"suffix": "evacuation", "label": "Evacuation"},
                {"suffix": "risk_matrix", "label": "Risk Matrix"},
                {"suffix": "barrier", "label": "Barrier"},
                {"suffix": "safety_observation", "label": "Safety Observation"},
                {"suffix": "unsafe_act", "label": "Unsafe Act"},
                {"suffix": "unsafe_condition", "label": "Unsafe Condition"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="maintenance",
            label="Maintenance",
            parent_type_id="entity",
            layer="maintenance",
            source_docs=(ONTOLOGY_SOURCE_DOCS[1],),
            entries=[
                {"suffix": "preventive_maintenance", "label": "Preventive Maintenance"},
                {"suffix": "predictive_maintenance", "label": "Predictive Maintenance"},
                {"suffix": "corrective_maintenance", "label": "Corrective Maintenance"},
                {"suffix": "failure", "label": "Failure"},
                {"suffix": "failure_mode", "label": "Failure Mode"},
                {"suffix": "failure_cause", "label": "Failure Cause"},
                {"suffix": "failure_effect", "label": "Failure Effect"},
                {"suffix": "fmea", "label": "FMEA"},
                {"suffix": "rca", "label": "RCA"},
                {"suffix": "mtbf", "label": "MTBF"},
                {"suffix": "mttr", "label": "MTTR"},
                {"suffix": "spare", "label": "Spare"},
                {"suffix": "inventory", "label": "Inventory"},
                {"suffix": "tool", "label": "Tool"},
                {"suffix": "lubricant", "label": "Lubricant"},
                {"suffix": "maintenance_plan", "label": "Maintenance Plan"},
                {"suffix": "task", "label": "Task"},
                {"suffix": "inspection_route", "label": "Inspection Route"},
                {"suffix": "schedule", "label": "Schedule"},
                {"suffix": "downtime", "label": "Downtime"},
                {"suffix": "uptime", "label": "Uptime"},
                {"suffix": "cost", "label": "Cost"},
                {"suffix": "labor", "label": "Labor"},
                {"suffix": "vendor_visit", "label": "Vendor Visit"},
                {"suffix": "warranty", "label": "Warranty"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="regulatory",
            label="Regulatory",
            parent_type_id="entity",
            layer="regulatory",
            source_docs=(ONTOLOGY_SOURCE_DOCS[1],),
            entries=[
                {"suffix": "iso_standard", "label": "ISO Standard"},
                {"suffix": "oisd", "label": "OISD"},
                {"suffix": "factory_act", "label": "Factory Act"},
                {"suffix": "peso", "label": "PESO"},
                {"suffix": "pollution_board", "label": "Pollution Board"},
                {"suffix": "audit", "label": "Audit"},
                {"suffix": "compliance_requirement", "label": "Compliance Requirement"},
                {"suffix": "violation", "label": "Violation"},
                {"suffix": "evidence", "label": "Evidence"},
                {"suffix": "certificate", "label": "Certificate"},
                {"suffix": "emission", "label": "Emission"},
                {"suffix": "waste", "label": "Waste"},
                {"suffix": "water", "label": "Water"},
                {"suffix": "noise", "label": "Noise"},
                {"suffix": "environmental_limit", "label": "Environmental Limit"},
                {"suffix": "fine", "label": "Fine"},
                {"suffix": "corrective_action", "label": "Corrective Action"},
                {"suffix": "review", "label": "Review"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="time",
            label="Time",
            parent_type_id="entity",
            layer="temporal",
            source_docs=(ONTOLOGY_SOURCE_DOCS[1],),
            entries=[
                {"suffix": "date", "label": "Date"},
                {"suffix": "timestamp", "label": "Timestamp"},
                {"suffix": "shift", "label": "Shift"},
                {"suffix": "week", "label": "Week"},
                {"suffix": "month", "label": "Month"},
                {"suffix": "quarter", "label": "Quarter"},
                {"suffix": "year", "label": "Year"},
                {"suffix": "maintenance_cycle", "label": "Maintenance Cycle"},
                {"suffix": "inspection_cycle", "label": "Inspection Cycle"},
                {"suffix": "calibration_cycle", "label": "Calibration Cycle"},
                {"suffix": "event_timeline", "label": "Event Timeline"},
                {"suffix": "history", "label": "History"},
                {"suffix": "version", "label": "Version"},
                {"suffix": "revision", "label": "Revision"},
                {"suffix": "lifecycle_stage", "label": "Lifecycle Stage"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="knowledge",
            label="Knowledge",
            parent_type_id="entity",
            layer="meta_ontology",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "fact", "label": "Fact", "examples": ["Pressure is 7.2 bar"]},
                {"suffix": "observation", "label": "Observation"},
                {"suffix": "assumption", "label": "Assumption"},
                {"suffix": "hypothesis", "label": "Hypothesis"},
                {"suffix": "rule", "label": "Rule"},
                {"suffix": "constraint", "label": "Constraint", "examples": ["Never run dry"]},
                {"suffix": "opinion", "label": "Opinion"},
                {"suffix": "best_practice", "label": "Best Practice"},
                {"suffix": "lesson_learned", "label": "Lesson Learned"},
                {"suffix": "recommendation", "label": "Recommendation", "examples": ["Replace bearing after 20,000 hours"]},
                {"suffix": "decision", "label": "Decision"},
                {"suffix": "requirement", "label": "Requirement"},
                {"suffix": "objective", "label": "Objective"},
                {"suffix": "evidence", "label": "Evidence"},
                {"suffix": "contradiction", "label": "Contradiction"},
                {"suffix": "exception", "label": "Exception"},
                {"suffix": "unknown", "label": "Unknown"},
                {"suffix": "open_question", "label": "Open Question"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="reasoning",
            label="Reasoning",
            parent_type_id="entity",
            layer="reasoning",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "deductive", "label": "Deductive"},
                {"suffix": "inductive", "label": "Inductive"},
                {"suffix": "abductive", "label": "Abductive"},
                {"suffix": "analogical", "label": "Analogical"},
                {"suffix": "statistical", "label": "Statistical"},
                {"suffix": "temporal", "label": "Temporal"},
                {"suffix": "spatial", "label": "Spatial"},
                {"suffix": "causal", "label": "Causal"},
                {"suffix": "counterfactual", "label": "Counterfactual"},
                {"suffix": "rule_based", "label": "Rule-based"},
                {"suffix": "constraint_satisfaction", "label": "Constraint Satisfaction"},
                {"suffix": "optimization", "label": "Optimization"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="decision",
            label="Decision",
            parent_type_id="entity",
            layer="decision_intelligence",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "decision_context", "label": "Decision Context"},
                {"suffix": "alternative", "label": "Alternative"},
                {"suffix": "criteria", "label": "Criteria"},
                {"suffix": "trade_off", "label": "Trade-offs"},
                {"suffix": "risk", "label": "Risk"},
                {"suffix": "benefit", "label": "Benefit"},
                {"suffix": "decision_owner", "label": "Decision Owner"},
                {"suffix": "approval", "label": "Approval"},
                {"suffix": "outcome", "label": "Outcome"},
                {"suffix": "postmortem", "label": "Postmortem"},
                {"suffix": "decision_confidence", "label": "Decision Confidence"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="intent",
            label="Intent",
            parent_type_id="entity",
            layer="intent",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "user_intent", "label": "User Intent"},
                {"suffix": "goal", "label": "Goal"},
                {"suffix": "objective", "label": "Objective"},
                {"suffix": "constraint", "label": "Constraint"},
                {"suffix": "priority", "label": "Priority"},
                {"suffix": "desired_outcome", "label": "Desired Outcome"},
                {"suffix": "context", "label": "Context"},
                {"suffix": "urgency", "label": "Urgency"},
                {"suffix": "audience", "label": "Audience"},
                {"suffix": "scope", "label": "Scope"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="simulation",
            label="Simulation",
            parent_type_id="entity",
            layer="simulation",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "scenario", "label": "Scenario"},
                {"suffix": "what_if_analysis", "label": "What-if Analysis"},
                {"suffix": "predicted_outcome", "label": "Predicted Outcome"},
                {"suffix": "assumption_set", "label": "Assumption Set"},
                {"suffix": "alternative_configuration", "label": "Alternative Configuration"},
                {"suffix": "monte_carlo_run", "label": "Monte Carlo Run"},
                {"suffix": "failure_simulation", "label": "Failure Simulation"},
                {"suffix": "capacity_simulation", "label": "Capacity Simulation"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="optimization",
            label="Optimization",
            parent_type_id="entity",
            layer="optimization",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "minimize_energy", "label": "Minimize Energy"},
                {"suffix": "maximize_throughput", "label": "Maximize Throughput"},
                {"suffix": "reduce_downtime", "label": "Reduce Downtime"},
                {"suffix": "reduce_maintenance_cost", "label": "Reduce Maintenance Cost"},
                {"suffix": "increase_safety", "label": "Increase Safety"},
                {"suffix": "increase_yield", "label": "Increase Yield"},
                {"suffix": "optimize_schedule", "label": "Optimize Schedule"},
                {"suffix": "optimize_spare_inventory", "label": "Optimize Spare Inventory"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="uncertainty",
            label="Uncertainty",
            parent_type_id="entity",
            layer="uncertainty",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "confidence", "label": "Confidence"},
                {"suffix": "probability", "label": "Probability"},
                {"suffix": "variance", "label": "Variance"},
                {"suffix": "data_quality", "label": "Data Quality"},
                {"suffix": "missing_data", "label": "Missing Data"},
                {"suffix": "unknown", "label": "Unknown"},
                {"suffix": "ambiguous", "label": "Ambiguous"},
                {"suffix": "conflicting_evidence", "label": "Conflicting Evidence"},
                {"suffix": "reliability", "label": "Reliability"},
                {"suffix": "trust_score", "label": "Trust Score"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="memory",
            label="Memory",
            parent_type_id="entity",
            layer="memory",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "short_term_memory", "label": "Short-term Memory"},
                {"suffix": "long_term_memory", "label": "Long-term Memory"},
                {"suffix": "episodic_memory", "label": "Episodic Memory"},
                {"suffix": "semantic_memory", "label": "Semantic Memory"},
                {"suffix": "working_memory", "label": "Working Memory"},
                {"suffix": "task_memory", "label": "Task Memory"},
                {"suffix": "expert_memory", "label": "Expert Memory"},
                {"suffix": "organization_memory", "label": "Organization Memory"},
                {"suffix": "plant_memory", "label": "Plant Memory"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="agent",
            label="Agent",
            parent_type_id="entity",
            layer="multi_agent",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "ocr_agent", "label": "OCR Agent"},
                {"suffix": "extraction_agent", "label": "Extraction Agent"},
                {"suffix": "graph_agent", "label": "Graph Agent"},
                {"suffix": "maintenance_agent", "label": "Maintenance Agent"},
                {"suffix": "compliance_agent", "label": "Compliance Agent"},
                {"suffix": "rca_agent", "label": "RCA Agent"},
                {"suffix": "planner_agent", "label": "Planner Agent"},
                {"suffix": "qa_agent", "label": "QA Agent"},
                {"suffix": "scheduler_agent", "label": "Scheduler Agent"},
                {"suffix": "alert_agent", "label": "Alert Agent"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="governance",
            label="Governance",
            parent_type_id="entity",
            layer="governance",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "data_steward", "label": "Data Steward"},
                {"suffix": "data_owner", "label": "Data Owner"},
                {"suffix": "knowledge_owner", "label": "Knowledge Owner"},
                {"suffix": "reviewer", "label": "Reviewer"},
                {"suffix": "approval_chain", "label": "Approval Chain"},
                {"suffix": "policy", "label": "Policy"},
                {"suffix": "classification", "label": "Classification"},
                {"suffix": "sensitivity", "label": "Sensitivity"},
                {"suffix": "audit_trail", "label": "Audit Trail"},
                {"suffix": "retention_rule", "label": "Retention Rule"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="security",
            label="Security",
            parent_type_id="entity",
            layer="security",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "user", "label": "User"},
                {"suffix": "group", "label": "Group"},
                {"suffix": "role", "label": "Role"},
                {"suffix": "permission", "label": "Permission"},
                {"suffix": "secret", "label": "Secret"},
                {"suffix": "credential_reference", "label": "Credential Reference"},
                {"suffix": "api_key_reference", "label": "API Key Reference"},
                {"suffix": "token", "label": "Token"},
                {"suffix": "access_policy", "label": "Access Policy"},
                {"suffix": "authentication", "label": "Authentication"},
                {"suffix": "authorization", "label": "Authorization"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="integration",
            label="Integration",
            parent_type_id="entity",
            layer="integration",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "connector", "label": "Connector"},
                {"suffix": "mapping", "label": "Mapping"},
                {"suffix": "synchronization", "label": "Synchronization"},
                {"suffix": "transformation", "label": "Transformation"},
                {"suffix": "schema", "label": "Schema"},
                {"suffix": "refresh_frequency", "label": "Refresh Frequency"},
                {"suffix": "sap", "label": "SAP"},
                {"suffix": "maximo", "label": "Maximo"},
                {"suffix": "scada", "label": "SCADA"},
                {"suffix": "dcs", "label": "DCS"},
                {"suffix": "plc", "label": "PLC"},
                {"suffix": "historian", "label": "Historian"},
                {"suffix": "erp", "label": "ERP"},
                {"suffix": "mes", "label": "MES"},
                {"suffix": "lims", "label": "LIMS"},
                {"suffix": "cmms", "label": "CMMS"},
                {"suffix": "qms", "label": "QMS"},
                {"suffix": "gis", "label": "GIS"},
                {"suffix": "bim", "label": "BIM"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="financial",
            label="Financial",
            parent_type_id="entity",
            layer="financial",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "cost", "label": "Cost"},
                {"suffix": "budget", "label": "Budget"},
                {"suffix": "purchase", "label": "Purchase"},
                {"suffix": "vendor", "label": "Vendor"},
                {"suffix": "invoice", "label": "Invoice"},
                {"suffix": "warranty", "label": "Warranty"},
                {"suffix": "amc", "label": "AMC"},
                {"suffix": "maintenance_cost", "label": "Maintenance Cost"},
                {"suffix": "downtime_cost", "label": "Downtime Cost"},
                {"suffix": "energy_cost", "label": "Energy Cost"},
                {"suffix": "roi", "label": "ROI"},
                {"suffix": "penalty", "label": "Penalty"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="supply_chain",
            label="Supply Chain",
            parent_type_id="entity",
            layer="supply_chain",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "supplier", "label": "Supplier"},
                {"suffix": "warehouse", "label": "Warehouse"},
                {"suffix": "inventory", "label": "Inventory"},
                {"suffix": "purchase_order", "label": "Purchase Order"},
                {"suffix": "delivery", "label": "Delivery"},
                {"suffix": "shipment", "label": "Shipment"},
                {"suffix": "stock", "label": "Stock"},
                {"suffix": "batch", "label": "Batch"},
                {"suffix": "lot", "label": "Lot"},
                {"suffix": "lead_time", "label": "Lead Time"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="manufacturing",
            label="Manufacturing",
            parent_type_id="entity",
            layer="manufacturing",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "product", "label": "Product"},
                {"suffix": "sku", "label": "SKU"},
                {"suffix": "recipe", "label": "Recipe"},
                {"suffix": "bom", "label": "BOM"},
                {"suffix": "operation", "label": "Operation"},
                {"suffix": "station", "label": "Station"},
                {"suffix": "cycle_time", "label": "Cycle Time"},
                {"suffix": "yield", "label": "Yield"},
                {"suffix": "scrap", "label": "Scrap"},
                {"suffix": "oee", "label": "OEE"},
                {"suffix": "batch", "label": "Batch"},
                {"suffix": "lot", "label": "Lot"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="communication",
            label="Communication",
            parent_type_id="entity",
            layer="communication",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "email", "label": "Email"},
                {"suffix": "teams_chat", "label": "Teams Chat"},
                {"suffix": "whatsapp", "label": "WhatsApp"},
                {"suffix": "call", "label": "Call"},
                {"suffix": "meeting", "label": "Meeting"},
                {"suffix": "minutes", "label": "Minutes"},
                {"suffix": "voice_note", "label": "Voice Note"},
                {"suffix": "transcript", "label": "Transcript"},
                {"suffix": "discussion", "label": "Discussion"},
                {"suffix": "comment", "label": "Comment"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="semantic",
            label="Semantic",
            parent_type_id="entity",
            layer="semantic",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "synonym", "label": "Synonym"},
                {"suffix": "alias", "label": "Alias"},
                {"suffix": "abbreviation", "label": "Abbreviation"},
                {"suffix": "translation", "label": "Translation"},
                {"suffix": "canonical_name", "label": "Canonical Name"},
                {"suffix": "concept", "label": "Concept"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="taxonomy",
            label="Taxonomy",
            parent_type_id="entity",
            layer="taxonomy",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "equipment_type", "label": "Equipment Type"},
                {"suffix": "failure_type", "label": "Failure Type"},
                {"suffix": "hazard_type", "label": "Hazard Type"},
                {"suffix": "maintenance_type", "label": "Maintenance Type"},
                {"suffix": "document_type", "label": "Document Type"},
                {"suffix": "inspection_type", "label": "Inspection Type"},
                {"suffix": "risk_type", "label": "Risk Type"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="value",
            label="Value",
            parent_type_id="entity",
            layer="value",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "downtime_prevented", "label": "Downtime Prevented"},
                {"suffix": "cost_saved", "label": "Cost Saved"},
                {"suffix": "energy_saved", "label": "Energy Saved"},
                {"suffix": "incidents_avoided", "label": "Incidents Avoided"},
                {"suffix": "compliance_risk_reduced", "label": "Compliance Risk Reduced"},
                {"suffix": "search_time_reduced", "label": "Search Time Reduced"},
                {"suffix": "knowledge_reuse", "label": "Knowledge Reuse"},
                {"suffix": "training_hours_saved", "label": "Training Hours Saved"},
                {"suffix": "mttr_improvement", "label": "MTTR Improvement"},
                {"suffix": "mtbf_improvement", "label": "MTBF Improvement"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="digital_twin",
            label="Digital Twin",
            parent_type_id="entity",
            layer="digital_twin",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "physical_pump", "label": "Physical Pump"},
                {"suffix": "digital_asset", "label": "Digital Asset"},
                {"suffix": "knowledge_graph_node", "label": "Knowledge Graph Node"},
                {"suffix": "simulation_model", "label": "Simulation Model"},
                {"suffix": "predictive_model", "label": "Predictive Model"},
                {"suffix": "maintenance_model", "label": "Maintenance Model"},
                {"suffix": "risk_model", "label": "Risk Model"},
                {"suffix": "financial_model", "label": "Financial Model"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="cognitive",
            label="Cognitive",
            parent_type_id="entity",
            layer="cognitive",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "mental_model", "label": "Mental Model"},
                {"suffix": "rule_of_thumb", "label": "Rule of Thumb"},
                {"suffix": "pattern_recognition", "label": "Pattern Recognition"},
                {"suffix": "diagnostic_strategy", "label": "Diagnostic Strategy"},
                {"suffix": "escalation_heuristic", "label": "Escalation Heuristic"},
                {"suffix": "failure_signature", "label": "Failure Signature"},
                {"suffix": "decision_shortcut", "label": "Decision Shortcut"},
                {"suffix": "expert_checklist", "label": "Expert Checklist"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="organizational_learning",
            label="Organizational Learning",
            parent_type_id="entity",
            layer="organizational_learning",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "improvement_initiative", "label": "Improvement Initiative"},
                {"suffix": "kaizen", "label": "Kaizen"},
                {"suffix": "capa", "label": "CAPA"},
                {"suffix": "innovation", "label": "Innovation"},
                {"suffix": "suggestion", "label": "Suggestion"},
                {"suffix": "experiment", "label": "Experiment"},
                {"suffix": "kpi_improvement", "label": "KPI Improvement"},
                {"suffix": "benchmark", "label": "Benchmark"},
                {"suffix": "maturity_assessment", "label": "Maturity Assessment"},
            ],
        )
    )

    types.extend(
        _group_types(
            prefix="self_improving_graph",
            label="Self Improving Knowledge Graph",
            parent_type_id="entity",
            layer="self_improving",
            source_docs=(ONTOLOGY_SOURCE_DOCS[2],),
            entries=[
                {"suffix": "draft", "label": "Draft"},
                {"suffix": "reviewed", "label": "Reviewed"},
                {"suffix": "approved", "label": "Approved"},
                {"suffix": "deprecated", "label": "Deprecated"},
                {"suffix": "archived", "label": "Archived"},
                {"suffix": "superseded", "label": "Superseded"},
                {"suffix": "branch", "label": "Branch"},
                {"suffix": "merge", "label": "Merge"},
                {"suffix": "version", "label": "Version"},
            ],
        )
    )

    return types


def _build_core_relation_definitions() -> List[OntologyRelationDefinition]:
    source_docs = ONTOLOGY_SOURCE_DOCS
    relations = [
        _relation("related_to", "related_to", aliases=["related"], source_docs=source_docs, examples=["Pump related_to Motor"]),
        _relation("contains", "contains", aliases=["has", "includes"], source_docs=source_docs, examples=["Plant contains Area"]),
        _relation("part_of", "part_of", aliases=["is_part_of", "belongs_to"], source_docs=source_docs, examples=["Motor part_of Pump"]),
        _relation("located_at", "located_at", aliases=["inside", "mounted_on", "installed_at"], source_docs=source_docs, examples=["Pump located_at Room"]),
        _relation("owns", "owns", aliases=["owned_by"], source_docs=source_docs, examples=["Department owns Equipment"]),
        _relation("belongs_to", "belongs_to", aliases=["belongs", "belongs_in"], source_docs=source_docs, examples=["Employee belongs_to Department"]),
        _relation("supplies", "supplies", aliases=["feeds", "delivers"], source_docs=source_docs, examples=["Vendor supplies Equipment"]),
        _relation("receives", "receives", aliases=["receives_input_from"], source_docs=source_docs, examples=["Customer receives Product"]),
        _relation("describes", "describes", aliases=["documents", "specifies"], source_docs=source_docs, examples=["SOP describes Pump"]),
        _relation("references", "references", aliases=["cites", "mentions"], source_docs=source_docs, examples=["Drawing references Tank"]),
        _relation("mentions", "mentions", source_docs=source_docs, examples=["Email mentions Work Order"]),
        _relation("authored_by", "authored_by", aliases=["written_by"], source_docs=source_docs, examples=["Engineer authored SOP"]),
        _relation("approved_by", "approved_by", aliases=["signed_off_by"], source_docs=source_docs, examples=["Review approved_by Manager"]),
        _relation("verified_by", "verified_by", aliases=["validated_by"], source_docs=source_docs, examples=["Audit verified_by Inspector"]),
        _relation("operates", "operates", aliases=["operates_on", "runs"], source_docs=source_docs, examples=["Operator operates Pump"]),
        _relation("controls", "controls", aliases=["regulates", "manages", "drives", "powers"], source_docs=source_docs, examples=["PLC controls Valve"]),
        _relation("monitors", "monitors", aliases=["observes", "tracks"], source_docs=source_docs, examples=["Sensor monitors Temperature"]),
        _relation("measures", "measures", aliases=["detects", "senses"], source_docs=source_docs, examples=["Sensor measures Pressure"]),
        _relation("causes", "causes", aliases=["triggered_by", "results_in"], source_docs=source_docs, examples=["Leak causes Incident"]),
        _relation("affected_by", "affected_by", aliases=["impacted_by"], source_docs=source_docs, examples=["Equipment affected_by Corrosion"]),
        _relation("triggers", "triggers", aliases=["initiates"], source_docs=source_docs, examples=["Alarm triggers Shutdown"]),
        _relation("resolved_by", "resolved_by", aliases=["resolved", "fixed_by"], source_docs=source_docs, examples=["Failure resolved_by Maintenance"]),
        _relation("requires", "requires", aliases=["needs", "depends_on"], source_docs=source_docs, examples=["Hot Work requires Permit"]),
        _relation("uses", "uses", aliases=["utilizes"], source_docs=source_docs, examples=["Process uses Catalyst"]),
        _relation("replaces", "replaces", aliases=["replaced_by", "substitutes"], source_docs=source_docs, examples=["Spare Part replaces Seal"]),
        _relation("repaired_by", "repaired_by", aliases=["repair_by"], source_docs=source_docs, examples=["Pump repaired_by Technician"]),
        _relation("inspected_by", "inspected_by", aliases=["reviewed_by"], source_docs=source_docs, examples=["Valve inspected_by Inspector"]),
        _relation("calibrated_by", "calibrated_by", aliases=["calibrated"], source_docs=source_docs, examples=["Sensor calibrated_by Technician"]),
        _relation("complies_with", "complies_with", aliases=["conforms_to"], source_docs=source_docs, examples=["Plant complies_with Standard"]),
        _relation("violates", "violates", aliases=["breaches"], source_docs=source_docs, examples=["Incident violates Policy"]),
        _relation("derived_from", "derived_from", aliases=["based_on"], source_docs=source_docs, examples=["Recommendation derived_from Evidence"]),
        _relation("supports", "supports", aliases=["backstops"], source_docs=source_docs, examples=["Evidence supports Decision"]),
        _relation("invalidates", "invalidates", aliases=["contradicts"], source_docs=source_docs, examples=["New data invalidates Hypothesis"]),
        _relation("similar_to", "similar_to", aliases=["analogous_to"], source_docs=source_docs, examples=["Pump A similar_to Pump B"]),
        _relation("version_of", "version_of", aliases=["revision_of"], source_docs=source_docs, examples=["Revision version_of Manual"]),
        _relation("precedes", "precedes", aliases=["before"], source_docs=source_docs, examples=["Inspection precedes Startup"]),
        _relation("follows", "follows", aliases=["after"], source_docs=source_docs, examples=["Repair follows Failure"]),
        _relation("observed_by", "observed_by", aliases=["seen_by"], source_docs=source_docs, examples=["Anomaly observed_by Sensor"]),
        _relation("predicted_by", "predicted_by", aliases=["forecast_by"], source_docs=source_docs, examples=["Failure predicted_by Model"]),
        _relation("explains", "explains", aliases=["clarifies"], source_docs=source_docs, examples=["RCA explains Incident"]),
        _relation("delegates_to", "delegates_to", aliases=["assigned_to"], source_docs=source_docs, examples=["OCR Agent delegates_to Extraction Agent"]),
        _relation("validates", "validates", aliases=["confirms"], source_docs=source_docs, examples=["QA Agent validates Extraction"]),
        _relation("critiques", "critiques", aliases=["reviews"], source_docs=source_docs, examples=["QA Agent critiques Recommendation"]),
        _relation("collaborates_with", "collaborates_with", aliases=["works_with"], source_docs=source_docs, examples=["Planner collaborates_with Maintenance Agent"]),
        _relation("supervises", "supervises", aliases=["oversees"], source_docs=source_docs, examples=["Manager supervises Employee"]),
        _relation("has_skill", "has_skill", aliases=["skilled_in"], source_docs=source_docs, examples=["Employee has_skill Welding"]),
        _relation("has_certification", "has_certification", aliases=["certified_in"], source_docs=source_docs, examples=["Employee has_certification Permit Training"]),
        _relation("works_shift", "works_shift", aliases=["assigned_shift"], source_docs=source_docs, examples=["Employee works_shift Night Shift"]),
        _relation("mentored_by", "mentored_by", aliases=["coached_by"], source_docs=source_docs, examples=["Employee mentored_by Expert"]),
        _relation("under_governance_of", "under_governance_of", aliases=["owned_by"], source_docs=source_docs, examples=["Knowledge under_governance_of Data Owner"]),
        _relation("has_provenance", "has_provenance", aliases=["cited_from"], source_docs=source_docs, examples=["Fact has_provenance Manual.pdf"]),
    ]
    return relations


ENTITY_TYPE_ROOT_MAP: Dict[str, str] = {
    "organization": "organization",
    "site": "organization.site",
    "plant": "organization.plant",
    "department": "organization.department",
    "division": "organization.division",
    "workshop": "organization.workshop",
    "area": "organization.area",
    "zone": "organization.zone",
    "building": "organization.building",
    "floor": "organization.floor",
    "room": "organization.room",
    "vendor": "organization.vendor",
    "customer": "organization.customer",
    "equipment": "asset.equipment",
    "machine": "asset.machine",
    "process": "process",
    "parameter": "process",
    "material": "process",
    "control_system": "asset.control_system",
    "location": "spatial",
    "failure_mode": "maintenance.failure_mode",
    "maintenance": "maintenance",
    "document": "document",
    "event": "event",
    "quality": "quality",
    "safety": "safety",
    "regulatory": "regulatory",
    "human": "human",
    "employee": "human.employee",
    "operator": "human.operator",
    "engineer": "human.engineer",
    "expert": "human.expert",
    "sensor": "asset.sensor",
    "actuator": "asset.actuator",
    "plc": "asset.plc",
    "dcs": "asset.dcs",
    "scada": "asset.scada_node",
    "iot": "iot",
}


DEFAULT_ENTITY_HINTS: Dict[str, str] = {
    "pump": "asset.equipment.pump",
    "valve": "asset.equipment.valve",
    "motor": "asset.equipment.motor",
    "sensor": "asset.sensor",
    "transmitter": "asset.sensor",
    "compressor": "asset.equipment.compressor",
    "tank": "asset.equipment.tank",
    "boiler": "asset.equipment.boiler",
    "generator": "asset.equipment.generator",
    "transformer": "asset.equipment.transformer",
    "bearing": "asset.equipment.bearing",
    "seal": "asset.equipment.seal",
    "gearbox": "asset.equipment.gearbox",
    "pipe": "asset.pipe",
    "pipeline": "asset.pipeline",
    "sop": "document.sop",
    "manual": "document.manual",
    "drawing": "document.drawing",
    "permit": "document.permit",
    "inspection": "document.inspection_report",
    "failure": "maintenance.failure",
    "incident": "safety.incident",
    "risk": "safety.risk",
    "hazard": "safety.hazard",
    "fact": "knowledge.fact",
    "hypothesis": "knowledge.hypothesis",
    "decision": "decision.decision",
    "recommendation": "knowledge.recommendation",
    "rule": "knowledge.rule",
    "constraint": "knowledge.constraint",
    "entity": "entity",
}


DEFAULT_RELATION_HINTS: Dict[str, str] = {
    "part_of": "part_of",
    "contains": "contains",
    "has": "contains",
    "located_at": "located_at",
    "inside": "located_at",
    "mounted_on": "located_at",
    "installed_at": "located_at",
    "controls": "controls",
    "regulates": "controls",
    "manages": "controls",
    "measures": "measures",
    "monitors": "monitors",
    "describes": "describes",
    "references": "references",
    "authored_by": "authored_by",
    "approved_by": "approved_by",
    "verified_by": "verified_by",
    "causes": "causes",
    "triggers": "triggers",
    "requires": "requires",
    "supplies": "supplies",
    "replaces": "replaces",
    "inspected_by": "inspected_by",
    "calibrated_by": "calibrated_by",
    "complies_with": "complies_with",
    "violates": "violates",
    "supports": "supports",
    "invalidates": "invalidates",
    "similar_to": "similar_to",
    "delegates_to": "delegates_to",
    "validates": "validates",
    "critiques": "critiques",
}


class OntologyExampleBank:
    """Minimal few-shot example retrieval over curated and observed examples."""

    def __init__(self, examples: Sequence[Dict[str, Any]] | None = None) -> None:
        self.examples: List[Dict[str, Any]] = list(examples or [])
        self.vectorizer = None
        self.matrix = None
        self._build_index()

    def _build_index(self) -> None:
        if not self.examples or TfidfVectorizer is None:
            return

        texts = [example.get("text", "") for example in self.examples]
        if not any(texts):
            return

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(texts)

    def extend(self, examples: Sequence[Dict[str, Any]]) -> None:
        self.examples.extend(examples)
        self._build_index()

    def retrieve(self, query: str, *, kind: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.examples:
            return []

        indexed_examples = [
            (idx, example)
            for idx, example in enumerate(self.examples)
            if kind is None or example.get("kind") == kind
        ]
        if not indexed_examples:
            return []

        if self.vectorizer is not None and self.matrix is not None and cosine_similarity is not None:
            try:
                query_vec = self.vectorizer.transform([query])
                scores = cosine_similarity(query_vec, self.matrix)[0]
                ranked = sorted(indexed_examples, key=lambda item: scores[item[0]], reverse=True)[:top_k]
                return [
                    {**example, "score": float(scores[idx])}
                    for idx, example in ranked
                    if float(scores[idx]) > 0.0
                ]
            except Exception:
                pass

        query_tokens = set(_tokenize(query))
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for _, example in indexed_examples:
            example_tokens = set(_tokenize(example.get("text", "")))
            overlap = len(query_tokens & example_tokens)
            score = overlap / max(1, len(query_tokens | example_tokens))
            scored.append((score, example))
        ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        return [{**example, "score": float(score)} for score, example in ranked if score > 0.0]


def build_domain_pack_from_markdown(path: Path | str) -> Dict[str, Any]:
    """Parse a markdown ontology document into a lightweight domain pack."""
    source_path = Path(path)
    if not source_path.exists():
        return {"name": source_path.stem, "path": str(source_path), "types": [], "relations": []}

    lines = [line.rstrip() for line in source_path.read_text(encoding="utf-8").splitlines()]
    name = re.sub(r"[^a-z0-9]+", "_", source_path.stem.lower()).strip("_") or source_path.stem

    types: List[str] = []
    relations: List[str] = []
    current_section: Optional[str] = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        lowered = line.lower()
        if lowered.startswith("## relations") or lowered.startswith("relationships"):
            current_section = "relations"
            continue
        if lowered.startswith("## types") or lowered.startswith("type"):
            current_section = "types"
            continue

        if line.startswith("-") or line.startswith("*"):
            item = re.sub(r"^[-*]\s*", "", line).strip()
            if not item:
                continue
            if current_section == "relations":
                relations.append(item)
            else:
                types.append(item)
            continue

        if re.match(r"^\d+\.", line):
            item = re.sub(r"^\d+\.\s*", "", line).strip()
            if not item:
                continue
            if current_section == "relations":
                relations.append(item)
            else:
                types.append(item)
            continue

        if current_section == "relations" and ("->" in line or "->" in lowered):
            relation_text = line.replace("->", " ").strip()
            if relation_text:
                relations.append(relation_text)

    def _clean_entry(entry: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", " ", entry.lower()).strip()
        return cleaned.replace(" ", "_").strip("_")

    unique_types = list(dict.fromkeys(_clean_entry(entry) for entry in types if entry and _clean_entry(entry)))
    unique_relations = list(dict.fromkeys(_clean_entry(entry) for entry in relations if entry and _clean_entry(entry)))

    return {
        "name": name,
        "path": str(source_path),
        "types": unique_types,
        "relations": unique_relations,
    }


def register_markdown_domain_packs(registry: "OntologyRegistry") -> None:
    """Register domain packs from markdown ontology documents into the runtime registry."""
    markdown_paths = discover_markdown_ontology_pack_files()
    if not markdown_paths:
        return

    for markdown_path in markdown_paths:
        pack = build_domain_pack_from_markdown(markdown_path)
        pack_name = pack.get("name") or markdown_path.stem
        for type_name in pack.get("types", []):
            safe_type_id = f"{pack_name}.{type_name}" if type_name else pack_name
            registry.add_type(
                _type(
                    safe_type_id,
                    type_name.replace("_", " ").title(),
                    parent_type_id="entity",
                    aliases=[type_name.replace("_", " ")],
                    keywords=[type_name],
                    layer="domain_pack",
                    pack=pack_name,
                    source_docs=(str(markdown_path),),
                    examples=[type_name.replace("_", " ")],
                )
            )
        for relation_name in pack.get("relations", []):
            safe_relation_id = f"{pack_name}.{relation_name}" if relation_name else pack_name
            registry.add_relation(
                _relation(
                    safe_relation_id,
                    relation_name.replace("_", " "),
                    aliases=[relation_name.replace("_", " ")],
                    keywords=[relation_name],
                    pack=pack_name,
                    source_docs=(str(markdown_path),),
                    examples=[relation_name.replace("_", " ")],
                )
            )


class OntologyRegistry:
    """Runtime registry for stable ontology types and relations."""

    schema_version = "1.0.0"

    def __init__(
        self,
        type_definitions: Sequence[OntologyTypeDefinition] | None = None,
        relation_definitions: Sequence[OntologyRelationDefinition] | None = None,
        *,
        registry_version: Optional[str] = None,
        source_docs: Sequence[str] = ONTOLOGY_SOURCE_DOCS,
    ) -> None:
        self.registry_version = registry_version or _now()
        self.source_docs = tuple(source_docs)
        self.type_definitions: List[OntologyTypeDefinition] = list(type_definitions or _build_core_type_definitions())
        self.relation_definitions: List[OntologyRelationDefinition] = list(relation_definitions or _build_core_relation_definitions())
        self.markdown_packs: List[OntologyMarkdownPack] = [
            _load_markdown_pack_manifest(Path(path))
            for path in discover_markdown_ontology_pack_files()
        ]
        self._type_by_id: Dict[str, OntologyTypeDefinition] = {}
        self._relation_by_id: Dict[str, OntologyRelationDefinition] = {}
        self._type_alias_index: Dict[str, List[str]] = defaultdict(list)
        self._relation_alias_index: Dict[str, List[str]] = defaultdict(list)
        self._children_by_parent: Dict[str, List[str]] = defaultdict(list)
        self._example_bank = OntologyExampleBank()
        self._rebuild_indexes()

    @classmethod
    def load_default(cls, state_path: Path | None = None) -> "OntologyRegistry":
        registry = cls()
        register_markdown_domain_packs(registry)
        registry.refresh_markdown_packs()
        if state_path is None:
            state_path = ONTOLOGY_STATE_PATH
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                registry.apply_state_snapshot(state)
            except Exception:
                pass
        return registry

    def refresh_markdown_packs(self) -> None:
        self.markdown_packs = [
            _load_markdown_pack_manifest(path)
            for path in discover_markdown_ontology_pack_files()
        ]

    def apply_state_snapshot(self, state: Dict[str, Any]) -> None:
        active_types = state.get("active_extensions", {}).get("types", {})
        for type_payload in active_types.values():
            try:
                self.add_type(OntologyTypeDefinition(**type_payload))
            except Exception:
                continue

        active_relations = state.get("active_extensions", {}).get("relations", {})
        for relation_payload in active_relations.values():
            try:
                self.add_relation(OntologyRelationDefinition(**relation_payload))
            except Exception:
                continue

        examples: List[Dict[str, Any]] = []
        examples.extend(state.get("examples", {}).get("type", []))
        examples.extend(state.get("examples", {}).get("relation", []))
        if examples:
            self._example_bank = OntologyExampleBank(examples)

    def refresh_examples(self, examples: Sequence[Dict[str, Any]]) -> None:
        self._example_bank = OntologyExampleBank(list(examples))

    def _rebuild_indexes(self) -> None:
        self._type_by_id = {definition.type_id: definition for definition in self.type_definitions}
        self._relation_by_id = {definition.relation_id: definition for definition in self.relation_definitions}
        self._type_alias_index.clear()
        self._relation_alias_index.clear()
        self._children_by_parent.clear()

        for definition in self.type_definitions:
            self._children_by_parent[definition.parent_type_id or ""].append(definition.type_id)
            aliases = {
                _normalize(definition.type_id),
                _normalize(definition.label),
                *(_normalize(alias) for alias in definition.aliases),
                *(_normalize(keyword) for keyword in definition.keywords),
            }
            for alias in aliases:
                if alias:
                    self._type_alias_index[alias].append(definition.type_id)

        for definition in self.relation_definitions:
            aliases = {
                _normalize(definition.relation_id),
                _normalize(definition.label),
                *(_normalize(alias) for alias in definition.aliases),
                *(_normalize(keyword) for keyword in definition.keywords),
            }
            for alias in aliases:
                if alias:
                    self._relation_alias_index[alias].append(definition.relation_id)

    def add_type(self, definition: OntologyTypeDefinition) -> None:
        if definition.type_id in self._type_by_id:
            return
        self.type_definitions.append(definition)
        self._rebuild_indexes()
        if definition.examples:
            self._example_bank.extend(
                [
                    {
                        "kind": "type",
                        "label": definition.type_id,
                        "text": _build_example_text(definition.label, definition.examples, definition.description),
                        "source": definition.pack,
                    }
                ]
            )

    def add_relation(self, definition: OntologyRelationDefinition) -> None:
        if definition.relation_id in self._relation_by_id:
            return
        self.relation_definitions.append(definition)
        self._rebuild_indexes()
        if definition.examples:
            self._example_bank.extend(
                [
                    {
                        "kind": "relation",
                        "label": definition.relation_id,
                        "text": _build_example_text(definition.label, definition.examples, definition.description),
                        "source": definition.pack,
                    }
                ]
            )

    def get_type(self, type_id: str) -> Optional[OntologyTypeDefinition]:
        return self._type_by_id.get(type_id)

    def get_relation(self, relation_id: str) -> Optional[OntologyRelationDefinition]:
        return self._relation_by_id.get(relation_id)

    def get_type_path(self, type_id: str) -> Tuple[str, ...]:
        path: List[str] = []
        current = self._type_by_id.get(type_id)
        guard = 0
        while current is not None and guard < 32:
            path.append(current.type_id)
            if not current.parent_type_id:
                break
            current = self._type_by_id.get(current.parent_type_id)
            guard += 1
        return tuple(reversed(path))

    def get_relation_path(self, relation_id: str) -> Tuple[str, ...]:
        relation = self._relation_by_id.get(relation_id)
        if relation is None:
            return tuple()
        return (relation.relation_id,)

    def descendants(self, type_id: str) -> List[str]:
        descendants: List[str] = []
        queue = [type_id]
        while queue:
            current = queue.pop(0)
            children = self._children_by_parent.get(current, [])
            descendants.extend(children)
            queue.extend(children)
        return descendants

    def list_packs(self) -> List[Dict[str, Any]]:
        type_counts = Counter(definition.pack or "core" for definition in self.type_definitions)
        relation_counts = Counter(definition.pack or "core" for definition in self.relation_definitions)
        pack_names = sorted(set(type_counts) | set(relation_counts))
        return [
            {
                "pack": pack_name,
                "type_count": int(type_counts.get(pack_name, 0)),
                "relation_count": int(relation_counts.get(pack_name, 0)),
                "total": int(type_counts.get(pack_name, 0) + relation_counts.get(pack_name, 0)),
            }
            for pack_name in pack_names
        ]

    def get_pack(self, pack_name: str) -> Dict[str, Any]:
        type_defs = [definition.to_dict() for definition in self.type_definitions if (definition.pack or "core") == pack_name]
        relation_defs = [definition.to_dict() for definition in self.relation_definitions if (definition.pack or "core") == pack_name]
        markdown_sources = [
            pack.to_dict()
            for pack in self.markdown_packs
            if pack.pack_id == pack_name or Path(pack.path).stem == pack_name
        ]
        return {
            "pack": pack_name,
            "type_count": len(type_defs),
            "relation_count": len(relation_defs),
            "types": type_defs,
            "relations": relation_defs,
            "markdown_sources": markdown_sources,
        }

    def list_markdown_packs(self) -> List[Dict[str, Any]]:
        return [pack.to_dict() for pack in self.markdown_packs]

    def describe_for_prompt(
        self,
        query: str,
        *,
        max_types: int = 24,
        max_relations: int = 16,
        max_markdown_packs: int = 4,
    ) -> Dict[str, Any]:
        ranked_types = sorted(
            self.type_definitions,
            key=lambda definition: self._score_text_against_definition(query, definition)[0],
            reverse=True,
        )
        ranked_relations = sorted(
            self.relation_definitions,
            key=lambda definition: self._score_relation_against_definition(query, definition)[0],
            reverse=True,
        )

        def _serialize_type(definition: OntologyTypeDefinition) -> Dict[str, Any]:
            return {
                "type_id": definition.type_id,
                "label": definition.label,
                "parent_type_id": definition.parent_type_id,
                "pack": definition.pack,
                "status": definition.status,
                "aliases": list(definition.aliases[:5]),
                "description": definition.description[:180],
                "examples": list(definition.examples[:3]),
            }

        def _serialize_relation(definition: OntologyRelationDefinition) -> Dict[str, Any]:
            return {
                "relation_id": definition.relation_id,
                "label": definition.label,
                "pack": definition.pack,
                "status": definition.status,
                "aliases": list(definition.aliases[:5]),
                "description": definition.description[:180],
                "examples": list(definition.examples[:3]),
                "source_type_ids": list(definition.source_type_ids[:5]),
                "target_type_ids": list(definition.target_type_ids[:5]),
            }

        selected_types = [_serialize_type(definition) for definition in ranked_types[:max_types]]
        selected_relations = [_serialize_relation(definition) for definition in ranked_relations[:max_relations]]
        markdown_packs = self.list_markdown_packs()[:max_markdown_packs]

        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "entity_types": selected_types,
            "relation_types": selected_relations,
            "markdown_packs": markdown_packs,
            "notes": [
                "Use active ontology types when the evidence is explicit.",
                "If the evidence does not map cleanly, emit unknown_candidate instead of forcing a label.",
                "Do not rename or delete existing type_ids.",
            ],
        }

    def _scope_for_entity_type(self, entity_type: str | None) -> Optional[str]:
        if not entity_type:
            return None
        return ENTITY_TYPE_ROOT_MAP.get(entity_type.strip().lower())

    def _score_text_against_definition(self, query: str, definition: OntologyTypeDefinition) -> Tuple[float, str]:
        query_norm = _normalize(query)
        label_norm = _normalize(definition.label)
        alias_norms = [_normalize(alias) for alias in definition.aliases]
        keywords = set(_tokenize(definition.label)) | set(_tokenize(" ".join(definition.aliases))) | set(_tokenize(" ".join(definition.keywords)))
        query_tokens = set(_tokenize(query))
        reason_parts: List[str] = []

        if query_norm == _normalize(definition.type_id) or query_norm == label_norm:
            return 1.0, "exact"
        if query_norm in alias_norms or any(alias and alias in query_norm for alias in alias_norms):
            return 0.97, "alias"
        if any(query_norm in alias for alias in alias_norms if alias):
            return 0.92, "alias_contains_query"

        overlap = len(query_tokens & keywords) / max(1, len(query_tokens | keywords))
        score = overlap * 0.7
        if query_tokens and any(token in keywords for token in query_tokens):
            reason_parts.append("keyword_overlap")

        ratio = difflib.SequenceMatcher(None, query_norm, label_norm).ratio()
        score = max(score, ratio * 0.6)
        if ratio > 0.65:
            reason_parts.append("sequence_match")

        example_score = 0.0
        examples = self._example_bank.retrieve(query, kind="type", top_k=3)
        for example in examples:
            if _normalize(example.get("label", "")) == _normalize(definition.type_id):
                example_score = max(example_score, float(example.get("score", 0.0)))
            elif _normalize(example.get("text", "")) and label_norm in _normalize(example.get("text", "")):
                example_score = max(example_score, float(example.get("score", 0.0)))
        if example_score:
            score = max(score, 0.5 + example_score * 0.45)
            reason_parts.append("few_shot")

        if definition.parent_type_id:
            parent_tokens = set(_tokenize(definition.parent_type_id))
            if query_tokens & parent_tokens:
                score = max(score, 0.55)
                reason_parts.append("parent_hint")

        return min(1.0, score), "+".join(reason_parts) if reason_parts else "heuristic"

    def _score_relation_against_definition(self, query: str, definition: OntologyRelationDefinition) -> Tuple[float, str]:
        query_norm = _normalize(query)
        relation_norm = _normalize(definition.relation_id)
        label_norm = _normalize(definition.label)
        alias_norms = [_normalize(alias) for alias in definition.aliases]
        keywords = set(_tokenize(definition.label)) | set(_tokenize(" ".join(definition.aliases))) | set(_tokenize(" ".join(definition.keywords)))
        query_tokens = set(_tokenize(query))

        if query_norm == relation_norm or query_norm == label_norm:
            return 1.0, "exact"
        if query_norm in alias_norms or any(alias and alias in query_norm for alias in alias_norms):
            return 0.97, "alias"
        if any(query_norm in alias for alias in alias_norms if alias):
            return 0.92, "alias_contains_query"

        overlap = len(query_tokens & keywords) / max(1, len(query_tokens | keywords))
        score = overlap * 0.7
        ratio = difflib.SequenceMatcher(None, query_norm, relation_norm).ratio()
        score = max(score, ratio * 0.6)

        examples = self._example_bank.retrieve(query, kind="relation", top_k=3)
        example_score = max((float(example.get("score", 0.0)) for example in examples), default=0.0)
        if example_score:
            score = max(score, 0.5 + example_score * 0.45)

        return min(1.0, score), "heuristic"

    def resolve_entity(
        self,
        name: str,
        *,
        entity_type: Optional[str] = None,
        context: str = "",
        threshold: float = 0.62,
    ) -> Optional[OntologyMatch]:
        query = " ".join(part for part in [name, entity_type or "", context[:120]] if part)
        scope = self._scope_for_entity_type(entity_type)
        candidate_ids = self.descendants(scope) if scope else list(self._type_by_id)
        if scope and scope in self._type_by_id:
            candidate_ids.insert(0, scope)

        if not candidate_ids:
            candidate_ids = list(self._type_by_id)

        query_norm = _normalize(query)
        alias_matches = self._type_alias_index.get(query_norm, [])
        if alias_matches:
            candidate_ids = list(dict.fromkeys([*alias_matches, *candidate_ids]))

        best: Optional[OntologyMatch] = None
        best_score = 0.0
        for type_id in candidate_ids:
            definition = self._type_by_id.get(type_id)
            if definition is None:
                continue
            score, reason = self._score_text_against_definition(query, definition)
            if entity_type and definition.type_id == self._scope_for_entity_type(entity_type):
                score = min(1.0, score + 0.18)
                reason = f"{reason}+entity_type"
            if score > best_score:
                best_score = score
                best = OntologyMatch(
                    kind="entity",
                    type_id=definition.type_id,
                    label=definition.label,
                    score=score,
                    status=definition.status,
                    parent_type_id=definition.parent_type_id,
                    source="registry",
                    reason=reason,
                    path=self.get_type_path(definition.type_id),
                    aliases=definition.aliases,
                    examples=definition.examples,
                )

        if best is not None and best.score >= threshold:
            return best

        return None

    def resolve_relation(
        self,
        relation_type: str,
        *,
        context: str = "",
        threshold: float = 0.58,
    ) -> Optional[OntologyMatch]:
        query = " ".join(part for part in [relation_type, context[:80]] if part)
        query_norm = _normalize(query)
        candidate_ids = list(self._relation_by_id)
        alias_matches = self._relation_alias_index.get(query_norm, [])
        if alias_matches:
            candidate_ids = list(dict.fromkeys([*alias_matches, *candidate_ids]))

        best: Optional[OntologyMatch] = None
        best_score = 0.0
        for relation_id in candidate_ids:
            definition = self._relation_by_id.get(relation_id)
            if definition is None:
                continue
            score, reason = self._score_relation_against_definition(query, definition)
            if score > best_score:
                best_score = score
                best = OntologyMatch(
                    kind="relation",
                    type_id=definition.relation_id,
                    label=definition.label,
                    score=score,
                    status=definition.status,
                    parent_type_id=None,
                    source="registry",
                    reason=reason,
                    path=self.get_relation_path(definition.relation_id),
                    aliases=definition.aliases,
                    examples=definition.examples,
                )

        if best is not None and best.score >= threshold:
            return best

        return None

    def propose_entity(
        self,
        name: str,
        *,
        entity_type: Optional[str] = None,
        context: str = "",
        confidence: float = 0.0,
    ) -> OntologyProposal:
        base_parent = self._scope_for_entity_type(entity_type) or "entity"
        name_slug = _normalize(name) or f"candidate_{_stable_hash(name)[:8]}"
        candidate_id = f"{base_parent}.{name_slug}" if not name_slug.startswith(base_parent) else name_slug
        return OntologyProposal(
            kind="entity",
            candidate_id=candidate_id,
            label=name or candidate_id.replace("_", " ").title(),
            parent_type_id=base_parent,
            confidence=float(confidence),
            source="zero_shot",
            evidence=context[:240],
            aliases=[name] if name else [],
        )

    def propose_relation(self, relation_type: str, *, context: str = "", confidence: float = 0.0) -> OntologyProposal:
        name_slug = _normalize(relation_type) or f"relation_{_stable_hash(relation_type)[:8]}"
        candidate_id = name_slug if name_slug in self._relation_by_id else f"relation.{name_slug}"
        return OntologyProposal(
            kind="relation",
            candidate_id=candidate_id,
            label=relation_type or candidate_id.replace("_", " ").title(),
            confidence=float(confidence),
            source="zero_shot",
            evidence=context[:240],
            aliases=[relation_type] if relation_type else [],
        )

    def match_or_propose_entity(
        self,
        name: str,
        *,
        entity_type: Optional[str] = None,
        context: str = "",
        threshold: float = 0.62,
    ) -> Tuple[OntologyMatch | OntologyProposal, bool]:
        match = self.resolve_entity(name, entity_type=entity_type, context=context, threshold=threshold)
        if match is not None:
            return match, False
        proposal = self.propose_entity(name, entity_type=entity_type, context=context, confidence=0.0)
        return proposal, True

    def match_or_propose_relation(
        self,
        relation_type: str,
        *,
        context: str = "",
        threshold: float = 0.58,
    ) -> Tuple[OntologyMatch | OntologyProposal, bool]:
        match = self.resolve_relation(relation_type, context=context, threshold=threshold)
        if match is not None:
            return match, False
        proposal = self.propose_relation(relation_type, context=context, confidence=0.0)
        return proposal, True

    def build_examples(self) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        for definition in self.type_definitions:
            if not definition.examples:
                continue
            examples.append(
                {
                    "kind": "type",
                    "label": definition.type_id,
                    "text": _build_example_text(definition.label, definition.examples, definition.description),
                    "source": definition.pack,
                }
            )
        for definition in self.relation_definitions:
            if not definition.examples:
                continue
            examples.append(
                {
                    "kind": "relation",
                    "label": definition.relation_id,
                    "text": _build_example_text(definition.label, definition.examples, definition.description),
                    "source": definition.pack,
                }
            )
        return examples

    def promote_from_state_snapshot(self, state: Dict[str, Any]) -> Dict[str, List[str]]:
        promotions = {"types": [], "relations": []}
        active_extensions = state.get("active_extensions", {})

        for candidate_id, candidate in state.get("proposals", {}).get("types", {}).items():
            if candidate_id in self._type_by_id:
                continue
            count = int(candidate.get("observed_count", 0) or 0)
            avg_confidence = float(candidate.get("average_confidence", 0.0) or 0.0)
            if count < 3 or avg_confidence < 0.85:
                continue

            definition = OntologyTypeDefinition(
                type_id=candidate_id,
                label=candidate.get("label", candidate_id.replace("_", " ").title()),
                parent_type_id=candidate.get("parent_type_id") or "entity",
                aliases=tuple(candidate.get("aliases", [])),
                keywords=tuple(candidate.get("keywords", [])),
                description=candidate.get("description", ""),
                layer=candidate.get("layer", "evolved"),
                pack=candidate.get("pack", "evolved"),
                status="active",
                source_docs=tuple(candidate.get("source_docs", [])),
                examples=tuple(candidate.get("examples", [])),
            )
            self.add_type(definition)
            active_extensions.setdefault("types", {})[candidate_id] = definition.to_dict()
            promotions["types"].append(candidate_id)

        for candidate_id, candidate in state.get("proposals", {}).get("relations", {}).items():
            if candidate_id in self._relation_by_id:
                continue
            count = int(candidate.get("observed_count", 0) or 0)
            avg_confidence = float(candidate.get("average_confidence", 0.0) or 0.0)
            if count < 5 or avg_confidence < 0.8:
                continue

            definition = OntologyRelationDefinition(
                relation_id=candidate_id,
                label=candidate.get("label", candidate_id.replace("_", " ").title()),
                aliases=tuple(candidate.get("aliases", [])),
                keywords=tuple(candidate.get("keywords", [])),
                source_type_ids=tuple(candidate.get("source_type_ids", [])),
                target_type_ids=tuple(candidate.get("target_type_ids", [])),
                description=candidate.get("description", ""),
                pack=candidate.get("pack", "evolved"),
                status="active",
                source_docs=tuple(candidate.get("source_docs", [])),
                examples=tuple(candidate.get("examples", [])),
            )
            self.add_relation(definition)
            active_extensions.setdefault("relations", {})[candidate_id] = definition.to_dict()
            promotions["relations"].append(candidate_id)

        return promotions


class OntologyStateStore:
    """Persistent store for ontology observations, proposals, and examples."""

    def __init__(self, state_path: Path | None = None) -> None:
        self.state_path = Path(state_path or ONTOLOGY_STATE_PATH)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
        return {
            "schema_version": OntologyRegistry.schema_version,
            "registry_version": _now(),
            "created_at": _now(),
            "updated_at": _now(),
            "type_observations": {},
            "relation_observations": {},
            "proposals": {"types": {}, "relations": {}},
            "active_extensions": {"types": {}, "relations": {}},
            "examples": {"type": [], "relation": []},
            "recent_observations": [],
        }

    def save(self) -> None:
        self.state["updated_at"] = _now()
        temp_path = self.state_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.state_path)

    def snapshot(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self.state))

    def _update_observation_bucket(
        self,
        bucket_name: str,
        key: str,
        *,
        label: str,
        confidence: float,
        evidence: str,
        source_document: Optional[str],
        kind: str,
        proposal: Optional[Dict[str, Any]] = None,
    ) -> None:
        bucket = self.state.setdefault(bucket_name, {})
        entry = bucket.setdefault(
            key,
            {
                "count": 0,
                "confidence_sum": 0.0,
                "average_confidence": 0.0,
                "first_seen": _now(),
                "last_seen": _now(),
                "label": label,
                "examples": [],
            },
        )
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["confidence_sum"] = float(entry.get("confidence_sum", 0.0)) + float(confidence)
        entry["average_confidence"] = entry["confidence_sum"] / max(1, entry["count"])
        entry["last_seen"] = _now()
        entry["label"] = label

        if evidence:
            entry.setdefault("examples", [])
            if evidence not in entry["examples"]:
                entry["examples"].append(evidence)
                entry["examples"] = entry["examples"][-5:]

        example_payload = {
            "kind": kind,
            "label": key,
            "text": evidence or label,
            "source": source_document or "unknown",
        }
        if kind == "type":
            examples = self.state.setdefault("examples", {}).setdefault("type", [])
        else:
            examples = self.state.setdefault("examples", {}).setdefault("relation", [])

        examples.append(example_payload)
        self.state["examples"]["type"] = self.state["examples"]["type"][-200:]
        self.state["examples"]["relation"] = self.state["examples"]["relation"][-200:]

        if proposal is not None:
            proposal_bucket = self.state.setdefault("proposals", {}).setdefault(kind + "s", {})
            existing = proposal_bucket.get(key, {})
            merged = {
                **existing,
                **proposal,
                "observed_count": int(existing.get("observed_count", 0)) + 1,
                "confidence_sum": float(existing.get("confidence_sum", 0.0)) + float(confidence),
                "average_confidence": (
                    (float(existing.get("confidence_sum", 0.0)) + float(confidence))
                    / max(1, int(existing.get("observed_count", 0)) + 1)
                ),
                "last_seen": _now(),
            }
            proposal_bucket[key] = merged

    def record_observations(
        self,
        *,
        entities: Sequence[Dict[str, Any]],
        relations: Sequence[Dict[str, Any]],
        proposals: Dict[str, List[Dict[str, Any]]],
        source_document: Optional[str] = None,
        text: str = "",
    ) -> Dict[str, Any]:
        for entity in entities:
            ontology_type = entity.get("ontology_type_id") or entity.get("type_id") or entity.get("ontology", {}).get("type_id")
            if not ontology_type:
                continue
            confidence = float(entity.get("ontology_confidence", entity.get("confidence", 0.0)) or 0.0)
            evidence = entity.get("ontology_evidence") or entity.get("name") or text[:200]
            proposal_payload = None
            if entity.get("ontology_status") == "proposed":
                proposal_payload = next((item for item in proposals.get("entities", []) if item.get("candidate_id") == ontology_type), None)
            self._update_observation_bucket(
                "type_observations",
                ontology_type,
                label=entity.get("ontology_label") or entity.get("name") or ontology_type,
                confidence=confidence,
                evidence=evidence,
                source_document=source_document,
                kind="type",
                proposal=proposal_payload,
            )

        for relation in relations:
            ontology_relation = relation.get("ontology_relation_id") or relation.get("relation_type_id") or relation.get("ontology", {}).get("relation_id") or relation.get("relation_type")
            if not ontology_relation:
                continue
            confidence = float(relation.get("ontology_confidence", relation.get("confidence", 0.0)) or 0.0)
            evidence = relation.get("ontology_evidence") or relation.get("evidence") or relation.get("relation_type") or text[:200]
            proposal_payload = None
            if relation.get("ontology_status") == "proposed":
                proposal_payload = next((item for item in proposals.get("relations", []) if item.get("candidate_id") == ontology_relation), None)
            self._update_observation_bucket(
                "relation_observations",
                ontology_relation,
                label=relation.get("ontology_label") or relation.get("relation_type") or ontology_relation,
                confidence=confidence,
                evidence=evidence,
                source_document=source_document,
                kind="relation",
                proposal=proposal_payload,
            )

        self.state.setdefault("recent_observations", []).append(
            {
                "timestamp": _now(),
                "source_document": source_document or "unknown",
                "entity_count": len(entities),
                "relation_count": len(relations),
                "proposal_count": len(proposals.get("entities", [])) + len(proposals.get("relations", [])),
                "text_excerpt": text[:240],
            }
        )
        self.state["recent_observations"] = self.state["recent_observations"][-100:]

        return self.snapshot()

    def register_promotions(self, promotions: Dict[str, List[str]], registry: OntologyRegistry) -> None:
        active_extensions = self.state.setdefault("active_extensions", {"types": {}, "relations": {}})
        for type_id in promotions.get("types", []):
            definition = registry.get_type(type_id)
            if definition is not None:
                active_extensions.setdefault("types", {})[type_id] = definition.to_dict()
        for relation_id in promotions.get("relations", []):
            definition = registry.get_relation(relation_id)
            if definition is not None:
                active_extensions.setdefault("relations", {})[relation_id] = definition.to_dict()

    def list_packs(self) -> List[Dict[str, Any]]:
        active_types = self.state.get("active_extensions", {}).get("types", {})
        active_relations = self.state.get("active_extensions", {}).get("relations", {})
        proposed_types = self.state.get("proposals", {}).get("types", {})
        proposed_relations = self.state.get("proposals", {}).get("relations", {})

        type_counts = Counter((payload.get("pack") or "core") for payload in active_types.values())
        relation_counts = Counter((payload.get("pack") or "core") for payload in active_relations.values())
        proposed_type_counts = Counter((payload.get("pack") or "core") for payload in proposed_types.values())
        proposed_relation_counts = Counter((payload.get("pack") or "core") for payload in proposed_relations.values())
        pack_names = sorted(set(type_counts) | set(relation_counts) | set(proposed_type_counts) | set(proposed_relation_counts))

        return [
            {
                "pack": pack_name,
                "active_types": int(type_counts.get(pack_name, 0)),
                "active_relations": int(relation_counts.get(pack_name, 0)),
                "proposed_types": int(proposed_type_counts.get(pack_name, 0)),
                "proposed_relations": int(proposed_relation_counts.get(pack_name, 0)),
                "total": int(
                    type_counts.get(pack_name, 0)
                    + relation_counts.get(pack_name, 0)
                    + proposed_type_counts.get(pack_name, 0)
                    + proposed_relation_counts.get(pack_name, 0)
                ),
            }
            for pack_name in pack_names
        ]

    def build_report(self) -> Dict[str, Any]:
        type_observations = self.state.get("type_observations", {})
        relation_observations = self.state.get("relation_observations", {})
        proposed_types = self.state.get("proposals", {}).get("types", {})
        proposed_relations = self.state.get("proposals", {}).get("relations", {})

        active_type_count = len(type_observations)
        active_relation_count = len(relation_observations)
        proposed_type_count = len(proposed_types)
        proposed_relation_count = len(proposed_relations)
        total_observations = active_type_count + active_relation_count + proposed_type_count + proposed_relation_count
        coverage = (active_type_count + active_relation_count) / max(1, total_observations)

        return {
            "schema_version": self.state.get("schema_version", OntologyRegistry.schema_version),
            "registry_version": self.state.get("registry_version", ""),
            "updated_at": self.state.get("updated_at"),
            "active_type_count": active_type_count,
            "active_relation_count": active_relation_count,
            "proposed_type_count": proposed_type_count,
            "proposed_relation_count": proposed_relation_count,
            "coverage": round(float(coverage), 3),
            "packs": self.list_packs(),
            "proposals": {
                "types": list(proposed_types.values()),
                "relations": list(proposed_relations.values()),
            },
            "active_extensions": self.state.get("active_extensions", {}),
        }


class OntologyEnricher:
    """Zero-shot + few-shot ontology enrichment and controlled evolution."""

    def __init__(
        self,
        registry: Optional[OntologyRegistry] = None,
        state_store: Optional[OntologyStateStore] = None,
        *,
        auto_promote: bool = True,
    ) -> None:
        self.state_store = state_store or OntologyStateStore()
        self.registry = registry or OntologyRegistry.load_default(self.state_store.state_path)
        self.auto_promote = auto_promote
        self.registry.refresh_examples(self.state_store.state.get("examples", {}).get("type", []) + self.state_store.state.get("examples", {}).get("relation", []))

    def _entity_context(self, text: str, entity: Dict[str, Any]) -> str:
        name = entity.get("name", "")
        entity_type = entity.get("entity_type", "")
        return " ".join(part for part in [name, entity_type, text[:240]] if part)

    def _relation_context(self, text: str, relation: Dict[str, Any]) -> str:
        source = relation.get("source", "")
        target = relation.get("target", "")
        relation_type = relation.get("relation_type", "")
        return " ".join(part for part in [source, relation_type, target, text[:240]] if part)

    def _entity_stable_id(self, entity: Dict[str, Any]) -> str:
        stable = entity.get("stable_id") or entity.get("canonical_name") or canonicalize_entity_name(entity.get("name", ""))
        return stable or _stable_hash(entity.get("name", ""))[:12]

    def _relation_stable_id(self, relation: Dict[str, Any]) -> str:
        stable = relation.get("stable_id") or relation.get("relation_id")
        if stable:
            return str(stable)

        source = relation.get("source_stable_id") or canonicalize_entity_name(relation.get("source", ""))
        target = relation.get("target_stable_id") or canonicalize_entity_name(relation.get("target", ""))
        relation_type = _normalize(str(relation.get("relation_type", ""))) or "related_to"
        if source or target:
            return "__".join(part for part in [source, relation_type, target] if part)

        stable = relation.get("relation_type")
        if stable:
            return str(stable)
        return _stable_hash(json.dumps(relation, sort_keys=True))[:12]

    def enrich(
        self,
        *,
        entities: Sequence[Dict[str, Any]],
        relations: Sequence[Dict[str, Any]],
        text: str,
        source_document: Optional[str] = None,
        page_map: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        enriched_entities: List[Dict[str, Any]] = []
        enriched_relations: List[Dict[str, Any]] = []
        proposals: Dict[str, List[Dict[str, Any]]] = {"entities": [], "relations": []}

        entity_lookup: Dict[str, Dict[str, Any]] = {}
        canonical_lookup: Dict[str, Dict[str, Any]] = {}

        for entity in entities:
            name = str(entity.get("name", "")).strip()
            entity_type = str(entity.get("entity_type", "")).strip()
            entity_context = self._entity_context(text, entity)
            match, proposed = self.registry.match_or_propose_entity(name, entity_type=entity_type, context=entity_context)
            stable_id = self._entity_stable_id(entity)
            raw_confidence = float(entity.get("confidence", 0.0) or 0.0)
            ontology_confidence = float(match.score if isinstance(match, OntologyMatch) else max(raw_confidence, 0.35))
            if proposed:
                proposal_dict = match.to_dict() if isinstance(match, OntologyProposal) else {}
                proposal_dict.update(
                    {
                        "candidate_id": match.candidate_id if isinstance(match, OntologyProposal) else stable_id,
                        "label": match.label if isinstance(match, OntologyProposal) else name,
                        "observed_count": 1,
                        "confidence_sum": ontology_confidence,
                        "average_confidence": ontology_confidence,
                        "parent_type_id": match.parent_type_id if isinstance(match, OntologyProposal) else entity_type or "entity",
                        "aliases": list(match.aliases) if isinstance(match, OntologyProposal) else [name],
                        "examples": [entity_context[:240]],
                        "source_docs": [source_document] if source_document else [],
                        "keywords": _tokenize(name),
                        "pack": "evolved",
                        "description": entity_context[:240],
                        "unknown_candidate": entity.get("unknown_candidate"),
                    }
                )
                proposals["entities"].append(proposal_dict)
                ontology_type_id = proposal_dict["candidate_id"]
                ontology_label = proposal_dict["label"]
                ontology_parent = proposal_dict.get("parent_type_id")
                ontology_status = "proposed"
                ontology_reason = "zero_shot_proposal"
                ontology_path = (ontology_parent,) if ontology_parent else tuple()
                ontology_source = "proposed"
            else:
                ontology_type_id = match.type_id
                ontology_label = match.label
                ontology_parent = match.parent_type_id
                ontology_status = match.status
                ontology_reason = match.reason
                ontology_path = match.path
                ontology_source = match.source

            enriched = {
                **entity,
                "stable_id": stable_id,
                "ontology_type_id": ontology_type_id,
                "ontology_label": ontology_label,
                "ontology_parent_type_id": ontology_parent,
                "ontology_status": ontology_status,
                "ontology_confidence": round(min(1.0, max(ontology_confidence, raw_confidence)), 3),
                "ontology_reason": ontology_reason,
                "ontology_path": list(ontology_path),
                "ontology_source": ontology_source,
                "ontology_evidence": entity_context[:240],
                "ontology_version": self.registry.registry_version,
                "schema_version": self.registry.schema_version,
                "ontology": {
                    "type_id": ontology_type_id,
                    "label": ontology_label,
                    "parent_type_id": ontology_parent,
                    "status": ontology_status,
                    "confidence": round(min(1.0, max(ontology_confidence, raw_confidence)), 3),
                    "path": list(ontology_path),
                    "source": ontology_source,
                    "reason": ontology_reason,
                },
            }
            enriched_entities.append(enriched)
            entity_lookup[stable_id] = enriched
            canonical_lookup[_normalize(name)] = enriched
            if name:
                canonical_lookup[_normalize(name.replace("-", " "))] = enriched

        for relation in relations:
            relation_type = str(relation.get("relation_type", "")).strip()
            relation_context = self._relation_context(text, relation)
            match, proposed = self.registry.match_or_propose_relation(relation_type, context=relation_context)
            stable_id = self._relation_stable_id(relation)
            ontology_confidence = float(match.score if isinstance(match, OntologyMatch) else max(float(relation.get("confidence", 0.0) or 0.0), 0.3))
            if proposed:
                proposal_dict = match.to_dict() if isinstance(match, OntologyProposal) else {}
                proposal_dict.update(
                    {
                        "candidate_id": match.candidate_id if isinstance(match, OntologyProposal) else stable_id,
                        "label": match.label if isinstance(match, OntologyProposal) else relation_type,
                        "observed_count": 1,
                        "confidence_sum": ontology_confidence,
                        "average_confidence": ontology_confidence,
                        "aliases": list(match.aliases) if isinstance(match, OntologyProposal) else [relation_type],
                        "examples": [relation_context[:240]],
                        "source_docs": [source_document] if source_document else [],
                        "keywords": _tokenize(relation_type),
                        "pack": "evolved",
                        "description": relation_context[:240],
                        "unknown_candidate": relation.get("unknown_candidate"),
                    }
                )
                proposals["relations"].append(proposal_dict)
                ontology_relation_id = proposal_dict["candidate_id"]
                ontology_label = proposal_dict["label"]
                ontology_status = "proposed"
                ontology_reason = "zero_shot_proposal"
                ontology_source = "proposed"
            else:
                ontology_relation_id = match.type_id
                ontology_label = match.label
                ontology_status = match.status
                ontology_reason = match.reason
                ontology_source = match.source

            source_ref = relation.get("source")
            target_ref = relation.get("target")
            source_key = _normalize(str(source_ref or ""))
            target_key = _normalize(str(target_ref or ""))
            source_entity = entity_lookup.get(str(relation.get("source_stable_id") or "")) or canonical_lookup.get(source_key)
            target_entity = entity_lookup.get(str(relation.get("target_stable_id") or "")) or canonical_lookup.get(target_key)

            enriched = {
                **relation,
                "stable_id": stable_id,
                "source_stable_id": (source_entity or {}).get("stable_id") or relation.get("source_stable_id") or _normalize(str(source_ref or "")),
                "target_stable_id": (target_entity or {}).get("stable_id") or relation.get("target_stable_id") or _normalize(str(target_ref or "")),
                "ontology_relation_id": ontology_relation_id,
                "ontology_label": ontology_label,
                "ontology_status": ontology_status,
                "ontology_confidence": round(ontology_confidence, 3),
                "ontology_reason": ontology_reason,
                "ontology_source": ontology_source,
                "ontology_evidence": relation_context[:240],
                "ontology_version": self.registry.registry_version,
                "schema_version": self.registry.schema_version,
                "ontology": {
                    "relation_id": ontology_relation_id,
                    "label": ontology_label,
                    "status": ontology_status,
                    "confidence": round(ontology_confidence, 3),
                    "source": ontology_source,
                    "reason": ontology_reason,
                },
            }
            enriched_relations.append(enriched)

        state_snapshot = self.state_store.record_observations(
            entities=enriched_entities,
            relations=enriched_relations,
            proposals=proposals,
            source_document=source_document,
            text=text,
        )
        self.state_store.save()

        promotions: Dict[str, List[str]] = {"types": [], "relations": []}
        if self.auto_promote:
            promotions = self.registry.promote_from_state_snapshot(state_snapshot)
            if promotions["types"] or promotions["relations"]:
                self.state_store.register_promotions(promotions, self.registry)
                self.state_store.save()

        self.registry.refresh_examples(
            self.state_store.state.get("examples", {}).get("type", [])
            + self.state_store.state.get("examples", {}).get("relation", [])
        )

        report = self.state_store.build_report()
        report.update(
            {
                "status": "completed",
                "source_document": source_document,
                "entity_count": len(enriched_entities),
                "relation_count": len(enriched_relations),
                "matched_entities": sum(1 for entity in enriched_entities if entity.get("ontology_status") == "active"),
                "proposed_entities": len(proposals["entities"]),
                "matched_relations": sum(1 for relation in enriched_relations if relation.get("ontology_status") == "active"),
                "proposed_relations": len(proposals["relations"]),
                "promotions": promotions,
                "substeps": {
                    "zero_shot_matching": {
                        "entity_count": len(enriched_entities),
                        "relation_count": len(enriched_relations),
                        "matched_entity_count": sum(1 for entity in enriched_entities if entity.get("ontology_status") == "active"),
                        "matched_relation_count": sum(1 for relation in enriched_relations if relation.get("ontology_status") == "active"),
                    },
                    "few_shot_retrieval": {
                        "example_count": len(self.state_store.state.get("examples", {}).get("type", [])) + len(self.state_store.state.get("examples", {}).get("relation", [])),
                        "registry_examples": len(self.registry.build_examples()),
                    },
                    "proposal_generation": {
                        "entity_proposals": len(proposals["entities"]),
                        "relation_proposals": len(proposals["relations"]),
                    },
                    "promotion": promotions,
                    "state_persistence": {
                        "path": str(self.state_store.state_path),
                        "saved": True,
                    },
                },
            }
        )

        return {
            "entities": enriched_entities,
            "relations": enriched_relations,
            "ontology_report": report,
            "ontology_proposals": proposals,
            "ontology_state_path": str(self.state_store.state_path),
            "ontology_version": self.registry.registry_version,
            "schema_version": self.registry.schema_version,
        }


def load_ontology_registry(state_path: Path | None = None) -> OntologyRegistry:
    return OntologyRegistry.load_default(state_path=state_path)


def get_default_ontology_enricher(state_path: Path | None = None) -> OntologyEnricher:
    store = OntologyStateStore(state_path=state_path)
    return OntologyEnricher(state_store=store)
