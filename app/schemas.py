from typing import Any, List, Optional
from pydantic import BaseModel, Field


class ProcessPdfRequest(BaseModel):
    file_name: Optional[str] = Field(default=None)


class EntityRecord(BaseModel):
    name: str
    entity_type: str
    confidence: float
    canonical_name: str
    stable_id: Optional[str] = None
    ontology: Optional[dict[str, Any]] = None
    ontology_type_id: Optional[str] = None
    ontology_label: Optional[str] = None
    ontology_parent_type_id: Optional[str] = None
    ontology_status: Optional[str] = None
    ontology_confidence: Optional[float] = None
    ontology_source: Optional[str] = None
    ontology_reason: Optional[str] = None
    ontology_path: Optional[List[str]] = None
    evidence_span: Optional[dict[str, Any]] = None
    unknown_candidate: Optional[dict[str, Any]] = None
    schema_version: Optional[str] = None
    status: Optional[str] = None
    type_id: Optional[str] = None
    parent_type_id: Optional[str] = None
    provenance: Optional[dict[str, Any]] = None


class RelationRecord(BaseModel):
    source: str
    target: str
    relation_type: str
    confidence: float
    stable_id: Optional[str] = None
    source_stable_id: Optional[str] = None
    target_stable_id: Optional[str] = None
    source_span: Optional[list[int]] = None
    target_span: Optional[list[int]] = None
    evidence_span: Optional[dict[str, Any]] = None
    unknown_candidate: Optional[dict[str, Any]] = None
    ontology: Optional[dict[str, Any]] = None
    ontology_relation_id: Optional[str] = None
    ontology_label: Optional[str] = None
    ontology_status: Optional[str] = None
    ontology_confidence: Optional[float] = None
    ontology_source: Optional[str] = None
    ontology_reason: Optional[str] = None
    schema_version: Optional[str] = None
    status: Optional[str] = None
    type_id: Optional[str] = None
    provenance: Optional[dict[str, Any]] = None


class SchemaProposal(BaseModel):
    proposal_id: Optional[str] = None
    kind: str
    candidate_id: str
    label: str
    parent_type_id: Optional[str] = None
    status: str = "proposed"
    confidence: float = 0.0
    source: str = "zero_shot"
    evidence: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    source_docs: List[str] = Field(default_factory=list)


class JobResult(BaseModel):
    job_id: str
    status: str
    message: str
    uploaded_filename: Optional[str] = None
    extraction_summary: Optional[str] = None
    text: Optional[str] = None
    entities: List[EntityRecord] = []
    relations: List[RelationRecord] = []
    layout: Optional[List[dict[str, Any]]] = None
    tables: Optional[List[dict[str, Any]]] = None
    formulas: Optional[List[str]] = None
    reading_order: Optional[List[dict[str, Any]]] = None
    doclayout_yolo: Optional[dict[str, Any]] = None
    table_transformer: Optional[dict[str, Any]] = None
    yolo_pid_insights: Optional[dict[str, Any]] = None
    vision_language: Optional[dict[str, Any]] = None
    graph_summary: Optional[str] = None
    neo4j_status: Optional[str] = None
    ontology_enrichment: Optional[dict[str, Any]] = None
    ontology_report: Optional[dict[str, Any]] = None
    ontology_proposals: Optional[dict[str, Any]] = None
    schema_proposals: Optional[List[SchemaProposal]] = None
    pipeline_metadata: Optional[dict[str, object]] = None


class OntologyMigrationRequest(BaseModel):
    dry_run: bool = True
    limit: Optional[int] = None
    include_all_nodes: bool = True
    create_typed_labels: bool = False
