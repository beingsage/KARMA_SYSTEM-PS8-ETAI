from typing import Any, List, Optional
from pydantic import BaseModel, Field


class ProcessPdfRequest(BaseModel):
    file_name: Optional[str] = Field(default=None)


class EntityRecord(BaseModel):
    name: str
    entity_type: str
    confidence: float
    canonical_name: str


class RelationRecord(BaseModel):
    source: str
    target: str
    relation_type: str
    confidence: float


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
    pipeline_metadata: Optional[dict[str, object]] = None
