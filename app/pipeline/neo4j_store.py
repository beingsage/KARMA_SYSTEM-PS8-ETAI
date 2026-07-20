"""
Neo4j Knowledge Graph Persistence
Stores industrial entities, relations, and reasoning in Neo4j
"""

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List
import time

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import AuthError, Neo4jError
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = None  # type: ignore

    class AuthError(Exception):
        pass

    class Neo4jError(Exception):
        pass

from app.config import settings
from app.pipeline.models import canonicalize_entity_name
from app.pipeline.ontology import OntologyRegistry, load_ontology_registry


class Neo4jGraphStore:
    """Persist industrial knowledge graph to Neo4j"""
    
    def __init__(self):
        self.driver = None
        self.connected = False
        self._connect()

    @staticmethod
    def _is_auth_failure(exc: Exception) -> bool:
        code = str(getattr(exc, "code", "")) or str(getattr(exc, "neo4j_code", ""))
        text = str(exc)
        return any(
            token in code or token in text
            for token in (
                "Unauthorized",
                "AuthenticationRateLimit",
                "CredentialsExpired",
            )
        )
    
    @staticmethod
    def _credential_candidates() -> list[tuple[str, str] | None]:
        candidates: list[tuple[str, str] | None] = []
        if settings.neo4j_user and settings.neo4j_password:
            candidates.append((settings.neo4j_user, settings.neo4j_password))
        default_user = settings.neo4j_user or "neo4j"
        default_password = "neo4j"
        fallback = (default_user, default_password)
        if fallback not in candidates:
            candidates.append(fallback)
        candidates.append(None)
        return candidates

    def _connect(self) -> bool:
        if GraphDatabase is None:
            print("⚠ Neo4j driver not installed; graph persistence disabled")
            return False

        last_error: Exception | None = None
        credentials = self._credential_candidates()
        if not credentials:
            print("⚠ Neo4j credentials are not configured. Graph persistence disabled.")
            return False

        for auth in credentials:
            user_label = "no-auth" if auth is None else auth[0]
            for attempt in range(3):
                try:
                    self.driver = GraphDatabase.driver(
                        settings.neo4j_uri,
                        auth=auth,
                        encrypted=False,
                    )
                    with self.driver.session() as session:
                        session.run("RETURN 1")
                    self.connected = True
                    if auth is None:
                        print(f"✓ Connected to Neo4j at {settings.neo4j_uri} without authentication")
                    else:
                        print(f"✓ Connected to Neo4j at {settings.neo4j_uri} as {user_label}")
                    return True
                except AuthError as e:
                    last_error = e
                    self.connected = False
                    break
                except Neo4jError as e:
                    last_error = e
                    self.connected = False
                    if self._is_auth_failure(e):
                        break
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    break
                except Exception as e:
                    last_error = e
                    self.connected = False
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    break
                finally:
                    if not self.connected and self.driver:
                        try:
                            self.driver.close()
                        except Exception:
                            pass
                        self.driver = None

        if last_error and settings.verbose:
            print(f"⚠ Neo4j unavailable; graph persistence disabled ({last_error})")
        else:
            print("⚠ Neo4j unavailable; graph persistence disabled")
        return False
    
    def create_indices(self) -> bool:
        if not self.driver:
            return False
        try:
            with self.driver.session() as session:
                session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.stable_id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.ontology_type_id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (t:OntologyType) ON (t.type_id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (r:RELATION) ON (r.stable_id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (r:RELATION) ON (r.type)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (j:Job) ON (j.job_id)")
            print("✓ Neo4j indices created")
            return True
        except Exception as e:
            print(f"⚠ Failed to create indices: {e}")
            return False

    @staticmethod
    def _entity_hash(name: str, entity_type: str) -> str:
        identifier = f"{canonicalize_entity_name(name)}|{canonicalize_entity_name(entity_type)}"
        return hashlib.md5(identifier.encode("utf-8")).hexdigest()

    @staticmethod
    def _relation_hash(source_id: str, relation_type: str, target_id: str) -> str:
        identifier = f"{source_id.strip().lower()}|{relation_type.strip().lower()}|{target_id.strip().lower()}"
        return hashlib.md5(identifier.encode("utf-8")).hexdigest()

    @staticmethod
    def _safe_neo4j_label(value: str, *, prefix: str = "Ontology") -> str:
        label = re.sub(r"[^A-Za-z0-9_]+", "_", value or "").strip("_")
        if not label:
            label = "Node"
        if label[0].isdigit():
            label = f"T_{label}"
        return f"{prefix}_{label}" if prefix else label

    @staticmethod
    def _resolve_entity_ontology_metadata(entity: Dict[str, Any], registry: OntologyRegistry) -> Dict[str, Any]:
        name = str(entity.get("name") or entity.get("canonical_name") or entity.get("stable_id") or "").strip()
        entity_type = str(entity.get("entity_type") or entity.get("type") or "").strip()
        canonical_name = str(entity.get("canonical_name") or canonicalize_entity_name(name) or canonicalize_entity_name(entity_type)).strip()
        stable_id = str(entity.get("stable_id") or canonical_name or canonicalize_entity_name(name) or canonicalize_entity_name(entity_type)).strip()
        context = " ".join(
            part
            for part in [
                str(entity.get("description") or "").strip(),
                str(entity.get("ontology_evidence") or "").strip(),
                str(entity.get("source_document") or "").strip(),
            ]
            if part
        )[:240]

        match = registry.resolve_entity(name or canonical_name, entity_type=entity_type or None, context=context)
        if match is not None:
            return {
                "stable_id": stable_id,
                "canonical_name": canonical_name or name,
                "ontology_type_id": match.type_id,
                "ontology_label": match.label,
                "ontology_parent_type_id": match.parent_type_id,
                "ontology_status": match.status,
                "ontology_confidence": match.score,
                "ontology_path": list(match.path),
                "ontology": {
                    "type_id": match.type_id,
                    "label": match.label,
                    "parent_type_id": match.parent_type_id,
                    "status": match.status,
                    "confidence": match.score,
                    "path": list(match.path),
                    "source": match.source,
                    "reason": match.reason,
                },
                "unknown_candidate": entity.get("unknown_candidate"),
            }

        proposal = registry.propose_entity(name or canonical_name, entity_type=entity_type or None, context=context, confidence=float(entity.get("confidence", 0.0) or 0.0))
        return {
            "stable_id": stable_id,
            "canonical_name": canonical_name or name,
            "ontology_type_id": proposal.candidate_id,
            "ontology_label": proposal.label,
            "ontology_parent_type_id": proposal.parent_type_id,
            "ontology_status": proposal.status,
            "ontology_confidence": proposal.confidence,
            "ontology_path": [proposal.parent_type_id] if proposal.parent_type_id else [],
            "ontology": {
                "type_id": proposal.candidate_id,
                "label": proposal.label,
                "parent_type_id": proposal.parent_type_id,
                "status": proposal.status,
                "confidence": proposal.confidence,
                "path": [proposal.parent_type_id] if proposal.parent_type_id else [],
                "source": proposal.source,
                "reason": "proposed",
            },
            "unknown_candidate": entity.get("unknown_candidate")
            or {
                "candidate_label": name or canonical_name or proposal.label,
                "candidate_type": entity_type or "unknown",
                "parent_type_id": proposal.parent_type_id or "entity",
                "reason": "proposed from backfill",
                "confidence": proposal.confidence,
            },
        }

    @staticmethod
    def _resolve_relation_ontology_metadata(
        relation: Dict[str, Any],
        registry: OntologyRegistry,
    ) -> Dict[str, Any]:
        source = str(relation.get("source") or "").strip()
        target = str(relation.get("target") or "").strip()
        relation_type = str(relation.get("relation_type") or "related_to").strip() or "related_to"
        source_stable_id = str(relation.get("source_stable_id") or canonicalize_entity_name(source)).strip()
        target_stable_id = str(relation.get("target_stable_id") or canonicalize_entity_name(target)).strip()
        relation_id = str(
            relation.get("stable_id")
            or f"{source_stable_id}__{canonicalize_entity_name(relation_type)}__{target_stable_id}"
        ).strip("_")
        context = " ".join(
            part
            for part in [
                source,
                relation_type,
                target,
                str(relation.get("evidence") or relation.get("ontology_evidence") or "").strip(),
            ]
            if part
        )[:240]

        match = registry.resolve_relation(relation_type, context=context)
        if match is not None:
            return {
                "stable_id": relation_id,
                "source_stable_id": source_stable_id,
                "target_stable_id": target_stable_id,
                "ontology_relation_id": match.type_id,
                "ontology_label": match.label,
                "ontology_status": match.status,
                "ontology_confidence": match.score,
                "ontology": {
                    "relation_id": match.type_id,
                    "label": match.label,
                    "status": match.status,
                    "confidence": match.score,
                    "source": match.source,
                    "reason": match.reason,
                },
                "unknown_candidate": relation.get("unknown_candidate"),
            }

        proposal = registry.propose_relation(relation_type, context=context, confidence=float(relation.get("confidence", 0.0) or 0.0))
        return {
            "stable_id": relation_id,
            "source_stable_id": source_stable_id,
            "target_stable_id": target_stable_id,
            "ontology_relation_id": proposal.candidate_id,
            "ontology_label": proposal.label,
            "ontology_status": proposal.status,
            "ontology_confidence": proposal.confidence,
            "ontology": {
                "relation_id": proposal.candidate_id,
                "label": proposal.label,
                "status": proposal.status,
                "confidence": proposal.confidence,
                "source": proposal.source,
                "reason": "proposed",
            },
            "unknown_candidate": relation.get("unknown_candidate")
            or {
                "candidate_label": relation_type or proposal.label,
                "candidate_type": relation_type or "unknown",
                "parent_type_id": "related_to",
                "reason": "proposed from backfill",
                "confidence": proposal.confidence,
            },
        }

    def persist_entities(
        self,
        entities: List[Dict[str, Any]],
        job_id: str,
        batch_size: int = None,
    ) -> bool:
        """Persist entities with memory-efficient batching."""
        from app.config import settings
        
        if batch_size is None:
            batch_size = settings.neo4j_entity_batch_size
        
        if not self.driver:
            return False

        valid_entities = [
            entity
            for entity in entities
            if entity.get("name") and entity.get("entity_type")
        ]
        if not valid_entities:
            print("⚠ No valid entities to persist")
            return False

        timestamp = datetime.now().isoformat()
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (j:Job {job_id: $job_id})
                    SET j.timestamp = $timestamp,
                        j.entity_count = $entity_count
                    """,
                    job_id=job_id,
                    timestamp=timestamp,
                    entity_count=len(valid_entities),
                )

            # Process entities in batches to avoid memory explosion
            for batch_start in range(0, len(valid_entities), batch_size):
                batch_end = min(batch_start + batch_size, len(valid_entities))
                batch = valid_entities[batch_start:batch_end]
                
                with self.driver.session() as session:
                    for entity in batch:
                        name = entity.get("name", "").strip()
                        entity_type = entity.get("entity_type", "").strip()
                        stable_id = str(
                            entity.get("stable_id")
                            or entity.get("canonical_name")
                            or canonicalize_entity_name(name)
                            or self._entity_hash(name, entity_type)
                        ).strip()
                        legacy_id = self._entity_hash(name, entity_type)
                        ontology = entity.get("ontology", {}) if isinstance(entity.get("ontology"), dict) else {}
                        ontology_type_id = entity.get("ontology_type_id") or ontology.get("type_id") or entity_type
                        ontology_label = entity.get("ontology_label") or ontology.get("label") or entity_type
                        ontology_parent_type_id = entity.get("ontology_parent_type_id") or ontology.get("parent_type_id")
                        ontology_status = entity.get("ontology_status") or ontology.get("status")
                        ontology_confidence = entity.get("ontology_confidence")
                        ontology_path = entity.get("ontology_path") or ontology.get("path") or []
                        source_document = entity.get("source_document")
                        evidence_span = entity.get("evidence_span")
                        unknown_candidate = entity.get("unknown_candidate")
                        type_id = entity.get("type_id") or entity.get("ontology_type_id") or entity_type
                        parent_type_id = entity.get("parent_type_id") or entity.get("ontology_parent_type_id")
                        status = entity.get("status") or entity.get("ontology_status") or "active"
                        schema_version = entity.get("schema_version") or "1.0.0"
                        provenance = entity.get("provenance") or {
                            "source_document": source_document,
                            "source_method": entity.get("source") or entity.get("source_method") or "pipeline",
                            "evidence": entity.get("evidence") or entity.get("context") or "",
                        }
                        session.run(
                            """
                            MERGE (e:Entity {stable_id: $stable_id})
                            SET e.name = $name,
                                e.canonical_name = $canonical_name,
                                e.id = coalesce(e.id, $legacy_id),
                                e.type = $entity_type,
                                e.entity_type = $entity_type,
                                e.confidence = $confidence,
                                e.ontology_type_id = $ontology_type_id,
                                e.ontology_label = $ontology_label,
                                e.ontology_parent_type_id = $ontology_parent_type_id,
                                e.ontology_status = $ontology_status,
                                e.ontology_confidence = $ontology_confidence,
                                e.ontology_path = $ontology_path,
                                e.ontology = $ontology,
                                e.type_id = $type_id,
                                e.parent_type_id = $parent_type_id,
                                e.schema_version = $schema_version,
                                e.status = $status,
                                e.provenance = $provenance,
                                e.source_document = $source_document,
                                e.evidence_span = $evidence_span,
                                e.unknown_candidate = $unknown_candidate,
                                e.timestamp = $timestamp
                            WITH e
                            MATCH (j:Job {job_id: $job_id})
                            MERGE (j)-[r:EXTRACTED_ENTITY]->(e)
                            SET r.timestamp = $timestamp
                            """,
                            stable_id=stable_id,
                            legacy_id=legacy_id,
                            name=name,
                            canonical_name=entity.get("canonical_name", name),
                            entity_type=entity_type,
                            confidence=entity.get("confidence", 0.5),
                            ontology=ontology,
                            ontology_type_id=ontology_type_id,
                            ontology_label=ontology_label,
                            ontology_parent_type_id=ontology_parent_type_id,
                            ontology_status=ontology_status,
                            ontology_confidence=ontology_confidence,
                            ontology_path=ontology_path,
                            source_document=source_document,
                            evidence_span=evidence_span,
                            unknown_candidate=unknown_candidate,
                            type_id=type_id,
                            parent_type_id=parent_type_id,
                            schema_version=schema_version,
                            status=status,
                            provenance=provenance,
                            job_id=job_id,
                            timestamp=timestamp,
                        )

                # Explicit cleanup after batch
                batch = None
                import gc
                gc.collect()
            
            print(f"✓ Persisted {len(valid_entities)} entities to Neo4j in batches of {batch_size}")
            return True
        except Exception as e:
            print(f"⚠ Failed to persist entities: {e}")
            return False
                            e.id = coalesce(e.id, $legacy_id),
                            e.type = $entity_type,
                            e.entity_type = $entity_type,
                            e.confidence = $confidence,
                            e.ontology_type_id = $ontology_type_id,
                            e.ontology_label = $ontology_label,
                            e.ontology_parent_type_id = $ontology_parent_type_id,
                            e.ontology_status = $ontology_status,
                            e.ontology_confidence = $ontology_confidence,
                            e.ontology_path = $ontology_path,
                            e.ontology = $ontology,
                            e.type_id = $type_id,
                            e.parent_type_id = $parent_type_id,
                            e.schema_version = $schema_version,
                            e.status = $status,
                            e.evidence_span = $evidence_span,
                            e.unknown_candidate = $unknown_candidate,
                            e.provenance = $provenance,
                            e.source_document = coalesce($source_document, e.source_document),
                            e.updated_at = $timestamp
                        WITH e
                        FOREACH (_ IN CASE WHEN $ontology_type_id IS NULL OR $ontology_type_id = '' THEN [] ELSE [1] END |
                            MERGE (t:OntologyType {type_id: $ontology_type_id})
                            SET t.label = $ontology_label,
                                t.parent_type_id = $ontology_parent_type_id,
                                t.status = $ontology_status,
                                t.updated_at = $timestamp
                            MERGE (e)-[:INSTANCE_OF]->(t)
                        )
                        WITH e
                        MATCH (j:Job {job_id: $job_id})
                        MERGE (j)-[r:EXTRACTED_ENTITY]->(e)
                        SET r.timestamp = $timestamp
                        """,
                        stable_id=stable_id,
                        legacy_id=legacy_id,
                        name=name,
                        canonical_name=entity.get("canonical_name", name),
                        entity_type=entity_type,
                        confidence=entity.get("confidence", 0.5),
                        ontology=ontology,
                        ontology_type_id=ontology_type_id,
                        ontology_label=ontology_label,
                        ontology_parent_type_id=ontology_parent_type_id,
                        ontology_status=ontology_status,
                        ontology_confidence=ontology_confidence,
                        ontology_path=ontology_path,
                        source_document=source_document,
                        evidence_span=evidence_span,
                        unknown_candidate=unknown_candidate,
                        type_id=type_id,
                        parent_type_id=parent_type_id,
                        schema_version=schema_version,
                        status=status,
                        provenance=provenance,
                        job_id=job_id,
                        timestamp=timestamp,
                    )

                # Explicit cleanup after batch
                batch = None
                import gc
                gc.collect()
            
            print(f"✓ Persisted {len(valid_entities)} entities to Neo4j in batches of {batch_size}")
                return True
        except Exception as e:
            print(f"⚠ Failed to persist entities: {e}")
            return False
    
    def persist_relations(
        self,
        relations: List[Dict[str, Any]],
        job_id: str,
        batch_size: int = None,
    ) -> bool:
        """Persist relations with memory-efficient batching."""
        from app.config import settings
        
        if batch_size is None:
            batch_size = settings.neo4j_relation_batch_size
                    source = relation.get("source", "").strip()
                    target = relation.get("target", "").strip()
                    relation_type = relation.get("relation_type", "").strip()
                    if not source or not target:
                        continue

                    source_stable_id = relation.get("source_stable_id") or canonicalize_entity_name(source) or None
                    target_stable_id = relation.get("target_stable_id") or canonicalize_entity_name(target) or None
                    relation_id = str(
                        relation.get("stable_id")
                        or self._relation_hash(source_stable_id or source, relation_type or "related_to", target_stable_id or target)
                    ).strip()
                    ontology = relation.get("ontology", {}) if isinstance(relation.get("ontology"), dict) else {}
                    ontology_relation_id = relation.get("ontology_relation_id") or ontology.get("relation_id") or relation_type or "related_to"
                    ontology_label = relation.get("ontology_label") or ontology.get("label") or relation_type or "related_to"
                    ontology_status = relation.get("ontology_status") or ontology.get("status")
                    ontology_confidence = relation.get("ontology_confidence")
                    type_id = relation.get("type_id") or ontology_relation_id
                    schema_version = relation.get("schema_version") or "1.0.0"
                    status = relation.get("status") or ontology_status or "active"
                    evidence_span = relation.get("evidence_span")
                    source_span = relation.get("source_span")
                    target_span = relation.get("target_span")
                    unknown_candidate = relation.get("unknown_candidate")
                    provenance = relation.get("provenance") or {
                        "source_document": relation.get("source_document"),
                        "source_method": relation.get("source_method") or relation.get("source") or "pipeline",
                        "evidence": relation.get("evidence") or relation.get("context") or "",
                    }

                    session.run(
                        """
                        MATCH (source:Entity)
                        WHERE ($source_stable_id IS NOT NULL AND (source.stable_id = $source_stable_id OR source.canonical_name = $source_stable_id))
                           OR ($source_name IS NOT NULL AND (source.name = $source_name OR source.canonical_name = $source_name))
                        MATCH (target:Entity)
                        WHERE ($target_stable_id IS NOT NULL AND (target.stable_id = $target_stable_id OR target.canonical_name = $target_stable_id))
                           OR ($target_name IS NOT NULL AND (target.name = $target_name OR target.canonical_name = $target_name))
                        MERGE (source)-[r:RELATION {stable_id: $relation_id}]->(target)
                        SET r.confidence = $confidence,
                            r.type = $relation_type,
                            r.relation_type = $relation_type,
                            r.job_id = $job_id,
                            r.timestamp = $timestamp,
                            r.ontology_relation_id = $ontology_relation_id,
                            r.ontology_label = $ontology_label,
                            r.ontology_status = $ontology_status,
                            r.ontology_confidence = $ontology_confidence,
                            r.ontology = $ontology,
                            r.type_id = $type_id,
                            r.schema_version = $schema_version,
                            r.status = $status,
                            r.evidence_span = $evidence_span,
                            r.source_span = $source_span,
                            r.target_span = $target_span,
                            r.unknown_candidate = $unknown_candidate,
                            r.provenance = $provenance,
                            r.source_stable_id = $source_stable_id,
                            r.target_stable_id = $target_stable_id
                        """,
                        source_name=source,
                        target_name=target,
                        source_stable_id=source_stable_id,
                        target_stable_id=target_stable_id,
                        relation_id=relation_id,
                        relation_type=relation_type,
                        confidence=relation.get("confidence", 0.5),
                        job_id=job_id,
                        ontology=ontology,
                        ontology_relation_id=ontology_relation_id,
                        ontology_label=ontology_label,
                        ontology_status=ontology_status,
                        ontology_confidence=ontology_confidence,
                        type_id=type_id,
                        schema_version=schema_version,
                        status=status,
                        evidence_span=evidence_span,
                        source_span=source_span,
                        target_span=target_span,
                        unknown_candidate=unknown_candidate,
                        provenance=provenance,
                        timestamp=timestamp,
                    )

                    total_persisted += 1
                
                # Explicit cleanup after batch
                batch = None
                import gc
                gc.collect()
            
            print(f"✓ Persisted {total_persisted} relations to Neo4j in batches of {batch_size}")
                return True
        except Exception as e:
            print(f"⚠ Failed to persist relations: {e}")
            return False

    def migrate_existing_nodes(
        self,
        *,
        dry_run: bool = False,
        limit: int | None = None,
        create_typed_labels: bool = True,
    ) -> Dict[str, Any]:
        return self.backfill_ontology_metadata(
            dry_run=dry_run,
            limit=limit,
            include_all_nodes=True,
            create_typed_labels=create_typed_labels,
        )

    def backfill_ontology_metadata(
        self,
        *,
        dry_run: bool = False,
        limit: int | None = None,
        include_all_nodes: bool = False,
        create_typed_labels: bool = False,
    ) -> Dict[str, Any]:
        if not self.driver:
            return {"status": "unavailable", "reason": "neo4j driver unavailable"}

        registry = load_ontology_registry()
        entity_updates = 0
        relation_updates = 0
        proposed_entities = 0
        proposed_relations = 0
        sample_entities: List[Dict[str, Any]] = []
        sample_relations: List[Dict[str, Any]] = []

        if include_all_nodes:
            entity_query = "MATCH (n) WHERE NOT n:Job AND NOT n:OntologyType RETURN elementId(n) AS node_id, labels(n) AS node_labels, n AS node"
        else:
            entity_query = "MATCH (n:Entity) RETURN elementId(n) AS node_id, labels(n) AS node_labels, n AS node"
        relation_query = "MATCH (source:Entity)-[r:RELATION]->(target:Entity) RETURN source, r, target"
        if limit is not None and limit > 0:
            entity_query += " LIMIT $limit"
            relation_query += " LIMIT $limit"

        try:
            with self.driver.session() as session:
                entity_result = session.run(entity_query, limit=limit) if limit is not None and limit > 0 else session.run(entity_query)
                for record in entity_result:
                    node_id = record["node_id"]
                    node = dict(record["node"])
                    metadata = self._resolve_entity_ontology_metadata(node, registry)
                    if metadata["ontology_status"] == "proposed":
                        proposed_entities += 1
                    sample_entities.append(
                        {
                            "name": node.get("name"),
                            "stable_id": metadata["stable_id"],
                            "ontology_type_id": metadata["ontology_type_id"],
                            "ontology_status": metadata["ontology_status"],
                        }
                    )
                    if dry_run:
                        entity_updates += 1
                        continue

                    typed_label = self._safe_neo4j_label(
                        metadata["ontology_type_id"] or metadata["ontology_label"] or node.get("entity_type") or "Entity"
                    )
                    typed_label_clause = f"SET e:{typed_label}" if create_typed_labels and typed_label else ""
                    legacy_id = node.get("id") or node.get("stable_id") or metadata["stable_id"]
                    session.run(
                        f"""
                        MATCH (e)
                        WHERE elementId(e) = $node_id
                        SET e:Entity,
                            e.id = coalesce(e.id, $legacy_id),
                            e.name = $name,
                            e.canonical_name = $canonical_name,
                            e.stable_id = $stable_id,
                            e.type = $entity_type,
                            e.entity_type = $entity_type,
                            e.ontology_type_id = $ontology_type_id,
                            e.ontology_label = $ontology_label,
                            e.ontology_parent_type_id = $ontology_parent_type_id,
                            e.ontology_status = $ontology_status,
                            e.ontology_confidence = $ontology_confidence,
                            e.ontology_path = $ontology_path,
                            e.ontology = $ontology,
                            e.type_id = $ontology_type_id,
                            e.parent_type_id = $ontology_parent_type_id,
                            e.schema_version = $schema_version,
                            e.status = $status,
                            e.unknown_candidate = $unknown_candidate,
                            e.provenance = $provenance,
                            e.updated_at = $timestamp
                        WITH e
                        {typed_label_clause}
                        FOREACH (_ IN CASE WHEN $ontology_type_id IS NULL OR $ontology_type_id = '' THEN [] ELSE [1] END |
                            MERGE (t:OntologyType {{type_id: $ontology_type_id}})
                            SET t.label = $ontology_label,
                                t.parent_type_id = $ontology_parent_type_id,
                                t.status = $ontology_status,
                                t.updated_at = $timestamp
                            MERGE (e)-[:INSTANCE_OF]->(t)
                        )
                        """,
                        node_id=node_id,
                        legacy_id=legacy_id,
                        name=node.get("name"),
                        stable_id=metadata["stable_id"],
                        canonical_name=metadata["canonical_name"],
                        entity_type=node.get("entity_type") or node.get("type") or "unknown",
                        ontology_type_id=metadata["ontology_type_id"],
                        ontology_label=metadata["ontology_label"],
                        ontology_parent_type_id=metadata["ontology_parent_type_id"],
                        ontology_status=metadata["ontology_status"],
                        ontology_confidence=metadata["ontology_confidence"],
                        ontology_path=metadata["ontology_path"],
                        ontology=metadata["ontology"],
                        schema_version=node.get("schema_version") or "1.0.0",
                        status=node.get("status") or metadata["ontology_status"] or "active",
                        unknown_candidate=metadata.get("unknown_candidate"),
                        provenance=node.get("provenance") or {
                            "source_document": node.get("source_document"),
                            "source_method": node.get("source") or node.get("source_method") or "migration",
                            "evidence": node.get("evidence") or node.get("context") or "",
                        },
                        timestamp=datetime.now().isoformat(),
                    )
                    entity_updates += 1

                relation_result = session.run(relation_query, limit=limit) if limit is not None and limit > 0 else session.run(relation_query)
                for record in relation_result:
                    source = dict(record["source"])
                    relation = dict(record["r"])
                    target = dict(record["target"])
                    relation_payload = {
                        **relation,
                        "source": source.get("name") or source.get("canonical_name") or "",
                        "target": target.get("name") or target.get("canonical_name") or "",
                        "source_stable_id": source.get("stable_id") or source.get("canonical_name") or canonicalize_entity_name(source.get("name", "")),
                        "target_stable_id": target.get("stable_id") or target.get("canonical_name") or canonicalize_entity_name(target.get("name", "")),
                    }
                    metadata = self._resolve_relation_ontology_metadata(relation_payload, registry)
                    if metadata["ontology_status"] == "proposed":
                        proposed_relations += 1
                    sample_relations.append(
                        {
                            "stable_id": metadata["stable_id"],
                            "ontology_relation_id": metadata["ontology_relation_id"],
                            "ontology_status": metadata["ontology_status"],
                        }
                    )
                    if dry_run:
                        relation_updates += 1
                        continue

                    session.run(
                        """
                        MATCH (source:Entity)
                        WHERE source.stable_id = $source_stable_id
                           OR source.canonical_name = $source_stable_id
                           OR source.name = $source_name
                        MATCH (target:Entity)
                        WHERE target.stable_id = $target_stable_id
                           OR target.canonical_name = $target_stable_id
                           OR target.name = $target_name
                        MATCH (source)-[r:RELATION]->(target)
                        SET r.stable_id = $stable_id,
                            r.source_stable_id = $source_stable_id,
                            r.target_stable_id = $target_stable_id,
                            r.type = $relation_type,
                            r.relation_type = $relation_type,
                            r.ontology_relation_id = $ontology_relation_id,
                            r.ontology_label = $ontology_label,
                            r.ontology_status = $ontology_status,
                            r.ontology_confidence = $ontology_confidence,
                            r.ontology = $ontology,
                            r.type_id = $type_id,
                            r.schema_version = $schema_version,
                            r.status = $status,
                            r.unknown_candidate = $unknown_candidate,
                            r.evidence_span = $evidence_span,
                            r.source_span = $source_span,
                            r.target_span = $target_span,
                            r.provenance = $provenance,
                            r.updated_at = $timestamp
                        """,
                        source_stable_id=metadata["source_stable_id"],
                        target_stable_id=metadata["target_stable_id"],
                        source_name=source.get("name"),
                        target_name=target.get("name"),
                        stable_id=metadata["stable_id"],
                        relation_type=relation.get("relation_type") or "related_to",
                        ontology_relation_id=metadata["ontology_relation_id"],
                        ontology_label=metadata["ontology_label"],
                        ontology_status=metadata["ontology_status"],
                        ontology_confidence=metadata["ontology_confidence"],
                        ontology=metadata["ontology"],
                        type_id=relation.get("type_id") or metadata["ontology_relation_id"],
                        schema_version=relation.get("schema_version") or "1.0.0",
                        status=relation.get("status") or metadata["ontology_status"] or "active",
                        unknown_candidate=metadata.get("unknown_candidate"),
                        evidence_span=relation.get("evidence_span"),
                        source_span=relation.get("source_span"),
                        target_span=relation.get("target_span"),
                        provenance=relation.get("provenance") or {
                            "source_document": relation.get("source_document"),
                            "source_method": relation.get("source_method") or relation.get("source") or "migration",
                            "evidence": relation.get("evidence") or relation.get("context") or "",
                        },
                        timestamp=datetime.now().isoformat(),
                    )
                    relation_updates += 1

            return {
                "status": "completed",
                "dry_run": dry_run,
                "include_all_nodes": include_all_nodes,
                "create_typed_labels": create_typed_labels,
                "entity_updates": entity_updates,
                "relation_updates": relation_updates,
                "proposed_entities": proposed_entities,
                "proposed_relations": proposed_relations,
                "samples": {
                    "entities": sample_entities[:20],
                    "relations": sample_relations[:20],
                },
            }
        except Exception as exc:
            return {
                "status": "failed",
                "dry_run": dry_run,
                "include_all_nodes": include_all_nodes,
                "create_typed_labels": create_typed_labels,
                "entity_updates": entity_updates,
                "relation_updates": relation_updates,
                "error": str(exc),
            }

    def close(self):
        if self.driver:
            self.driver.close()


# Legacy function interface
_store = None

def persist_to_neo4j(entities: List[Dict[str, Any]], relations: List[Dict[str, Any]], job_id: str) -> str:
    global _store
    
    if _store is None:
        _store = Neo4jGraphStore()
        _store.create_indices()
    
    if not _store.driver:
        return "neo4j:skipped (Connection unavailable)"
    
    try:
        _store.persist_entities(entities, job_id)
        _store.persist_relations(relations, job_id)
        return "neo4j:stored"
    except Exception as exc:
        return f"neo4j:skipped ({exc})"
