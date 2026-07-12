"""
Neo4j Knowledge Graph Persistence
Stores industrial entities, relations, and reasoning in Neo4j
"""

import hashlib
from datetime import datetime
from typing import Any, Dict, List
import time
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError
from app.config import settings


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
                session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (r:Relation) ON (r.type)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (j:Job) ON (j.job_id)")
            print("✓ Neo4j indices created")
            return True
        except Exception as e:
            print(f"⚠ Failed to create indices: {e}")
            return False

    @staticmethod
    def _entity_hash(name: str, entity_type: str) -> str:
        identifier = f"{name.strip().lower()}|{entity_type.strip().lower()}"
        return hashlib.md5(identifier.encode("utf-8")).hexdigest()

    def persist_entities(
        self,
        entities: List[Dict[str, Any]],
        job_id: str,
    ) -> bool:
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

                for entity in valid_entities:
                    name = entity.get("name", "").strip()
                    entity_type = entity.get("entity_type", "").strip()
                    entity_id = self._entity_hash(name, entity_type)
                    session.run(
                        """
                        MERGE (e:Entity {id: $entity_id})
                        SET e.name = $name,
                            e.type = $entity_type,
                            e.confidence = $confidence,
                            e.canonical_name = $canonical_name,
                            e.updated_at = $timestamp
                        WITH e
                        MATCH (j:Job {job_id: $job_id})
                        MERGE (j)-[r:EXTRACTED_ENTITY]->(e)
                        SET r.timestamp = $timestamp
                        """,
                        entity_id=entity_id,
                        name=name,
                        entity_type=entity_type,
                        confidence=entity.get("confidence", 0.5),
                        canonical_name=entity.get("canonical_name", name),
                        job_id=job_id,
                        timestamp=timestamp,
                    )

                print(f"✓ Persisted {len(valid_entities)} entities to Neo4j")
                return True
        except Exception as e:
            print(f"⚠ Failed to persist entities: {e}")
            return False
    
    def persist_relations(
        self,
        relations: List[Dict[str, Any]],
        job_id: str
    ) -> bool:
        if not self.driver:
            return False

        valid_relations = [
            relation
            for relation in relations
            if relation.get("source") and relation.get("target")
        ]
        if not valid_relations:
            print("⚠ No valid relations to persist")
            return False

        timestamp = datetime.now().isoformat()
        try:
            with self.driver.session() as session:
                for relation in valid_relations:
                    source = relation.get("source", "").strip()
                    target = relation.get("target", "").strip()
                    relation_type = relation.get("relation_type", "").strip()
                    if not source or not target:
                        continue

                    session.run(
                        """
                        MATCH (source:Entity {name: $source})
                        MATCH (target:Entity {name: $target})
                        MERGE (source)-[r:RELATION {type: $relation_type}]->(target)
                        SET r.confidence = $confidence,
                            r.job_id = $job_id,
                            r.timestamp = $timestamp
                        """,
                        source=source,
                        target=target,
                        relation_type=relation_type,
                        confidence=relation.get("confidence", 0.5),
                        job_id=job_id,
                        timestamp=timestamp,
                    )

                print(f"✓ Persisted {len(valid_relations)} relations to Neo4j")
                return True
        except Exception as e:
            print(f"⚠ Failed to persist relations: {e}")
            return False
    
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
