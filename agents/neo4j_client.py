from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.config import settings

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import Neo4jError
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = None  # type: ignore
    Neo4jError = Exception  # type: ignore

logger = logging.getLogger(__name__)


def _serialize_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list):
        return [_serialize_value(item) for item in value]

    if isinstance(value, dict):
        return {key: _serialize_value(value[key]) for key in value}

    try:
        if hasattr(value, "items"):
            return {key: _serialize_value(item) for key, item in dict(value).items()}
    except Exception:
        pass

    if hasattr(value, "labels") and hasattr(value, "id"):
        node = dict(value)
        node["id"] = getattr(value, "id", None)
        node["labels"] = list(value.labels)
        return {k: _serialize_value(v) for k, v in node.items()}

    if hasattr(value, "type") and hasattr(value, "start_node"):
        relation = dict(value)
        relation["id"] = getattr(value, "id", None)
        relation["type"] = getattr(value, "type", None)
        relation["start_node_id"] = getattr(value, "start_node", None).id if getattr(value, "start_node", None) else None
        relation["end_node_id"] = getattr(value, "end_node", None).id if getattr(value, "end_node", None) else None
        return {k: _serialize_value(v) for k, v in relation.items()}

    try:
        return str(value)
    except Exception:
        return None


class Neo4jClient:
    """Lightweight Neo4j client for agent retrieval."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.driver = None
        self.connected = False
        self._connect()

    def _connect(self) -> bool:
        if GraphDatabase is None:
            logger.warning("Neo4j driver not installed; Neo4j client unavailable")
            return False

        if not self.uri:
            logger.warning("Neo4j URI is not configured; Neo4j client unavailable")
            return False

        try:
            auth = (self.user, self.password) if self.user and self.password else None
            self.driver = GraphDatabase.driver(self.uri, auth=auth, encrypted=False)
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.connected = True
            logger.info(f"✓ Connected to Neo4j at {self.uri}")
            return True
        except Exception as exc:
            logger.warning(f"Neo4j connection failed: {exc}")
            self.connected = False
            self.driver = None
            return False

    def close(self) -> None:
        if self.driver:
            try:
                self.driver.close()
            except Exception:
                pass
            self.driver = None
            self.connected = False

    def run_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.connected or not self.driver:
            logger.warning("Neo4j client is not connected; returning empty result")
            return []

        try:
            with self.driver.session() as session:
                result = session.run(query, params or {})
                return [_serialize_value(record.data()) for record in result]
        except Neo4jError as exc:
            logger.error(f"Neo4j query failed: {exc}")
            return []
        except Exception as exc:
            logger.error(f"Unexpected Neo4j error: {exc}")
            return []

    def search_nodes(self, labels: List[str], keyword: str = "", limit: int = 20) -> List[Dict[str, Any]]:
        text_filter = ""
        params = {
            "labels": labels,
            "keyword": keyword.lower().strip(),
            "limit": limit,
        }

        if params["keyword"]:
            text_filter = (
                "AND ("
                "toLower(coalesce(n.name, '')) CONTAINS $keyword OR "
                "toLower(coalesce(n.description, '')) CONTAINS $keyword OR "
                "toLower(coalesce(n.notes, '')) CONTAINS $keyword OR "
                "toLower(coalesce(n.type, '')) CONTAINS $keyword OR "
                "toLower(coalesce(n.category, '')) CONTAINS $keyword)"
            )

        query = (
            "MATCH (n) "
            "WHERE (size($labels) = 0 OR any(lbl IN labels(n) WHERE lbl IN $labels)) "
            f"{text_filter} "
            "RETURN n, labels(n) AS labels LIMIT $limit"
        )
        return self.run_query(query, params)

    def search_relationships(
        self,
        source_labels: List[str],
        target_labels: List[str],
        keyword: str = "",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        text_filter = ""
        params = {
            "source_labels": source_labels,
            "target_labels": target_labels,
            "keyword": keyword.lower().strip(),
            "limit": limit,
        }

        if params["keyword"]:
            text_filter = (
                "AND ("
                "toLower(coalesce(a.name, '')) CONTAINS $keyword OR "
                "toLower(coalesce(b.name, '')) CONTAINS $keyword OR "
                "toLower(coalesce(r.type, '')) CONTAINS $keyword OR "
                "toLower(coalesce(a.description, '')) CONTAINS $keyword OR "
                "toLower(coalesce(b.description, '')) CONTAINS $keyword)"
            )

        query = (
            "MATCH (a)-[r]->(b) "
            "WHERE (size($source_labels) = 0 OR any(lbl IN labels(a) WHERE lbl IN $source_labels)) "
            "AND (size($target_labels) = 0 OR any(lbl IN labels(b) WHERE lbl IN $target_labels)) "
            f"{text_filter} "
            "RETURN a, r, b LIMIT $limit"
        )
        return self.run_query(query, params)


__all__ = ["Neo4jClient"]
