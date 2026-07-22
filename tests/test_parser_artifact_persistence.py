from __future__ import annotations

from typing import Any

from app.pipeline.neo4j_store import Neo4jGraphStore


class FakeSession:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def run(self, query: str, **params: Any) -> None:
        self.calls.append({"query": query, "params": params})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeDriver:
    def __init__(self):
        self.session_obj = FakeSession()

    def session(self):
        return self.session_obj


def test_persist_source_artifacts_creates_source_artifact_nodes_and_job_links():
    store = object.__new__(Neo4jGraphStore)
    store.driver = FakeDriver()

    artifacts = [
        {
            "artifact_id": "art-001",
            "source_document": "manual.pdf",
            "kind": "page_image",
            "page": 1,
            "mime_type": "image/png",
            "caption": "Generator page 1",
            "content": "base64:image",
        }
    ]

    assert store.persist_source_artifacts(artifacts, "job-123") is True
    session = store.driver.session_obj
    assert any("MERGE (a:SourceArtifact" in call["query"] for call in session.calls)
    assert any("MATCH (j:Job" in call["query"] for call in session.calls)
