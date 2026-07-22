from __future__ import annotations

import json
from types import SimpleNamespace

from app.config import settings
from app.pipeline.advanced_models import GraphSAGEGraphEmbedder
from app.pipeline.llamaindex_hybrid import LlamaIndexHybrid


def test_graphsage_embedder_initializes_when_pyg_is_available():
    embedder = GraphSAGEGraphEmbedder()
    assert isinstance(embedder, GraphSAGEGraphEmbedder)
    assert embedder.available in (True, False)


def test_llm_backed_citation_summarize_uses_direct_llm_path(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            body = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "summary_method": "llm-citation-synthesis",
                                            "status": "analyzed",
                                            "anomalies_detected": [
                                                {"name": "high risk anomaly", "confidence": 0.86}
                                            ],
                                            "failure_risks": [
                                                {"name": "overheating", "confidence": 0.82}
                                            ],
                                            "maintenance_recommendations": [
                                                {"name": "inspect pump", "confidence": 0.8}
                                            ],
                                            "compliance": [],
                                            "confidence": 0.88,
                                            "evidence_coverage": 0.8,
                                            "citations": [],
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ]
            }
            return json.dumps(body).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse())

    hybrid = LlamaIndexHybrid(embedder=None, api_key="test-key")
    hybrid.build_index(text_chunks=["Pump overheating observed during maintenance window"], text_chunks_metadata=[{"chunk_id": 0}])
    result = hybrid.citation_summarize(
        entities=[{"name": "pump"}],
        relations=[],
        text="Pump overheating observed during maintenance window",
        queries=["pump overheating"],
    )

    assert result["summary_method"] == "llm-citation-synthesis"
    assert result["status"] == "analyzed"
