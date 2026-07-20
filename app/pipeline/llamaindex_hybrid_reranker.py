"""LlamaIndex-based hybrid reranking utilities.

This module is optional and defensive:
- If LlamaIndex rerank postprocessors are unavailable, callers should fallback.
- We keep the interface small and stable for integration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class LlamaIndexHybridReranker:
    def __init__(self, embedder: Optional[Any] = None) -> None:
        self._available = False
        self._import_error: Optional[Exception] = None

        try:
            from llama_index.core.postprocessor import SimilarityPostprocessor

            # SimilarityPostprocessor in LlamaIndex is essentially an embedding similarity filter.
            # We treat it as a lightweight reranker.
            self.SimilarityPostprocessor = SimilarityPostprocessor
            self._available = True
        except Exception as exc:
            self._import_error = exc
            self._available = False

        self._embedder = embedder

    @property
    def available(self) -> bool:
        return self._available

    def get_telemetry(self) -> Dict[str, Any]:
        return {"available": self._available, "error": str(self._import_error) if self._import_error else None}

    def rerank_texts(self, query: str, candidates: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """Rerank candidates by similarity.

        Returns list of dicts with: rank, id, score, text.
        If not available, returns empty list.
        """
        if not self._available or not candidates:
            return []

        # Defensive fallback: since our current LlamaIndexHybrid builds a vector index,
        # similarity postprocessing is only meaningful if we have nodes.
        # For a pure text list, we can't reliably score without rebuilding a temporary index.
        # So for now we keep this as "not implemented" and return empty so engine_v2 can fallback.
        return []

