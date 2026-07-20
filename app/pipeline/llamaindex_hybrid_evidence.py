"""Thin wrapper around LlamaIndexHybrid to retrieve evidence contexts.

We keep this separated from llamaindex_hybrid.py to avoid modifying its behavior.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.pipeline.llamaindex_context_scorer import EvidenceSelection, choose_evidence_mode
from app.pipeline.llamaindex_hybrid import LlamaIndexHybrid


def build_evidence_from_entities(
    *,
    llamaindex_hybrid: LlamaIndexHybrid,
    entities: Optional[List[Dict[str, Any]]],
    relations: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    if not llamaindex_hybrid or not getattr(llamaindex_hybrid, "available", False):
        return []

    surface_queries: List[str] = []
    if entities:
        for e in entities[:12]:
            name = (e.get("name") or "").strip()
            if name:
                surface_queries.append(name)

    # Optional relation queries (can be empty if relations are empty)
    if relations:
        for r in relations[:8]:
            s = (r.get("source") or "").strip()
            t = (r.get("target") or "").strip()
            if s and t:
                surface_queries.append(f"{s} {r.get('relation_type','related_to')} {t}")

    surface_queries = surface_queries[:10] or []
    if not surface_queries:
        return []

    evidence = llamaindex_hybrid.retrieve(queries=surface_queries, entities=entities or [])
    combined = evidence.combined_text
    if not combined:
        return []
    # split back into chunks for scoring convenience
    return [combined]


def select_evidence_mode_for_synthesis(
    *,
    llamaindex_hybrid: Optional[LlamaIndexHybrid],
    legacy_text_chunks: List[str],
    entities: Optional[List[Dict[str, Any]]],
    relations: Optional[List[Dict[str, Any]]],
) -> EvidenceSelection:
    if not llamaindex_hybrid or not getattr(llamaindex_hybrid, "available", False):
        # No LlamaIndex: pick legacy by definition
        return EvidenceSelection(
            mode="legacy",
            legacy_coverage=0.0,
            llamaindex_coverage=0.0,
            chosen_coverage=0.0,
        )

    evidence_contexts = build_evidence_from_entities(
        llamaindex_hybrid=llamaindex_hybrid,
        entities=entities,
        relations=relations,
    )
    li_text = evidence_contexts[0] if evidence_contexts else ""

    return choose_evidence_mode(
        legacy_text_chunks=legacy_text_chunks,
        llamaindex_evidence_context_text=li_text,
        entities=entities,
    )

