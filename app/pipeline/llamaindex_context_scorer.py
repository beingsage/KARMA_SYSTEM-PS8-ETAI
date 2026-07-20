"""Utilities to score evidence sources between legacy chunks and LlamaIndex.

Goal:
- Do not break the existing pipeline.
- Provide a cheap heuristic to decide which retrieved contexts are likely more grounded.

This is used for:
- Stage 13/14 hybrid: deciding which evidence source should be preferred
  in downstream synthesis.

Notes:
- We keep it purely heuristic (no LLM) to reduce variability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvidenceSelection:
    mode: str  # "legacy" | "llamaindex" | "hybrid"
    legacy_coverage: float
    llamaindex_coverage: float
    chosen_coverage: float


def _entity_surface_forms(entities: Optional[List[Dict[str, Any]]]) -> List[str]:
    if not entities:
        return []
    names: List[str] = []
    for e in entities:
        name = (e.get("name") or "").strip()
        if name:
            names.append(name)
    return names[:50]


def _coverage_score_from_text(text: str, surface_forms: List[str]) -> float:
    if not text or not surface_forms:
        return 0.0
    tl = text.lower()
    hits = 0
    denom = max(1, min(len(surface_forms), 25))
    for n in surface_forms[:25]:
        if not n:
            continue
        nn = re.sub(r"\s+", " ", n.strip().lower())
        nn2 = nn.replace("-", " ")
        if nn in tl or nn2 in tl:
            hits += 1
    return hits / denom


def choose_evidence_mode(
    *,
    legacy_text_chunks: List[str],
    llamaindex_evidence_context_text: str,
    entities: Optional[List[Dict[str, Any]]],
    legacy_top_n_chunks: int = 4,
) -> EvidenceSelection:
    """Heuristic selection.

    We compute coverage of entity surface forms inside:
    - legacy: concatenation of top-N chunks (by position; this caller can prefilter)
    - llamaindex: concatenated retrieved contexts (caller-provided)

    Since we don't have retrieval scores for legacy in this repo, we treat legacy
    as the first N chunks.
    """

    surface_forms = _entity_surface_forms(entities)
    legacy_text = "\n---\n".join(legacy_text_chunks[: max(1, legacy_top_n_chunks)])

    legacy_cov = _coverage_score_from_text(legacy_text, surface_forms)
    li_cov = _coverage_score_from_text(llamaindex_evidence_context_text, surface_forms)

    if li_cov <= 0.01 and legacy_cov <= 0.01:
        return EvidenceSelection(
            mode="hybrid",
            legacy_coverage=legacy_cov,
            llamaindex_coverage=li_cov,
            chosen_coverage=max(legacy_cov, li_cov),
        )

    # If one clearly dominates, pick it.
    if li_cov > legacy_cov * 1.15:
        mode = "llamaindex"
        chosen = li_cov
    elif legacy_cov > li_cov * 1.15:
        mode = "legacy"
        chosen = legacy_cov
    else:
        mode = "hybrid"
        chosen = max(legacy_cov, li_cov)

    return EvidenceSelection(
        mode=mode,
        legacy_coverage=legacy_cov,
        llamaindex_coverage=li_cov,
        chosen_coverage=chosen,
    )

