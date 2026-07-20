"""Prompt-based zero-shot extraction helpers.

This module keeps the prompt logic separate from the GLiNER/GLiREL stack so the
pipeline can use a true prompt-first path when an instruction-tuned LLM is
available. The extractor returns structured JSON and preserves unknown
candidates explicitly so ontology evolution can promote them later.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from app.config import settings
from app.pipeline.advanced_models import Qwen3LLM
from app.pipeline.document_utils import chunk_text, normalize_text
from app.pipeline.models import canonicalize_entity_name
from app.pipeline.ontology import OntologyRegistry, load_ontology_registry


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for item in items:
        normalized = str(item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _strip_code_fences(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_json_blob(text: str) -> str:
    stripped = _strip_code_fences(text)
    if not stripped:
        return ""

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped

    start_candidates = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0]
    if not start_candidates:
        return stripped

    start = min(start_candidates)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if end > start:
        return stripped[start : end + 1]
    return stripped[start:]


def _coerce_span(span: Any, *, text: str, fallback_quote: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {"start": None, "end": None, "quote": ""}
    if isinstance(span, dict):
        result["start"] = span.get("start", span.get("start_char"))
        result["end"] = span.get("end", span.get("end_char"))
        result["quote"] = str(span.get("quote") or span.get("text") or fallback_quote or "").strip()
    elif isinstance(span, (list, tuple)) and len(span) >= 2:
        result["start"] = span[0]
        result["end"] = span[1]
    elif isinstance(span, str):
        result["quote"] = span.strip()

    quote = str(result.get("quote") or fallback_quote or "").strip()
    start = result.get("start")
    end = result.get("end")
    try:
        if start is not None:
            start = int(start)
    except Exception:
        start = None
    try:
        if end is not None:
            end = int(end)
    except Exception:
        end = None

    if quote and (start is None or end is None):
        idx = text.lower().find(quote.lower())
        if idx >= 0:
            start = idx
            end = idx + len(quote)

    if (start is None or end is None) and fallback_quote:
        idx = text.lower().find(fallback_quote.lower())
        if idx >= 0:
            start = idx
            end = idx + len(fallback_quote)

    result["start"] = start
    result["end"] = end
    result["quote"] = quote
    return result


def _normalize_unknown_candidate(
    candidate: Any,
    *,
    name: str,
    detected_type: str,
    evidence_quote: str,
) -> Optional[Dict[str, Any]]:
    if candidate is None and detected_type not in {"unknown", "candidate", "unspecified", ""}:
        return None

    payload: Dict[str, Any] = {}
    if isinstance(candidate, dict):
        payload.update(candidate)
    elif isinstance(candidate, str):
        payload["candidate_label"] = candidate
    elif candidate is not None:
        payload["candidate_label"] = str(candidate)

    payload.setdefault("candidate_label", name or detected_type or "unknown")
    payload.setdefault("candidate_type", detected_type or "unknown")
    payload.setdefault("parent_type_id", payload.get("parent_type_id") or "entity")
    payload.setdefault("reason", payload.get("reason") or "uncertain mapping from prompt output")
    aliases = payload.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    payload["aliases"] = _dedupe_preserve_order([*aliases, name, evidence_quote])
    payload["confidence"] = _safe_float(payload.get("confidence"), 0.0)
    return payload


class PromptZeroShotExtractor:
    """Prompt-based JSON extractor for entities and relations."""

    def __init__(
        self,
        registry: Optional[OntologyRegistry] = None,
        llm: Optional[Qwen3LLM] = None,
    ) -> None:
        self.registry = registry or load_ontology_registry()
        self.llm = llm or Qwen3LLM(load_model=False)

    @staticmethod
    def _looks_unavailable(response: str) -> bool:
        response = (response or "").strip()
        return not response or response.startswith("[LLM Response to:") or response.startswith("Error:")

    def _prompt_context(self, query: str) -> Dict[str, Any]:
        if hasattr(self.registry, "describe_for_prompt"):
            return self.registry.describe_for_prompt(query)
        return {
            "schema_version": getattr(self.registry, "schema_version", "1.0.0"),
            "registry_version": getattr(self.registry, "registry_version", ""),
            "entity_types": [],
            "relation_types": [],
            "markdown_packs": [],
        }

    def _build_prompt(
        self,
        *,
        text: str,
        mode: str,
        entities: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> str:
        context = self._prompt_context(text)
        entity_context = [
            {
                "name": str(entity.get("name") or entity.get("canonical_name") or "").strip(),
                "entity_type": str(entity.get("entity_type") or "").strip(),
                "canonical_name": str(entity.get("canonical_name") or "").strip(),
                "confidence": _safe_float(entity.get("confidence"), 0.0),
            }
            for entity in (entities or [])
            if str(entity.get("name") or entity.get("canonical_name") or "").strip()
        ]

        instructions = (
            "Return JSON only. Do not add markdown fences, commentary, or code blocks.\n"
            "If a span or type is uncertain, keep the primary field conservative and set unknown_candidate.\n"
            "Never invent unsupported ontology labels. Prefer 'unknown' over a wrong label.\n"
            "Use the ontology context to map spans to the closest active types when possible.\n"
        )

        schema = {
            "entities": [
                {
                    "name": "exact mention text",
                    "entity_type": "ontology type id or unknown",
                    "canonical_name": "stable canonical name",
                    "confidence": 0.0,
                    "evidence_span": {"start": 0, "end": 0, "quote": "short evidence quote"},
                    "unknown_candidate": {
                        "candidate_label": "candidate label when uncertain",
                        "candidate_type": "unknown or provisional type",
                        "parent_type_id": "entity or parent ontology type",
                        "reason": "why the type is uncertain",
                        "aliases": ["alias one", "alias two"],
                    },
                }
            ],
            "relations": [
                {
                    "source": "exact source mention",
                    "target": "exact target mention",
                    "relation_type": "ontology relation id or related_to",
                    "confidence": 0.0,
                    "evidence_span": {"start": 0, "end": 0, "quote": "short evidence quote"},
                    "unknown_candidate": {
                        "candidate_label": "candidate relation label",
                        "candidate_type": "unknown or provisional relation",
                        "parent_type_id": "related_to",
                        "reason": "why the relation is uncertain",
                    },
                }
            ],
        }

        return (
            f"{instructions}\n"
            "Task: extract industrial entities and relations from the document text.\n"
            "Mode: "
            f"{mode}\n"
            "Document text:\n"
            f"{normalize_text(text)[:6000]}\n\n"
            "Known entity candidates from earlier stages (if any):\n"
            f"{json.dumps(entity_context, indent=2, sort_keys=True)}\n\n"
            "Ontology context:\n"
            f"{json.dumps(context, indent=2, sort_keys=True)}\n\n"
            "JSON schema example:\n"
            f"{json.dumps(schema, indent=2, sort_keys=True)}\n\n"
            "Return a single JSON object with keys 'entities' and 'relations'."
        )

    def _parse_response(self, response: str) -> Dict[str, Any]:
        blob = _extract_json_blob(response)
        if not blob:
            return {}
        try:
            parsed = json.loads(blob)
            return parsed if isinstance(parsed, dict) else {"entities": [], "relations": parsed}
        except Exception:
            # Some models emit single quotes or trailing commas. Try a conservative cleanup.
            cleaned = blob.strip()
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            try:
                parsed = json.loads(cleaned)
                return parsed if isinstance(parsed, dict) else {"entities": [], "relations": parsed}
            except Exception:
                return {}

    def _normalize_entity(self, item: Dict[str, Any], *, chunk_text: str) -> Optional[Dict[str, Any]]:
        name = str(item.get("name") or item.get("text") or item.get("mention") or "").strip()
        if not name:
            return None
        entity_type = str(item.get("entity_type") or item.get("type") or item.get("label") or "unknown").strip() or "unknown"
        canonical_name = str(item.get("canonical_name") or canonicalize_entity_name(name)).strip()
        evidence_span = _coerce_span(item.get("evidence_span") or item.get("span"), text=chunk_text, fallback_quote=name)
        evidence_quote = str(evidence_span.get("quote") or item.get("evidence_text") or "").strip()
        unknown_candidate = _normalize_unknown_candidate(
            item.get("unknown_candidate"),
            name=name,
            detected_type=entity_type,
            evidence_quote=evidence_quote,
        )
        return {
            "name": name,
            "entity_type": entity_type,
            "canonical_name": canonical_name,
            "confidence": round(_safe_float(item.get("confidence"), 0.0), 3),
            "start": evidence_span.get("start"),
            "end": evidence_span.get("end"),
            "evidence_span": evidence_span,
            "unknown_candidate": unknown_candidate,
            "source": "prompt_zero_shot",
            "source_method": "prompt_zero_shot",
        }

    def _normalize_relation(
        self,
        item: Dict[str, Any],
        *,
        chunk_text: str,
    ) -> Optional[Dict[str, Any]]:
        source = str(item.get("source") or item.get("source_text") or item.get("head") or "").strip()
        target = str(item.get("target") or item.get("target_text") or item.get("tail") or "").strip()
        if not source or not target:
            return None
        relation_type = str(item.get("relation_type") or item.get("type") or item.get("label") or "related_to").strip() or "related_to"
        evidence_span = _coerce_span(item.get("evidence_span") or item.get("span"), text=chunk_text, fallback_quote=f"{source} {relation_type} {target}")
        evidence_quote = str(evidence_span.get("quote") or item.get("evidence_text") or "").strip()
        unknown_candidate = _normalize_unknown_candidate(
            item.get("unknown_candidate"),
            name=relation_type,
            detected_type=relation_type if relation_type != "related_to" else "unknown",
            evidence_quote=evidence_quote,
        )
        source_id = canonicalize_entity_name(source)
        target_id = canonicalize_entity_name(target)
        return {
            "source": source,
            "target": target,
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "confidence": round(_safe_float(item.get("confidence"), 0.0), 3),
            "source_span": item.get("source_span"),
            "target_span": item.get("target_span"),
            "evidence_span": evidence_span,
            "unknown_candidate": unknown_candidate,
            "source_method": "prompt_zero_shot",
        }

    def _extract_chunk(self, *, chunk: str, mode: str, entities: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
        prompt = self._build_prompt(text=chunk, mode=mode, entities=entities)
        response = self.llm.generate(prompt, max_tokens=min(settings.qwen3_max_tokens, 1200), temperature=0.0)
        if self._looks_unavailable(response):
            return {"entities": [], "relations": [], "status": "unavailable", "raw_response": response}

        parsed = self._parse_response(response)
        if not parsed:
            return {"entities": [], "relations": [], "status": "unparseable", "raw_response": response}

        entities_out = [
            normalized
            for item in parsed.get("entities", [])
            if isinstance(item, dict)
            for normalized in [self._normalize_entity(item, chunk_text=chunk)]
            if normalized is not None
        ]
        relations_out = [
            normalized
            for item in parsed.get("relations", [])
            if isinstance(item, dict)
            for normalized in [self._normalize_relation(item, chunk_text=chunk)]
            if normalized is not None
        ]
        return {
            "entities": entities_out,
            "relations": relations_out,
            "status": "ok",
            "raw_response": response,
        }

    def extract_structured(
        self,
        text: str,
        *,
        entities: Optional[Sequence[Dict[str, Any]]] = None,
        max_chunks: int = 6,
    ) -> Dict[str, Any]:
        normalized_text = normalize_text(text)
        if not normalized_text:
            return {"status": "empty", "entities": [], "relations": [], "chunks": []}

        chunks = chunk_text(normalized_text, max_chars=2400, overlap=220)
        if max_chunks > 0:
            chunks = chunks[:max_chunks]

        merged_entities: Dict[str, Dict[str, Any]] = {}
        merged_relations: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        chunk_results: List[Dict[str, Any]] = []

        for chunk in chunks:
            result = self._extract_chunk(chunk=chunk, mode="entity_and_relation", entities=entities)
            chunk_results.append(result)
            for entity in result.get("entities", []):
                canonical = str(entity.get("canonical_name") or canonicalize_entity_name(entity.get("name", ""))).strip()
                if not canonical:
                    continue
                existing = merged_entities.get(canonical)
                if existing is None:
                    merged_entities[canonical] = dict(entity)
                    continue
                existing_confidence = _safe_float(existing.get("confidence"), 0.0)
                candidate_confidence = _safe_float(entity.get("confidence"), 0.0)
                merged = dict(existing if existing_confidence >= candidate_confidence else entity)
                merged.setdefault("unknown_candidate", existing.get("unknown_candidate") or entity.get("unknown_candidate"))
                merged.setdefault("evidence_span", existing.get("evidence_span") or entity.get("evidence_span"))
                merged_entities[canonical] = merged

            for relation in result.get("relations", []):
                key = (
                    str(relation.get("source_id") or canonicalize_entity_name(relation.get("source", ""))).strip(),
                    str(relation.get("target_id") or canonicalize_entity_name(relation.get("target", ""))).strip(),
                    str(relation.get("relation_type") or "related_to").strip() or "related_to",
                )
                existing = merged_relations.get(key)
                if existing is None:
                    merged_relations[key] = dict(relation)
                    continue
                existing_confidence = _safe_float(existing.get("confidence"), 0.0)
                candidate_confidence = _safe_float(relation.get("confidence"), 0.0)
                merged = dict(existing if existing_confidence >= candidate_confidence else relation)
                merged.setdefault("unknown_candidate", existing.get("unknown_candidate") or relation.get("unknown_candidate"))
                merged.setdefault("evidence_span", existing.get("evidence_span") or relation.get("evidence_span"))
                merged_relations[key] = merged

        return {
            "status": "completed" if (merged_entities or merged_relations) else "empty",
            "entities": list(merged_entities.values()),
            "relations": list(merged_relations.values()),
            "chunks": chunk_results,
            "chunk_count": len(chunks),
            "model": getattr(self.llm, "model_name", "unknown"),
            "backend": "prompt_zero_shot",
        }

    def extract_entities(self, text: str, *, max_chunks: int = 6) -> List[Dict[str, Any]]:
        result = self.extract_structured(text, max_chunks=max_chunks)
        return list(result.get("entities", []))

    def extract_relations(
        self,
        text: str,
        entities: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        max_chunks: int = 6,
    ) -> List[Dict[str, Any]]:
        result = self.extract_structured(text, entities=entities, max_chunks=max_chunks)
        return list(result.get("relations", []))
