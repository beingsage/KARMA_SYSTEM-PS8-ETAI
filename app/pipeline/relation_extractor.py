"""Industrial relation extraction using GLiREL and robust heuristics."""

import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.pipeline.compat import allow_trusted_torch_pickle
from app.pipeline.models import canonicalize_entity_name, normalize_relation_payload
from app.pipeline.prompt_zero_shot import PromptZeroShotExtractor
from app.pipeline.runtime import select_device


class GLiRELRelationExtractor:
    """Extract relations using GLiREL when available, with enhanced heuristic fallback."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model = None
        self.model_name = model_name or "jackboyla/glirel-large-v0"
        self.device = select_device()
        self.is_ready = False
        self.backend = "heuristic"
        self._debug_enabled = os.getenv("RELATION_EXTRACTION_DEBUG", "").lower() in {"1", "true", "yes", "on"}
        self.prompt_zero_shot = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            with allow_trusted_torch_pickle():
                from glirel import GLiREL

                self.model = GLiREL.from_pretrained(self.model_name, map_location=self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            self.is_ready = self.model is not None
            self.backend = "glirel"
            print(f"✓ GLiREL relation extraction model loaded on {self.device}: {self.model_name}")
        except Exception as exc:
            msg = str(exc)
            print(f"✗ GLiREL initialization failed: {msg}")
            if "PyExtensionType" in msg or "pyarrow" in msg:
                print("  ⚠ Detected pyarrow API mismatch. The runtime now shims `PyExtensionType` when possible; otherwise install a pyarrow build that still exposes that alias.")
            if "weights_only" in msg or "pickle" in msg or "torch.load" in msg:
                print("  ⚠ Detected torch loading/pickle issue. Ensure PyTorch >= 2.6 or use safetensors checkpoints; consider installing `safetensors` and using .safetensors model files.")
            print("  ⚠ Falling back to heuristic relation extraction.")
            self.model = None
            self.is_ready = True
            self.backend = "heuristic"

        try:
            self.prompt_zero_shot = PromptZeroShotExtractor()
        except Exception as exc:
            print(f"⚠ Prompt zero-shot relation extractor initialization failed: {exc}")
            self.prompt_zero_shot = None

    def extract(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        if not text:
            return []
        has_prompt_fallback = settings.enable_prompt_zero_shot_extraction and self.prompt_zero_shot is not None
        if len(entities) < 2 and not has_prompt_fallback:
            return []

        normalized_entities = self._normalize_entities(entities)
        relation_candidates: List[Dict[str, Any]] = []

        if self.model is not None:
            relation_candidates.extend(self._model_extract(text, normalized_entities, threshold=threshold))

        relation_candidates.extend(self._heuristic_extract(text, normalized_entities, threshold=threshold))
        relation_candidates.extend(self._graph_inference(text, normalized_entities, threshold=threshold))
        if settings.enable_prompt_zero_shot_extraction and self.prompt_zero_shot is not None:
            try:
                prompt_relations = self.prompt_zero_shot.extract_relations(text, normalized_entities)
                relation_candidates.extend(normalize_relation_payload(relation) for relation in prompt_relations)
            except Exception as exc:
                if self._debug_enabled:
                    print(f"[relation_debug] prompt zero-shot relation extraction failed: {exc}")
        merged = self._merge_relations(relation_candidates)
        if self._debug_enabled and merged:
            print(f"[relation_debug] emitted {len(merged)} relations with threshold={threshold}")
        return merged

    def _normalize_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for entity in entities or []:
            name = (entity.get("name") or "").strip()
            if not name:
                continue
            start = entity.get("start")
            end = entity.get("end")
            try:
                start_value = int(start) if start is not None else None
            except (TypeError, ValueError):
                start_value = None
            try:
                end_value = int(end) if end is not None else None
            except (TypeError, ValueError):
                end_value = None
            if start_value is not None and end_value is None:
                end_value = start_value + max(1, len(name))
            elif start_value is None and end_value is not None:
                start_value = max(0, end_value - max(1, len(name)))
            elif start_value is None and end_value is None:
                start_value = 0
                end_value = max(1, len(name))
            normalized.append(
                {
                    "name": name,
                    "canonical_name": entity.get("canonical_name") or entity.get("canonical_id") or canonicalize_entity_name(name),
                    "entity_type": entity.get("entity_type", "unknown"),
                    "start": start_value,
                    "end": end_value,
                    "unknown_candidate": entity.get("unknown_candidate"),
                }
            )
        return normalized

    def _model_extract(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        entity_names = [e["name"] for e in entities if e.get("name")]
        if len(entity_names) < 2:
            return []

        relation_types = [
            "connected_to",
            "controls",
            "measures",
            "receives_input_from",
            "sends_output_to",
            "is_part_of",
            "operates_at",
            "related_to",
        ]

        ner = [
            [
                int(entity.get("start", 0)),
                max(int(entity.get("end", 1)) - 1, int(entity.get("start", 0))),
                entity.get("entity_type", "unknown"),
                entity.get("name", ""),
            ]
            for entity in entities
            if entity.get("name")
        ]

        predictions = self.model.predict_relations(
            text,
            relation_types,
            threshold=threshold,
            ner=ner,
            top_k=1,
        )

        relations: List[Dict[str, Any]] = []
        for relation in predictions or []:
            source = " ".join(relation.get("head_text", [])).strip()
            target = " ".join(relation.get("tail_text", [])).strip()
            if not source or not target:
                continue

            relations.append(
                normalize_relation_payload(
                    {
                        "source": source,
                        "source_id": canonicalize_entity_name(source),
                        "target": target,
                        "target_id": canonicalize_entity_name(target),
                        "relation_type": relation.get("label", "related_to"),
                        "confidence": float(relation.get("score", 0.0)),
                        "source_method": "glirel",
                        "evidence": "",
                    }
                )
            )

        return relations

    @staticmethod
    def _heuristic_extract(text: str, entities: List[Dict[str, Any]], threshold: float = 0.35) -> List[Dict[str, Any]]:
        """Extract relations from industrial text using lexical patterns, sentence context, and weak supervision."""
        relations: List[Dict[str, Any]] = []
        if len(entities) < 2:
            return []

        normalized_entities = []
        for entity in entities[:20]:
            name = (entity.get("name") or "").strip()
            if not name:
                continue
            normalized_entities.append(
                {
                    "name": name,
                    "canonical_name": entity.get("canonical_name") or entity.get("canonical_id") or canonicalize_entity_name(name),
                    "entity_type": entity.get("entity_type", "unknown"),
                    "start": entity.get("start"),
                    "end": entity.get("end"),
                }
            )

        if len(normalized_entities) < 2:
            return []

        segments = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        if not segments:
            segments = [text.strip()]

        for segment in segments:
            sentence_entities = [
                entity
                for entity in normalized_entities
                if GLiRELRelationExtractor._entity_appears_in_text(entity["name"], segment)
            ]
            if len(sentence_entities) < 2:
                continue

            for idx, source in enumerate(sentence_entities):
                for target in sentence_entities[idx + 1 :]:
                    if source["canonical_name"] == target["canonical_name"]:
                        continue

                    relation_type, confidence, evidence = GLiRELRelationExtractor._infer_relation(segment, source, target)
                    if relation_type is None:
                        continue

                    relation = normalize_relation_payload(
                        {
                            "source": source["name"],
                            "source_id": source["canonical_name"],
                            "target": target["name"],
                            "target_id": target["canonical_name"],
                            "relation_type": relation_type,
                            "confidence": max(float(threshold), float(confidence)),
                            "source_method": "heuristic_enhanced",
                            "evidence": evidence,
                            "source_span": [source.get("start"), source.get("end")],
                            "target_span": [target.get("start"), target.get("end")],
                        }
                    )
                    relations.append(relation)

        # Weak supervision: if entities co-occur in the same paragraph/section and are not already linked,
        # add a low-confidence related_to edge to bootstrap relation coverage.
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph_entities = [
                entity for entity in normalized_entities if GLiRELRelationExtractor._entity_appears_in_text(entity["name"], paragraph)
            ]
            if len(paragraph_entities) < 2:
                continue
            for idx, source in enumerate(paragraph_entities):
                for target in paragraph_entities[idx + 1 :]:
                    if source["canonical_name"] == target["canonical_name"]:
                        continue
                    if any(
                        existing["source_id"] == source["canonical_name"] and existing["target_id"] == target["canonical_name"]
                        for existing in relations
                    ):
                        continue
                    relations.append(
                        normalize_relation_payload(
                            {
                                "source": source["name"],
                                "source_id": source["canonical_name"],
                                "target": target["name"],
                                "target_id": target["canonical_name"],
                                "relation_type": "related_to",
                                "confidence": max(float(threshold), 0.45),
                                "source_method": "weak_supervision",
                                "evidence": paragraph[:160],
                                "source_span": [source.get("start"), source.get("end")],
                                "target_span": [target.get("start"), target.get("end")],
                            }
                        )
                    )

        return relations

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2]

    @staticmethod
    def _entity_appears_in_text(entity_name: str, sentence: str) -> bool:
        if not entity_name:
            return False
        pattern = rf"\b{re.escape(entity_name.lower())}\b"
        return re.search(pattern, sentence.lower()) is not None

    @staticmethod
    def _infer_relation(
        sentence: str,
        source: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Tuple[Optional[str], float, str]:
        normalized_sentence = re.sub(r"\s+", " ", sentence).strip()
        lower_sentence = normalized_sentence.lower()

        phrase_rules: List[Tuple[re.Pattern[str], str, float]] = [
            (re.compile(r"\b(connected|link|linked|coupled|attached|mounted)\b.*\b(to|with|onto)\b"), "connected_to", 0.9),
            (re.compile(r"\b(drives|powers|operates|controls|regulates|manages)\b"), "controls", 0.88),
            (re.compile(r"\b(measures|monitors|detects|senses|tracks)\b"), "measures", 0.9),
            (re.compile(r"\b(receives|gets)\b.*\b(from|through)\b"), "receives_input_from", 0.86),
            (re.compile(r"\b(sends|feeds|delivers|outputs?)\b"), "sends_output_to", 0.84),
            (re.compile(r"\b(part of|is part of|included in|installed in|located in|mounted on|placed in|contained in|consists of|contains|includes)\b"), "is_part_of", 0.82),
            (re.compile(r"\b(operates at|operates on|runs at)\b"), "operates_at", 0.8),
        ]

        for pattern, relation_type, base_confidence in phrase_rules:
            if pattern.search(lower_sentence):
                # Reward sensor-to-entity measurement relations.
                if relation_type == "measures" and source.get("entity_type", "").lower() in {"sensor", "measurement"}:
                    base_confidence = min(0.97, base_confidence + 0.05)
                return relation_type, base_confidence, normalized_sentence

        # Fallback: if the sentence is short and contains both entities, mark as related.
        if len(normalized_sentence.split()) <= 25:
            return "related_to", 0.55, normalized_sentence

        return None, 0.0, ""

    @staticmethod
    def _cosine_similarity(left: Counter, right: Counter) -> float:
        if not left or not right:
            return 0.0
        common = set(left) & set(right)
        if not common:
            return 0.0
        dot = sum(left[token] * right[token] for token in common)
        left_norm = sum(value * value for value in left.values()) ** 0.5
        right_norm = sum(value * value for value in right.values()) ** 0.5
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _graph_inference(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """Apply a lightweight graph-style message passing step over entity embeddings."""
        if len(entities) < 2:
            return []

        segments = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", text) if segment.strip()]
        if not segments:
            segments = [text.strip()]

        features: List[Counter] = []
        for entity in entities:
            name_tokens = Counter(GLiRELRelationExtractor._tokenize(entity.get("name", "")))
            type_tokens = Counter(GLiRELRelationExtractor._tokenize(entity.get("entity_type", "")))
            context_tokens = Counter()
            for segment in segments:
                if GLiRELRelationExtractor._entity_appears_in_text(entity.get("name", ""), segment):
                    context_tokens.update(GLiRELRelationExtractor._tokenize(segment))
            feature = name_tokens + type_tokens + context_tokens
            features.append(feature)

        adjacency = [[False for _ in entities] for _ in entities]
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if any(
                    GLiRELRelationExtractor._entity_appears_in_text(entities[i].get("name", ""), segment)
                    and GLiRELRelationExtractor._entity_appears_in_text(entities[j].get("name", ""), segment)
                    for segment in segments
                ):
                    adjacency[i][j] = adjacency[j][i] = True
                    continue
                similarity = self._cosine_similarity(features[i], features[j])
                if similarity > 0.12:
                    adjacency[i][j] = adjacency[j][i] = True

        states = [feature.copy() for feature in features]
        for _ in range(2):
            updated_states: List[Counter] = []
            for i, state in enumerate(states):
                neighbors = [states[j] for j in range(len(states)) if adjacency[i][j] and j != i]
                if not neighbors:
                    updated_states.append(state)
                    continue
                aggregated = Counter()
                for neighbor in neighbors:
                    aggregated.update({token: count / len(neighbors) for token, count in neighbor.items()})
                for token, count in state.items():
                    aggregated[token] += count
                updated_states.append(aggregated)
            states = updated_states

        relations: List[Dict[str, Any]] = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if not adjacency[i][j]:
                    continue
                score = self._cosine_similarity(states[i], states[j])
                if score < max(threshold, 0.15):
                    continue
                relation_type = "related_to"
                evidence = ""
                for segment in segments:
                    if GLiRELRelationExtractor._entity_appears_in_text(entities[i].get("name", ""), segment) and GLiRELRelationExtractor._entity_appears_in_text(entities[j].get("name", ""), segment):
                        evidence = segment[:160]
                        break
                if not evidence:
                    evidence = text[:160]
                confidence = max(float(threshold), min(0.95, 0.45 + score * 0.5))
                relations.append(
                    {
                        "source": entities[i].get("name"),
                        "source_id": entities[i].get("canonical_name") or entities[i].get("canonical_id") or canonicalize_entity_name(entities[i].get("name", "")),
                        "target": entities[j].get("name"),
                        "target_id": entities[j].get("canonical_name") or entities[j].get("canonical_id") or canonicalize_entity_name(entities[j].get("name", "")),
                        "relation_type": relation_type,
                        "confidence": confidence,
                        "source_method": "graph_inference",
                        "evidence": evidence,
                        "source_span": [entities[i].get("start"), entities[i].get("end")],
                        "target_span": [entities[j].get("start"), entities[j].get("end")],
                    }
                )

        return relations

    @staticmethod
    def _merge_relations(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for relation in relations:
            key = (
                relation.get("source_id") or relation.get("source") or "",
                relation.get("target_id") or relation.get("target") or "",
                relation.get("relation_type") or "related_to",
            )
            existing = merged.get(key)
            if existing is None:
                merged[key] = relation
                continue

            new_confidence = float(relation.get("confidence", 0.0) or 0.0)
            existing_confidence = float(existing.get("confidence", 0.0) or 0.0)
            if new_confidence > existing_confidence:
                if existing.get("unknown_candidate") and not relation.get("unknown_candidate"):
                    relation["unknown_candidate"] = existing.get("unknown_candidate")
                merged[key] = relation
            elif new_confidence == existing_confidence and relation.get("source_method") == "heuristic_enhanced":
                if existing.get("unknown_candidate") and not relation.get("unknown_candidate"):
                    relation["unknown_candidate"] = existing.get("unknown_candidate")
                merged[key] = relation
            elif relation.get("unknown_candidate") and not existing.get("unknown_candidate"):
                existing["unknown_candidate"] = relation.get("unknown_candidate")

        # Collapse duplicate source-target pairs by preferring more specific relation types over generic related_to.
        pair_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for relation in merged.values():
            pair = (
                relation.get("source_id") or relation.get("source") or "",
                relation.get("target_id") or relation.get("target") or "",
            )
            existing = pair_map.get(pair)
            if existing is None:
                pair_map[pair] = relation
                continue

            existing_type = existing.get("relation_type") or "related_to"
            candidate_type = relation.get("relation_type") or "related_to"
            existing_confidence = float(existing.get("confidence", 0.0) or 0.0)
            candidate_confidence = float(relation.get("confidence", 0.0) or 0.0)
            if existing_type == "related_to" and candidate_type != "related_to":
                pair_map[pair] = relation
            elif candidate_type != "related_to" and existing_type != "related_to" and candidate_confidence > existing_confidence:
                pair_map[pair] = relation
            elif candidate_type == existing_type and candidate_confidence > existing_confidence:
                pair_map[pair] = relation

        return sorted(
            pair_map.values(),
            key=lambda item: (
                0 if (item.get("relation_type") or "related_to") != "related_to" else 1,
                float(item.get("confidence", 0.0) or 0.0),
            ),
            reverse=True,
        )


class RebelRelationExtractor(GLiRELRelationExtractor):
    """Backward compatible alias for legacy imports."""

    pass
