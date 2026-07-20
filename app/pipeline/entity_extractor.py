"""Industrial entity extraction using GLiNER."""

import re
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings
from app.pipeline.compat import allow_trusted_torch_pickle, ensure_pyarrow_compat
from app.pipeline.document_utils import chunk_text, normalize_text
from app.pipeline.models import canonicalize_entity_name, detect_pid_components, normalize_entity_payload
from app.pipeline.prompt_zero_shot import PromptZeroShotExtractor
from app.pipeline.runtime import select_device


class GlinerEntityExtractor:
    """Extract entities using GLiNER."""

    INDUSTRIAL_ENTITIES = {
        "equipment": "Pumps, motors, compressors, valves, sensors",
        "process": "Distillation, compression, heating, cooling",
        "parameter": "Pressure, temperature, flow rate, viscosity",
        "material": "Oil, water, gas, chemical compound",
        "control_system": "PLC, SCADA, HMI, DCS, automation system",
        "location": "Reactor, vessel, pipeline, heat exchanger",
        "failure_mode": "Cavitation, corrosion, fouling, seal failure",
        "maintenance": "Inspection, repair, replacement, calibration",
    }

    HEURISTIC_ENTITY_TERMS = {
        "pump": "equipment",
        "valve": "equipment",
        "motor": "equipment",
        "compressor": "equipment",
        "tank": "equipment",
        "boiler": "equipment",
        "generator": "equipment",
        "transformer": "equipment",
        "bearing": "equipment",
        "seal": "equipment",
        "gearbox": "equipment",
        "pipe": "equipment",
        "pipeline": "equipment",
        "sensor": "sensor",
        "transmitter": "sensor",
        "pressure": "parameter",
        "temperature": "parameter",
        "flow rate": "parameter",
        "flow": "parameter",
        "level": "parameter",
        "viscosity": "parameter",
        "density": "parameter",
        "plc": "control_system",
        "scada": "control_system",
        "dcs": "control_system",
        "hmi": "control_system",
        "control system": "control_system",
        "startup": "process",
        "shutdown": "process",
        "maintenance": "maintenance",
        "inspection": "maintenance",
        "repair": "maintenance",
        "calibration": "maintenance",
        "commissioning": "process",
        "shutdown": "process",
        "failure": "failure_mode",
        "incident": "maintenance",
        "incident report": "maintenance",
        "catalyst": "material",
        "chemical": "material",
        "water": "material",
        "oil": "material",
        "gas": "material",
    }

    def __init__(self) -> None:
        self.model = None
        self.fine_tuned_model_path = Path("./models/gliner-industrial-v1")
        self.device = select_device()
        self.prompt_zero_shot = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            ensure_pyarrow_compat()
            from gliner import GLiNER

            with allow_trusted_torch_pickle():
                if self.fine_tuned_model_path.exists():
                    self.model = GLiNER.from_pretrained(str(self.fine_tuned_model_path), map_location=self.device)
                    print(f"✓ GLiNER loaded from fine-tuned industrial model on {self.device}")
                else:
                    self.model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1", map_location=self.device)
                    print(f"✓ GLiNER loaded (medium-v2.1) on {self.device}")

            if hasattr(self.model, "eval"):
                self.model.eval()
        except Exception as exc:
            print(f"✗ GLiNER initialization failed: {exc}")
            self.model = None

        try:
            self.prompt_zero_shot = PromptZeroShotExtractor()
        except Exception as exc:
            print(f"⚠ Prompt zero-shot extractor initialization failed: {exc}")
            self.prompt_zero_shot = None

    @staticmethod
    def _merge_candidate(
        merged_entities: Dict[str, Dict[str, Any]],
        entity: Dict[str, Any],
    ) -> None:
        canonical = str(entity.get("canonical_name") or canonicalize_entity_name(entity.get("name", ""))).strip()
        if not canonical:
            return

        existing = merged_entities.get(canonical)
        if existing is None:
            merged_entities[canonical] = dict(entity)
            return

        existing_confidence = float(existing.get("confidence", 0.0) or 0.0)
        candidate_confidence = float(entity.get("confidence", 0.0) or 0.0)
        preferred = dict(existing if existing_confidence >= candidate_confidence else entity)

        if existing.get("unknown_candidate") and not preferred.get("unknown_candidate"):
            preferred["unknown_candidate"] = existing.get("unknown_candidate")
        if entity.get("unknown_candidate"):
            preferred["unknown_candidate"] = entity.get("unknown_candidate")
        if existing.get("evidence_span") and not preferred.get("evidence_span"):
            preferred["evidence_span"] = existing.get("evidence_span")
        if entity.get("evidence_span") and not preferred.get("evidence_span"):
            preferred["evidence_span"] = entity.get("evidence_span")

        merged_entities[canonical] = preferred

    def extract(self, text: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        try:
            text = normalize_text(text)
            merged_entities: Dict[str, Dict[str, Any]] = {}

            if settings.enable_prompt_zero_shot_extraction and self.prompt_zero_shot is not None:
                try:
                    prompt_entities = self.prompt_zero_shot.extract_entities(text)
                    for entity in prompt_entities:
                        self._merge_candidate(merged_entities, normalize_entity_payload(entity))
                except Exception as exc:
                    print(f"⚠ Prompt zero-shot entity extraction failed: {exc}")

            if self.model is None:
                if merged_entities:
                    return list(merged_entities.values())
                return self._heuristic_extract(text)

            # GLiNER internally tokenizes and may truncate very long sentences.
            # Use conservative chunk sizes to avoid internal truncation warnings
            # and ensure each model input fits typical NER token limits (~384).
            segment_texts = chunk_text(text, max_chars=350, overlap=80)
            for segment in segment_texts:
                entities = self._extract_with_model(segment, threshold)
                for entity in entities:
                    self._merge_candidate(merged_entities, entity)

            if merged_entities:
                return list(merged_entities.values())
            return self._heuristic_extract(text)
        except Exception as exc:
            print(f"⚠ GLiNER extraction failed: {exc}; falling back to heuristics")
            return self._heuristic_extract(text)

    def _heuristic_extract(self, text: str) -> List[Dict[str, Any]]:
        normalized_text = normalize_text(text)
        lowered = normalized_text.lower()
        candidates: Dict[str, Dict[str, Any]] = {}

        for term, entity_type in self.HEURISTIC_ENTITY_TERMS.items():
            pattern = rf"\b{re.escape(term)}\b"
            for match in re.finditer(pattern, lowered):
                entity_text = normalized_text[match.start() : match.end()]
                canonical = canonicalize_entity_name(entity_text)
                confidence = 0.72 if " " not in term else 0.8
                existing = candidates.get(canonical)
                if existing and existing.get("confidence", 0.0) >= confidence:
                    continue
                candidate_payload = {
                    "name": entity_text.strip(),
                    "entity_type": entity_type,
                    "confidence": round(confidence, 3),
                    "canonical_name": canonical,
                    "start": match.start(),
                    "end": match.end(),
                    "source": "heuristic",
                    "source_document": None,
                    "context": normalized_text[match.start() : match.end()],
                }
                candidates[canonical] = normalize_entity_payload(candidate_payload)

        if candidates:
            return sorted(candidates.values(), key=lambda item: (item["start"], item["name"]))

        fallback_terms = detect_pid_components(text)
        heuristic_entities = []
        for term in fallback_terms:
            canonical = canonicalize_entity_name(term)
            heuristic_entities.append(
                normalize_entity_payload(
                    {
                        "name": term,
                        "entity_type": "equipment",
                        "confidence": 0.55,
                        "canonical_name": canonical,
                        "start": lowered.find(term.lower()),
                        "end": lowered.find(term.lower()) + len(term),
                        "source": "heuristic",
                        "source_document": None,
                        "context": term,
                    }
                )
            )
        return heuristic_entities

    def _extract_with_model(self, text: str, threshold: float) -> List[Dict[str, Any]]:
        entities = self.model.predict_entities(
            text,
            labels=list(self.INDUSTRIAL_ENTITIES.keys()),
            threshold=threshold,
        )

        formatted_entities = []
        for entity in entities:
            name = entity["text"].strip()
            canonical = canonicalize_entity_name(name)
            formatted_entities.append(
                normalize_entity_payload(
                    {
                        "name": name,
                        "entity_type": entity["label"],
                        "confidence": round(entity.get("score", 0.0), 3),
                        "canonical_name": canonical,
                        "start": entity.get("start", 0),
                        "end": entity.get("end", 0),
                        "source": "gliner",
                        "source_document": None,
                        "context": name,
                    }
                )
            )
        return formatted_entities

    def fine_tune_on_domain(self, training_data: List[Dict[str, Any]]) -> bool:
        if self.model is None:
            raise RuntimeError("GLiNER model is not initialized. Cannot fine-tune.")
        self.model.save_pretrained(str(self.fine_tuned_model_path))
        print("✓ Fine-tuned GLiNER model saved")
        return True
