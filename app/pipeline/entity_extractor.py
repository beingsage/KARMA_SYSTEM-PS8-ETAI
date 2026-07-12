"""Industrial entity extraction using GLiNER."""

from pathlib import Path
from typing import Any, Dict, List

from app.pipeline.compat import allow_trusted_torch_pickle, ensure_pyarrow_compat
from app.pipeline.document_utils import chunk_text, normalize_text
from app.pipeline.models import canonicalize_entity_name, detect_pid_components
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

    def __init__(self) -> None:
        self.model = None
        self.fine_tuned_model_path = Path("./models/gliner-industrial-v1")
        self.device = select_device()
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

    def extract(self, text: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("GLiNER entity extractor is unavailable.")

        text = normalize_text(text)
        merged_entities: Dict[str, Dict[str, Any]] = {}

        # GLiNER internally tokenizes and may truncate very long sentences.
        # Use conservative chunk sizes to avoid internal truncation warnings
        # and ensure each model input fits typical NER token limits (~384).
        segment_texts = chunk_text(text, max_chars=350, overlap=80)
        for segment in segment_texts:
            entities = self._extract_with_model(segment, threshold)
            for entity in entities:
                canonical = entity["canonical_name"]
                existing = merged_entities.get(canonical)
                if not existing or entity["confidence"] > existing["confidence"]:
                    merged_entities[canonical] = entity

        return list(merged_entities.values())

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
                {
                    "name": name,
                    "entity_type": entity["label"],
                    "confidence": round(entity.get("score", 0.0), 3),
                    "canonical_name": canonical,
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0),
                    "source": "gliner",
                }
            )
        return formatted_entities

    def fine_tune_on_domain(self, training_data: List[Dict[str, Any]]) -> bool:
        if self.model is None:
            raise RuntimeError("GLiNER model is not initialized. Cannot fine-tune.")
        self.model.save_pretrained(str(self.fine_tuned_model_path))
        print("✓ Fine-tuned GLiNER model saved")
        return True
