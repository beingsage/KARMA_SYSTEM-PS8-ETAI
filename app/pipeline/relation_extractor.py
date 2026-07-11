"""Industrial relation extraction using GLiREL."""

from typing import Any, Dict, List, Optional

from app.pipeline.models import canonicalize_entity_name
from app.pipeline.runtime import select_device


class GLiRELRelationExtractor:
    """Extract relations using GLiREL."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model = None
        self.model_name = model_name or "jackboyla/glirel-large-v0"
        self.device = select_device()
        self.is_ready = False
        self._initialize()

    def _initialize(self) -> None:
        try:
            from glirel import GLiREL

            self.model = GLiREL.from_pretrained(self.model_name, map_location=self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            self.is_ready = self.model is not None
            print(f"✓ GLiREL relation extraction model loaded on {self.device}: {self.model_name}")
        except Exception as exc:
            print(f"✗ GLiREL initialization failed: {exc}")
            self.model = None
            self.is_ready = False

    def extract(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("GLiREL relation extractor is unavailable.")

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
                {
                    "source": source,
                    "source_id": canonicalize_entity_name(source),
                    "target": target,
                    "target_id": canonicalize_entity_name(target),
                    "relation_type": relation.get("label", "related_to"),
                    "confidence": float(relation.get("score", 0.0)),
                    "source_method": "glirel",
                }
            )

        return relations


class RebelRelationExtractor(GLiRELRelationExtractor):
    """Backward compatible alias for legacy imports."""

    pass
