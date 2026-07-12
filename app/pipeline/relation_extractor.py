"""Industrial relation extraction using GLiREL."""

from typing import Any, Dict, List, Optional

from app.pipeline.compat import allow_trusted_torch_pickle
from app.pipeline.models import canonicalize_entity_name
from app.pipeline.runtime import select_device


class GLiRELRelationExtractor:
    """Extract relations using GLiREL."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model = None
        self.model_name = model_name or "jackboyla/glirel-large-v0"
        self.device = select_device()
        self.is_ready = False
        self.backend = "heuristic"
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
            # Provide actionable guidance for common environment issues
            if "PyExtensionType" in msg or "pyarrow" in msg:
                print("  ⚠ Detected pyarrow API mismatch. The runtime now shims `PyExtensionType` when possible; otherwise install a pyarrow build that still exposes that alias.")
            if "weights_only" in msg or "pickle" in msg or "torch.load" in msg:
                print("  ⚠ Detected torch loading/pickle issue. Ensure PyTorch >= 2.6 or use safetensors checkpoints; consider installing `safetensors` and using .safetensors model files.")
            print("  ⚠ Falling back to heuristic relation extraction.")
            self.model = None
            self.is_ready = True
            self.backend = "heuristic"

    def extract(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        if self.model is None:
            return self._heuristic_extract(text, entities)

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

    @staticmethod
    def _heuristic_extract(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback heuristic relation extraction."""
        import re

        relations: List[Dict[str, Any]] = []
        entity_names = [e["name"] for e in entities if e.get("name")]

        if len(entity_names) < 2:
            return []

        for i, entity1 in enumerate(entities[:20]):
            for entity2 in entities[i + 1 : 20]:
                name1 = entity1.get("name", "")
                name2 = entity2.get("name", "")

                if name1 and name2:
                    pattern = f"({re.escape(name1)}.*{re.escape(name2)}|{re.escape(name2)}.*{re.escape(name1)})"
                    if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                        relations.append(
                            {
                                "source": name1,
                                "source_id": entity1.get("canonical_name", name1),
                                "target": name2,
                                "target_id": entity2.get("canonical_name", name2),
                                "relation_type": "related_to",
                                "confidence": 0.5,
                                "source_method": "heuristic",
                            }
                        )

        return relations


class RebelRelationExtractor(GLiRELRelationExtractor):
    """Backward compatible alias for legacy imports."""

    pass
