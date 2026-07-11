import re
from typing import Any, Dict, List

from app.pipeline.models import canonicalize_entity_name


class BlinkEntityLinker:
    """Entity linker that uses BLINK if available, otherwise a lightweight fallback."""

    def __init__(self) -> None:
        self.blink_available = False

        try:
            import blink  # type: ignore

            self.blink_available = True
        except Exception:
            self.blink_available = False
            self._embedder = None

    @staticmethod
    def _fallback_confidence(name: str, canonical: str) -> float:
        if not name:
            return 0.55

        token_count = len([token for token in canonical.split("_") if token])
        if token_count >= 3:
            return 0.7
        if token_count == 2:
            return 0.65
        if len(name) > 10:
            return 0.6
        return 0.55

    def link_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        linked: List[Dict[str, Any]] = []
        seen = set()

        for entity in entities:
            name = entity.get("name", "")
            canonical = canonicalize_entity_name(name)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)

            if self.blink_available:
                linked_id = f"blink:{canonical}"
                link_source = "blink"
                confidence = 0.75
            else:
                linked_id = f"blink_fallback:{canonical}"
                link_source = "blink_fallback"
                confidence = self._fallback_confidence(name, canonical)

            linked_entity = {
                **entity,
                "canonical_name": canonical,
                "linked_id": linked_id,
                "linked_name": name,
                "link_confidence": float(confidence),
                "link_source": link_source,
            }
            linked.append(linked_entity)

        return linked
