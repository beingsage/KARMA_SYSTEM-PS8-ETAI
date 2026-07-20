"""
Improved PID Component Detection (Stage 12)
Handles both text-based and entity-based component extraction with canonical mapping.
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Load component taxonomy
TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "component_taxonomy.json"

try:
    with open(TAXONOMY_PATH, "r") as f:
        COMPONENT_TAXONOMY = json.load(f)
except FileNotFoundError:
    logger.warning(f"Component taxonomy not found at {TAXONOMY_PATH}, using defaults")
    COMPONENT_TAXONOMY = {
        "component_keywords": {},
        "specific_components": {},
        "synonym_mappings": {}
    }


class ComponentDetector:
    """
    Enhanced component detection with multimodal support and canonical mapping.
    """

    def __init__(self, taxonomy: Dict[str, Any] = None):
        self.taxonomy = taxonomy or COMPONENT_TAXONOMY
        self.component_keywords = self.taxonomy.get("component_keywords", {})
        self.specific_components = self.taxonomy.get("specific_components", {})
        self.synonym_mappings = self.taxonomy.get("synonym_mappings", {})
        self.schema_version = self.taxonomy.get("schema_version", "1.0.0")
        self.domain_packs = self.taxonomy.get("domain_packs") or {
            "core": {
                "pack_id": "core_component_pack",
                "title": "Core industrial component pack",
                "schema_version": self.schema_version,
                "parent_type_id": "asset.component",
                "status": "active",
                "description": "Industry-agnostic component taxonomy used for extraction and normalization",
            }
        }
        self._build_keyword_index()

    def _build_ontology_payload(self, component_id: str, component_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        base_label = component_data.get("name") if component_data else None
        if not base_label:
            base_label = component_id.replace("_", " ").title()
        keywords = component_data.get("keywords", []) if component_data else []
        aliases = [keyword for keyword in keywords if isinstance(keyword, str)]
        raw_parent = (component_data or {}).get("parent") if component_data else None
        parent_type_id = (component_data or {}).get("parent_type_id") if component_data else None
        if not parent_type_id:
            parent_type_id = raw_parent or "asset.component"
        if parent_type_id and not str(parent_type_id).startswith("asset"):
            parent_type_id = f"asset.{parent_type_id}"
        type_id = (component_data or {}).get("ontology_type_id") or (component_data or {}).get("canonical_id") or component_id
        if not str(type_id).startswith("asset"):
            type_id = f"asset.component.{type_id}"
        if type_id.endswith(".None"):
            type_id = type_id[:-5]
        return {
            "type_id": type_id,
            "parent_type_id": parent_type_id,
            "status": "active",
            "schema_version": self.schema_version,
            "aliases": aliases or [base_label],
            "pack_id": next(iter(self.domain_packs.keys()), "core"),
            "source": "component_taxonomy",
            "confidence": 0.85,
        }

    def _enrich_component_record(self, component_id: str, component_data: Optional[Dict[str, Any]] = None, *, name: Optional[str] = None) -> Dict[str, Any]:
        ontology = self._build_ontology_payload(component_id, component_data)
        effective_name = name or (component_data.get("model_number") if component_data else None) or component_id.replace("_", " ").title()
        return {
            "canonical_id": component_id,
            "name": effective_name,
            "entity_type": component_data.get("entity_type") if component_data else "component",
            "ontology": ontology,
            "type_id": ontology["type_id"],
            "parent_type_id": ontology["parent_type_id"],
            "schema_version": ontology["schema_version"],
            "status": ontology["status"],
        }

    def _build_keyword_index(self):
        """Build efficient lookup structures for keyword matching."""
        self.keyword_to_component = {}
        
        # Map specific keywords
        for comp_id, comp_data in self.specific_components.items():
            for keyword in comp_data.get("keywords", []):
                self.keyword_to_component[keyword.lower()] = comp_id

        # Map generic keywords
        for comp_type, comp_data in self.component_keywords.items():
            for keyword in comp_data.get("keywords", []):
                if keyword.lower() not in self.keyword_to_component:
                    self.keyword_to_component[keyword.lower()] = comp_type

        # Add synonym mappings
        for synonym, target in self.synonym_mappings.items():
            self.keyword_to_component[synonym.lower()] = target

    def detect_from_text(self, text: str, page_map: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
        """
        Extract components from OCR text with localization.
        
        Args:
            text: OCR extracted text from document
            page_map: Optional dict mapping page_id -> page_text for per-page localization
        
        Returns:
            Dict with detected components and occurrence metadata
        """
        components = {}
        
        if not text:
            return {
                "components": [],
                "summary": {
                    "total_components": 0,
                    "total_mentions": 0,
                    "detection_methods": {"text_entity": 0, "visual": 0},
                    "status": "empty_input"
                }
            }

        # Search for specific components first (higher priority)
        for comp_id, comp_data in self.specific_components.items():
            occurrences = self._find_component_occurrences(
                text, 
                comp_data.get("keywords", []), 
                page_map
            )
            if occurrences:
                components[comp_id] = {
                    "canonical_id": comp_id,
                    "name": comp_data.get("model_number", comp_id),
                    "entity_type": "equipment_variant",
                    "detected_via": "text_entity",
                    "confidence": min(0.95, 0.8 + len(occurrences) * 0.05),  # Boost with frequency
                    "occurrences": occurrences,
                    "manufacturer": comp_data.get("manufacturer"),
                    "specifications": comp_data.get("specifications", {}),
                    "parent_component": comp_data.get("parent"),
                    "related_entities": comp_data.get("related_to", []),
                    **self._enrich_component_record(comp_id, comp_data, name=comp_data.get("model_number", comp_id)),
                }

        # Search for generic component types
        for comp_type, comp_data in self.component_keywords.items():
            canonical_id = comp_data.get("canonical_id", comp_type)
            
            # Skip if specific variant already detected
            if canonical_id in components:
                continue

            occurrences = self._find_component_occurrences(
                text,
                comp_data.get("keywords", []),
                page_map
            )
            if occurrences:
                components[canonical_id] = {
                    "canonical_id": canonical_id,
                    "name": comp_type.replace("_", " ").title(),
                    "entity_type": comp_data.get("entity_types", ["component"])[0],
                    "detected_via": "text_entity",
                    "confidence": min(0.90, 0.6 + len(occurrences) * 0.05),
                    "occurrences": occurrences,
                    "category": comp_data.get("category", ""),
                    "specifications": comp_data.get("typical_specs", {}),
                    **self._enrich_component_record(canonical_id, comp_data, name=comp_type.replace("_", " ").title()),
                }

        total_mentions = sum(len(c["occurrences"]) for c in components.values())
        
        return {
            "components": list(components.values()),
            "summary": {
                "total_components": len(components),
                "total_mentions": total_mentions,
                "detection_methods": {
                    "text_entity": len(components),
                    "visual": 0
                },
                "coverage_notes": "Text-based extraction from OCR"
            }
        }

    def _find_component_occurrences(
        self, 
        text: str, 
        keywords: List[str],
        page_map: Optional[Dict[int, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find all occurrences of component keywords in text.
        Handles plural forms and case-insensitive matching.
        
        Returns list of occurrence objects with page/offset info.
        """
        occurrences = []
        
        for keyword in keywords:
            # Build patterns to match singular, plural, and variations
            patterns_to_try = [
                rf"\b{re.escape(keyword)}\b",  # Exact match
                rf"\b{re.escape(keyword)}s\b",  # Plural form (add 's')
                rf"\b{re.escape(keyword)}es\b",  # Plural form (add 'es')
            ]
            
            # Also try without trailing 's' to catch base form
            if keyword.endswith('s'):
                patterns_to_try.append(rf"\b{re.escape(keyword[:-1])}\b")
            
            for pattern in patterns_to_try:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start_idx = match.start()
                    end_idx = match.end()
                    
                    # Extract context window (50 chars before and after)
                    context_start = max(0, start_idx - 50)
                    context_end = min(len(text), end_idx + 50)
                    context_snippet = text[context_start:context_end].strip()
                    
                    # Determine page (if page_map provided)
                    page_id = self._get_page_from_offset(text, start_idx, page_map)
                    
                    occurrences.append({
                        "keyword": keyword,
                        "matched_text": text[start_idx:end_idx],
                        "char_offset": [start_idx, end_idx],
                        "context_snippet": context_snippet[:200],  # Truncate very long snippets
                        "page": page_id or "unknown",
                        "confidence": 0.95  # High confidence for exact keyword match
                    })
        
        return occurrences

    def _get_page_from_offset(
        self, 
        text: str, 
        offset: int,
        page_map: Optional[Dict[int, str]] = None
    ) -> Optional[int]:
        """
        Estimate page number from character offset.
        """
        if not page_map:
            return None
        
        cumulative = 0
        for page_id in sorted(page_map.keys()):
            page_text = page_map.get(page_id, "")
            cumulative += len(page_text)
            if offset <= cumulative:
                return page_id
        
        return list(page_map.keys())[-1] if page_map else None

    def detect_from_entities(
        self,
        entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract components from pre-extracted entities (fallback from stage 15).
        
        Args:
            entities: List of entity dicts from stage 15 (entity_extraction)
                     Expected keys: name, canonical_name, entity_type, confidence, page, start, end
        
        Returns:
            Dict with detected components linked to entity metadata
        """
        components = {}
        
        for entity in entities:
            entity_name = entity.get("name", "").lower()
            entity_canonical = entity.get("canonical_name", "").lower()
            
            # Check for direct component match
            component_id = self._match_entity_to_component(entity_name, entity_canonical)
            
            if component_id:
                if component_id not in components:
                    components[component_id] = {
                        "canonical_id": component_id,
                        "name": entity.get("name"),
                        "entity_type": entity.get("entity_type", "component"),
                        "detected_via": "entity_extraction",
                        "confidence": entity.get("confidence", 0.7),
                        "occurrences": [],
                        "parent_entity_ids": [],
                        **self._enrich_component_record(component_id, None, name=entity.get("name")),
                    }
                
                # Add occurrence with entity offset
                components[component_id]["occurrences"].append({
                    "entity_name": entity.get("name"),
                    "char_offset": [entity.get("start", -1), entity.get("end", -1)],
                    "page": entity.get("page", "unknown"),
                    "entity_canonical": entity_canonical,
                    "entity_id": entity.get("canonical_name"),
                    "confidence": entity.get("confidence", 0.7)
                })
                
                components[component_id]["parent_entity_ids"].append(
                    entity.get("canonical_name")
                )

        total_mentions = sum(len(c["occurrences"]) for c in components.values())
        
        return {
            "components": list(components.values()),
            "summary": {
                "total_components": len(components),
                "total_mentions": total_mentions,
                "detection_methods": {
                    "text_entity": len(components),
                    "visual": 0
                },
                "coverage_notes": "Entity-based extraction from stage 15"
            }
        }

    def _match_entity_to_component(self, entity_name: str, entity_canonical: str) -> Optional[str]:
        """
        Match an entity to a known component type/ID.
        Uses fuzzy matching to handle variations in naming.
        """
        search_terms = [entity_name.lower(), entity_canonical.lower()]
        
        # Check specific components first (higher priority)
        for comp_id, comp_data in self.specific_components.items():
            comp_keywords = [kw.lower() for kw in comp_data.get("keywords", [])]
            for term in search_terms:
                for keyword in comp_keywords:
                    # Exact match or substring match
                    if term == keyword or keyword in term or term in keyword:
                        return comp_id
        
        # Check generic components
        for comp_type, comp_data in self.component_keywords.items():
            comp_keywords = [kw.lower() for kw in comp_data.get("keywords", [])]
            canonical_id = comp_data.get("canonical_id", comp_type)
            for term in search_terms:
                for keyword in comp_keywords:
                    # Handle plural forms by checking base keyword
                    base_keyword = keyword.rstrip('s')  # Remove trailing 's' for plural
                    if (term == keyword or keyword in term or term in keyword or
                        base_keyword in term or term in base_keyword):
                        return canonical_id
        
        # Check synonym mappings with fuzzy matching
        for search_term in search_terms:
            for synonym, target in self.synonym_mappings.items():
                if search_term == synonym or synonym in search_term or search_term in synonym:
                    return target
        
        return None

    def fuse_text_and_entity_results(
        self,
        text_result: Dict[str, Any],
        entity_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fuse results from text extraction and entity extraction.
        Prefers entity data when available, supplements with text data.
        """
        all_components = {}
        
        # Start with entity-based detections (higher quality)
        for comp in entity_result.get("components", []):
            canonical_id = comp.get("canonical_id")
            all_components[canonical_id] = comp
        
        # Add text-based detections not covered by entity extraction
        for comp in text_result.get("components", []):
            canonical_id = comp.get("canonical_id")
            if canonical_id not in all_components:
                all_components[canonical_id] = comp
            else:
                # Merge occurrences if component detected by both methods
                existing = all_components[canonical_id]
                existing["detected_via"] = ["text_entity", "entity_extraction"]
                existing["occurrences"].extend(comp.get("occurrences", []))
                # Boost confidence if detected multiple ways
                existing["confidence"] = min(0.99, existing["confidence"] + 0.05)

        total_mentions = sum(len(c["occurrences"]) for c in all_components.values())
        detection_methods = {"text_entity": 0, "visual": 0}
        for comp in all_components.values():
            via = comp.get("detected_via", "unknown")
            if isinstance(via, list):
                for method in via:
                    if "text" in method:
                        detection_methods["text_entity"] += 1
                        break
            elif "text" in via:
                detection_methods["text_entity"] += 1

        return {
            "components": list(all_components.values()),
            "summary": {
                "total_components": len(all_components),
                "total_mentions": total_mentions,
                "detection_methods": detection_methods,
                "coverage_notes": "Fused text and entity extraction"
            }
        }

    def to_output_format(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert internal representation to pipeline output format.
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "stage": "pid_component_detection",
            "status": "completed",
            "full_output": detection_result
        }


def detect_pid_components_v2(
    text: str,
    entities: Optional[List[Dict[str, Any]]] = None,
    page_map: Optional[Dict[int, str]] = None,
    use_entity_fallback: bool = True
) -> Dict[str, Any]:
    """
    Improved PID component detection (Stage 12 replacement).
    
    Args:
        text: Full OCR text
        entities: Optional list of entities from stage 15
        page_map: Optional dict mapping page_id -> page_text
        use_entity_fallback: If True and entities provided, fuse results
    
    Returns:
        Pipeline output dict with detected components
    """
    detector = ComponentDetector()
    
    # Always do text-based detection
    text_result = detector.detect_from_text(text, page_map)
    
    # If entities provided, also extract from them and optionally fuse
    if entities and use_entity_fallback:
        entity_result = detector.detect_from_entities(entities)
        final_result = detector.fuse_text_and_entity_results(text_result, entity_result)
    else:
        final_result = text_result
    
    return detector.to_output_format(final_result)
