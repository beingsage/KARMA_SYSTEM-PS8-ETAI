"""
Unit tests for improved PID Component Detection (Stage 12)
Tests the ComponentDetector class and related functions.
"""

import pytest
import json
from pathlib import Path
from app.pipeline.component_detector import (
    ComponentDetector,
    detect_pid_components_v2
)


class TestComponentDetectorBasics:
    """Test basic component detection functionality."""

    def setup_method(self):
        """Initialize detector for each test."""
        self.detector = ComponentDetector()

    def test_detector_loads_taxonomy(self):
        """Ensure component taxonomy is loaded."""
        assert len(self.detector.component_keywords) > 0
        assert len(self.detector.specific_components) > 0
        assert len(self.detector.keyword_to_component) > 0

    def test_domain_pack_metadata_is_exposed(self):
        """Detected components should carry ontology metadata from the domain-pack taxonomy."""
        result = self.detector.detect_from_text("The centrifugal pump and PLC are installed at the site.")
        component = next((item for item in result["components"] if "pump" in item.get("canonical_id", "")), None)

        assert component is not None
        assert component["ontology"]["status"] == "active"
        assert component["ontology"]["type_id"].startswith("asset")

    def test_detect_pump_from_text(self):
        """Test detection of 'pump' keyword in text."""
        text = "The Hydro MPC booster systems with CR 120 pump are secured."
        result = self.detector.detect_from_text(text)
        
        assert result is not None
        assert "components" in result
        assert len(result["components"]) > 0
        
        # Check if pump is detected
        component_names = [c["name"] for c in result["components"]]
        assert any("pump" in name.lower() for name in component_names)

    def test_detect_valve_from_text(self):
        """Test detection of 'valve' keyword."""
        text = "The expansion valve must be installed on the inlet pipe."
        result = self.detector.detect_from_text(text)
        
        component_names = [c["name"].lower() for c in result["components"]]
        assert any("valve" in name for name in component_names)

    def test_detect_multiple_components(self):
        """Test detection of multiple component types in one document."""
        text = """
        The Hydro MPC booster system includes a CR 120 pump, electric motor,
        pressure sensor, control cabinet, and expansion joint. The manifold
        connects the inlet and outlet pipes.
        """
        result = self.detector.detect_from_text(text)
        
        component_types = [c["canonical_id"].split("_")[0] for c in result["components"]]
        
        # We should detect at least pump, motor, sensor, control, expansion_joint, manifold
        expected_types = ["pump", "motor", "sensor", "control"]
        for expected in expected_types:
            assert any(expected in ctype for ctype in component_types), \
                f"Missing expected component type: {expected}"

    def test_empty_text_returns_empty_result(self):
        """Test handling of empty text input."""
        result = self.detector.detect_from_text("")
        
        assert result["components"] == []
        assert result["summary"]["total_components"] == 0

    def test_no_components_in_text(self):
        """Test text with no component keywords."""
        text = "This is a sample document about cooking recipes and weather."
        result = self.detector.detect_from_text(text)
        
        assert len(result["components"]) == 0

    def test_component_has_required_fields(self):
        """Test that detected components have all required metadata."""
        text = "The pump and valve are installed together."
        result = self.detector.detect_from_text(text)
        
        for component in result["components"]:
            assert "canonical_id" in component
            assert "name" in component
            assert "entity_type" in component
            assert "detected_via" in component
            assert "confidence" in component
            assert "occurrences" in component
            assert isinstance(component["occurrences"], list)

    def test_occurrence_has_location_info(self):
        """Test that component occurrences include location data."""
        text = "Install the pump carefully. The pump must be primed before starting."
        result = self.detector.detect_from_text(text)
        
        for component in result["components"]:
            for occurrence in component["occurrences"]:
                assert "keyword" in occurrence
                assert "char_offset" in occurrence
                assert len(occurrence["char_offset"]) == 2
                assert "context_snippet" in occurrence
                assert "page" in occurrence


class TestSpecificComponentDetection:
    """Test detection of specific component variants."""

    def setup_method(self):
        self.detector = ComponentDetector()

    def test_detect_cr120_pump(self):
        """Test specific detection of CR 120 pump variant."""
        text = "The Hydro MPC booster systems with CR 120 pumps are secured by transport straps."
        result = self.detector.detect_from_text(text)
        
        # Should detect CR 120 pump specifically
        component_ids = [c["canonical_id"] for c in result["components"]]
        assert "cr_120_pump" in component_ids or any("cr_120" in cid for cid in component_ids)

    def test_detect_hydro_mpc_booster(self):
        """Test detection of Hydro MPC booster system."""
        text = "These installation instructions apply to the Grundfos Hydro MPC booster systems."
        result = self.detector.detect_from_text(text)
        
        component_ids = [c["canonical_id"] for c in result["components"]]
        assert "hydro_mpc_booster" in component_ids or any("hydro" in cid.lower() for cid in component_ids)

    def test_synonym_mapping_expansion_joint(self):
        """Test that vibration damper is mapped to expansion joint."""
        text = "Use vibration dampeners on the inlet and outlet pipes."
        result = self.detector.detect_from_text(text)
        
        # Should detect expansion_joint via synonym
        component_ids = [c["canonical_id"] for c in result["components"]]
        assert any("expansion" in cid or "damper" in cid for cid in component_ids)


class TestEntityBasedDetection:
    """Test detection from extracted entities (stage 15 output)."""

    def setup_method(self):
        self.detector = ComponentDetector()

    def test_entity_matching_pump(self):
        """Test matching entity to pump component."""
        entities = [
            {
                "name": "CR 120 pump",
                "canonical_name": "cr_120_pump",
                "entity_type": "equipment_variant",
                "confidence": 0.95,
                "page": 3,
                "start": 450,
                "end": 462
            }
        ]
        result = self.detector.detect_from_entities(entities)
        
        assert len(result["components"]) > 0
        component_ids = [c["canonical_id"] for c in result["components"]]
        assert any("pump" in cid.lower() or "cr_120" in cid for cid in component_ids)

    def test_entity_with_multiple_occurrences(self):
        """Test entity that appears multiple times in document."""
        entities = [
            {
                "name": "motor",
                "canonical_name": "motor",
                "entity_type": "equipment",
                "confidence": 0.85,
                "page": 3,
                "start": 100,
                "end": 105
            },
            {
                "name": "electric motor",
                "canonical_name": "motor_electric",
                "entity_type": "equipment",
                "confidence": 0.90,
                "page": 5,
                "start": 800,
                "end": 814
            }
        ]
        result = self.detector.detect_from_entities(entities)
        
        # Should aggregate multiple motor mentions
        motor_comps = [c for c in result["components"] if "motor" in c["canonical_id"].lower()]
        assert len(motor_comps) > 0

    def test_empty_entity_list(self):
        """Test handling of empty entity list."""
        result = self.detector.detect_from_entities([])
        
        assert result["components"] == []
        assert result["summary"]["total_components"] == 0


class TestFusion:
    """Test fusion of text-based and entity-based results."""

    def setup_method(self):
        self.detector = ComponentDetector()

    def test_fuse_text_and_entity_results(self):
        """Test merging of text and entity extraction."""
        text = "The pump and valve are connected."
        text_result = self.detector.detect_from_text(text)
        
        entities = [
            {
                "name": "pump",
                "canonical_name": "pump",
                "entity_type": "equipment",
                "confidence": 0.9,
                "page": 1,
                "start": 4,
                "end": 8
            }
        ]
        entity_result = self.detector.detect_from_entities(entities)
        
        fused = self.detector.fuse_text_and_entity_results(text_result, entity_result)
        
        assert fused["components"] is not None
        assert fused["summary"]["detection_methods"]["text_entity"] > 0
        
        # Fused result should contain both text and entity detections
        assert len(fused["components"]) >= 1

    def test_fusion_boosts_confidence_for_multimodal(self):
        """Test that components detected by multiple methods get boosted confidence."""
        text = "The pump is critical."
        text_result = self.detector.detect_from_text(text)
        
        entities = [
            {
                "name": "pump",
                "canonical_name": "pump",
                "entity_type": "equipment",
                "confidence": 0.85,
                "page": 1,
                "start": 4,
                "end": 8
            }
        ]
        entity_result = self.detector.detect_from_entities(entities)
        
        fused = self.detector.fuse_text_and_entity_results(text_result, entity_result)
        
        # Find pump in fused results
        pump_comps = [c for c in fused["components"] if "pump" in c.get("name", "").lower()]
        for pump in pump_comps:
            # Confidence should be high (boosted if multimodal)
            assert pump["confidence"] > 0.7


class TestOutputFormat:
    """Test output format compliance."""

    def setup_method(self):
        self.detector = ComponentDetector()

    def test_output_format_structure(self):
        """Test that output follows expected pipeline format."""
        text = "The pump and sensor are installed."
        result = self.detector.detect_from_text(text)
        
        output = self.detector.to_output_format(result)
        
        assert "timestamp" in output
        assert "stage" in output
        assert output["stage"] == "pid_component_detection"
        assert "status" in output
        assert output["status"] == "completed"
        assert "full_output" in output

    def test_summary_fields(self):
        """Test that summary contains expected statistics."""
        text = "Pump, valve, motor, and sensor are needed."
        result = self.detector.detect_from_text(text)
        
        assert "summary" in result
        summary = result["summary"]
        
        assert "total_components" in summary
        assert "total_mentions" in summary
        assert "detection_methods" in summary
        assert "coverage_notes" in summary
        
        assert summary["total_components"] >= 1
        assert summary["total_mentions"] >= 1


class TestIntegration:
    """Integration tests with the v2 function."""

    def test_detect_pid_components_v2_basic(self):
        """Test the main v2 function."""
        text = "The Hydro MPC pump and expansion joint need maintenance."
        result = detect_pid_components_v2(text)
        
        assert result is not None
        assert "full_output" in result
        assert "components" in result["full_output"]
        
        components = result["full_output"]["components"]
        assert len(components) > 0

    def test_v2_with_entities(self):
        """Test v2 function with optional entity input."""
        text = "The pump operates at high pressure."
        entities = [
            {
                "name": "pump",
                "canonical_name": "pump",
                "entity_type": "equipment",
                "confidence": 0.9,
                "page": 1,
                "start": 4,
                "end": 8
            }
        ]
        
        result = detect_pid_components_v2(text, entities=entities)
        
        assert result is not None
        components = result["full_output"]["components"]
        assert len(components) >= 1

    def test_v2_without_entity_fallback(self):
        """Test v2 function with entity fallback disabled."""
        text = "The pump and valve are required."
        entities = [
            {
                "name": "pump",
                "canonical_name": "pump",
                "entity_type": "equipment",
                "confidence": 0.9,
                "page": 1,
                "start": 4,
                "end": 8
            }
        ]
        
        result = detect_pid_components_v2(text, entities=entities, use_entity_fallback=False)
        
        assert result is not None
        # Without entity fallback, should still detect from text
        components = result["full_output"]["components"]
        assert len(components) >= 1


class TestHydroMPCManualSpecific:
    """Tests specific to the Hydro MPC manual test document."""

    def setup_method(self):
        self.detector = ComponentDetector()

    def test_detect_manual_components(self):
        """Test on actual manual text snippets."""
        manual_text = """
        Hydro MPC Installation and operating instructions
        These installation and operating instructions apply to the
        Grundfos Hydro MPC booster systems.
        
        The Hydro MPC booster systems with CR 120 or CR 150 pumps are secured
        by means of transport straps.
        
        Connect the pipes to the manifolds of the booster system.
        Apply sealing compound to the unused end of the manifold.
        Fit expansion joints on the inlet and outlet pipes to prevent
        vibrations being transmitted through the pipes.
        """
        
        result = self.detector.detect_from_text(manual_text)
        components = result["components"]
        
        # Expected components in this text
        expected_keywords = ["pump", "manifold", "expansion", "sealing"]
        found = set()
        
        for comp in components:
            name_lower = comp["name"].lower()
            canonical_lower = comp["canonical_id"].lower()
            for kw in expected_keywords:
                if kw in name_lower or kw in canonical_lower:
                    found.add(kw)
        
        # Should find most expected components
        assert len(found) >= 3, f"Found only {found}, expected at least 3 from {expected_keywords}"

    def test_no_hallucinated_components(self):
        """Ensure detector doesn't hallucinate components not in text."""
        text = "This is general text about document management systems."
        result = self.detector.detect_from_text(text)
        
        # Should not detect any industrial components
        assert len(result["components"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
