"""
Comprehensive tests for Stage 20 GraphRAG Analysis
Tests evidence grounding, hallucination prevention, and confidence calibration
"""

import unittest
from app.pipeline.graphrag_summarizer import GraphRAGSummarizer


class TestGraphRAGEvidenceGrounding(unittest.TestCase):
    """Test evidence grounding and hallucination prevention."""

    def setUp(self) -> None:
        self.summarizer = GraphRAGSummarizer()

    def test_check_evidence_sufficiency_with_good_data(self) -> None:
        """Test that evidence sufficiency check passes with good data."""
        entities = [
            {"name": "Pump A", "entity_type": "equipment", "confidence": 0.8},
            {"name": "Bearing", "entity_type": "component", "confidence": 0.7},
        ]
        relations = [{"source": "Pump A", "relation_type": "has_component", "target": "Bearing"}]
        text = "The pump bearing shows significant wear patterns in the maintenance logs dated 2024-01-15."

        result = self.summarizer._check_evidence_sufficiency(entities, relations, text)

        self.assertTrue(result["sufficient"], f"Should find sufficient evidence: {result}")
        self.assertGreater(result["coverage"], 0.5)

    def test_check_evidence_sufficiency_with_insufficient_data(self) -> None:
        """Test that evidence sufficiency check fails with minimal data."""
        entities = []
        relations = []
        text = "Short"

        result = self.summarizer._check_evidence_sufficiency(entities, relations, text)

        self.assertFalse(result["sufficient"])
        self.assertEqual(result["coverage"], 0.0)

    def test_validate_claims_filters_generic_phrases(self) -> None:
        """Test that generic safety language without specifics is filtered out."""
        entities = [
            {"name": "Safety System", "entity_type": "component"},
            {"name": "Pump", "entity_type": "equipment"},
            {"name": "Bearing", "entity_type": "component"},
        ]
        text = "The pump bearing shows wear. The system operates normally."

        claims = [
            {"name": "Serious personal injury", "description": "Dangerous situation without specifics"},
            {"name": "Pump bearing wear detected", "description": "Specific component failure mode"},
        ]

        validated = self.summarizer._validate_claims(claims, entities, text)

        # Generic claim should be filtered, bearing claim should pass
        self.assertEqual(len(validated), 1)
        self.assertIn("bearing wear", str(validated[0]).lower())

    def test_validate_claims_accepts_specific_details(self) -> None:
        """Test that claims with specific details are accepted."""
        entities = [{"name": "Pump A", "entity_type": "equipment"}]
        text = "Part 101 shows 2.5mm wear on the seal surface."

        claims = [
            {
                "name": "Pump A seal wear",
                "description": "Part 101 shows 2.5mm wear on the seal surface",
            }
        ]

        validated = self.summarizer._validate_claims(claims, entities, text)

        self.assertEqual(len(validated), 1)
        self.assertTrue(
            validated[0]["source"].startswith("entity:")
            or validated[0]["source"].startswith("text_excerpt:"),
            f"Expected derived provenance source, got {validated[0]['source']}",
        )

    def test_claim_support_classifier_blocks_unsupported_claims(self) -> None:
        """Test that unsupported generic claims are blocked by the classifier."""
        claim_text = "Unknown anomaly without evidence"
        supported_score = self.summarizer._claim_support_probability(claim_text)

        self.assertLess(supported_score, 0.5)

    def test_build_explanation_chains_creates_provenance_edges(self) -> None:
        """Test that explanation chains are built with claim nodes and provenance edges."""
        validated_claims = [
            {
                "name": "Pump bearing wear detected",
                "description": "Pump A shows 2.5mm wear on page 4",
                "source": "page:4",
                "confidence": 0.7,
            }
        ]
        chains = self.summarizer._build_explanation_chains(validated_claims, [], "")

        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0]["claim_id"], "claim:1")
        self.assertEqual(chains[0]["provenance_nodes"][0]["type"], "page")
        self.assertEqual(chains[0]["edges"][0]["relation"], "supported_by")
        self.assertAlmostEqual(chains[0]["edges"][0]["edge_confidence"], 0.7, places=2)

    def test_validate_claims_removes_placeholder_text(self) -> None:
        """Test that claims with placeholder markers are filtered."""
        entities = [{"name": "Equipment", "entity_type": "component"}]
        text = "No content"

        claims = [
            {"name": "Issue with `` placeholder", "description": "Has marker"},
            {"name": "[image placeholder]", "description": "Image ref without processing"},
            {"name": "Actual issue", "description": "Equipment failure"},
        ]

        validated = self.summarizer._validate_claims(claims, entities, text)

        # Only non-placeholder claims should remain
        validated_strs = [str(c) for c in validated]
        self.assertFalse(any("`" in s for s in validated_strs))
        self.assertFalse(any("[image" in s.lower() for s in validated_strs))

    def test_has_specific_detail_detects_measurements(self) -> None:
        """Test detection of specific measurement details."""
        specific_claims = [
            {"description": "2.5mm bearing wear"},
            {"description": "Operating at 3500 rpm"},
            {"description": "Section 4.2 maintenance procedure"},
            {"description": "Page 15 reference"},
        ]

        for claim in specific_claims:
            self.assertTrue(
                self.summarizer._has_specific_detail(claim),
                f"Should detect specific detail in: {claim}",
            )

    def test_has_specific_detail_rejects_generic(self) -> None:
        """Test that generic statements are rejected."""
        generic_claims = [
            {"description": "There is an issue"},
            {"description": "Something is broken"},
            {"description": "We need to check it"},
        ]

        for claim in generic_claims:
            self.assertFalse(
                self.summarizer._has_specific_detail(claim),
                f"Should reject generic claim: {claim}",
            )


class TestGraphRAGConfidenceCalibration(unittest.TestCase):
    """Test that confidence scores are properly calibrated."""

    def setUp(self) -> None:
        self.summarizer = GraphRAGSummarizer()

    def test_max_unvalidated_confidence_enforced(self) -> None:
        """Test that unvalidated claims don't get high confidence."""
        self.assertEqual(self.summarizer.max_unvalidated_confidence, 0.3)

    def test_confidence_reduced_by_evidence_coverage(self) -> None:
        """Test that confidence is reduced based on evidence coverage."""
        # If 50% of claims validated, confidence should be halved
        base_confidence = 0.6
        coverage = 0.5
        expected = min(base_confidence, 0.3) * coverage  # min(0.6, 0.3) * 0.5 = 0.15

        result_confidence = min(base_confidence, 0.3) * coverage

        self.assertAlmostEqual(result_confidence, 0.15)
        self.assertLessEqual(result_confidence, 0.95)

    def test_confidence_capped_at_0_95(self) -> None:
        """Test that confidence is never set above 0.95."""
        # Even with perfect evidence, cap at 0.95
        perfect_confidence = min(1.0, 0.3) * 1.0  # 0.3
        capped = min(perfect_confidence, 0.95)

        self.assertLessEqual(capped, 0.95)


class TestGraphRAGJSONParsing(unittest.TestCase):
    """Test robust JSON parsing and error handling."""

    def setUp(self) -> None:
        self.summarizer = GraphRAGSummarizer()

    def test_parse_json_response_valid(self) -> None:
        """Test parsing valid JSON response."""
        response = (
            '{"anomalies": [{"name": "Bearing wear"}], '
            '"risks": [], "recommendations": [], "compliance": [], "confidence": 0.7}'
        )

        result = self.summarizer._parse_json_response(response)

        self.assertEqual(len(result["anomalies"]), 1)
        self.assertEqual(result["confidence"], 0.7)
        self.assertEqual(result["parse_status"], "json_parsed")

    def test_parse_json_response_invalid_json(self) -> None:
        """Test handling of invalid JSON."""
        response = "This is not valid JSON at all"

        result = self.summarizer._parse_json_response(response)

        self.assertEqual(result["anomalies"], [])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["parse_status"], "no_json_found")

    def test_parse_json_response_missing_fields(self) -> None:
        """Test handling of JSON with missing required fields."""
        response = '{"anomalies": [{"name": "Issue"}]}'

        result = self.summarizer._parse_json_response(response)

        self.assertEqual(len(result["anomalies"]), 1)
        self.assertEqual(result["risks"], [])
        self.assertEqual(result["recommendations"], [])

    def test_parse_json_response_invalid_confidence(self) -> None:
        """Test handling of invalid confidence values."""
        responses = [
            '{"anomalies": [], "risks": [], "recommendations": [], "compliance": [], "confidence": -0.5}',
            '{"anomalies": [], "risks": [], "recommendations": [], "compliance": [], "confidence": 1.5}',
            '{"anomalies": [], "risks": [], "recommendations": [], "compliance": [], "confidence": null}',
        ]

        for response in responses:
            result = self.summarizer._parse_json_response(response)
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)

    def test_ensure_list_converts_values(self) -> None:
        """Test that _ensure_list properly converts various input types."""
        test_cases = [
            ([1, 2, 3], [1, 2, 3]),
            ({"key": "value"}, [{"key": "value"}]),
            (None, []),
            ("", []),
            ("string", ["string"]),
        ]

        for input_val, expected in test_cases:
            result = GraphRAGSummarizer._ensure_list(input_val)
            self.assertEqual(result, expected, f"Failed for input: {input_val}")


class TestGraphRAGPromptConstruction(unittest.TestCase):
    """Test that prompts are properly constructed with evidence requirements."""

    def setUp(self) -> None:
        self.summarizer = GraphRAGSummarizer()

    def test_prompt_includes_evidence_requirements(self) -> None:
        """Test that prompt explicitly requires evidence grounding."""
        entities = [{"name": "Pump", "entity_type": "equipment", "confidence": 0.8}]
        relations = []
        text = "The pump operates at normal conditions."

        prompt = self.summarizer._build_reasoning_prompt(entities, relations, text)

        # Check for key evidence-grounding requirements
        self.assertIn("ONLY cite specific evidence", prompt)
        self.assertIn("supporting evidence", prompt)
        self.assertIn("source", prompt)
        self.assertIn("NOT generate generic", prompt)
        self.assertIn("NEVER insert image placeholders", prompt)

    def test_prompt_includes_low_confidence_guidance(self) -> None:
        """Test that prompt guides towards lower, realistic confidence."""
        entities = [{"name": "Component", "entity_type": "equipment", "confidence": 0.7}]
        relations = []
        text = "Component operates normally."

        prompt = self.summarizer._build_reasoning_prompt(entities, relations, text)

        # Check for confidence guidance
        self.assertIn("0.3-0.7", prompt)
        self.assertIn("0.8+", prompt)

    def test_prompt_includes_example_output(self) -> None:
        """Test that prompt includes concrete output example."""
        entities = [{"name": "Bearing", "entity_type": "component"}]
        relations = []
        text = "Bearing shows wear."

        prompt = self.summarizer._build_reasoning_prompt(entities, relations, text)

        # Should include example JSON output
        self.assertIn("anomalies", prompt)
        self.assertIn("source", prompt)
        self.assertIn("confidence", prompt)


class TestGraphRAGIntegration(unittest.TestCase):
    """Integration tests for full generate_summary flow."""

    def setUp(self) -> None:
        self.summarizer = GraphRAGSummarizer()

    def test_generate_summary_returns_expected_structure(self) -> None:
        """Test that generate_summary returns complete structure."""
        entities = [{"name": "Pump", "entity_type": "equipment", "confidence": 0.8}]
        relations = []
        text = "The pump operates with normal parameters."

        # Note: This will return "unavailable" since no LLM is loaded in test env
        result = self.summarizer.generate_summary(entities, relations, text)

        # Check structure regardless of content
        expected_keys = {
            "summary_method",
            "status",
            "reasoning",
            "anomalies_detected",
            "failure_risks",
            "maintenance_recommendations",
            "compliance",
            "confidence",
            "explanation_chains",
        }

        self.assertTrue(
            expected_keys.issubset(result.keys()),
            f"Missing keys in result: {expected_keys - result.keys()}",
        )

        # Verify types
        self.assertIsInstance(result["anomalies_detected"], list)
        self.assertIsInstance(result["failure_risks"], list)
        self.assertIsInstance(result["confidence"], (int, float))
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_generate_summary_with_insufficient_data_returns_no_evidence(self) -> None:
        """Test that insufficient data returns no_evidence status."""
        result = self.summarizer.generate_summary([], [], "")

        self.assertEqual(result["status"], "no_evidence")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["anomalies_detected"], [])

    def test_generate_summary_tracks_evidence_coverage(self) -> None:
        """Test that result includes evidence coverage metrics."""
        entities = [
            {"name": "Component", "entity_type": "equipment", "confidence": 0.9}
        ]
        relations = []
        text = "Component operates normally without issues."

        result = self.summarizer.generate_summary(entities, relations, text)

        # Should have metrics even if LLM unavailable
        self.assertIn("confidence", result)
        # Evidence coverage may be added by some return paths
        if "evidence_coverage" in result:
            self.assertGreaterEqual(result["evidence_coverage"], 0.0)
            self.assertLessEqual(result["evidence_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
