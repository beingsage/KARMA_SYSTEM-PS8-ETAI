import unittest
import os
import sys
import types
from unittest.mock import patch

from app.pipeline.relation_extractor import GLiRELRelationExtractor
from app.config import settings


class RelationExtractorFallbackTests(unittest.TestCase):
    def test_relation_extractor_uses_heuristic_fallback_when_glirel_fails(self) -> None:
        fake_glirel_module = types.ModuleType("glirel")

        class FakeGLiREL:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("forced failure")

        fake_glirel_module.GLiREL = FakeGLiREL

        with patch.dict(sys.modules, {"glirel": fake_glirel_module}):
            extractor = GLiRELRelationExtractor()

        self.assertTrue(hasattr(extractor, "is_ready"))
        self.assertTrue(extractor.is_ready)
        self.assertIsNone(getattr(extractor, "model", None))

        text = "Pump A is connected to Valve B."
        entities = [
            {"name": "Pump A", "entity_type": "equipment", "canonical_name": "pump_a", "start": 0, "end": 6},
            {"name": "Valve B", "entity_type": "equipment", "canonical_name": "valve_b", "start": 21, "end": 28},
        ]
        relations = extractor.extract(text, entities)

        self.assertGreaterEqual(len(relations), 1)
        self.assertEqual(relations[0]["source"], "Pump A")
        self.assertEqual(relations[0]["target"], "Valve B")


class ExecutionModeTests(unittest.TestCase):
    def test_execution_mode_defaults_to_cpu(self) -> None:
        """Ensure execution mode follows the loaded environment configuration."""
        expected = os.environ.get("EXECUTION_MODE")
        if expected is None:
            from app import config as app_config

            expected = app_config.load_env_file().get("EXECUTION_MODE", "cpu")

        mode = getattr(settings, "execution_mode", "cpu")
        self.assertEqual(mode.lower(), expected.lower())

    def test_device_configuration_cpu_mode(self) -> None:
        """Verify device placement in CPU mode."""
        # Save current mode
        original_mode = getattr(settings, "execution_mode", "cpu")
        
        # Force CPU mode
        if original_mode == "cpu":
            self.assertEqual(settings.device_for_detection, "cpu")
            self.assertEqual(settings.device_for_extraction, "cpu")
            self.assertEqual(settings.device_for_embedding, "cpu")

    def test_timeout_settings_available(self) -> None:
        """Verify timeout configuration is available."""
        self.assertTrue(hasattr(settings, "table_extraction_timeout"))
        self.assertTrue(hasattr(settings, "ocr_processing_timeout"))
        self.assertTrue(hasattr(settings, "pdf_render_timeout"))


if __name__ == "__main__":
    unittest.main()
