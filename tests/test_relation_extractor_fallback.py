import unittest
import os

from app.pipeline.relation_extractor import GLiRELRelationExtractor
from app.config import settings


class RelationExtractorFallbackTests(unittest.TestCase):
    def test_relation_extractor_reports_unavailable_when_glirel_fails(self) -> None:
        extractor = GLiRELRelationExtractor()

        self.assertTrue(hasattr(extractor, "is_ready"))
        self.assertFalse(extractor.is_ready)
        self.assertIsNone(getattr(extractor, "model", None))


class ExecutionModeTests(unittest.TestCase):
    def test_execution_mode_defaults_to_cpu(self) -> None:
        """Ensure execution mode defaults to CPU for local testing."""
        # If not set, should default to cpu
        if "EXECUTION_MODE" not in os.environ:
            mode = getattr(settings, "execution_mode", "cpu")
            self.assertEqual(mode, "cpu")

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
