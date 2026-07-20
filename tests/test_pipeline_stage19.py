import unittest

from app.pipeline.engine_v2 import IndustrialGraphPipeline


class PipelineStage19Tests(unittest.TestCase):
    def test_vision_language_fallback_uses_ocr_proxy_when_model_unavailable(self) -> None:
        pipeline = IndustrialGraphPipeline.__new__(IndustrialGraphPipeline)
        pipeline.copilot_agent = None
        pipeline.fallback_usage = {"lexical_fallback": 0, "vl_fallback": 0}

        ocr_result = {
            "text": "Figure 1: Pump installation steps and technical overview. The diagram shows the pump inlet and outlet.",
            "layout": [],
            "tables": [],
        }

        result = pipeline._vision_language_understanding(
            entities=[],
            relations=[],
            text=ocr_result["text"],
            layout=[],
            tables=[],
            reading_order=[],
            pdf_bytes=None,
            ocr_result=ocr_result,
        )

        self.assertEqual(result["status"], "vl_fallback")
        self.assertEqual(result["images_processed"], 1)
        self.assertTrue(result["captions"])
        self.assertEqual(result["captions"][0]["method"], "ocr_proxy")
        self.assertIn("Pump installation", result["captions"][0]["caption"])
        self.assertEqual(result["telemetry"]["fallback_usage"]["vl_fallback"], 1)

    def test_vision_language_pipeline_tried_model_flag_when_transformers_missing(self) -> None:
        pipeline = IndustrialGraphPipeline.__new__(IndustrialGraphPipeline)
        pipeline.copilot_agent = None
        pipeline.fallback_usage = {"lexical_fallback": 0, "vl_fallback": 0}

        result = pipeline._vision_language_understanding(
            entities=[],
            relations=[],
            text="No OCR text available.",
            layout=[],
            tables=[],
            reading_order=[],
            pdf_bytes=None,
            ocr_result={"text": ""},
        )

        self.assertFalse(result["method_tried_model"])
        self.assertEqual(result["status"], "vl_fallback")
        self.assertIn("caption", result["captions"][0])


if __name__ == "__main__":
    unittest.main()
