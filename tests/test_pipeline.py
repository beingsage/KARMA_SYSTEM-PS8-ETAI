import asyncio
import unittest
from io import BytesIO

from PIL import Image, ImageDraw

from app.pipeline.engine_v2 import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_sample_pdf_generates_entities_and_relations(self) -> None:
        text = "Industrial pump controls a motor and monitors a pressure sensor."
        image = Image.new("RGB", (800, 200), color="white")
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), text, fill="black")

        buffer = BytesIO()
        image.save(buffer, format="PDF")
        sample_pdf = buffer.getvalue()

        result = asyncio.run(run_pipeline("sample.pdf", sample_pdf))

        self.assertEqual(result["status"], "completed")
        self.assertGreaterEqual(len(result["entities"]), 1)
        self.assertGreaterEqual(len(result["relations"]), 1)
        self.assertIn("pipeline_metadata", result)
        self.assertIn("docling_surya_ocr", result["pipeline_metadata"]["stages"])
        self.assertIn("doclayout_yolo_analysis", result["pipeline_metadata"]["stages"])
        self.assertIn("table_transformer_extraction", result["pipeline_metadata"]["stages"])
        self.assertIn("yolo_pid_detector", result["pipeline_metadata"]["stages"])
        self.assertIn("entity_linking", result["pipeline_metadata"]["stages"])
        self.assertIn(result["pipeline_metadata"]["model_mode"], ["best-model-stack", "partial-stack", "unavailable"])


if __name__ == "__main__":
    unittest.main()
