import unittest

from app.pipeline.engine_v2 import IndustrialGraphPipeline


class PipelineStage3Tests(unittest.TestCase):
    def test_ocr_fallback_layout_builds_headings(self) -> None:
        pipeline = IndustrialGraphPipeline.__new__(IndustrialGraphPipeline)
        ocr_result = {
            "text": "## Hydro MPC\n\nInstallation and operating instructions\n\n## CONTENTS\n\nSection 1: Safety",
            "layout": [],
            "tables": [],
        }

        fallback_layout = pipeline._build_ocr_layout_fallback(ocr_result)

        self.assertTrue(fallback_layout)
        self.assertEqual(fallback_layout[0]["label"], "heading")
        self.assertIn("Hydro MPC", fallback_layout[0]["text"])

    def test_structural_stage3_summary_uses_ocr_fallback(self) -> None:
        pipeline = IndustrialGraphPipeline.__new__(IndustrialGraphPipeline)
        ocr_result = {
            "text": "## Hydro MPC\n\nInstallation and operating instructions\n\n## CONTENTS",
            "layout": [],
            "tables": [],
        }

        summary = pipeline._build_structural_stage3_summary(ocr_result, b"pdf")

        self.assertEqual(summary["stage"], "surya_layout_understanding")
        self.assertGreaterEqual(summary["detected_objects"], 1)
        self.assertIn("Hydro MPC", " ".join(summary["headings"]))


if __name__ == "__main__":
    unittest.main()
