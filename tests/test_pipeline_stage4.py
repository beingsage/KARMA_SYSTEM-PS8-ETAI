import unittest

from app.pipeline.engine_v2 import IndustrialGraphPipeline


class PipelineStage4Tests(unittest.TestCase):
    def test_grounding_prompt_uses_stage3_headings(self) -> None:
        pipeline = IndustrialGraphPipeline.__new__(IndustrialGraphPipeline)
        ocr_result = {"text": "Installation and operating instructions"}
        stage3_summary = {
            "full_output": {
                "headings": ["Hydro MPC", "CONTENTS", "Safety"]
            }
        }

        prompt = pipeline._build_groundingdino_prompt(ocr_result, stage3_summary)

        self.assertIn("Hydro MPC", prompt)
        self.assertIn("CONTENTS", prompt)
        self.assertIn("Safety", prompt)


if __name__ == "__main__":
    unittest.main()
