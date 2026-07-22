from pathlib import Path

from app.pipeline.engine_v2 import IndustrialGraphPipeline
from app.pipeline.model_helpers import PandidRCNNDetector


class DummyPandidDetector:
    def detect(self, images):
        return [{"page": 1, "label": "pandid_rcnn_component", "confidence": 0.8, "bbox": [0, 0, 10, 10], "source": "pandid_rcnn"}]


def test_pandid_rcnn_detector_resolves_training_artifact():
    artifact_path = Path("/media/sagesujal/DEV1/bytes/structured/training/pandid_rcnn (1).t7")
    detector = PandidRCNNDetector(artifact_path=artifact_path)

    assert detector.artifact_path == artifact_path
    assert detector.artifact_path.exists()
    assert detector.backend in {"pandid_rcnn_t7", "fallback"}


def test_pipeline_detect_pandid_rcnn_objects_uses_detector():
    pipeline = IndustrialGraphPipeline.__new__(IndustrialGraphPipeline)
    pipeline.pandid_rcnn_detector = DummyPandidDetector()

    result = pipeline._detect_pandid_rcnn_objects(pdf_bytes=b"", text="")

    assert result["source"] == "pandid_rcnn"
    assert result["count"] == 1
    assert result["detections"][0]["label"] == "pandid_rcnn_component"
