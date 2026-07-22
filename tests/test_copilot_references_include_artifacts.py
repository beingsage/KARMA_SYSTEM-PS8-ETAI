from __future__ import annotations

from app.main import _build_references


def test_build_references_includes_structured_artifact_evidence():
    agent_result = {
        "records": [],
        "structured_claims": [
            {
                "id": "claim:1",
                "type": "recommendation",
                "text": "Inspect pump seal",
                "description": "Seal inspection is recommended after overheating.",
                "provenance": "page:image:1",
                "verification_score": 0.91,
                "confidence": 0.8,
            }
        ],
        "source_artifacts": [
            {
                "artifact_id": "page-image-1",
                "kind": "page_image",
                "page": 1,
                "mime_type": "image/png",
                "caption": "Pump seal image",
                "content": "data:image/png;base64,abc123",
            }
        ],
    }

    references = _build_references(agent_result, [])
    assert any(item.get("source_type") == "artifact" for item in references)
    image_ref = next(item for item in references if item.get("source_type") == "artifact")
    assert image_ref["title"] == "Pump seal image"
    assert image_ref["image_data"] == "data:image/png;base64,abc123"
