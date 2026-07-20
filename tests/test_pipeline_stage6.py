from app.pipeline.engine_v2 import IndustrialGraphPipeline


def test_stage6_summary_ratings_and_synthesis():
    pipeline = IndustrialGraphPipeline.__new__(IndustrialGraphPipeline)

    ocr_result = {
        "text": "Hydro MPC System\n\nSafety Warning\n\nInstallation Guide\n",
        "layout": [{"page": 1, "label": "heading", "text": "Hydro MPC System"}],
    }
    stage3_result = {
        "layout": [{"page": 1, "label": "heading", "text": "Hydro MPC System"}],
        "headings": ["Hydro MPC System"],
    }
    stage4_result = {
        "table_transformer_detections": [{"page": 1, "label": "table", "confidence": 0.92}],
    }
    stage5_result = {
        "detections": [{"label": "valve", "confidence": 0.9}],
        "count": 1,
    }

    summary = pipeline._build_structural_stage6_summary(
        ocr_result=ocr_result,
        pdf_bytes=b"%PDF-1.4",
        stage3_result=stage3_result,
        stage4_result=stage4_result,
        stage5_result=stage5_result,
    )

    assert summary["status"] == "completed"
    assert summary["stage"] == "cross_stage_synthesis"
    assert summary["overall_quality_score"] >= 0
    assert summary["overall_quality_score"] <= 10
    assert summary["overall_quality_label"] in {"good", "very good", "excellent"}
