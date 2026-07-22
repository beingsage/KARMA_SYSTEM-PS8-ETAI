from pathlib import Path


def test_copilot_page_exposes_voice_controls() -> None:
    html_path = Path(__file__).resolve().parents[1] / "app" / "frontend" / "copilot.html"
    html = html_path.read_text(encoding="utf-8").lower()

    assert "voice mode" in html or "mic" in html
    assert "listen" in html or "speak" in html
    assert "speechrecognition" in html or "speechsynthesis" in html


def test_copilot_page_exposes_advanced_dashboard_sections() -> None:
    html_path = Path(__file__).resolve().parents[1] / "app" / "frontend" / "copilot.html"
    html = html_path.read_text(encoding="utf-8")

    assert "advanced runtime view" in html.lower()
    assert "neo4j-style graph" in html.lower()
    assert "root cause analysis" in html.lower()
    assert "anomaly detection" in html.lower()
    assert "rul prediction" in html.lower()
    assert "vector search" in html.lower()
    assert "reasoning layers" in html.lower()
    assert "confidence bar" in html.lower() or "confidence-bar" in html.lower()
