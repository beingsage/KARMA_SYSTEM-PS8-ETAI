from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_voice_assistant_endpoint_exists() -> None:
    client = TestClient(app)
    response = client.get('/health')
    assert response.status_code == 200

    html_path = Path(__file__).resolve().parents[1] / 'app' / 'frontend' / 'copilot.html'
    html = html_path.read_text(encoding='utf-8').lower()
    assert 'api/v1/voice/assistant' in html
