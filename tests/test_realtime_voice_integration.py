from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app
from app import voice_runtime


def test_realtime_voice_route_serves_frontend_assets() -> None:
    client = TestClient(app)
    html_response = client.get('/voice-chat')
    assert html_response.status_code == 200
    html = html_response.text.lower()
    assert 'real-time voice chat' in html or 'voice chat' in html

    js_response = client.get('/static/app.js')
    assert js_response.status_code == 200
    assert 'websocket' in js_response.text.lower() or 'navigator' in js_response.text.lower()


def test_voice_runtime_builds_copilot_bridge(monkeypatch) -> None:
    calls = {}

    def fake_generate(text: str, job_id: str | None = None, use_web: bool = True) -> dict[str, object]:
        calls["text"] = text
        calls["job_id"] = job_id
        calls["use_web"] = use_web
        return {"answer": f"copilot:{text}"}

    monkeypatch.setattr(voice_runtime, "_get_copilot_response_payload", fake_generate)

    provider = voice_runtime.build_copilot_response_provider()
    assert provider("hello there", job_id="job-123", use_web=False) == "copilot:hello there"
    assert calls == {"text": "hello there", "job_id": "job-123", "use_web": False}
