import sys
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
REALTIME_DIR = ROOT / "agents" / "RealtimeVoiceChat" / "code"


def _load_realtime_voice_server_module():
    if str(REALTIME_DIR) not in sys.path:
        sys.path.insert(0, str(REALTIME_DIR))
    try:
        import server as realtime_server
    except Exception:
        return None
    return realtime_server


def get_realtime_voice_app():
    module = _load_realtime_voice_server_module()
    return getattr(module, "app", None)


def _get_copilot_response_payload(question: str, job_id: str | None = None, use_web: bool = True) -> dict[str, Any]:
    from app.main import _generate_copilot_response

    return _generate_copilot_response(question, job_id=job_id, use_web=use_web)


def build_copilot_response_provider() -> Callable[[str, Optional[str], bool], str]:
    def provider(question: str, job_id: str | None = None, use_web: bool = True) -> str:
        payload = _get_copilot_response_payload(question, job_id=job_id, use_web=use_web)
        return str(payload.get("answer") or "")

    return provider


def register_realtime_voice_routes(app):
    realtime_app = get_realtime_voice_app()
    if realtime_app is None:
        return False

    for route in getattr(realtime_app, "routes", []):
        if getattr(route, "path", None) == "/ws":
            app.add_websocket_route("/voice-ws", route.endpoint)
            return True
    return False
