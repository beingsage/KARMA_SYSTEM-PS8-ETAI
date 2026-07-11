import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import settings


def _job_path(job_id: str) -> Path:
    return settings.jobs_dir / f"{job_id}.json"


def create_job(uploaded_filename: Optional[str]) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "status": "queued",
        "uploaded_filename": uploaded_filename,
        "message": "Queued for processing",
    }
    _job_path(job_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def update_job(job_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    payload = load_job(job_id)
    payload.update(updates)
    _job_path(job_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_job(job_id: str) -> Dict[str, Any]:
    path = _job_path(job_id)
    if not path.exists():
        raise FileNotFoundError(job_id)
    return json.loads(path.read_text(encoding="utf-8"))


def list_jobs() -> list[Dict[str, Any]]:
    jobs = []
    for path in sorted(settings.jobs_dir.glob("*.json")):
        jobs.append(json.loads(path.read_text(encoding="utf-8")))
    return jobs
