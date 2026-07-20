import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import settings


class NpEncoder(json.JSONEncoder):
    """JSON encoder that converts common numpy and pandas types to native Python types."""
    def default(self, obj):
        try:
            import numpy as _np
            import pandas as _pd
            import datetime
            import decimal

            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, (_np.ndarray,)):
                return obj.tolist()
            if isinstance(obj, (_np.bool_,)):
                return bool(obj)
            if isinstance(obj, (_np.generic,)):
                return obj.item()
            if isinstance(obj, (_pd.Timestamp, _pd.Timedelta, _pd.Period)):
                return str(obj)
            if isinstance(obj, _pd.Series):
                return obj.tolist()
            if isinstance(obj, _pd.Index):
                return obj.tolist()
            if isinstance(obj, decimal.Decimal):
                return float(obj)
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            if isinstance(obj, datetime.date):
                return obj.isoformat()
            if isinstance(obj, datetime.timedelta):
                return obj.total_seconds()
        except Exception:
            # numpy/pandas may not be installed or import may fail; fall through
            pass

        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="ignore")
        if hasattr(obj, "tolist"):
            try:
                return obj.tolist()
            except Exception:
                pass

        if hasattr(obj, "__str__"):
            return str(obj)

        return super().default(obj)


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
    _job_path(job_id).write_text(json.dumps(payload, indent=2, cls=NpEncoder), encoding="utf-8")
    return payload


def update_job(job_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    payload = load_job(job_id)
    payload.update(updates)
    _job_path(job_id).write_text(json.dumps(payload, indent=2, cls=NpEncoder), encoding="utf-8")
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
