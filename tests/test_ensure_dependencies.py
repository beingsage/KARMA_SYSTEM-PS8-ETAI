from pathlib import Path

import scripts.ensure_dependencies as ensure_dependencies


def test_ensure_dependencies_handles_missing_packages(tmp_path, monkeypatch):
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("fakepkg==1.0\n")

    call_count = {"count": 0}

    def fake_find_missing_dependencies(required_packages=None, *, include_optional=False):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return ["fakepkg==1.0"]
        return []

    monkeypatch.setattr(
        ensure_dependencies,
        "find_missing_dependencies",
        fake_find_missing_dependencies,
    )
    monkeypatch.setattr(ensure_dependencies.subprocess, "check_call", lambda *args, **kwargs: None)

    result = ensure_dependencies.ensure_dependencies(req_file, python_executable="python")

    assert result == []
