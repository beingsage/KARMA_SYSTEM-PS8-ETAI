import subprocess
import sys
import types
from pathlib import Path

import scripts.ensure_dependencies as ensure_dependencies
from app.pipeline.compat import ensure_pyarrow_compat


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
    monkeypatch.setattr(
        ensure_dependencies.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )

    result = ensure_dependencies.ensure_dependencies(req_file, python_executable="python")

    assert result == []


def test_ensure_dependencies_continues_after_failed_install(tmp_path, monkeypatch):
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("fakepkg==1.0\nanotherpkg==2.0\n")

    call_count = {"count": 0}

    def fake_find_missing_dependencies(required_packages=None, *, include_optional=False):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return ["fakepkg==1.0", "anotherpkg==2.0"]
        return ["fakepkg==1.0"]

    monkeypatch.setattr(
        ensure_dependencies,
        "find_missing_dependencies",
        fake_find_missing_dependencies,
    )

    def fake_run(cmd, **kwargs):
        if cmd[-1] == "fakepkg==1.0":
            return subprocess.CompletedProcess(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ensure_dependencies.subprocess, "run", fake_run)

    result = ensure_dependencies.ensure_dependencies(req_file, python_executable="python")

    assert result == ["fakepkg==1.0"]


def test_main_returns_nonzero_when_dependency_install_fails(tmp_path, monkeypatch):
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("fakepkg==1.0\n")

    monkeypatch.setattr(
        ensure_dependencies,
        "parse_requirements_file",
        lambda path: ["fakepkg==1.0"],
    )
    monkeypatch.setattr(
        ensure_dependencies,
        "find_missing_dependencies",
        lambda required_packages, include_optional=False: ["fakepkg==1.0"],
    )
    monkeypatch.setattr(
        ensure_dependencies,
        "ensure_dependencies",
        lambda requirements_path, python_executable=None, include_optional=False: ["fakepkg==1.0"],
    )
    monkeypatch.setattr(sys, "argv", ["ensure_dependencies.py", "--requirements", str(req_file)])

    result = ensure_dependencies.main()

    assert result == 1


def test_find_missing_dependencies_accepts_compatible_pyarrow(monkeypatch):
    fake_pyarrow = types.ModuleType("pyarrow")
    fake_pyarrow.ExtensionType = object
    fake_pyarrow.__version__ = "25.0.0"

    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)

    assert ensure_pyarrow_compat() is True
    assert hasattr(fake_pyarrow, "PyExtensionType")
    assert ensure_dependencies.find_missing_dependencies(["pyarrow==15.0.2"]) == []
