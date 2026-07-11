import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ensure_dependencies import find_missing_dependencies


def test_find_missing_dependencies_reports_missing_modules() -> None:
    missing = find_missing_dependencies(["definitely_missing_pkg_123", "another_missing_pkg_456"])

    assert missing == ["definitely_missing_pkg_123", "another_missing_pkg_456"]


def test_optional_dependency_packages_are_skipped_by_default() -> None:
    missing = find_missing_dependencies(["bertopic", "hdbscan"])

    assert missing == []
