from pathlib import Path

from app import config


def test_env_file_loader_uses_repo_root() -> None:
    env_file = Path(__file__).resolve().parents[1] / ".env.local"
    assert env_file.exists()
    loaded = config.load_env_file(env_file)
    assert loaded.get("USE_NATIVE_BACKENDS", "").lower() == "true"
