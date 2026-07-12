from pathlib import Path

import torch

from app import config


def test_env_file_loader_uses_repo_root() -> None:
    env_file = Path(__file__).resolve().parents[1] / ".env.local"
    assert env_file.exists()
    loaded = config.load_env_file(env_file)
    assert loaded.get("USE_NATIVE_BACKENDS", "").lower() == "true"


def test_device_properties_fall_back_to_cpu_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(config.settings, "execution_mode", "gpu")
    monkeypatch.setattr(config.settings, "_device_detection", "AUTO")
    monkeypatch.setattr(config.settings, "_device_segmentation", "AUTO")
    monkeypatch.setattr(config.settings, "_device_extraction", "AUTO")
    monkeypatch.setattr(config.settings, "_device_embedding", "AUTO")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert config.settings.device_for_detection == "cpu"
    assert config.settings.device_for_segmentation == "cpu"
    assert config.settings.device_for_extraction == "cpu"
    assert config.settings.device_for_embedding == "cpu"
