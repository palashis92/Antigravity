"""Unit tests for configuration loading and validation."""

import os
from pathlib import Path
from lumi.config.settings import load_settings, LumiSettings


def test_default_settings_load() -> None:
    settings = load_settings()
    assert isinstance(settings, LumiSettings)
    assert settings.app.name == "LUMI"
    assert settings.app.owner_name == "Palash"
    assert settings.memory.wal_mode is True
    assert settings.display.width == 240
    assert settings.display.height == 240
    assert settings.hardware.servo_driver_model == "PCA9685"


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("LUMI_ENV", "production")
    monkeypatch.setenv("LUMI_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LUMI_DATA_DIR", "custom_data")

    settings = load_settings()
    assert settings.app.environment == "production"
    assert settings.app.log_level == "DEBUG"
    assert settings.app.data_dir == "custom_data"
