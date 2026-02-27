import os
from pathlib import Path
import tempfile
import yaml

import pytest

from src.config.settings import Settings


def write_yaml(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def test_settings_load_with_env_override(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        write_yaml(config_dir / "paths.yml", {
            "marketlake_root": "/yaml/path",
            "features_root": "yaml_data",
            "artifacts_root": "yaml_artifacts",
        })
        write_yaml(config_dir / "universe.yml", {})
        write_yaml(config_dir / "features.yml", {})

        monkeypatch.setenv("MARKETLAKE_ROOT", "/env/path")
        monkeypatch.setenv("FEATURES_ROOT", "env_data")

        settings = Settings.load(config_dir=str(config_dir))

        assert settings.marketlake_root == Path("/env/path")
        assert settings.features_root == Path("env_data")


def test_settings_fails_without_marketlake_root(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        write_yaml(config_dir / "paths.yml", {})
        write_yaml(config_dir / "universe.yml", {})
        write_yaml(config_dir / "features.yml", {})

        monkeypatch.delenv("MARKETLAKE_ROOT", raising=False)

        with pytest.raises(ValueError):
            Settings.load(config_dir=str(config_dir), load_env=False)