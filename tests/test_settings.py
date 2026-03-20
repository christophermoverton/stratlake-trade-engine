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


def test_settings_load_without_paths_yaml_uses_env_and_defaults(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        write_yaml(config_dir / "universe.yml", {})
        write_yaml(config_dir / "features.yml", {})

        monkeypatch.setenv("MARKETLAKE_ROOT", "/env/marketlake")
        monkeypatch.setenv("DUCKDB_PATH", "/env/db.duckdb")
        monkeypatch.setenv("FEATURES_ROOT", "env_features")
        monkeypatch.setenv("ARTIFACTS_ROOT", "env_artifacts")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("DEFAULT_TIMEZONE", "America/Chicago")

        settings = Settings.load(config_dir=str(config_dir), load_env=False)

        assert settings.marketlake_root == Path("/env/marketlake")
        assert settings.features_root == Path("env_features")
        assert settings.artifacts_root == Path("env_artifacts")
        assert settings.duckdb_path == "/env/db.duckdb"
        assert settings.log_level == "DEBUG"
        assert settings.default_timezone == "America/Chicago"
        assert settings.paths_config == {}
