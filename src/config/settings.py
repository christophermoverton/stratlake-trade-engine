from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class Settings:
    marketlake_root: Path
    features_root: Path
    artifacts_root: Path
    duckdb_path: str
    log_level: str
    default_timezone: str
    paths_config: Dict[str, Any]
    universe_config: Dict[str, Any]
    features_config: Dict[str, Any]

    @staticmethod
    def load(config_dir: str = "configs", *, load_env: bool = True) -> "Settings":
        """
        load_env=True  -> load from .env (normal usage)
        load_env=False -> do not load .env (tests / controlled environments)
        """
        if load_env:
            # Do not override variables already set by the environment/test runner
            load_dotenv(override=False)

        config_path = Path(config_dir)

        paths_file = config_path / "paths.yml"
        paths_config = load_yaml_config(paths_file) if paths_file.exists() else {}
        universe_config = load_yaml_config(config_path / "universe.yml")
        features_config = load_yaml_config(config_path / "features.yml")

        marketlake_root_raw = os.getenv("MARKETLAKE_ROOT") or paths_config.get("marketlake_root", "")
        if not marketlake_root_raw:
            raise ValueError("MARKETLAKE_ROOT must be set in .env or configs/paths.yml")

        marketlake_root = Path(marketlake_root_raw)

        features_root = Path(os.getenv("FEATURES_ROOT") or paths_config.get("features_root", "data"))
        artifacts_root = Path(os.getenv("ARTIFACTS_ROOT") or paths_config.get("artifacts_root", "artifacts"))

        duckdb_path = os.getenv("DUCKDB_PATH", ":memory:")
        log_level = os.getenv("LOG_LEVEL", "INFO")
        default_timezone = os.getenv("DEFAULT_TIMEZONE", "UTC")

        return Settings(
            marketlake_root=marketlake_root,
            features_root=features_root,
            artifacts_root=artifacts_root,
            duckdb_path=duckdb_path,
            log_level=log_level,
            default_timezone=default_timezone,
            paths_config=paths_config,
            universe_config=universe_config,
            features_config=features_config,
        )
