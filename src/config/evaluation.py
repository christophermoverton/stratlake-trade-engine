from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.config.settings import load_yaml_config

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_CONFIG = REPO_ROOT / "configs" / "evaluation.yml"
SUPPORTED_EVALUATION_MODES = {"fixed", "rolling", "expanding"}


@dataclass(frozen=True)
class EvaluationConfig:
    """Typed evaluation split configuration loaded from YAML."""

    mode: str
    timeframe: str
    start: str | None = None
    end: str | None = None
    train_window: str | None = None
    test_window: str | None = None
    step: str | None = None
    train_start: str | None = None
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "EvaluationConfig":
        """Build a config object from an `evaluation` YAML mapping."""

        mode = payload.get("mode")
        if not isinstance(mode, str) or mode not in SUPPORTED_EVALUATION_MODES:
            supported = ", ".join(sorted(SUPPORTED_EVALUATION_MODES))
            raise ValueError(f"Evaluation mode must be one of: {supported}.")

        timeframe = payload.get("timeframe", "1d")
        if not isinstance(timeframe, str) or not timeframe.strip():
            raise ValueError("Evaluation timeframe must be a non-empty string when provided.")

        return cls(
            mode=mode,
            timeframe=timeframe,
            start=_normalize_date_field(payload.get("start")),
            end=_normalize_date_field(payload.get("end")),
            train_window=payload.get("train_window"),
            test_window=payload.get("test_window"),
            step=payload.get("step"),
            train_start=_normalize_date_field(payload.get("train_start")),
            train_end=_normalize_date_field(payload.get("train_end")),
            test_start=_normalize_date_field(payload.get("test_start")),
            test_end=_normalize_date_field(payload.get("test_end")),
        )


def load_evaluation_config(path: Path = EVALUATION_CONFIG) -> EvaluationConfig:
    """Load the evaluation split configuration from YAML."""

    payload = load_yaml_config(path)
    if not isinstance(payload, dict):
        raise ValueError("Evaluation configuration file must contain a top-level mapping.")

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("Evaluation configuration must define an 'evaluation' mapping.")

    return EvaluationConfig.from_mapping(evaluation)


def _normalize_date_field(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value
