from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import pandas as pd

from src.config.evaluation import EvaluationConfig, load_evaluation_config

SUPPORTED_WINDOW_UNITS = {"D", "W", "M", "Y"}


class EvaluationSplitConfigError(ValueError):
    """Raised when evaluation split configuration is invalid."""


@dataclass(frozen=True)
class EvaluationSplit:
    """Serializable metadata describing one deterministic evaluation split."""

    split_id: str
    mode: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    def to_dict(self) -> dict[str, str]:
        """Return the split as a plain dictionary for logging or serialization."""

        return {
            "split_id": self.split_id,
            "mode": self.mode,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


def generate_evaluation_splits(config: EvaluationConfig | dict[str, Any]) -> list[EvaluationSplit]:
    """Generate deterministic evaluation splits using half-open date ranges.

    All windows use inclusive `start` and exclusive `end` semantics:
    `[start, end)`. A split is valid only when `test_end` is less than or equal
    to the configured overall evaluation `end`.
    """

    normalized = _coerce_config(config)
    if normalized.mode == "fixed":
        return [_build_fixed_split(normalized)]

    return _build_walk_forward_splits(normalized)


def load_and_generate_evaluation_splits() -> list[EvaluationSplit]:
    """Load `configs/evaluation.yml` and generate evaluation splits."""

    return generate_evaluation_splits(load_evaluation_config())


def _coerce_config(config: EvaluationConfig | dict[str, Any]) -> EvaluationConfig:
    if isinstance(config, EvaluationConfig):
        return config
    if isinstance(config, dict):
        return EvaluationConfig.from_mapping(config)
    raise TypeError("Evaluation config must be an EvaluationConfig or a dictionary.")


def _build_fixed_split(config: EvaluationConfig) -> EvaluationSplit:
    required = {
        "train_start": config.train_start,
        "train_end": config.train_end,
        "test_start": config.test_start,
        "test_end": config.test_end,
    }
    _require_fields(required, "Fixed evaluation config")

    train_start = _parse_date(config.train_start, "train_start")
    train_end = _parse_date(config.train_end, "train_end")
    test_start = _parse_date(config.test_start, "test_start")
    test_end = _parse_date(config.test_end, "test_end")

    if train_start >= train_end:
        raise EvaluationSplitConfigError("Fixed split requires train_start to be before train_end.")
    if test_start >= test_end:
        raise EvaluationSplitConfigError("Fixed split requires test_start to be before test_end.")
    if train_end > test_start:
        raise EvaluationSplitConfigError(
            "Fixed split windows must not overlap: train_end must be less than or equal to test_start."
        )

    return _make_split(
        split_id="fixed_0000",
        mode=config.mode,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )


def _build_walk_forward_splits(config: EvaluationConfig) -> list[EvaluationSplit]:
    required = {
        "start": config.start,
        "end": config.end,
        "train_window": config.train_window,
        "test_window": config.test_window,
        "step": config.step,
    }
    _require_fields(required, f"{config.mode.title()} evaluation config")

    overall_start = _parse_date(config.start, "start")
    overall_end = _parse_date(config.end, "end")
    if overall_start >= overall_end:
        raise EvaluationSplitConfigError("Evaluation start must be before evaluation end.")

    train_window = _parse_window(config.train_window, "train_window")
    test_window = _parse_window(config.test_window, "test_window")
    step = _parse_window(config.step, "step")

    initial_train_start = overall_start
    initial_train_end = overall_start + train_window
    initial_test_start = initial_train_end
    initial_test_end = initial_test_start + test_window

    if initial_test_end > overall_end:
        raise EvaluationSplitConfigError(
            "Evaluation range does not contain enough data to form at least one train/test split."
        )

    splits: list[EvaluationSplit] = []
    split_index = 0
    train_start = initial_train_start
    train_end = initial_train_end
    test_start = initial_test_start
    test_end = initial_test_end

    while test_end <= overall_end:
        splits.append(
            _make_split(
                split_id=f"{config.mode}_{split_index:04d}",
                mode=config.mode,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        split_index += 1

        if config.mode == "rolling":
            train_start = train_start + step
        train_end = train_end + step
        test_start = test_start + step
        test_end = test_end + step

    if not splits:
        raise EvaluationSplitConfigError(
            "Evaluation range does not contain enough data to form at least one train/test split."
        )

    return splits


def _make_split(
    *,
    split_id: str,
    mode: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> EvaluationSplit:
    return EvaluationSplit(
        split_id=split_id,
        mode=mode,
        train_start=_format_date(train_start),
        train_end=_format_date(train_end),
        test_start=_format_date(test_start),
        test_end=_format_date(test_end),
    )


def _require_fields(fields: dict[str, Any], label: str) -> None:
    missing = [name for name, value in fields.items() if value is None]
    if missing:
        joined = ", ".join(sorted(missing))
        raise EvaluationSplitConfigError(f"{label} is missing required fields: {joined}.")


def _parse_date(value: str | None, field_name: str) -> pd.Timestamp:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationSplitConfigError(f"{field_name} must be a non-empty YYYY-MM-DD string.")

    try:
        timestamp = pd.Timestamp(value)
    except ValueError as exc:
        raise EvaluationSplitConfigError(f"{field_name} must be a valid YYYY-MM-DD date.") from exc

    if timestamp.tzinfo is not None:
        raise EvaluationSplitConfigError(f"{field_name} must not include a timezone.")

    return timestamp.normalize()


def _parse_window(value: str | None, field_name: str) -> pd.DateOffset:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationSplitConfigError(
            f"{field_name} must be a non-empty duration string such as '30D', '4W', '3M', or '1Y'."
        )

    compact = value.replace(" ", "")
    match = re.fullmatch(r"(?P<count>\d+)(?P<unit>[DWMYdwmy])", compact)
    if match is None:
        raise EvaluationSplitConfigError(
            f"{field_name} must use a positive duration like '30D', '4W', '3M', or '1Y'."
        )

    count = int(match.group("count"))
    unit = match.group("unit").upper()
    if unit not in SUPPORTED_WINDOW_UNITS:
        supported = ", ".join(sorted(SUPPORTED_WINDOW_UNITS))
        raise EvaluationSplitConfigError(f"{field_name} must use one of the supported units: {supported}.")
    if count <= 0:
        raise EvaluationSplitConfigError(f"{field_name} must be greater than zero.")

    if unit == "D":
        return pd.DateOffset(days=count)
    if unit == "W":
        return pd.DateOffset(weeks=count)
    if unit == "M":
        return pd.DateOffset(months=count)
    return pd.DateOffset(years=count)


def _format_date(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")
