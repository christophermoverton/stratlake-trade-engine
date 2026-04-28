from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class MarketSimulationConfigError(ValueError):
    """Raised when a market simulation framework configuration is malformed."""


def required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise MarketSimulationConfigError(
            f"Market simulation config field '{field_name}' must be a non-empty string."
        )
    normalized = value.strip()
    if not normalized:
        raise MarketSimulationConfigError(
            f"Market simulation config field '{field_name}' must be a non-empty string."
        )
    return normalized


def optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return required_string(value, field_name)


def bool_value(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise MarketSimulationConfigError(f"Market simulation config field '{field_name}' must be boolean.")
    return value


def nonnegative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MarketSimulationConfigError(
            f"Market simulation config field '{field_name}' must be a non-negative integer."
        )
    return int(value)


def int_value(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MarketSimulationConfigError(f"Market simulation config field '{field_name}' must be an integer.")
    return int(value)


def path_string(value: Any, field_name: str) -> str:
    return Path(required_string(value, field_name)).as_posix()


def optional_path_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return path_string(value, field_name)


def string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise MarketSimulationConfigError(
            f"Market simulation config field '{field_name}' must be a sequence."
        )
    return tuple(required_string(item, field_name) for item in value)


def optional_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MarketSimulationConfigError(
            f"Market simulation config field '{field_name}' must be a mapping when provided."
        )
    return dict(value)
