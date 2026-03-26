from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.config.settings import load_yaml_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SANITY_CONFIG = REPO_ROOT / "configs" / "sanity.yml"

_DEFAULT_MAX_ABS_PERIOD_RETURN = 1.0
_DEFAULT_MAX_ANNUALIZED_RETURN = 25.0
_DEFAULT_MAX_SHARPE_RATIO = 10.0
_DEFAULT_MAX_EQUITY_MULTIPLE = 1_000_000.0
_DEFAULT_MIN_VOLATILITY_FLOOR = 0.02
_DEFAULT_MIN_VOLATILITY_TRIGGER_SHARPE = 4.0
_DEFAULT_MIN_VOLATILITY_TRIGGER_ANNUALIZED_RETURN = 1.0
_DEFAULT_SMOOTHNESS_MIN_SHARPE = 3.0
_DEFAULT_SMOOTHNESS_MIN_ANNUALIZED_RETURN = 0.75
_DEFAULT_SMOOTHNESS_MAX_DRAWDOWN = 0.02
_DEFAULT_SMOOTHNESS_MIN_POSITIVE_RETURN_FRACTION = 0.95


@dataclass(frozen=True)
class SanityCheckConfig:
    """Deterministic thresholds for suspicious-performance detection."""

    max_abs_period_return: float | None = _DEFAULT_MAX_ABS_PERIOD_RETURN
    max_annualized_return: float | None = _DEFAULT_MAX_ANNUALIZED_RETURN
    max_sharpe_ratio: float | None = _DEFAULT_MAX_SHARPE_RATIO
    max_equity_multiple: float | None = _DEFAULT_MAX_EQUITY_MULTIPLE
    strict_sanity_checks: bool = False
    min_annualized_volatility_floor: float | None = _DEFAULT_MIN_VOLATILITY_FLOOR
    min_volatility_trigger_sharpe: float | None = _DEFAULT_MIN_VOLATILITY_TRIGGER_SHARPE
    min_volatility_trigger_annualized_return: float | None = _DEFAULT_MIN_VOLATILITY_TRIGGER_ANNUALIZED_RETURN
    smoothness_min_sharpe: float | None = _DEFAULT_SMOOTHNESS_MIN_SHARPE
    smoothness_min_annualized_return: float | None = _DEFAULT_SMOOTHNESS_MIN_ANNUALIZED_RETURN
    smoothness_max_drawdown: float | None = _DEFAULT_SMOOTHNESS_MAX_DRAWDOWN
    smoothness_min_positive_return_fraction: float | None = _DEFAULT_SMOOTHNESS_MIN_POSITIVE_RETURN_FRACTION

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs_period_return": self.max_abs_period_return,
            "max_annualized_return": self.max_annualized_return,
            "max_sharpe_ratio": self.max_sharpe_ratio,
            "max_equity_multiple": self.max_equity_multiple,
            "strict_sanity_checks": self.strict_sanity_checks,
            "min_annualized_volatility_floor": self.min_annualized_volatility_floor,
            "min_volatility_trigger_sharpe": self.min_volatility_trigger_sharpe,
            "min_volatility_trigger_annualized_return": self.min_volatility_trigger_annualized_return,
            "smoothness_min_sharpe": self.smoothness_min_sharpe,
            "smoothness_min_annualized_return": self.smoothness_min_annualized_return,
            "smoothness_max_drawdown": self.smoothness_max_drawdown,
            "smoothness_min_positive_return_fraction": self.smoothness_min_positive_return_fraction,
        }


def load_sanity_config(path: Path = SANITY_CONFIG) -> SanityCheckConfig:
    """Load shared research sanity thresholds from YAML when present."""

    if not path.exists():
        return SanityCheckConfig()

    payload = load_yaml_config(path)
    if not isinstance(payload, dict):
        raise ValueError("Sanity configuration file must contain a top-level mapping.")

    sanity_payload = payload.get("sanity", payload)
    if not isinstance(sanity_payload, Mapping):
        raise ValueError("Sanity configuration must define a 'sanity' mapping.")
    return resolve_sanity_check_config(sanity_payload, base=SanityCheckConfig())


def resolve_sanity_check_config(
    payload: SanityCheckConfig | Mapping[str, Any] | None,
    *,
    base: SanityCheckConfig | None = None,
) -> SanityCheckConfig:
    """Resolve sanity settings from defaults plus optional overrides."""

    if payload is None:
        return base or load_sanity_config()
    if isinstance(payload, SanityCheckConfig):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError("Sanity configuration must be a mapping when provided.")

    seed = base or load_sanity_config()
    strict_value = payload.get("strict_sanity_checks", seed.strict_sanity_checks)
    if not isinstance(strict_value, bool):
        raise ValueError("Sanity configuration field 'strict_sanity_checks' must be a boolean.")

    config = SanityCheckConfig(
        max_abs_period_return=_coerce_optional_non_negative_float(
            payload.get("max_abs_period_return", seed.max_abs_period_return),
            field_name="max_abs_period_return",
        ),
        max_annualized_return=_coerce_optional_non_negative_float(
            payload.get("max_annualized_return", seed.max_annualized_return),
            field_name="max_annualized_return",
        ),
        max_sharpe_ratio=_coerce_optional_non_negative_float(
            payload.get("max_sharpe_ratio", seed.max_sharpe_ratio),
            field_name="max_sharpe_ratio",
        ),
        max_equity_multiple=_coerce_optional_non_negative_float(
            payload.get("max_equity_multiple", seed.max_equity_multiple),
            field_name="max_equity_multiple",
        ),
        strict_sanity_checks=strict_value,
        min_annualized_volatility_floor=_coerce_optional_non_negative_float(
            payload.get("min_annualized_volatility_floor", seed.min_annualized_volatility_floor),
            field_name="min_annualized_volatility_floor",
        ),
        min_volatility_trigger_sharpe=_coerce_optional_non_negative_float(
            payload.get("min_volatility_trigger_sharpe", seed.min_volatility_trigger_sharpe),
            field_name="min_volatility_trigger_sharpe",
        ),
        min_volatility_trigger_annualized_return=_coerce_optional_non_negative_float(
            payload.get(
                "min_volatility_trigger_annualized_return",
                seed.min_volatility_trigger_annualized_return,
            ),
            field_name="min_volatility_trigger_annualized_return",
        ),
        smoothness_min_sharpe=_coerce_optional_non_negative_float(
            payload.get("smoothness_min_sharpe", seed.smoothness_min_sharpe),
            field_name="smoothness_min_sharpe",
        ),
        smoothness_min_annualized_return=_coerce_optional_non_negative_float(
            payload.get("smoothness_min_annualized_return", seed.smoothness_min_annualized_return),
            field_name="smoothness_min_annualized_return",
        ),
        smoothness_max_drawdown=_coerce_optional_non_negative_float(
            payload.get("smoothness_max_drawdown", seed.smoothness_max_drawdown),
            field_name="smoothness_max_drawdown",
        ),
        smoothness_min_positive_return_fraction=_coerce_optional_probability(
            payload.get(
                "smoothness_min_positive_return_fraction",
                seed.smoothness_min_positive_return_fraction,
            ),
            field_name="smoothness_min_positive_return_fraction",
        ),
    )
    return config


def _coerce_optional_non_negative_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sanity configuration field '{field_name}' must be a non-negative number.") from exc
    if numeric < 0.0:
        raise ValueError(f"Sanity configuration field '{field_name}' must be non-negative.")
    return numeric


def _coerce_optional_probability(value: Any, *, field_name: str) -> float | None:
    numeric = _coerce_optional_non_negative_float(value, field_name=field_name)
    if numeric is None:
        return None
    if numeric > 1.0:
        raise ValueError(f"Sanity configuration field '{field_name}' must be between 0.0 and 1.0.")
    return numeric

