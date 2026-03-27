from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.settings import load_yaml_config

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS_CONFIG = REPO_ROOT / "configs" / "robustness.yml"
SUPPORTED_STABILITY_MODES = {"disabled", "subperiods", "walk_forward"}


@dataclass(frozen=True)
class MetricThreshold:
    """Deterministic threshold rule used in robustness pass-rate summaries."""

    metric: str
    min_value: float | None = None
    max_value: float | None = None

    @classmethod
    def from_mapping(cls, metric: str, payload: Any) -> "MetricThreshold":
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("Robustness threshold names must be non-empty strings.")

        normalized_metric = metric.strip()
        if isinstance(payload, int | float):
            return cls(metric=normalized_metric, min_value=float(payload))
        if not isinstance(payload, dict):
            raise ValueError(
                f"Robustness threshold for '{normalized_metric}' must be a number or a mapping with min/max."
            )

        min_value = payload.get("min")
        max_value = payload.get("max")
        if min_value is None and max_value is None:
            raise ValueError(
                f"Robustness threshold for '{normalized_metric}' must define at least one of 'min' or 'max'."
            )
        if min_value is not None and not isinstance(min_value, int | float):
            raise ValueError(f"Robustness threshold min for '{normalized_metric}' must be numeric.")
        if max_value is not None and not isinstance(max_value, int | float):
            raise ValueError(f"Robustness threshold max for '{normalized_metric}' must be numeric.")
        return cls(
            metric=normalized_metric,
            min_value=None if min_value is None else float(min_value),
            max_value=None if max_value is None else float(max_value),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "min": self.min_value,
            "max": self.max_value,
        }


@dataclass(frozen=True)
class SweepDefinition:
    """One parameter sweep definition with explicit ordered values."""

    parameter: str
    values: tuple[Any, ...]

    @classmethod
    def from_mapping(cls, payload: Any) -> "SweepDefinition":
        if not isinstance(payload, dict):
            raise ValueError("Each robustness sweep entry must be a mapping.")

        parameter = payload.get("parameter")
        values = payload.get("values")
        if not isinstance(parameter, str) or not parameter.strip():
            raise ValueError("Robustness sweep entries must define a non-empty parameter name.")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Robustness sweep '{parameter}' must define a non-empty values list.")
        return cls(parameter=parameter.strip(), values=tuple(values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "values": list(self.values),
        }


@dataclass(frozen=True)
class StabilityConfig:
    """Optional time-slice stability analysis configuration."""

    mode: str = "disabled"
    periods: int | None = None
    evaluation_path: str | None = None

    @classmethod
    def from_mapping(cls, payload: Any) -> "StabilityConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("Robustness stability configuration must be a mapping.")

        mode = payload.get("mode", "disabled")
        if not isinstance(mode, str) or mode not in SUPPORTED_STABILITY_MODES:
            supported = ", ".join(sorted(SUPPORTED_STABILITY_MODES))
            raise ValueError(f"Robustness stability mode must be one of: {supported}.")

        periods = payload.get("periods")
        if periods is not None:
            if not isinstance(periods, int):
                raise ValueError("Robustness stability periods must be an integer when provided.")
            if periods < 2:
                raise ValueError("Robustness stability periods must be at least 2.")

        evaluation_path = payload.get("evaluation_path")
        if evaluation_path is not None and not isinstance(evaluation_path, str):
            raise ValueError("Robustness stability evaluation_path must be a string when provided.")

        if mode == "subperiods" and periods is None:
            raise ValueError("Robustness subperiod stability requires a periods value.")

        return cls(mode=mode, periods=periods, evaluation_path=evaluation_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "periods": self.periods,
            "evaluation_path": self.evaluation_path,
        }


@dataclass(frozen=True)
class RobustnessConfig:
    """Typed robustness-run configuration loaded from YAML."""

    strategy_name: str | None
    ranking_metric: str
    higher_is_better: bool | None
    sweep: tuple[SweepDefinition, ...]
    stability: StabilityConfig
    thresholds: tuple[MetricThreshold, ...]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "RobustnessConfig":
        strategy_name = payload.get("strategy_name")
        if strategy_name is not None and (not isinstance(strategy_name, str) or not strategy_name.strip()):
            raise ValueError("Robustness strategy_name must be a non-empty string when provided.")

        ranking_metric = payload.get("ranking_metric", "sharpe_ratio")
        if not isinstance(ranking_metric, str) or not ranking_metric.strip():
            raise ValueError("Robustness ranking_metric must be a non-empty string.")

        higher_is_better = payload.get("higher_is_better")
        if higher_is_better is not None and not isinstance(higher_is_better, bool):
            raise ValueError("Robustness higher_is_better must be a boolean when provided.")

        raw_sweep = payload.get("sweep")
        if not isinstance(raw_sweep, list) or not raw_sweep:
            raise ValueError("Robustness configuration must define a non-empty sweep list.")
        sweep = tuple(SweepDefinition.from_mapping(entry) for entry in raw_sweep)

        seen_parameters: set[str] = set()
        for definition in sweep:
            if definition.parameter in seen_parameters:
                raise ValueError(f"Robustness sweep contains duplicate parameter '{definition.parameter}'.")
            seen_parameters.add(definition.parameter)

        stability = StabilityConfig.from_mapping(payload.get("stability"))

        raw_thresholds = payload.get("thresholds", {})
        if not isinstance(raw_thresholds, dict):
            raise ValueError("Robustness thresholds must be a mapping when provided.")
        thresholds = tuple(
            MetricThreshold.from_mapping(metric_name, raw_thresholds[metric_name])
            for metric_name in sorted(raw_thresholds)
        )

        return cls(
            strategy_name=None if strategy_name is None else strategy_name.strip(),
            ranking_metric=ranking_metric.strip(),
            higher_is_better=higher_is_better,
            sweep=sweep,
            stability=stability,
            thresholds=thresholds,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "ranking_metric": self.ranking_metric,
            "higher_is_better": self.higher_is_better,
            "sweep": [definition.to_dict() for definition in self.sweep],
            "stability": self.stability.to_dict(),
            "thresholds": {
                threshold.metric: {
                    key: value
                    for key, value in (("min", threshold.min_value), ("max", threshold.max_value))
                    if value is not None
                }
                for threshold in self.thresholds
            },
        }

    def resolve_strategy_name(self, cli_strategy_name: str | None) -> str:
        if cli_strategy_name and self.strategy_name and cli_strategy_name != self.strategy_name:
            raise ValueError(
                f"Robustness strategy mismatch: CLI requested '{cli_strategy_name}' but config targets "
                f"'{self.strategy_name}'."
            )
        resolved = cli_strategy_name or self.strategy_name
        if resolved is None:
            raise ValueError("Robustness runs require a strategy_name in the config or via --strategy.")
        return resolved


def load_robustness_config(path: Path = ROBUSTNESS_CONFIG) -> RobustnessConfig:
    """Load a robustness configuration from YAML."""

    payload = load_yaml_config(path)
    if not isinstance(payload, dict):
        raise ValueError("Robustness configuration file must contain a top-level mapping.")

    robustness_payload = payload.get("robustness", payload)
    if not isinstance(robustness_payload, dict):
        raise ValueError("Robustness configuration must define a 'robustness' mapping.")

    return RobustnessConfig.from_mapping(robustness_payload)
