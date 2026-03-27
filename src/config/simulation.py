from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_CONFIG = REPO_ROOT / "configs" / "simulation.yml"
SUPPORTED_SIMULATION_METHODS = frozenset({"bootstrap", "monte_carlo"})


@dataclass(frozen=True)
class SimulationConfig:
    """Typed deterministic configuration for return-path simulation workflows."""

    method: str
    num_paths: int
    path_length: int | None = None
    seed: int = 0
    monte_carlo_mean: float | None = None
    monte_carlo_volatility: float | None = None
    distribution: str = "normal"
    min_samples: int = 2
    drawdown_threshold: float | None = None
    metrics: tuple[str, ...] | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SimulationConfig":
        method = payload.get("method")
        if not isinstance(method, str) or method not in SUPPORTED_SIMULATION_METHODS:
            supported = ", ".join(sorted(SUPPORTED_SIMULATION_METHODS))
            raise ValueError(f"Simulation method must be one of: {supported}.")

        num_paths = payload.get("num_paths", payload.get("paths"))
        if not isinstance(num_paths, int) or num_paths <= 0:
            raise ValueError("Simulation num_paths must be a positive integer.")

        path_length = payload.get("path_length")
        if path_length is not None and (not isinstance(path_length, int) or path_length <= 0):
            raise ValueError("Simulation path_length must be a positive integer when provided.")

        seed = payload.get("seed", 0)
        if not isinstance(seed, int):
            raise ValueError("Simulation seed must be an integer.")

        monte_carlo_mean = _optional_float(payload.get("monte_carlo_mean"), field_name="monte_carlo_mean")
        monte_carlo_volatility = _optional_non_negative_float(
            payload.get("monte_carlo_volatility"),
            field_name="monte_carlo_volatility",
        )

        distribution = payload.get("distribution", "normal")
        if not isinstance(distribution, str) or distribution.strip().lower() != "normal":
            raise ValueError("Simulation distribution must currently be 'normal'.")

        min_samples = payload.get("min_samples", 2)
        if not isinstance(min_samples, int) or min_samples <= 0:
            raise ValueError("Simulation min_samples must be a positive integer.")

        drawdown_threshold = _optional_probability(
            payload.get("drawdown_threshold"),
            field_name="drawdown_threshold",
        )

        metrics_payload = payload.get("metrics")
        if metrics_payload is not None:
            if not isinstance(metrics_payload, list) or not metrics_payload:
                raise ValueError("Simulation metrics must be a non-empty list when provided.")
            normalized_metrics: tuple[str, ...] = tuple(_normalize_metric_name(metric) for metric in metrics_payload)
        else:
            normalized_metrics = None

        return cls(
            method=method,
            num_paths=num_paths,
            path_length=path_length,
            seed=seed,
            monte_carlo_mean=monte_carlo_mean,
            monte_carlo_volatility=monte_carlo_volatility,
            distribution="normal",
            min_samples=min_samples,
            drawdown_threshold=drawdown_threshold,
            metrics=normalized_metrics,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "num_paths": self.num_paths,
            "path_length": self.path_length,
            "seed": self.seed,
            "monte_carlo_mean": self.monte_carlo_mean,
            "monte_carlo_volatility": self.monte_carlo_volatility,
            "distribution": self.distribution,
            "min_samples": self.min_samples,
            "drawdown_threshold": self.drawdown_threshold,
            "metrics": None if self.metrics is None else list(self.metrics),
        }


def resolve_simulation_config(
    payload: SimulationConfig | dict[str, Any] | None,
    *,
    base: SimulationConfig | dict[str, Any] | None = None,
) -> SimulationConfig | None:
    if payload is None and base is None:
        return None
    if payload is None:
        return base if isinstance(base, SimulationConfig) else SimulationConfig.from_mapping(dict(base or {}))
    if isinstance(payload, SimulationConfig):
        if base is None:
            return payload
        merged = {
            **(base.to_dict() if isinstance(base, SimulationConfig) else dict(base or {})),
            **payload.to_dict(),
        }
        return SimulationConfig.from_mapping(merged)
    if not isinstance(payload, dict):
        raise ValueError("Simulation config must be a dictionary when provided.")
    merged = {
        **({} if base is None else (base.to_dict() if isinstance(base, SimulationConfig) else dict(base))),
        **payload,
    }
    return SimulationConfig.from_mapping(merged)


def load_simulation_config(path: Path = SIMULATION_CONFIG) -> SimulationConfig:
    if not path.exists():
        raise ValueError(f"Simulation config file does not exist: {path}.")

    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            payload = json.load(handle)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle)
        else:
            raise ValueError(
                f"Unsupported simulation config format {path.suffix!r}. Use JSON, YAML, or YML."
            )

    if not isinstance(payload, dict):
        raise ValueError("Simulation configuration file must contain a top-level mapping.")

    simulation_payload = payload.get("simulation", payload)
    if not isinstance(simulation_payload, dict):
        raise ValueError("Simulation configuration must define a 'simulation' mapping.")
    return SimulationConfig.from_mapping(simulation_payload)


def _normalize_metric_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Simulation metric names must be non-empty strings.")
    return value.strip()


def _optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Simulation {field_name} must be a finite float when provided.") from exc


def _optional_non_negative_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    numeric = _optional_float(value, field_name=field_name)
    if numeric is None or numeric < 0.0:
        raise ValueError(f"Simulation {field_name} must be a non-negative float when provided.")
    return numeric


def _optional_probability(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    numeric = _optional_float(value, field_name=field_name)
    if numeric is None or not (0.0 <= numeric <= 1.0):
        raise ValueError(f"Simulation {field_name} must be within [0, 1] when provided.")
    return numeric


__all__ = [
    "SIMULATION_CONFIG",
    "SUPPORTED_SIMULATION_METHODS",
    "SimulationConfig",
    "load_simulation_config",
    "resolve_simulation_config",
]
