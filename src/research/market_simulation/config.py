from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.research.market_simulation.ids import generate_scenario_id
from src.research.market_simulation.registry import get_simulation_type_metadata
from src.research.market_simulation.validation import (
    MarketSimulationConfigError,
    bool_value,
    int_value,
    nonnegative_int,
    optional_mapping,
    optional_string,
    path_string,
    required_string,
    string_sequence,
)
from src.research.registry import canonicalize_value

DEFAULT_RANDOM_SEED = 0

_ROOT_KEYS = frozenset(
    {
        "simulation_name",
        "output_root",
        "random_seed",
        "source_review_pack",
        "baseline_policy",
        "source_policy_candidates",
        "market_simulations",
        "stress_metrics",
        "metadata",
    }
)
_SCENARIO_KEYS = frozenset(
    {
        "scenario_id",
        "name",
        "type",
        "enabled",
        "random_seed",
        "seed",
        "path_count",
        "source_window_start",
        "source_window_end",
        "notes",
        "method_config",
        "source_config_name",
    }
)


@dataclass(frozen=True)
class MarketSimulationScenarioConfig:
    scenario_id: str
    name: str
    simulation_type: str
    enabled: bool = True
    seed: int = DEFAULT_RANDOM_SEED
    path_count: int = 0
    source_window_start: str | None = None
    source_window_end: str | None = None
    notes: str | None = None
    method_config: dict[str, Any] = field(default_factory=dict)
    source_config_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "scenario_id": self.scenario_id,
                "name": self.name,
                "type": self.simulation_type,
                "enabled": self.enabled,
                "seed": self.seed,
                "path_count": self.path_count,
                "source_window_start": self.source_window_start,
                "source_window_end": self.source_window_end,
                "notes": self.notes,
                "method_config": self.method_config,
                "source_config_name": self.source_config_name,
            }
        )


@dataclass(frozen=True)
class StressMetricsConfig:
    enabled: bool = True
    output_dir_name: str = "simulation_metrics"
    failure_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "max_drawdown_limit": -0.10,
            "min_total_return": -0.05,
            "max_transition_count": 50.0,
            "max_stress_regime_share": 0.50,
            "max_policy_underperformance": -0.02,
        }
    )
    leaderboard: dict[str, Any] = field(
        default_factory=lambda: {"ranking_metric": "mean_stress_score", "ascending": True}
    )
    tail_quantile: float = 0.05
    stress_regimes: tuple[str, ...] = ("stress",)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "enabled": self.enabled,
                "output_dir_name": self.output_dir_name,
                "failure_thresholds": dict(self.failure_thresholds),
                "leaderboard": dict(self.leaderboard),
                "tail_quantile": self.tail_quantile,
                "stress_regimes": list(self.stress_regimes),
            }
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "StressMetricsConfig":
        if payload is None:
            return cls()
        mapping = optional_mapping(payload, "stress_metrics")
        thresholds = dict(cls().failure_thresholds)
        thresholds.update(
            {
                str(key): float(value)
                for key, value in optional_mapping(
                    mapping.get("failure_thresholds"), "stress_metrics.failure_thresholds"
                ).items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
        )
        leaderboard = dict(cls().leaderboard)
        leaderboard.update(optional_mapping(mapping.get("leaderboard"), "stress_metrics.leaderboard"))
        output_dir_name = optional_string(
            mapping.get("output_dir_name"), "stress_metrics.output_dir_name"
        ) or "simulation_metrics"
        tail_quantile_raw = mapping.get("tail_quantile", 0.05)
        if not isinstance(tail_quantile_raw, int | float) or isinstance(tail_quantile_raw, bool):
            raise MarketSimulationConfigError("stress_metrics.tail_quantile must be numeric.")
        tail_quantile = float(tail_quantile_raw)
        if not 0.0 < tail_quantile < 1.0:
            raise MarketSimulationConfigError("stress_metrics.tail_quantile must be between 0 and 1.")
        return cls(
            enabled=bool_value(mapping.get("enabled", True), "stress_metrics.enabled"),
            output_dir_name=output_dir_name,
            failure_thresholds=thresholds,
            leaderboard=leaderboard,
            tail_quantile=tail_quantile,
            stress_regimes=string_sequence(mapping.get("stress_regimes", ["stress"]), "stress_metrics.stress_regimes"),
        )


@dataclass(frozen=True)
class MarketSimulationConfig:
    simulation_name: str
    output_root: str = "artifacts/regime_stress_tests"
    random_seed: int = DEFAULT_RANDOM_SEED
    source_review_pack: str | None = None
    baseline_policy: str = ""
    source_policy_candidates: tuple[str, ...] = ()
    market_simulations: tuple[MarketSimulationScenarioConfig, ...] = ()
    stress_metrics: StressMetricsConfig = field(default_factory=StressMetricsConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "simulation_name": self.simulation_name,
                "output_root": Path(self.output_root).as_posix(),
                "random_seed": self.random_seed,
                "source_review_pack": None
                if self.source_review_pack is None
                else Path(self.source_review_pack).as_posix(),
                "baseline_policy": self.baseline_policy,
                "source_policy_candidates": list(self.source_policy_candidates),
                "market_simulations": [scenario.to_dict() for scenario in self.market_simulations],
                "stress_metrics": self.stress_metrics.to_dict(),
                "metadata": dict(self.metadata),
            }
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MarketSimulationConfig":
        unknown = sorted(set(payload) - _ROOT_KEYS)
        if unknown:
            raise MarketSimulationConfigError(
                f"Market simulation config contains unsupported keys: {unknown}."
            )
        simulation_name = required_string(payload.get("simulation_name"), "simulation_name")
        random_seed = int_value(payload.get("random_seed", DEFAULT_RANDOM_SEED), "random_seed")
        baseline_policy = required_string(payload.get("baseline_policy"), "baseline_policy")
        source_policy_candidates = string_sequence(
            payload.get("source_policy_candidates"), "source_policy_candidates"
        )
        scenarios = _resolve_market_simulations(
            payload.get("market_simulations"),
            simulation_name=simulation_name,
            global_seed=random_seed,
        )
        if not scenarios:
            raise MarketSimulationConfigError(
                "Market simulation config requires at least one market_simulations entry."
            )
        return cls(
            simulation_name=simulation_name,
            output_root=path_string(payload.get("output_root", "artifacts/regime_stress_tests"), "output_root"),
            random_seed=random_seed,
            source_review_pack=(
                None
                if payload.get("source_review_pack") is None
                else path_string(payload.get("source_review_pack"), "source_review_pack")
            ),
            baseline_policy=baseline_policy,
            source_policy_candidates=source_policy_candidates,
            market_simulations=scenarios,
            stress_metrics=StressMetricsConfig.from_mapping(payload.get("stress_metrics")),
            metadata=canonicalize_value(optional_mapping(payload.get("metadata"), "metadata")),
        )


def load_market_simulation_config(path: Path) -> MarketSimulationConfig:
    if not path.exists():
        raise MarketSimulationConfigError(f"Market simulation config does not exist: {path.as_posix()}")
    with path.open("r", encoding="utf-8") as handle:
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yml", ".yaml"}:
            payload = yaml.safe_load(handle) or {}
        else:
            raise MarketSimulationConfigError(
                f"Unsupported market simulation config format {path.suffix!r}. Use JSON, YAML, or YML."
            )
    if not isinstance(payload, Mapping):
        raise MarketSimulationConfigError("Market simulation config must contain a top-level mapping.")
    return MarketSimulationConfig.from_mapping(payload)


def _resolve_market_simulations(
    value: Any,
    *,
    simulation_name: str,
    global_seed: int,
) -> tuple[MarketSimulationScenarioConfig, ...]:
    if not isinstance(value, list):
        raise MarketSimulationConfigError("Market simulation config field 'market_simulations' must be a sequence.")
    scenarios: list[MarketSimulationScenarioConfig] = []
    names_seen: set[str] = set()
    ids_seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise MarketSimulationConfigError(
                f"Market simulation entry at index {index} must be a mapping."
            )
        unknown = sorted(set(item) - _SCENARIO_KEYS)
        if unknown:
            raise MarketSimulationConfigError(
                f"Market simulation entry {index} contains unsupported keys: {unknown}."
            )
        name = required_string(item.get("name"), f"market_simulations[{index}].name")
        if name in names_seen:
            raise MarketSimulationConfigError(
                f"Market simulation scenario names must be unique. Duplicate: {name!r}."
            )
        names_seen.add(name)
        simulation_type = required_string(item.get("type"), f"market_simulations[{index}].type")
        get_simulation_type_metadata(simulation_type)
        seed = _resolve_seed(item, global_seed=global_seed, index=index)
        path_count = nonnegative_int(item.get("path_count", 0), f"market_simulations[{index}].path_count")
        method_config = canonicalize_value(
            optional_mapping(item.get("method_config"), f"market_simulations[{index}].method_config")
        )
        scenario_id = generate_scenario_id(
            simulation_name=simulation_name,
            scenario_name=name,
            simulation_type=simulation_type,
            seed=seed,
            path_count=path_count,
            source_window_start=optional_string(
                item.get("source_window_start"), f"market_simulations[{index}].source_window_start"
            ),
            source_window_end=optional_string(
                item.get("source_window_end"), f"market_simulations[{index}].source_window_end"
            ),
            method_config=method_config,
        )
        provided_scenario_id = optional_string(
            item.get("scenario_id"), f"market_simulations[{index}].scenario_id"
        )
        if provided_scenario_id is not None and provided_scenario_id != scenario_id:
            raise MarketSimulationConfigError(
                f"Market simulation entry {index} scenario_id does not match deterministic scenario id {scenario_id!r}."
            )
        if scenario_id in ids_seen:
            raise MarketSimulationConfigError(
                "Market simulation config produced duplicate scenario IDs; adjust scenario definitions."
            )
        ids_seen.add(scenario_id)
        scenarios.append(
            MarketSimulationScenarioConfig(
                scenario_id=scenario_id,
                name=name,
                simulation_type=simulation_type,
                enabled=bool_value(item.get("enabled", True), f"market_simulations[{index}].enabled"),
                seed=seed,
                path_count=path_count,
                source_window_start=optional_string(
                    item.get("source_window_start"), f"market_simulations[{index}].source_window_start"
                ),
                source_window_end=optional_string(
                    item.get("source_window_end"), f"market_simulations[{index}].source_window_end"
                ),
                notes=optional_string(item.get("notes"), f"market_simulations[{index}].notes"),
                method_config=method_config,
                source_config_name=optional_string(
                    item.get("source_config_name"), f"market_simulations[{index}].source_config_name"
                ),
            )
        )
    return tuple(scenarios)


def _resolve_seed(payload: Mapping[str, Any], *, global_seed: int, index: int) -> int:
    has_random_seed = payload.get("random_seed") is not None
    has_seed = payload.get("seed") is not None
    if has_random_seed and has_seed:
        raise MarketSimulationConfigError(
            f"Market simulation entry {index} may use 'random_seed' or 'seed', not both."
        )
    if has_random_seed:
        return int_value(payload.get("random_seed"), f"market_simulations[{index}].random_seed")
    if has_seed:
        return int_value(payload.get("seed"), f"market_simulations[{index}].seed")
    return global_seed
