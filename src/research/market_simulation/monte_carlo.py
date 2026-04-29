"""Deterministic regime-transition Monte Carlo simulation.

This module generates regime-label paths only. It does not simulate return or
price paths.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.research.market_simulation.config import (
    MarketSimulationConfig,
    MarketSimulationScenarioConfig,
)
from src.research.market_simulation.ids import generate_path_id
from src.research.market_simulation.validation import MarketSimulationConfigError
from src.research.registry import canonicalize_value

MONTE_CARLO_SCHEMA_VERSION = "1.0"
DEFAULT_VALIDATION_TOLERANCE = 1e-8

MONTE_CARLO_PATH_COLUMNS = (
    "scenario_id",
    "path_id",
    "path_index",
    "path_step",
    "ts_utc",
    "regime_label",
    "previous_regime_label",
    "transitioned",
    "duration_in_regime",
    "transition_probability",
    "seed",
    "stress_bias_applied",
    "sticky_adjustment_applied",
    "initial_regime",
    "source_matrix_type",
)

MONTE_CARLO_PATH_CATALOG_COLUMNS = (
    "scenario_id",
    "path_id",
    "simulation_type",
    "seed",
    "path_index",
    "path_length_bars",
    "path_start",
    "path_end",
    "initial_regime",
    "unique_regime_count",
    "transition_count",
    "max_duration_observed",
    "min_duration_observed",
    "terminal_regime",
    "stress_state_count",
    "notes",
)

TRANSITION_COUNT_COLUMNS = (
    "source_regime",
    "destination_regime",
    "transition_count",
)


@dataclass(frozen=True)
class MonteCarloResult:
    scenario_id: str
    scenario_name: str
    output_dir: Path
    transition_matrix_path: Path
    adjusted_transition_matrix_path: Path | None
    transition_counts_path: Path | None
    regime_paths_path: Path
    path_catalog_path: Path
    summary_path: Path
    manifest_path: Path
    path_count: int
    generated_row_count: int


@dataclass(frozen=True)
class _DurationConstraints:
    min_duration_bars: int
    max_duration_bars: int | None


@dataclass(frozen=True)
class _BiasConfig:
    stress_bias_enabled: bool
    stress_target_regimes: tuple[str, ...]
    stress_multiplier: float
    sticky_enabled: bool
    sticky_self_transition_multiplier: float


@dataclass(frozen=True)
class _MonteCarloConfig:
    path_count: int
    path_length_bars: int
    path_start: str
    initial_regime: str | None
    initial_regime_distribution: dict[str, float] | None
    normalize_transition_rows: bool
    validation_tolerance: float
    duration_constraints: _DurationConstraints
    bias_config: _BiasConfig

    @classmethod
    def from_scenario(cls, scenario: MarketSimulationScenarioConfig) -> "_MonteCarloConfig":
        method_config = scenario.method_config
        duration_config = _optional_mapping(
            method_config.get("duration_constraints"),
            "method_config.duration_constraints",
        )
        min_duration = _positive_int(
            duration_config.get("min_duration_bars", 1),
            "method_config.duration_constraints.min_duration_bars",
        )
        max_duration = _optional_positive_int(
            duration_config.get("max_duration_bars"),
            "method_config.duration_constraints.max_duration_bars",
        )
        if max_duration is not None and max_duration < min_duration:
            raise MarketSimulationConfigError(
                "Regime transition Monte Carlo duration_constraints.max_duration_bars "
                "must be greater than or equal to min_duration_bars."
            )
        return cls(
            path_count=_positive_int(
                method_config.get("path_count", scenario.path_count or 1),
                "method_config.path_count",
            ),
            path_length_bars=_positive_int(
                method_config.get("path_length_bars"),
                "method_config.path_length_bars",
            ),
            path_start=_optional_string(method_config.get("path_start"), "method_config.path_start")
            or "2000-01-01",
            initial_regime=_optional_string(
                method_config.get("initial_regime"),
                "method_config.initial_regime",
            ),
            initial_regime_distribution=_optional_probability_mapping(
                method_config.get("initial_regime_distribution"),
                "method_config.initial_regime_distribution",
            ),
            normalize_transition_rows=_bool(
                method_config.get("normalize_transition_rows", False),
                "method_config.normalize_transition_rows",
            ),
            validation_tolerance=_positive_float(
                method_config.get("validation_tolerance", DEFAULT_VALIDATION_TOLERANCE),
                "method_config.validation_tolerance",
            ),
            duration_constraints=_DurationConstraints(
                min_duration_bars=min_duration,
                max_duration_bars=max_duration,
            ),
            bias_config=_bias_config(method_config),
        )


@dataclass(frozen=True)
class _MatrixBuildResult:
    base_matrix: dict[str, dict[str, float]]
    transition_counts: list[dict[str, Any]]
    source_type: str
    source_metadata: dict[str, Any]


def run_regime_transition_monte_carlo_scenarios(
    config: MarketSimulationConfig,
    *,
    simulation_run_id: str,
    market_simulations_output_dir: Path,
) -> list[MonteCarloResult]:
    results: list[MonteCarloResult] = []
    for scenario in config.market_simulations:
        if scenario.simulation_type != "regime_transition_monte_carlo" or not scenario.enabled:
            continue
        if not _has_monte_carlo_method_config(scenario.method_config):
            continue
        results.append(
            run_regime_transition_monte_carlo(
                scenario,
                simulation_run_id=simulation_run_id,
                market_simulations_output_dir=market_simulations_output_dir,
            )
        )
    return results


def _has_monte_carlo_method_config(method_config: Mapping[str, Any]) -> bool:
    return any(
        key in method_config
        for key in (
            "transition_matrix",
            "transition_source",
            "path_length_bars",
            "initial_regime",
            "initial_regime_distribution",
        )
    )


def run_regime_transition_monte_carlo(
    scenario: MarketSimulationScenarioConfig,
    *,
    simulation_run_id: str,
    market_simulations_output_dir: Path,
) -> MonteCarloResult:
    mc_config = _MonteCarloConfig.from_scenario(scenario)
    matrix_build = _build_transition_matrix(scenario.method_config, mc_config)
    base_matrix = matrix_build.base_matrix
    adjusted_matrix = _apply_transition_adjustments(base_matrix, mc_config.bias_config)
    regimes = tuple(sorted(base_matrix))
    _validate_regime_references(mc_config, regimes)

    rng = np.random.default_rng(scenario.seed)
    path_start = _parse_path_start(mc_config.path_start)
    path_rows: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    for path_index in range(mc_config.path_count):
        path_id = generate_path_id(
            scenario_id=scenario.scenario_id,
            path_index=path_index,
            seed=scenario.seed,
            metadata={
                "path_length_bars": mc_config.path_length_bars,
                "source_matrix_type": matrix_build.source_type,
                "duration_constraints": {
                    "min_duration_bars": mc_config.duration_constraints.min_duration_bars,
                    "max_duration_bars": mc_config.duration_constraints.max_duration_bars,
                },
                "stress_bias_enabled": mc_config.bias_config.stress_bias_enabled,
                "sticky_enabled": mc_config.bias_config.sticky_enabled,
            },
        )
        rows = _generate_path(
            scenario=scenario,
            mc_config=mc_config,
            matrix=adjusted_matrix,
            rng=rng,
            path_id=path_id,
            path_index=path_index,
            path_start=path_start,
            regimes=regimes,
            source_matrix_type=matrix_build.source_type,
        )
        path_rows.extend(rows)
        catalog_rows.append(
            _path_catalog_row(
                scenario=scenario,
                mc_config=mc_config,
                path_id=path_id,
                path_index=path_index,
                rows=rows,
            )
        )

    scenario_dir = market_simulations_output_dir / scenario.scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    adjustment_applied = (
        mc_config.bias_config.stress_bias_enabled or mc_config.bias_config.sticky_enabled
    )
    paths = {
        "transition_matrix_json": scenario_dir / "transition_matrix.json",
        "monte_carlo_regime_paths_parquet": scenario_dir / "monte_carlo_regime_paths.parquet",
        "monte_carlo_path_catalog_csv": scenario_dir / "monte_carlo_path_catalog.csv",
        "monte_carlo_summary_json": scenario_dir / "monte_carlo_summary.json",
        "manifest_json": scenario_dir / "manifest.json",
    }
    adjusted_path = scenario_dir / "adjusted_transition_matrix.json" if adjustment_applied else None
    counts_path = scenario_dir / "transition_counts.csv" if matrix_build.transition_counts else None
    generated_files = {key: path.name for key, path in paths.items()}
    if adjusted_path is not None:
        generated_files["adjusted_transition_matrix_json"] = adjusted_path.name
    if counts_path is not None:
        generated_files["transition_counts_csv"] = counts_path.name

    matrix_payload = _matrix_payload(
        base_matrix=base_matrix,
        adjusted_matrix=adjusted_matrix,
        source_type=matrix_build.source_type,
        source_metadata=matrix_build.source_metadata,
        mc_config=mc_config,
    )
    summary = _summary_payload(
        scenario=scenario,
        mc_config=mc_config,
        regimes=regimes,
        matrix_source=matrix_build.source_type,
        path_rows=path_rows,
        catalog_rows=catalog_rows,
        generated_files=generated_files,
    )
    manifest = _manifest_payload(
        scenario=scenario,
        simulation_run_id=simulation_run_id,
        mc_config=mc_config,
        matrix_source=matrix_build.source_type,
        generated_files=generated_files,
        path_row_count=len(catalog_rows),
        regime_row_count=len(path_rows),
        transition_count_row_count=len(matrix_build.transition_counts),
    )

    _write_json(paths["transition_matrix_json"], matrix_payload)
    if adjusted_path is not None:
        _write_json(adjusted_path, adjusted_matrix)
    if counts_path is not None:
        _write_csv(counts_path, matrix_build.transition_counts, TRANSITION_COUNT_COLUMNS)
    _write_parquet(paths["monte_carlo_regime_paths_parquet"], path_rows, MONTE_CARLO_PATH_COLUMNS)
    _write_csv(paths["monte_carlo_path_catalog_csv"], catalog_rows, MONTE_CARLO_PATH_CATALOG_COLUMNS)
    _write_json(paths["monte_carlo_summary_json"], summary)
    _write_json(paths["manifest_json"], manifest)

    return MonteCarloResult(
        scenario_id=scenario.scenario_id,
        scenario_name=scenario.name,
        output_dir=scenario_dir,
        transition_matrix_path=paths["transition_matrix_json"],
        adjusted_transition_matrix_path=adjusted_path,
        transition_counts_path=counts_path,
        regime_paths_path=paths["monte_carlo_regime_paths_parquet"],
        path_catalog_path=paths["monte_carlo_path_catalog_csv"],
        summary_path=paths["monte_carlo_summary_json"],
        manifest_path=paths["manifest_json"],
        path_count=len(catalog_rows),
        generated_row_count=len(path_rows),
    )


def validate_transition_matrix(
    matrix: Mapping[str, Any],
    *,
    normalize_transition_rows: bool = False,
    validation_tolerance: float = DEFAULT_VALIDATION_TOLERANCE,
) -> dict[str, dict[str, float]]:
    if not isinstance(matrix, Mapping) or not matrix:
        raise MarketSimulationConfigError(
            "Regime transition Monte Carlo transition_matrix must be a non-empty mapping."
        )
    regimes = tuple(sorted(_required_string(regime, "transition_matrix source regime") for regime in matrix))
    normalized: dict[str, dict[str, float]] = {}
    for source in regimes:
        row = matrix[source]
        if not isinstance(row, Mapping) or not row:
            raise MarketSimulationConfigError(
                f"Regime transition Monte Carlo transition row for {source!r} must be non-empty."
            )
        row_values: dict[str, float] = {destination: 0.0 for destination in regimes}
        for raw_destination, raw_probability in row.items():
            destination = _required_string(raw_destination, "transition_matrix destination regime")
            if destination not in row_values:
                raise MarketSimulationConfigError(
                    f"Regime transition Monte Carlo destination regime {destination!r} "
                    f"from source {source!r} is not in the known regime set."
                )
            probability = _probability(raw_probability, f"transition_matrix.{source}.{destination}")
            row_values[destination] = probability
        row_sum = sum(row_values.values())
        if row_sum <= 0:
            raise MarketSimulationConfigError(
                f"Regime transition Monte Carlo transition row for {source!r} must have positive row sum."
            )
        if normalize_transition_rows:
            row_values = {destination: probability / row_sum for destination, probability in row_values.items()}
        elif abs(row_sum - 1.0) > validation_tolerance:
            raise MarketSimulationConfigError(
                f"Regime transition Monte Carlo transition row for {source!r} sums to {row_sum:.12g}; "
                "set normalize_transition_rows=true to normalize positive rows."
            )
        normalized[source] = row_values
    return canonicalize_value(normalized)


def _build_transition_matrix(
    method_config: Mapping[str, Any],
    mc_config: _MonteCarloConfig,
) -> _MatrixBuildResult:
    has_inline = method_config.get("transition_matrix") is not None
    has_source = method_config.get("transition_source") is not None
    if has_inline and has_source:
        raise MarketSimulationConfigError(
            "Regime transition Monte Carlo supports either transition_matrix or transition_source, not both."
        )
    if has_inline:
        matrix = validate_transition_matrix(
            _optional_mapping(method_config.get("transition_matrix"), "method_config.transition_matrix"),
            normalize_transition_rows=mc_config.normalize_transition_rows,
            validation_tolerance=mc_config.validation_tolerance,
        )
        return _MatrixBuildResult(
            base_matrix=matrix,
            transition_counts=[],
            source_type="inline",
            source_metadata={"transition_matrix_source": "inline"},
        )
    if has_source:
        return _empirical_transition_matrix(
            _optional_mapping(method_config.get("transition_source"), "method_config.transition_source"),
            mc_config,
        )
    raise MarketSimulationConfigError(
        "Regime transition Monte Carlo requires method_config.transition_matrix or method_config.transition_source."
    )


def _empirical_transition_matrix(
    transition_source: Mapping[str, Any],
    mc_config: _MonteCarloConfig,
) -> _MatrixBuildResult:
    dataset_path_value = _required_string(
        transition_source.get("dataset_path"),
        "method_config.transition_source.dataset_path",
    )
    dataset_path = Path(dataset_path_value)
    resolved_dataset_path = dataset_path if dataset_path.is_absolute() else Path.cwd() / dataset_path
    timestamp_column = _required_string(
        transition_source.get("timestamp_column"),
        "method_config.transition_source.timestamp_column",
    )
    regime_column = _required_string(
        transition_source.get("regime_column"),
        "method_config.transition_source.regime_column",
    )
    symbol_column = _optional_string(
        transition_source.get("symbol_column"),
        "method_config.transition_source.symbol_column",
    )
    frame = _load_dataset(resolved_dataset_path)
    _require_columns(frame, (timestamp_column, regime_column))
    if symbol_column is not None:
        _require_columns(frame, (symbol_column,))
    normalized = frame.copy()
    normalized["_source_ts_utc"] = pd.to_datetime(normalized[timestamp_column], errors="raise", utc=True)
    if normalized["_source_ts_utc"].isna().any():
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo timestamp column {timestamp_column!r} contains null values."
        )
    normalized["_regime_label"] = normalized[regime_column].astype("string")
    if normalized["_regime_label"].isna().any():
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo regime column {regime_column!r} contains null values."
        )
    sort_columns = [symbol_column, "_source_ts_utc"] if symbol_column is not None else ["_source_ts_utc"]
    normalized = normalized.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    regimes = tuple(sorted(str(value) for value in normalized["_regime_label"].dropna().unique()))
    counts: dict[str, dict[str, int]] = {
        source: {destination: 0 for destination in regimes} for source in regimes
    }
    groups = normalized.groupby(symbol_column, sort=True) if symbol_column is not None else [(None, normalized)]
    for _, group in groups:
        labels = [str(value) for value in group["_regime_label"].tolist()]
        for source, destination in zip(labels, labels[1:], strict=False):
            counts[source][destination] += 1
    matrix: dict[str, dict[str, float]] = {}
    for source in regimes:
        row_sum = sum(counts[source].values())
        if row_sum <= 0:
            raise MarketSimulationConfigError(
                f"Regime transition Monte Carlo empirical transition row for {source!r} has no transitions."
            )
        matrix[source] = {
            destination: counts[source][destination] / row_sum for destination in regimes
        }
    count_rows = [
        {
            "source_regime": source,
            "destination_regime": destination,
            "transition_count": counts[source][destination],
        }
        for source in regimes
        for destination in regimes
    ]
    return _MatrixBuildResult(
        base_matrix=validate_transition_matrix(
            matrix,
            normalize_transition_rows=mc_config.normalize_transition_rows,
            validation_tolerance=mc_config.validation_tolerance,
        ),
        transition_counts=count_rows,
        source_type="empirical",
        source_metadata={
            "dataset_path": _display_path(resolved_dataset_path),
            "format": resolved_dataset_path.suffix.lower().lstrip("."),
            "timestamp_column": timestamp_column,
            "regime_column": regime_column,
            "symbol_column": symbol_column,
            "source_observation_count": len(normalized),
        },
    )


def _apply_transition_adjustments(
    matrix: Mapping[str, Mapping[str, float]],
    bias_config: _BiasConfig,
) -> dict[str, dict[str, float]]:
    adjusted = {source: dict(row) for source, row in matrix.items()}
    for source, row in adjusted.items():
        if bias_config.sticky_enabled and source in row:
            row[source] *= bias_config.sticky_self_transition_multiplier
        if bias_config.stress_bias_enabled:
            for target in bias_config.stress_target_regimes:
                if target in row:
                    row[target] *= bias_config.stress_multiplier
        row_sum = sum(row.values())
        if row_sum <= 0:
            raise MarketSimulationConfigError(
                f"Regime transition Monte Carlo adjusted transition row for {source!r} has non-positive row sum."
            )
        adjusted[source] = {destination: probability / row_sum for destination, probability in row.items()}
    return canonicalize_value(adjusted)


def _generate_path(
    *,
    scenario: MarketSimulationScenarioConfig,
    mc_config: _MonteCarloConfig,
    matrix: Mapping[str, Mapping[str, float]],
    rng: np.random.Generator,
    path_id: str,
    path_index: int,
    path_start: pd.Timestamp,
    regimes: tuple[str, ...],
    source_matrix_type: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current = _initial_regime(mc_config, regimes, rng)
    previous: str | None = None
    duration = 1
    transition_probability = 1.0
    for path_step in range(mc_config.path_length_bars):
        if path_step > 0:
            previous = current
            next_regime, transition_probability = _next_regime(
                current=current,
                duration=duration,
                matrix=matrix,
                regimes=regimes,
                rng=rng,
                constraints=mc_config.duration_constraints,
            )
            if next_regime == current:
                duration += 1
            else:
                current = next_regime
                duration = 1
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "path_id": path_id,
                "path_index": path_index,
                "path_step": path_step,
                "ts_utc": _iso(path_start + pd.Timedelta(days=path_step)),
                "regime_label": current,
                "previous_regime_label": previous,
                "transitioned": previous is not None and current != previous,
                "duration_in_regime": duration,
                "transition_probability": transition_probability,
                "seed": scenario.seed,
                "stress_bias_applied": mc_config.bias_config.stress_bias_enabled,
                "sticky_adjustment_applied": mc_config.bias_config.sticky_enabled,
                "initial_regime": rows[0]["regime_label"] if rows else current,
                "source_matrix_type": source_matrix_type,
            }
        )
    return rows


def _next_regime(
    *,
    current: str,
    duration: int,
    matrix: Mapping[str, Mapping[str, float]],
    regimes: tuple[str, ...],
    rng: np.random.Generator,
    constraints: _DurationConstraints,
) -> tuple[str, float]:
    row = dict(matrix[current])
    if duration < constraints.min_duration_bars:
        return current, 1.0
    if constraints.max_duration_bars is not None and duration >= constraints.max_duration_bars:
        alternatives = [regime for regime in regimes if regime != current]
        if alternatives:
            row[current] = 0.0
            row = _renormalized_row(row, current)
    probabilities = np.array([row[regime] for regime in regimes], dtype=float)
    selected = str(rng.choice(regimes, p=probabilities))
    return selected, float(row[selected])


def _renormalized_row(row: Mapping[str, float], source: str) -> dict[str, float]:
    row_sum = sum(row.values())
    if row_sum <= 0:
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo cannot force transition away from {source!r}; "
            "all alternative transition probabilities are zero."
        )
    return {destination: probability / row_sum for destination, probability in row.items()}


def _initial_regime(
    mc_config: _MonteCarloConfig,
    regimes: tuple[str, ...],
    rng: np.random.Generator,
) -> str:
    if mc_config.initial_regime is not None:
        return mc_config.initial_regime
    if mc_config.initial_regime_distribution is not None:
        labels = tuple(sorted(mc_config.initial_regime_distribution))
        probabilities = np.array([mc_config.initial_regime_distribution[label] for label in labels])
        return str(rng.choice(labels, p=probabilities))
    return regimes[0]


def _path_catalog_row(
    *,
    scenario: MarketSimulationScenarioConfig,
    mc_config: _MonteCarloConfig,
    path_id: str,
    path_index: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    durations = [int(row["duration_in_regime"]) for row in rows]
    stress_count = sum(str(row["regime_label"]).lower() == "stress" for row in rows)
    return {
        "scenario_id": scenario.scenario_id,
        "path_id": path_id,
        "simulation_type": scenario.simulation_type,
        "seed": scenario.seed,
        "path_index": path_index,
        "path_length_bars": mc_config.path_length_bars,
        "path_start": rows[0]["ts_utc"],
        "path_end": rows[-1]["ts_utc"],
        "initial_regime": rows[0]["regime_label"],
        "unique_regime_count": len({row["regime_label"] for row in rows}),
        "transition_count": sum(bool(row["transitioned"]) for row in rows),
        "max_duration_observed": max(durations),
        "min_duration_observed": min(durations),
        "terminal_regime": rows[-1]["regime_label"],
        "stress_state_count": stress_count,
        "notes": scenario.notes or "",
    }


def _matrix_payload(
    *,
    base_matrix: Mapping[str, Mapping[str, float]],
    adjusted_matrix: Mapping[str, Mapping[str, float]],
    source_type: str,
    source_metadata: Mapping[str, Any],
    mc_config: _MonteCarloConfig,
) -> dict[str, Any]:
    regimes = tuple(sorted(base_matrix))
    return canonicalize_value(
        {
            "source_type": source_type,
            "source_metadata": dict(source_metadata),
            "regime_list": list(regimes),
            "base_transition_matrix": base_matrix,
            "adjusted_transition_matrix": adjusted_matrix,
            "row_sums": {
                source: sum(float(value) for value in base_matrix[source].values()) for source in regimes
            },
            "adjusted_row_sums": {
                source: sum(float(value) for value in adjusted_matrix[source].values()) for source in regimes
            },
            "normalize_transition_rows": mc_config.normalize_transition_rows,
            "validation_tolerance": mc_config.validation_tolerance,
            "stress_bias": {
                "enabled": mc_config.bias_config.stress_bias_enabled,
                "target_regimes": list(mc_config.bias_config.stress_target_regimes),
                "multiplier": mc_config.bias_config.stress_multiplier,
            },
            "sticky_regime": {
                "enabled": mc_config.bias_config.sticky_enabled,
                "self_transition_multiplier": mc_config.bias_config.sticky_self_transition_multiplier,
            },
        }
    )


def _summary_payload(
    *,
    scenario: MarketSimulationScenarioConfig,
    mc_config: _MonteCarloConfig,
    regimes: tuple[str, ...],
    matrix_source: str,
    path_rows: list[dict[str, Any]],
    catalog_rows: list[dict[str, Any]],
    generated_files: Mapping[str, str],
) -> dict[str, Any]:
    transition_counts = [int(row["transition_count"]) for row in catalog_rows]
    stress_count = sum(int(row["stress_state_count"]) for row in catalog_rows)
    return canonicalize_value(
        {
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "simulation_type": scenario.simulation_type,
            "seed": scenario.seed,
            "path_count": mc_config.path_count,
            "path_length_bars": mc_config.path_length_bars,
            "regime_count": len(regimes),
            "transition_matrix_source": matrix_source,
            "total_generated_rows": len(path_rows),
            "mean_transition_count": float(np.mean(transition_counts)) if transition_counts else 0.0,
            "min_transition_count": min(transition_counts) if transition_counts else 0,
            "max_transition_count": max(transition_counts) if transition_counts else 0,
            "stress_regime_share": stress_count / len(path_rows) if path_rows else None,
            "generated_files": dict(generated_files),
            "limitations": _limitations(),
        }
    )


def _manifest_payload(
    *,
    scenario: MarketSimulationScenarioConfig,
    simulation_run_id: str,
    mc_config: _MonteCarloConfig,
    matrix_source: str,
    generated_files: Mapping[str, str],
    path_row_count: int,
    regime_row_count: int,
    transition_count_row_count: int,
) -> dict[str, Any]:
    row_counts = {
        "monte_carlo_path_catalog_csv": path_row_count,
        "monte_carlo_regime_paths_parquet": regime_row_count,
    }
    if transition_count_row_count:
        row_counts["transition_counts_csv"] = transition_count_row_count
    return canonicalize_value(
        {
            "artifact_type": "regime_transition_monte_carlo",
            "schema_version": MONTE_CARLO_SCHEMA_VERSION,
            "simulation_run_id": simulation_run_id,
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "simulation_type": scenario.simulation_type,
            "generated_files": dict(generated_files),
            "row_counts": row_counts,
            "transition_matrix_source": matrix_source,
            "path_count": mc_config.path_count,
            "path_length_bars": mc_config.path_length_bars,
            "seed": scenario.seed,
            "relative_paths": {
                "scenario_dir": scenario.scenario_id,
                **dict(generated_files),
            },
            "limitations": _limitations(),
        }
    )


def _limitations() -> list[str]:
    return [
        "Monte Carlo paths simulate regime labels only and do not generate full market return paths.",
        "Synthetic path timestamps are deterministic daily bars for path ordering.",
        "Artifacts are for research stress testing, not live trading or market forecasting.",
    ]


def _bias_config(method_config: Mapping[str, Any]) -> _BiasConfig:
    stress_bias = _optional_mapping(method_config.get("stress_bias"), "method_config.stress_bias")
    sticky = _optional_mapping(method_config.get("sticky_regime"), "method_config.sticky_regime")
    return _BiasConfig(
        stress_bias_enabled=_bool(stress_bias.get("enabled", False), "method_config.stress_bias.enabled"),
        stress_target_regimes=_string_tuple(
            stress_bias.get("target_regimes", ()),
            "method_config.stress_bias.target_regimes",
        ),
        stress_multiplier=_positive_float(
            stress_bias.get("multiplier", 1.0),
            "method_config.stress_bias.multiplier",
        ),
        sticky_enabled=_bool(sticky.get("enabled", False), "method_config.sticky_regime.enabled"),
        sticky_self_transition_multiplier=_positive_float(
            sticky.get("self_transition_multiplier", 1.0),
            "method_config.sticky_regime.self_transition_multiplier",
        ),
    )


def _validate_regime_references(mc_config: _MonteCarloConfig, regimes: tuple[str, ...]) -> None:
    regime_set = set(regimes)
    if mc_config.initial_regime is not None and mc_config.initial_regime not in regime_set:
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo initial_regime {mc_config.initial_regime!r} "
            "is not in the transition matrix regime set."
        )
    if mc_config.initial_regime_distribution is not None:
        unknown = sorted(set(mc_config.initial_regime_distribution) - regime_set)
        if unknown:
            raise MarketSimulationConfigError(
                f"Regime transition Monte Carlo initial_regime_distribution contains unknown regimes: {unknown}."
            )
    if mc_config.bias_config.stress_bias_enabled:
        if not mc_config.bias_config.stress_target_regimes:
            raise MarketSimulationConfigError(
                "Regime transition Monte Carlo stress_bias.target_regimes is required when stress_bias is enabled."
            )
        unknown = sorted(set(mc_config.bias_config.stress_target_regimes) - regime_set)
        if unknown:
            raise MarketSimulationConfigError(
                f"Regime transition Monte Carlo stress_bias.target_regimes contains unknown regimes: {unknown}."
            )


def _optional_probability_mapping(value: Any, field_name: str) -> dict[str, float] | None:
    if value is None:
        return None
    mapping = _optional_mapping(value, field_name)
    probabilities = {
        _required_string(key, field_name): _probability(probability, f"{field_name}.{key}")
        for key, probability in mapping.items()
    }
    row_sum = sum(probabilities.values())
    if row_sum <= 0 or abs(row_sum - 1.0) > DEFAULT_VALIDATION_TOLERANCE:
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo {field_name} probabilities must sum to 1.0."
        )
    return probabilities


def _load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo dataset does not exist: {_display_path(dataset_path)}"
        )
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(dataset_path)
    raise MarketSimulationConfigError(
        f"Unsupported regime transition Monte Carlo dataset format {dataset_path.suffix!r}. Use CSV or Parquet."
    )


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo dataset is missing required columns: {missing}."
        )


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo field '{field_name}' must be a non-empty string."
        )
    return value.strip()


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name)


def _optional_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo field '{field_name}' must be a mapping."
        )
    return value


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo field '{field_name}' must be a sequence."
        )
    return tuple(_required_string(item, field_name) for item in value)


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo field '{field_name}' must be a positive integer."
        )
    return int(value)


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _positive_float(value: Any, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or float(value) <= 0:
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo field '{field_name}' must be a positive number."
        )
    return float(value)


def _probability(value: Any, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo probability '{field_name}' must be numeric."
        )
    probability = float(value)
    if probability < 0:
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo probability '{field_name}' must be non-negative."
        )
    return probability


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo field '{field_name}' must be boolean."
        )
    return value


def _parse_path_start(value: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except ValueError as exc:
        raise MarketSimulationConfigError(
            f"Regime transition Monte Carlo path_start timestamp {value!r} is invalid."
        ) from exc
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _iso(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _write_parquet(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    frame = pd.DataFrame([{column: row.get(column) for column in columns} for row in rows])
    if frame.empty:
        frame = pd.DataFrame(columns=list(columns))
    frame.to_parquet(path, index=False)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(canonicalize_value(payload), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _display_path(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        digest = hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()[:12]
        return f"external/{path.name}_{digest}"
