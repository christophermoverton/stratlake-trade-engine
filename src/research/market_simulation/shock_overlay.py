from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.research.market_simulation.config import (
    MarketSimulationConfig,
    MarketSimulationScenarioConfig,
)
from src.research.market_simulation.historical_replay import HistoricalEpisodeReplayResult
from src.research.market_simulation.validation import MarketSimulationConfigError
from src.research.registry import canonicalize_value

SHOCK_OVERLAY_SCHEMA_VERSION = "1.0"

SHOCK_OVERLAY_RESULT_COLUMNS = (
    "scenario_id",
    "source_scenario_id",
    "source_episode_id",
    "source_episode_name",
    "ts_utc",
    "symbol",
    "source_return",
    "stressed_source_return",
    "adaptive_policy_return",
    "stressed_adaptive_policy_return",
    "static_baseline_return",
    "stressed_static_baseline_return",
    "gmm_confidence",
    "stressed_gmm_confidence",
    "gmm_entropy",
    "stressed_gmm_entropy",
    "overlay_count",
    "overlay_stack",
)

SHOCK_OVERLAY_LOG_COLUMNS = (
    "scenario_id",
    "overlay_index",
    "overlay_name",
    "overlay_type",
    "target_column",
    "stressed_column",
    "parameter_name",
    "parameter_value",
    "rows_affected",
    "before_mean",
    "after_mean",
    "before_min",
    "after_min",
    "before_max",
    "after_max",
    "status",
    "message",
)


@dataclass(frozen=True)
class ShockOverlayResult:
    scenario_id: str
    scenario_name: str
    output_dir: Path
    shock_overlay_config_path: Path
    shock_overlay_results_path: Path
    shock_overlay_log_path: Path
    shock_overlay_summary_path: Path
    manifest_path: Path
    source_scenario_id: str
    source_scenario_name: str
    overlay_count: int
    row_count: int


@dataclass(frozen=True)
class _SourceInput:
    scenario_id: str
    scenario_name: str
    path: Path
    frame: pd.DataFrame


@dataclass(frozen=True)
class _OverlayConfig:
    name: str
    overlay_type: str
    columns: tuple[str, ...]
    parameter_name: str
    parameter_value: float
    missing_column_policy: str


@dataclass(frozen=True)
class _ShockOverlayConfig:
    input_source: Mapping[str, Any]
    timestamp_column: str
    symbol_column: str | None
    base_return_column: str | None
    adaptive_policy_return_column: str | None
    static_baseline_return_column: str | None
    confidence_column: str | None
    entropy_column: str | None
    overlays: tuple[_OverlayConfig, ...]

    @classmethod
    def from_scenario(cls, scenario: MarketSimulationScenarioConfig) -> "_ShockOverlayConfig":
        method_config = scenario.method_config
        input_source = _required_mapping(method_config.get("input_source"), "method_config.input_source")
        overlays = _resolve_overlays(method_config.get("overlays"))
        return cls(
            input_source=input_source,
            timestamp_column=_required_string(
                method_config.get("timestamp_column"), "method_config.timestamp_column"
            ),
            symbol_column=_optional_string(method_config.get("symbol_column"), "method_config.symbol_column"),
            base_return_column=_optional_string(
                method_config.get("base_return_column"), "method_config.base_return_column"
            ),
            adaptive_policy_return_column=_optional_string(
                method_config.get("adaptive_policy_return_column"),
                "method_config.adaptive_policy_return_column",
            ),
            static_baseline_return_column=_optional_string(
                method_config.get("static_baseline_return_column"),
                "method_config.static_baseline_return_column",
            ),
            confidence_column=_optional_string(
                method_config.get("confidence_column"), "method_config.confidence_column"
            ),
            entropy_column=_optional_string(method_config.get("entropy_column"), "method_config.entropy_column"),
            overlays=overlays,
        )


def run_shock_overlay_scenarios(
    config: MarketSimulationConfig,
    *,
    simulation_run_id: str,
    market_simulations_output_dir: Path,
    historical_episode_replay_results: list[HistoricalEpisodeReplayResult],
) -> list[ShockOverlayResult]:
    results: list[ShockOverlayResult] = []
    for scenario in config.market_simulations:
        if scenario.simulation_type != "shock_overlay" or not scenario.enabled:
            continue
        overlay_config = _ShockOverlayConfig.from_scenario(scenario)
        source = _resolve_input_source(overlay_config.input_source, historical_episode_replay_results)
        results.append(
            run_shock_overlay_scenario(
                scenario,
                overlay_config=overlay_config,
                source=source,
                simulation_run_id=simulation_run_id,
                market_simulations_output_dir=market_simulations_output_dir,
            )
        )
    return results


def run_shock_overlay_scenario(
    scenario: MarketSimulationScenarioConfig,
    *,
    overlay_config: _ShockOverlayConfig,
    source: _SourceInput,
    simulation_run_id: str,
    market_simulations_output_dir: Path,
) -> ShockOverlayResult:
    frame = _normalize_source_frame(source.frame, overlay_config)
    overlay_log: list[dict[str, Any]] = []
    for overlay_index, overlay in enumerate(overlay_config.overlays, start=1):
        _apply_overlay(frame, scenario=scenario, overlay=overlay, overlay_index=overlay_index, log_rows=overlay_log)

    result_rows = _result_rows(frame, scenario=scenario, source=source, overlay_config=overlay_config)
    summary = _summary_payload(
        scenario=scenario,
        source=source,
        overlay_config=overlay_config,
        frame=frame,
        log_rows=overlay_log,
    )

    scenario_dir = market_simulations_output_dir / scenario.scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "shock_overlay_config_json": scenario_dir / "shock_overlay_config.json",
        "shock_overlay_results_csv": scenario_dir / "shock_overlay_results.csv",
        "shock_overlay_log_csv": scenario_dir / "shock_overlay_log.csv",
        "shock_overlay_summary_json": scenario_dir / "shock_overlay_summary.json",
        "manifest_json": scenario_dir / "manifest.json",
    }
    generated_files = {key: path.name for key, path in paths.items()}
    manifest = _manifest_payload(
        scenario=scenario,
        simulation_run_id=simulation_run_id,
        source=source,
        result_rows=result_rows,
        overlay_config=overlay_config,
        log_row_count=len(overlay_log),
        generated_files=generated_files,
    )

    _write_json(paths["shock_overlay_config_json"], scenario.to_dict())
    _write_csv(paths["shock_overlay_results_csv"], result_rows, _result_columns(result_rows))
    _write_csv(paths["shock_overlay_log_csv"], overlay_log, SHOCK_OVERLAY_LOG_COLUMNS)
    _write_json(paths["shock_overlay_summary_json"], summary)
    _write_json(paths["manifest_json"], manifest)

    return ShockOverlayResult(
        scenario_id=scenario.scenario_id,
        scenario_name=scenario.name,
        output_dir=scenario_dir,
        shock_overlay_config_path=paths["shock_overlay_config_json"],
        shock_overlay_results_path=paths["shock_overlay_results_csv"],
        shock_overlay_log_path=paths["shock_overlay_log_csv"],
        shock_overlay_summary_path=paths["shock_overlay_summary_json"],
        manifest_path=paths["manifest_json"],
        source_scenario_id=source.scenario_id,
        source_scenario_name=source.scenario_name,
        overlay_count=len(overlay_config.overlays),
        row_count=len(result_rows),
    )


def _resolve_input_source(
    input_source: Mapping[str, Any],
    historical_episode_replay_results: list[HistoricalEpisodeReplayResult],
) -> _SourceInput:
    source_type = _required_string(input_source.get("type"), "method_config.input_source.type")
    if source_type != "historical_episode_replay":
        raise MarketSimulationConfigError(
            "Shock overlay input_source.type must be 'historical_episode_replay'. "
            "File inputs are reserved for a follow-up issue."
        )
    scenario_name = _optional_string(
        input_source.get("scenario_name"), "method_config.input_source.scenario_name"
    )
    scenario_id = _optional_string(input_source.get("scenario_id"), "method_config.input_source.scenario_id")
    if scenario_name is None and scenario_id is None:
        raise MarketSimulationConfigError(
            "Shock overlay input_source requires scenario_name or scenario_id."
        )
    matches = [
        result
        for result in historical_episode_replay_results
        if (scenario_name is None or result.scenario_name == scenario_name)
        and (scenario_id is None or result.scenario_id == scenario_id)
    ]
    if not matches:
        target = scenario_name if scenario_name is not None else scenario_id
        raise MarketSimulationConfigError(
            f"Shock overlay input source historical replay scenario was not found: {target!r}."
        )
    if len(matches) > 1:
        raise MarketSimulationConfigError(
            f"Shock overlay input source matched multiple historical replay scenarios: {scenario_name!r}."
        )
    match = matches[0]
    return _SourceInput(
        scenario_id=match.scenario_id,
        scenario_name=match.scenario_name,
        path=match.episode_replay_results_path,
        frame=pd.read_csv(match.episode_replay_results_path),
    )


def _resolve_overlays(value: Any) -> tuple[_OverlayConfig, ...]:
    if not isinstance(value, list) or not value:
        raise MarketSimulationConfigError("Shock overlay requires at least one method_config.overlays entry.")
    overlays: list[_OverlayConfig] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise MarketSimulationConfigError(f"Shock overlay entry at index {index} must be a mapping.")
        overlay_type = _required_string(item.get("type"), f"overlays[{index}].type")
        name = _required_string(item.get("name"), f"overlays[{index}].name")
        columns = _overlay_columns(item, index=index)
        missing_policy = _optional_string(
            item.get("missing_column_policy"), f"overlays[{index}].missing_column_policy"
        ) or "fail"
        if missing_policy not in {"fail", "ignore"}:
            raise MarketSimulationConfigError(
                f"Shock overlay {name!r} missing_column_policy must be 'fail' or 'ignore'."
            )
        parameter_name, parameter_value = _overlay_parameter(item, overlay_type=overlay_type, index=index)
        overlays.append(
            _OverlayConfig(
                name=name,
                overlay_type=overlay_type,
                columns=columns,
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                missing_column_policy=missing_policy,
            )
        )
    return tuple(overlays)


def _overlay_columns(item: Mapping[str, Any], *, index: int) -> tuple[str, ...]:
    if item.get("columns") is not None:
        raw_columns = item.get("columns")
        if not isinstance(raw_columns, list) or not raw_columns:
            raise MarketSimulationConfigError(f"overlays[{index}].columns must be a non-empty sequence.")
        return tuple(_required_string(column, f"overlays[{index}].columns") for column in raw_columns)
    column = _required_string(item.get("column"), f"overlays[{index}].column")
    return (column,)


def _overlay_parameter(item: Mapping[str, Any], *, overlay_type: str, index: int) -> tuple[str, float]:
    parameter_by_type = {
        "return_bps": "bps",
        "volatility_multiplier": "multiplier",
        "transaction_cost_multiplier": "multiplier",
        "slippage_multiplier": "multiplier",
        "confidence_multiplier": "multiplier",
        "entropy_multiplier": "multiplier",
    }
    if overlay_type not in parameter_by_type:
        expected = ", ".join(sorted(parameter_by_type))
        raise MarketSimulationConfigError(
            f"Shock overlay type {overlay_type!r} is unsupported. Expected one of: {expected}."
        )
    parameter_name = parameter_by_type[overlay_type]
    value = item.get(parameter_name)
    if not isinstance(value, int | float):
        raise MarketSimulationConfigError(
            f"Shock overlay overlays[{index}].{parameter_name} must be numeric."
        )
    return parameter_name, float(value)


def _normalize_source_frame(frame: pd.DataFrame, overlay_config: _ShockOverlayConfig) -> pd.DataFrame:
    _require_columns(frame, (overlay_config.timestamp_column,))
    normalized = frame.copy()
    normalized["source_scenario_id"] = normalized.get("scenario_id")
    normalized["source_episode_id"] = normalized.get("episode_id")
    normalized["source_episode_name"] = normalized.get("episode_name")
    sort_columns = [overlay_config.timestamp_column]
    if overlay_config.symbol_column is not None and overlay_config.symbol_column in normalized.columns:
        sort_columns.append(overlay_config.symbol_column)
    if "source_episode_id" in normalized.columns:
        sort_columns.insert(0, "source_episode_id")
    return normalized.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _apply_overlay(
    frame: pd.DataFrame,
    *,
    scenario: MarketSimulationScenarioConfig,
    overlay: _OverlayConfig,
    overlay_index: int,
    log_rows: list[dict[str, Any]],
) -> None:
    for target_column in overlay.columns:
        stressed_column = f"stressed_{target_column}"
        if target_column not in frame.columns:
            if overlay.missing_column_policy == "ignore":
                log_rows.append(
                    _log_row(
                        scenario=scenario,
                        overlay=overlay,
                        overlay_index=overlay_index,
                        target_column=target_column,
                        stressed_column=stressed_column,
                        status="skipped",
                        message=f"Column {target_column!r} is unavailable.",
                    )
                )
                continue
            raise MarketSimulationConfigError(
                f"Shock overlay {overlay.name!r} requires missing column {target_column!r}."
            )
        if stressed_column not in frame.columns:
            frame[stressed_column] = frame[target_column]
        before = pd.to_numeric(frame[stressed_column], errors="coerce")
        after = _apply_overlay_values(before, overlay)
        frame[stressed_column] = after
        log_rows.append(
            _log_row(
                scenario=scenario,
                overlay=overlay,
                overlay_index=overlay_index,
                target_column=target_column,
                stressed_column=stressed_column,
                status="applied",
                message="",
                before=before,
                after=after,
            )
        )


def _apply_overlay_values(values: pd.Series, overlay: _OverlayConfig) -> pd.Series:
    if overlay.overlay_type == "return_bps":
        return values + (overlay.parameter_value / 10000.0)
    if overlay.overlay_type == "volatility_multiplier":
        mean = values.mean()
        return mean + overlay.parameter_value * (values - mean)
    if overlay.overlay_type in {"transaction_cost_multiplier", "slippage_multiplier"}:
        return values * overlay.parameter_value
    if overlay.overlay_type == "confidence_multiplier":
        return (values * overlay.parameter_value).clip(lower=0.0, upper=1.0)
    if overlay.overlay_type == "entropy_multiplier":
        return (values * overlay.parameter_value).clip(lower=0.0)
    raise MarketSimulationConfigError(f"Unsupported shock overlay type {overlay.overlay_type!r}.")


def _result_rows(
    frame: pd.DataFrame,
    *,
    scenario: MarketSimulationScenarioConfig,
    source: _SourceInput,
    overlay_config: _ShockOverlayConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overlay_stack = "|".join(overlay.name for overlay in overlay_config.overlays)
    for _, row in frame.iterrows():
        result_row = {
            "scenario_id": scenario.scenario_id,
            "source_scenario_id": _value(row, "source_scenario_id") or source.scenario_id,
            "source_episode_id": _value(row, "source_episode_id"),
            "source_episode_name": _value(row, "source_episode_name"),
            "ts_utc": _value(row, overlay_config.timestamp_column),
            "symbol": _value(row, overlay_config.symbol_column),
            "source_return": _value(row, overlay_config.base_return_column),
            "stressed_source_return": _value(row, _stressed(overlay_config.base_return_column)),
            "adaptive_policy_return": _value(row, overlay_config.adaptive_policy_return_column),
            "stressed_adaptive_policy_return": _value(
                row, _stressed(overlay_config.adaptive_policy_return_column)
            ),
            "static_baseline_return": _value(row, overlay_config.static_baseline_return_column),
            "stressed_static_baseline_return": _value(
                row, _stressed(overlay_config.static_baseline_return_column)
            ),
            "gmm_confidence": _value(row, overlay_config.confidence_column),
            "stressed_gmm_confidence": _value(row, _stressed(overlay_config.confidence_column)),
            "gmm_entropy": _value(row, overlay_config.entropy_column),
            "stressed_gmm_entropy": _value(row, _stressed(overlay_config.entropy_column)),
            "overlay_count": len(overlay_config.overlays),
            "overlay_stack": overlay_stack,
        }
        for column in frame.columns:
            if column in _INTERNAL_SOURCE_COLUMNS or column in result_row:
                continue
            result_row[column] = _value(row, column)
        rows.append(result_row)
    return rows


_INTERNAL_SOURCE_COLUMNS = frozenset(
    {
        "scenario_id",
        "episode_id",
        "episode_name",
        "source_scenario_id",
        "source_episode_id",
        "source_episode_name",
    }
)


def _result_columns(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    extra_columns = sorted(
        {column for row in rows for column in row if column not in SHOCK_OVERLAY_RESULT_COLUMNS}
    )
    return (*SHOCK_OVERLAY_RESULT_COLUMNS, *extra_columns)


def _summary_payload(
    *,
    scenario: MarketSimulationScenarioConfig,
    source: _SourceInput,
    overlay_config: _ShockOverlayConfig,
    frame: pd.DataFrame,
    log_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    affected_columns = sorted({row["target_column"] for row in log_rows if row["status"] == "applied"})
    adaptive_total = _total_return(frame, _stressed(overlay_config.adaptive_policy_return_column))
    static_total = _total_return(frame, _stressed(overlay_config.static_baseline_return_column))
    limitations = [
        "Shock overlays are deterministic research stress tests, not market forecasts.",
        "Overlay operations preserve source columns and write adjusted values to stressed_* columns.",
        "File input sources are reserved for a follow-up issue.",
    ]
    return canonicalize_value(
        {
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "simulation_type": scenario.simulation_type,
            "source_scenario_id": source.scenario_id,
            "source_scenario_name": source.scenario_name,
            "overlay_count": len(overlay_config.overlays),
            "row_count": len(frame),
            "affected_columns": affected_columns,
            "overlay_types": [overlay.overlay_type for overlay in overlay_config.overlays],
            "return_columns_stressed": [
                column for column in affected_columns if "return" in column
            ],
            "confidence_columns_stressed": [
                column for column in affected_columns if "confidence" in column
            ],
            "entropy_columns_stressed": [column for column in affected_columns if "entropy" in column],
            "policy_comparison_available": adaptive_total is not None and static_total is not None,
            "stressed_adaptive_return_total": adaptive_total,
            "stressed_static_baseline_return_total": static_total,
            "stressed_adaptive_vs_static_return_delta": None
            if adaptive_total is None or static_total is None
            else adaptive_total - static_total,
            "limitations": limitations,
            "generated_files": {
                "shock_overlay_config_json": "shock_overlay_config.json",
                "shock_overlay_results_csv": "shock_overlay_results.csv",
                "shock_overlay_log_csv": "shock_overlay_log.csv",
                "shock_overlay_summary_json": "shock_overlay_summary.json",
                "manifest_json": "manifest.json",
            },
        }
    )


def _manifest_payload(
    *,
    scenario: MarketSimulationScenarioConfig,
    simulation_run_id: str,
    source: _SourceInput,
    result_rows: list[dict[str, Any]],
    overlay_config: _ShockOverlayConfig,
    log_row_count: int,
    generated_files: Mapping[str, str],
) -> dict[str, Any]:
    return canonicalize_value(
        {
            "artifact_type": "shock_overlay",
            "schema_version": SHOCK_OVERLAY_SCHEMA_VERSION,
            "simulation_run_id": simulation_run_id,
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "simulation_type": scenario.simulation_type,
            "source_scenario": {
                "scenario_id": source.scenario_id,
                "scenario_name": source.scenario_name,
                "artifact_path": _display_path(source.path),
            },
            "generated_files": dict(generated_files),
            "row_counts": {
                "shock_overlay_results_csv": len(result_rows),
                "shock_overlay_log_csv": log_row_count,
            },
            "overlay_count": len(overlay_config.overlays),
            "relative_paths": {
                "scenario_dir": scenario.scenario_id,
                **dict(generated_files),
            },
            "limitations": [
                "Shock overlays are deterministic robustness tests, not forecasts.",
                "No live trading, broker, real-time feed, or order-book behavior is modeled.",
                "File input sources are reserved for a follow-up issue.",
            ],
        }
    )


def _log_row(
    *,
    scenario: MarketSimulationScenarioConfig,
    overlay: _OverlayConfig,
    overlay_index: int,
    target_column: str,
    stressed_column: str,
    status: str,
    message: str,
    before: pd.Series | None = None,
    after: pd.Series | None = None,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "overlay_index": overlay_index,
        "overlay_name": overlay.name,
        "overlay_type": overlay.overlay_type,
        "target_column": target_column,
        "stressed_column": stressed_column,
        "parameter_name": overlay.parameter_name,
        "parameter_value": overlay.parameter_value,
        "rows_affected": 0 if before is None else int(before.notna().sum()),
        "before_mean": _series_stat(before, "mean"),
        "after_mean": _series_stat(after, "mean"),
        "before_min": _series_stat(before, "min"),
        "after_min": _series_stat(after, "min"),
        "before_max": _series_stat(before, "max"),
        "after_max": _series_stat(after, "max"),
        "status": status,
        "message": message,
    }


def _total_return(frame: pd.DataFrame, column: str | None) -> float | None:
    if column is None or column not in frame.columns or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        return None
    return float((1.0 + values.astype(float)).prod() - 1.0)


def _series_stat(series: pd.Series | None, stat: str) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    if stat == "mean":
        return float(values.mean())
    if stat == "min":
        return float(values.min())
    if stat == "max":
        return float(values.max())
    raise ValueError(stat)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise MarketSimulationConfigError(f"Shock overlay input is missing required columns: {missing}.")


def _required_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MarketSimulationConfigError(f"Shock overlay field '{field_name}' must be a mapping.")
    return value


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MarketSimulationConfigError(
            f"Shock overlay field '{field_name}' must be a non-empty string."
        )
    return value.strip()


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name)


def _stressed(column: str | None) -> str | None:
    return None if column is None else f"stressed_{column}"


def _value(row: pd.Series, column: str | None) -> Any:
    if column is None or column not in row:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


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
