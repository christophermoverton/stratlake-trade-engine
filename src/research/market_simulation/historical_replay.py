from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from src.research.market_simulation.config import (
    MarketSimulationConfig,
    MarketSimulationScenarioConfig,
)
from src.research.market_simulation.validation import MarketSimulationConfigError
from src.research.registry import canonicalize_value, serialize_canonical_json

HISTORICAL_REPLAY_SCHEMA_VERSION = "1.0"

HISTORICAL_EPISODE_CATALOG_COLUMNS = (
    "scenario_id",
    "episode_id",
    "episode_name",
    "episode_type",
    "start",
    "end",
    "dataset_path",
    "row_count",
    "symbol_count",
    "regime_count",
    "has_policy_returns",
    "has_confidence",
    "has_entropy",
    "description",
)

EPISODE_REPLAY_RESULT_COLUMNS = (
    "scenario_id",
    "episode_id",
    "episode_name",
    "ts_utc",
    "symbol",
    "source_return",
    "regime_label",
    "gmm_confidence",
    "gmm_entropy",
    "adaptive_policy_return",
    "static_baseline_return",
    "adaptive_vs_static_return_delta",
)

EPISODE_POLICY_COMPARISON_COLUMNS = (
    "scenario_id",
    "episode_id",
    "episode_name",
    "episode_type",
    "start",
    "end",
    "row_count",
    "symbol_count",
    "adaptive_return_total",
    "static_baseline_return_total",
    "adaptive_vs_static_return_delta",
    "adaptive_volatility",
    "static_baseline_volatility",
    "adaptive_max_drawdown",
    "static_baseline_max_drawdown",
    "adaptive_vs_static_drawdown_delta",
    "mean_confidence",
    "mean_entropy",
    "fallback_data_available",
    "comparison_status",
    "primary_reason",
)


@dataclass(frozen=True)
class HistoricalEpisodeReplayResult:
    scenario_id: str
    scenario_name: str
    output_dir: Path
    historical_episode_catalog_path: Path
    episode_replay_results_path: Path
    episode_policy_comparison_path: Path
    episode_summary_path: Path
    manifest_path: Path
    episode_count: int
    replayed_row_count: int


def run_historical_episode_replays(
    config: MarketSimulationConfig,
    *,
    simulation_run_id: str,
    market_simulations_output_dir: Path,
) -> list[HistoricalEpisodeReplayResult]:
    results: list[HistoricalEpisodeReplayResult] = []
    for scenario in config.market_simulations:
        if scenario.simulation_type != "historical_episode_replay" or not scenario.enabled:
            continue
        if not _has_replay_method_config(scenario.method_config):
            continue
        results.append(
            run_historical_episode_replay(
                scenario,
                simulation_run_id=simulation_run_id,
                market_simulations_output_dir=market_simulations_output_dir,
            )
        )
    return results


def _has_replay_method_config(method_config: Mapping[str, Any]) -> bool:
    return any(
        key in method_config for key in ("dataset_path", "episodes", "episode_catalog_path")
    )


def run_historical_episode_replay(
    scenario: MarketSimulationScenarioConfig,
    *,
    simulation_run_id: str,
    market_simulations_output_dir: Path,
) -> HistoricalEpisodeReplayResult:
    replay_config = _ReplayConfig.from_scenario(scenario)
    source_frame = _load_dataset(replay_config.dataset_path)
    normalized = _normalize_source_frame(source_frame, replay_config)

    scenario_dir = market_simulations_output_dir / scenario.scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    catalog_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []

    for episode in replay_config.episodes:
        episode_frame = _select_episode_rows(normalized, episode)
        episode_id = generate_episode_id(
            scenario_id=scenario.scenario_id,
            episode_name=episode["episode_name"],
            start=episode["start"],
            end=episode["end"],
            dataset_path=replay_config.display_dataset_path,
        )
        availability = _column_availability(replay_config, normalized)
        catalog_rows.append(
            _catalog_row(
                scenario=scenario,
                episode=episode,
                episode_id=episode_id,
                frame=episode_frame,
                replay_config=replay_config,
                dataset_path=replay_config.display_dataset_path,
                availability=availability,
            )
        )
        replay_rows.extend(
            _replay_rows(
                scenario=scenario,
                episode=episode,
                episode_id=episode_id,
                frame=episode_frame,
                replay_config=replay_config,
                availability=availability,
            )
        )
        comparison_row = _comparison_row(
            scenario=scenario,
            episode=episode,
            episode_id=episode_id,
            frame=episode_frame,
            replay_config=replay_config,
            availability=availability,
        )
        comparison_rows.append(comparison_row)
        episode_summaries.append(
            {
                "episode_id": episode_id,
                "episode_name": episode["episode_name"],
                "episode_type": episode["episode_type"],
                "start": episode["start"],
                "end": episode["end"],
                "row_count": len(episode_frame),
                "comparison_status": comparison_row["comparison_status"],
                "primary_reason": comparison_row["primary_reason"],
            }
        )

    paths = {
        "historical_episode_catalog_csv": scenario_dir / "historical_episode_catalog.csv",
        "episode_replay_results_csv": scenario_dir / "episode_replay_results.csv",
        "episode_policy_comparison_csv": scenario_dir / "episode_policy_comparison.csv",
        "episode_summary_json": scenario_dir / "episode_summary.json",
        "manifest_json": scenario_dir / "manifest.json",
    }
    _write_csv(paths["historical_episode_catalog_csv"], catalog_rows, HISTORICAL_EPISODE_CATALOG_COLUMNS)
    _write_csv(paths["episode_replay_results_csv"], replay_rows, EPISODE_REPLAY_RESULT_COLUMNS)
    _write_csv(paths["episode_policy_comparison_csv"], comparison_rows, EPISODE_POLICY_COMPARISON_COLUMNS)

    summary = _summary_payload(
        scenario=scenario,
        catalog_rows=catalog_rows,
        comparison_rows=comparison_rows,
        episode_summaries=episode_summaries,
    )
    manifest = _manifest_payload(
        scenario=scenario,
        simulation_run_id=simulation_run_id,
        replay_config=replay_config,
        catalog_rows=catalog_rows,
        replay_rows=replay_rows,
        comparison_rows=comparison_rows,
        generated_files={key: path.name for key, path in paths.items()},
    )
    _write_json(paths["episode_summary_json"], summary)
    _write_json(paths["manifest_json"], manifest)

    return HistoricalEpisodeReplayResult(
        scenario_id=scenario.scenario_id,
        scenario_name=scenario.name,
        output_dir=scenario_dir,
        historical_episode_catalog_path=paths["historical_episode_catalog_csv"],
        episode_replay_results_path=paths["episode_replay_results_csv"],
        episode_policy_comparison_path=paths["episode_policy_comparison_csv"],
        episode_summary_path=paths["episode_summary_json"],
        manifest_path=paths["manifest_json"],
        episode_count=len(catalog_rows),
        replayed_row_count=len(replay_rows),
    )


def generate_episode_id(
    *,
    scenario_id: str,
    episode_name: str,
    start: str,
    end: str,
    dataset_path: str,
) -> str:
    payload = {
        "scenario_id": scenario_id,
        "episode_name": episode_name,
        "start": start,
        "end": end,
        "dataset_path": dataset_path,
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{_slugify(episode_name)}_{digest}"


@dataclass(frozen=True)
class _ReplayConfig:
    dataset_path: Path
    display_dataset_path: str
    timestamp_column: str
    return_column: str
    symbol_column: str | None
    regime_column: str | None
    confidence_column: str | None
    entropy_column: str | None
    adaptive_policy_return_column: str | None
    static_baseline_return_column: str | None
    episodes: tuple[dict[str, str], ...]

    @classmethod
    def from_scenario(cls, scenario: MarketSimulationScenarioConfig) -> "_ReplayConfig":
        method_config = scenario.method_config
        dataset_path_value = _required_string(method_config.get("dataset_path"), "method_config.dataset_path")
        dataset_path = Path(dataset_path_value)
        resolved_dataset_path = dataset_path if dataset_path.is_absolute() else Path.cwd() / dataset_path
        timestamp_column = _required_string(
            method_config.get("timestamp_column"), "method_config.timestamp_column"
        )
        return_column = _required_string(method_config.get("return_column"), "method_config.return_column")
        episodes = _resolve_episodes(method_config)
        return cls(
            dataset_path=resolved_dataset_path,
            display_dataset_path=_display_path(dataset_path),
            timestamp_column=timestamp_column,
            return_column=return_column,
            symbol_column=_optional_string(method_config.get("symbol_column"), "method_config.symbol_column"),
            regime_column=_optional_string(method_config.get("regime_column"), "method_config.regime_column"),
            confidence_column=_optional_string(
                method_config.get("confidence_column"), "method_config.confidence_column"
            ),
            entropy_column=_optional_string(method_config.get("entropy_column"), "method_config.entropy_column"),
            adaptive_policy_return_column=_optional_string(
                method_config.get("adaptive_policy_return_column"),
                "method_config.adaptive_policy_return_column",
            ),
            static_baseline_return_column=_optional_string(
                method_config.get("static_baseline_return_column"),
                "method_config.static_baseline_return_column",
            ),
            episodes=episodes,
        )


def _resolve_episodes(method_config: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    inline_episodes = method_config.get("episodes")
    catalog_path_value = method_config.get("episode_catalog_path")
    if inline_episodes is not None and catalog_path_value is not None:
        raise MarketSimulationConfigError(
            "Historical episode replay may configure either method_config.episodes or "
            "method_config.episode_catalog_path, not both."
        )
    if catalog_path_value is not None:
        raw_episodes = _load_episode_catalog(_required_string(catalog_path_value, "method_config.episode_catalog_path"))
    else:
        raw_episodes = inline_episodes
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise MarketSimulationConfigError(
            "Historical episode replay requires at least one method_config.episodes entry."
        )

    episodes: list[dict[str, str]] = []
    for index, item in enumerate(raw_episodes):
        if not isinstance(item, Mapping):
            raise MarketSimulationConfigError(
                f"Historical episode entry at index {index} must be a mapping."
            )
        episode_name = _required_string(item.get("episode_name"), f"episodes[{index}].episode_name")
        episodes.append(
            {
                "episode_name": episode_name,
                "episode_type": _optional_string(item.get("episode_type"), f"episodes[{index}].episode_type")
                or "historical_episode",
                "start": _required_string(item.get("start"), f"episodes[{index}].start"),
                "end": _required_string(item.get("end"), f"episodes[{index}].end"),
                "description": _optional_string(item.get("description"), f"episodes[{index}].description") or "",
            }
        )
    return tuple(episodes)


def _load_episode_catalog(catalog_path_value: str) -> list[dict[str, Any]]:
    catalog_path = Path(catalog_path_value)
    resolved = catalog_path if catalog_path.is_absolute() else Path.cwd() / catalog_path
    if not resolved.exists():
        raise MarketSimulationConfigError(
            f"Historical episode catalog does not exist: {_display_path(catalog_path)}"
        )
    suffix = resolved.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        if isinstance(payload, Mapping):
            payload = payload.get("episodes")
        return payload
    if suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            payload = payload.get("episodes")
        return payload
    if suffix == ".csv":
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise MarketSimulationConfigError(
        f"Unsupported historical episode catalog format {resolved.suffix!r}. Use CSV, JSON, YAML, or YML."
    )


def _load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise MarketSimulationConfigError(
            f"Historical episode replay dataset does not exist: {_display_path(dataset_path)}"
        )
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(dataset_path)
    raise MarketSimulationConfigError(
        f"Unsupported historical episode replay dataset format {dataset_path.suffix!r}. Use CSV or Parquet."
    )


def _normalize_source_frame(frame: pd.DataFrame, replay_config: _ReplayConfig) -> pd.DataFrame:
    _require_columns(frame, (replay_config.timestamp_column, replay_config.return_column))
    normalized = frame.copy()
    normalized["_replay_ts_utc"] = pd.to_datetime(
        normalized[replay_config.timestamp_column],
        errors="raise",
        utc=True,
    )
    if normalized["_replay_ts_utc"].isna().any():
        raise MarketSimulationConfigError(
            f"Historical episode replay timestamp column {replay_config.timestamp_column!r} contains null values."
        )
    sort_columns = ["_replay_ts_utc"]
    if replay_config.symbol_column is not None and replay_config.symbol_column in normalized.columns:
        sort_columns.append(replay_config.symbol_column)
    normalized = normalized.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    return normalized


def _select_episode_rows(frame: pd.DataFrame, episode: Mapping[str, str]) -> pd.DataFrame:
    start = _parse_window_timestamp(episode["start"], "start")
    end = _parse_window_timestamp(episode["end"], "end")
    if end < start:
        raise MarketSimulationConfigError(
            f"Historical episode {episode['episode_name']!r} end must be on or after start."
        )
    mask = (frame["_replay_ts_utc"] >= start) & (frame["_replay_ts_utc"] <= end)
    return frame.loc[mask].copy().reset_index(drop=True)


def _parse_window_timestamp(value: str, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value).tz_localize("UTC")
    except TypeError:
        try:
            timestamp = pd.Timestamp(value).tz_convert("UTC")
        except TypeError as exc:
            raise MarketSimulationConfigError(
                f"Historical episode replay {field_name} timestamp {value!r} is invalid."
            ) from exc
    except ValueError as exc:
        raise MarketSimulationConfigError(
            f"Historical episode replay {field_name} timestamp {value!r} is invalid."
        ) from exc
    if field_name == "end" and _is_date_only(value):
        timestamp = timestamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return timestamp


def _is_date_only(value: str) -> bool:
    return len(value) == 10 and value[4] == "-" and value[7] == "-"


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise MarketSimulationConfigError(
            f"Historical episode replay dataset is missing required columns: {missing}."
        )


def _column_availability(replay_config: _ReplayConfig, frame: pd.DataFrame) -> dict[str, bool]:
    adaptive = _has_column(frame, replay_config.adaptive_policy_return_column)
    static = _has_column(frame, replay_config.static_baseline_return_column)
    return {
        "has_symbol": _has_column(frame, replay_config.symbol_column),
        "has_regime": _has_column(frame, replay_config.regime_column),
        "has_confidence": _has_column(frame, replay_config.confidence_column),
        "has_entropy": _has_column(frame, replay_config.entropy_column),
        "has_adaptive_policy_return": adaptive,
        "has_static_baseline_return": static,
        "has_policy_returns": adaptive and static,
    }


def _catalog_row(
    *,
    scenario: MarketSimulationScenarioConfig,
    episode: Mapping[str, str],
    episode_id: str,
    frame: pd.DataFrame,
    replay_config: _ReplayConfig,
    dataset_path: str,
    availability: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "episode_id": episode_id,
        "episode_name": episode["episode_name"],
        "episode_type": episode["episode_type"],
        "start": episode["start"],
        "end": episode["end"],
        "dataset_path": dataset_path,
        "row_count": len(frame),
        "symbol_count": _nunique(frame, replay_config.symbol_column) if availability["has_symbol"] else 0,
        "regime_count": _nunique(frame, replay_config.regime_column) if availability["has_regime"] else 0,
        "has_policy_returns": availability["has_policy_returns"],
        "has_confidence": availability["has_confidence"],
        "has_entropy": availability["has_entropy"],
        "description": episode.get("description", ""),
    }


def _replay_rows(
    *,
    scenario: MarketSimulationScenarioConfig,
    episode: Mapping[str, str],
    episode_id: str,
    frame: pd.DataFrame,
    replay_config: _ReplayConfig,
    availability: Mapping[str, bool],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        adaptive = _value(row, replay_config.adaptive_policy_return_column, availability["has_adaptive_policy_return"])
        static = _value(row, replay_config.static_baseline_return_column, availability["has_static_baseline_return"])
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "episode_id": episode_id,
                "episode_name": episode["episode_name"],
                "ts_utc": row["_replay_ts_utc"].isoformat().replace("+00:00", "Z"),
                "symbol": _value(row, replay_config.symbol_column, availability["has_symbol"]),
                "source_return": _value(row, replay_config.return_column, True),
                "regime_label": _value(row, replay_config.regime_column, availability["has_regime"]),
                "gmm_confidence": _value(row, replay_config.confidence_column, availability["has_confidence"]),
                "gmm_entropy": _value(row, replay_config.entropy_column, availability["has_entropy"]),
                "adaptive_policy_return": adaptive,
                "static_baseline_return": static,
                "adaptive_vs_static_return_delta": None
                if adaptive is None or static is None
                else float(adaptive) - float(static),
            }
        )
    return rows


def _comparison_row(
    *,
    scenario: MarketSimulationScenarioConfig,
    episode: Mapping[str, str],
    episode_id: str,
    frame: pd.DataFrame,
    replay_config: _ReplayConfig,
    availability: Mapping[str, bool],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "episode_id": episode_id,
        "episode_name": episode["episode_name"],
        "episode_type": episode["episode_type"],
        "start": episode["start"],
        "end": episode["end"],
        "row_count": len(frame),
        "symbol_count": _nunique(frame, replay_config.symbol_column) if availability["has_symbol"] else 0,
        "adaptive_return_total": None,
        "static_baseline_return_total": None,
        "adaptive_vs_static_return_delta": None,
        "adaptive_volatility": None,
        "static_baseline_volatility": None,
        "adaptive_max_drawdown": None,
        "static_baseline_max_drawdown": None,
        "adaptive_vs_static_drawdown_delta": None,
        "mean_confidence": _mean(frame, replay_config.confidence_column) if availability["has_confidence"] else None,
        "mean_entropy": _mean(frame, replay_config.entropy_column) if availability["has_entropy"] else None,
        "fallback_data_available": availability["has_policy_returns"],
        "comparison_status": "insufficient_policy_columns",
        "primary_reason": "Adaptive and static baseline policy return columns are required for comparison.",
    }
    if not availability["has_policy_returns"]:
        missing = [
            column
            for column, has_column in (
                (
                    replay_config.adaptive_policy_return_column or "adaptive_policy_return_column",
                    availability["has_adaptive_policy_return"],
                ),
                (
                    replay_config.static_baseline_return_column or "static_baseline_return_column",
                    availability["has_static_baseline_return"],
                ),
            )
            if not has_column
        ]
        row["primary_reason"] = f"Missing policy return columns: {', '.join(missing)}."
        return row

    adaptive_returns = pd.to_numeric(frame[replay_config.adaptive_policy_return_column], errors="coerce")
    static_returns = pd.to_numeric(frame[replay_config.static_baseline_return_column], errors="coerce")
    if adaptive_returns.isna().any() or static_returns.isna().any():
        row["comparison_status"] = "insufficient_policy_values"
        row["primary_reason"] = "Policy return columns contain null or non-numeric values in the episode window."
        return row

    adaptive_total = _total_return(adaptive_returns)
    static_total = _total_return(static_returns)
    adaptive_drawdown = _max_drawdown(adaptive_returns)
    static_drawdown = _max_drawdown(static_returns)
    row.update(
        {
            "adaptive_return_total": adaptive_total,
            "static_baseline_return_total": static_total,
            "adaptive_vs_static_return_delta": adaptive_total - static_total,
            "adaptive_volatility": float(adaptive_returns.std(ddof=0)),
            "static_baseline_volatility": float(static_returns.std(ddof=0)),
            "adaptive_max_drawdown": adaptive_drawdown,
            "static_baseline_max_drawdown": static_drawdown,
            "adaptive_vs_static_drawdown_delta": adaptive_drawdown - static_drawdown,
            "comparison_status": "available",
            "primary_reason": "",
        }
    )
    return row


def _summary_payload(
    *,
    scenario: MarketSimulationScenarioConfig,
    catalog_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    episode_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    available_rows = [row for row in comparison_rows if row["comparison_status"] == "available"]
    best_delta = (
        max(available_rows, key=lambda row: row["adaptive_vs_static_return_delta"])
        if available_rows
        else None
    )
    worst_drawdown = min(available_rows, key=lambda row: row["adaptive_max_drawdown"]) if available_rows else None
    limitations = [
        "Episode windows use inclusive start and inclusive end timestamps; date-only end values include the full UTC date.",
        "Historical episode replay replays configured source rows only; it does not mutate shocks or generate paths.",
        "Policy comparison is limited to rows with configured adaptive and static return columns.",
    ]
    if len(available_rows) < len(comparison_rows):
        limitations.append("One or more episodes lack policy columns or valid policy values for comparison.")
    return canonicalize_value(
        {
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "simulation_type": scenario.simulation_type,
            "episode_count": len(catalog_rows),
            "total_replayed_rows": sum(int(row["row_count"]) for row in catalog_rows),
            "episodes": episode_summaries,
            "policy_comparison_available": bool(available_rows),
            "best_episode_by_adaptive_delta": _episode_pick(best_delta, "adaptive_vs_static_return_delta"),
            "worst_episode_by_drawdown": _episode_pick(worst_drawdown, "adaptive_max_drawdown"),
            "limitations": limitations,
            "generated_files": {
                "historical_episode_catalog_csv": "historical_episode_catalog.csv",
                "episode_replay_results_csv": "episode_replay_results.csv",
                "episode_policy_comparison_csv": "episode_policy_comparison.csv",
                "episode_summary_json": "episode_summary.json",
                "manifest_json": "manifest.json",
            },
        }
    )


def _manifest_payload(
    *,
    scenario: MarketSimulationScenarioConfig,
    simulation_run_id: str,
    replay_config: _ReplayConfig,
    catalog_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    generated_files: Mapping[str, str],
) -> dict[str, Any]:
    return canonicalize_value(
        {
            "artifact_type": "historical_episode_replay",
            "schema_version": HISTORICAL_REPLAY_SCHEMA_VERSION,
            "simulation_run_id": simulation_run_id,
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "simulation_type": scenario.simulation_type,
            "generated_files": dict(generated_files),
            "row_counts": {
                "historical_episode_catalog_csv": len(catalog_rows),
                "episode_replay_results_csv": len(replay_rows),
                "episode_policy_comparison_csv": len(comparison_rows),
            },
            "source_dataset_metadata": {
                "dataset_path": replay_config.display_dataset_path,
                "format": replay_config.dataset_path.suffix.lower().lstrip("."),
                "timestamp_column": replay_config.timestamp_column,
                "return_column": replay_config.return_column,
                "symbol_column": replay_config.symbol_column,
                "regime_column": replay_config.regime_column,
                "confidence_column": replay_config.confidence_column,
                "entropy_column": replay_config.entropy_column,
                "adaptive_policy_return_column": replay_config.adaptive_policy_return_column,
                "static_baseline_return_column": replay_config.static_baseline_return_column,
            },
            "episode_count": len(catalog_rows),
            "total_replayed_rows": len(replay_rows),
            "relative_paths": {
                "scenario_dir": scenario.scenario_id,
                **dict(generated_files),
            },
            "limitations": [
                "Episode windows use inclusive start and inclusive end timestamps; date-only end values include the full UTC date.",
                "Replay preserves source rows and deterministic sorting; it does not add shock overlays or synthetic paths.",
                "Artifacts are for research stress testing, not live trading or market forecasting.",
            ],
        }
    )


def _episode_pick(row: Mapping[str, Any] | None, metric: str) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "episode_id": row["episode_id"],
        "episode_name": row["episode_name"],
        "metric": metric,
        "value": row[metric],
    }


def _total_return(returns: pd.Series) -> float:
    return float((1.0 + returns.astype(float)).prod() - 1.0)


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.astype(float)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _mean(frame: pd.DataFrame, column: str | None) -> float | None:
    if column is None or column not in frame.columns or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _nunique(frame: pd.DataFrame, column: str | None) -> int:
    if column is None or column not in frame.columns:
        return 0
    return int(frame[column].dropna().nunique())


def _has_column(frame: pd.DataFrame, column: str | None) -> bool:
    return column is not None and column in frame.columns


def _value(row: pd.Series, column: str | None, available: bool) -> Any:
    if not available or column is None:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return value


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MarketSimulationConfigError(
            f"Historical episode replay field '{field_name}' must be a non-empty string."
        )
    return value.strip()


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name)


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


def _slugify(value: str) -> str:
    chars = [character.lower() if character.isalnum() else "_" for character in value.strip()]
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "episode"
