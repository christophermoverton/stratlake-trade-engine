from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.research.market_simulation.block_bootstrap import BlockBootstrapResult
from src.research.market_simulation.config import MarketSimulationConfig, StressMetricsConfig
from src.research.market_simulation.historical_replay import HistoricalEpisodeReplayResult
from src.research.market_simulation.monte_carlo import MonteCarloResult
from src.research.market_simulation.shock_overlay import ShockOverlayResult
from src.research.registry import canonicalize_value

SIMULATION_STRESS_METRICS_SCHEMA_VERSION = "1.0"

PATH_METRIC_COLUMNS = (
    "simulation_run_id",
    "scenario_id",
    "scenario_name",
    "simulation_type",
    "source_artifact_type",
    "path_id",
    "episode_id",
    "path_index",
    "path_length",
    "row_count",
    "policy_name",
    "tail_quantile",
    "has_return_metrics",
    "has_policy_metrics",
    "has_regime_metrics",
    "total_return",
    "annualized_return",
    "volatility",
    "max_drawdown",
    "min_period_return",
    "max_period_return",
    "mean_period_return",
    "tail_quantile_return",
    "adaptive_return_total",
    "static_baseline_return_total",
    "adaptive_vs_static_return_delta",
    "adaptive_vs_static_win",
    "policy_failure",
    "failure_reason",
    "regime_transition_count",
    "unique_regime_count",
    "stress_regime_share",
    "max_regime_duration",
    "mean_regime_duration",
    "fallback_activation_count",
    "overlay_count",
    "stress_score",
)

SUMMARY_COLUMNS = (
    "simulation_run_id",
    "scenario_id",
    "scenario_name",
    "simulation_type",
    "path_count",
    "row_count",
    "tail_quantile",
    "paths_with_return_metrics",
    "paths_with_policy_metrics",
    "paths_with_regime_metrics",
    "mean_total_return",
    "median_total_return",
    "tail_quantile_total_return",
    "worst_total_return",
    "mean_max_drawdown",
    "worst_max_drawdown",
    "mean_volatility",
    "policy_failure_rate",
    "adaptive_vs_static_win_rate",
    "mean_adaptive_vs_static_delta",
    "mean_regime_transition_count",
    "mean_stress_regime_share",
    "worst_stress_score",
    "mean_stress_score",
    "ranking_metric",
    "notes",
)

LEADERBOARD_RANKING_METRICS = frozenset(
    {
        "path_count",
        "row_count",
        "paths_with_return_metrics",
        "paths_with_policy_metrics",
        "paths_with_regime_metrics",
        "mean_total_return",
        "median_total_return",
        "tail_quantile_total_return",
        "worst_total_return",
        "mean_max_drawdown",
        "worst_max_drawdown",
        "mean_volatility",
        "policy_failure_rate",
        "adaptive_vs_static_win_rate",
        "mean_adaptive_vs_static_delta",
        "mean_regime_transition_count",
        "mean_stress_regime_share",
        "worst_stress_score",
        "mean_stress_score",
    }
)

LEADERBOARD_COLUMNS = (
    "rank",
    "simulation_run_id",
    "scenario_id",
    "scenario_name",
    "simulation_type",
    "policy_name",
    "path_count",
    "tail_quantile",
    "policy_failure_rate",
    "adaptive_vs_static_win_rate",
    "tail_quantile_total_return",
    "worst_max_drawdown",
    "mean_stress_score",
    "ranking_metric",
    "ranking_value",
    "tie_breaker",
    "decision_label",
    "primary_reason",
)


@dataclass(frozen=True)
class SimulationStressMetricsResult:
    output_dir: Path
    path_metrics_path: Path
    summary_path: Path
    leaderboard_path: Path
    policy_failure_summary_path: Path
    manifest_path: Path
    metric_config_path: Path
    path_metric_row_count: int
    summary_row_count: int
    leaderboard_row_count: int
    policy_failure_rate: float


def run_simulation_stress_metrics(
    config: MarketSimulationConfig,
    *,
    simulation_run_id: str,
    market_simulations_output_dir: Path,
    historical_episode_replay_results: list[HistoricalEpisodeReplayResult],
    block_bootstrap_results: list[BlockBootstrapResult],
    monte_carlo_results: list[MonteCarloResult],
    shock_overlay_results: list[ShockOverlayResult],
) -> SimulationStressMetricsResult | None:
    metrics_config = config.stress_metrics
    if not metrics_config.enabled:
        return None
    output_dir = market_simulations_output_dir / _safe_dir_name(metrics_config.output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_names = {scenario.scenario_id: scenario.name for scenario in config.market_simulations}
    path_rows: list[dict[str, Any]] = []
    path_rows.extend(
        _historical_rows(simulation_run_id, historical_episode_replay_results, scenario_names, metrics_config)
    )
    path_rows.extend(_shock_rows(simulation_run_id, shock_overlay_results, scenario_names, metrics_config))
    path_rows.extend(_bootstrap_rows(simulation_run_id, block_bootstrap_results, scenario_names, metrics_config))
    path_rows.extend(_monte_carlo_rows(simulation_run_id, monte_carlo_results, scenario_names, metrics_config))
    path_rows = sorted(
        path_rows,
        key=lambda row: (
            str(row["scenario_name"]),
            str(row["simulation_type"]),
            str(row["scenario_id"]),
            str(row.get("policy_name") or ""),
            str(row.get("episode_id") or ""),
            int(row.get("path_index") or 0),
            str(row.get("path_id") or ""),
        ),
    )
    summary_rows = _summary_rows(path_rows, metrics_config)
    leaderboard_rows = _leaderboard_rows(summary_rows, metrics_config)
    failure_summary = _policy_failure_summary(
        simulation_run_id=simulation_run_id,
        path_rows=path_rows,
        summary_rows=summary_rows,
        leaderboard_rows=leaderboard_rows,
        metrics_config=metrics_config,
    )
    paths = {
        "simulation_path_metrics_csv": output_dir / "simulation_path_metrics.csv",
        "simulation_summary_csv": output_dir / "simulation_summary.csv",
        "simulation_leaderboard_csv": output_dir / "simulation_leaderboard.csv",
        "policy_failure_summary_json": output_dir / "policy_failure_summary.json",
        "simulation_metric_config_json": output_dir / "simulation_metric_config.json",
        "manifest_json": output_dir / "manifest.json",
    }
    manifest = _manifest_payload(
        simulation_run_id=simulation_run_id,
        output_dir_name=output_dir.name,
        paths=paths,
        path_rows=path_rows,
        summary_rows=summary_rows,
        leaderboard_rows=leaderboard_rows,
        metrics_config=metrics_config,
    )
    _write_csv(paths["simulation_path_metrics_csv"], path_rows, PATH_METRIC_COLUMNS)
    _write_csv(paths["simulation_summary_csv"], summary_rows, SUMMARY_COLUMNS)
    _write_csv(paths["simulation_leaderboard_csv"], leaderboard_rows, LEADERBOARD_COLUMNS)
    _write_json(paths["policy_failure_summary_json"], failure_summary)
    _write_json(paths["simulation_metric_config_json"], metrics_config.to_dict())
    _write_json(paths["manifest_json"], manifest)
    return SimulationStressMetricsResult(
        output_dir=output_dir,
        path_metrics_path=paths["simulation_path_metrics_csv"],
        summary_path=paths["simulation_summary_csv"],
        leaderboard_path=paths["simulation_leaderboard_csv"],
        policy_failure_summary_path=paths["policy_failure_summary_json"],
        manifest_path=paths["manifest_json"],
        metric_config_path=paths["simulation_metric_config_json"],
        path_metric_row_count=len(path_rows),
        summary_row_count=len(summary_rows),
        leaderboard_row_count=len(leaderboard_rows),
        policy_failure_rate=_rate(row["policy_failure"] for row in path_rows),
    )


def _historical_rows(
    simulation_run_id: str,
    results: list[HistoricalEpisodeReplayResult],
    scenario_names: Mapping[str, str],
    config: StressMetricsConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        frame = pd.read_csv(result.episode_policy_comparison_path)
        for _, item in frame.iterrows():
            policy_available = str(item.get("comparison_status", "")) == "available"
            adaptive_total = _number(item.get("adaptive_return_total"))
            static_total = _number(item.get("static_baseline_return_total"))
            metrics = {
                "total_return": adaptive_total,
                "volatility": _number(item.get("adaptive_volatility")),
                "max_drawdown": _number(item.get("adaptive_max_drawdown")),
            }
            row = _base_row(
                simulation_run_id,
                result.scenario_id,
                scenario_names.get(result.scenario_id, result.scenario_name),
                "historical_episode_replay",
                "episode_policy_comparison",
                config,
                episode_id=_string_or_none(item.get("episode_id")),
                row_count=_int_or_none(item.get("row_count")),
                path_length=_int_or_none(item.get("row_count")),
                policy_name="adaptive_policy",
                has_return_metrics=adaptive_total is not None,
                has_policy_metrics=policy_available,
                **metrics,
                adaptive_return_total=adaptive_total,
                static_baseline_return_total=static_total,
                adaptive_vs_static_return_delta=_number(item.get("adaptive_vs_static_return_delta")),
                adaptive_vs_static_win=_bool_from_delta(_number(item.get("adaptive_vs_static_return_delta"))),
            )
            rows.append(_finalize_row(row, config))
    return rows


def _shock_rows(
    simulation_run_id: str,
    results: list[ShockOverlayResult],
    scenario_names: Mapping[str, str],
    config: StressMetricsConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        frame = pd.read_csv(result.shock_overlay_results_path)
        group_cols = ["source_episode_id"] if "source_episode_id" in frame.columns else [None]
        groups = frame.groupby(group_cols[0], sort=True) if group_cols[0] is not None else [(None, frame)]
        for episode_id, group in groups:
            return_metrics = _return_metrics(group.get("stressed_source_return"), config.tail_quantile)
            adaptive = _series_total(group.get("stressed_adaptive_policy_return"))
            static = _series_total(group.get("stressed_static_baseline_return"))
            delta = None if adaptive is None or static is None else adaptive - static
            row = _base_row(
                simulation_run_id,
                result.scenario_id,
                scenario_names.get(result.scenario_id, result.scenario_name),
                "shock_overlay",
                "shock_overlay_results",
                config,
                episode_id=None if pd.isna(episode_id) else str(episode_id),
                row_count=len(group),
                path_length=len(group),
                policy_name="stressed_adaptive_policy",
                has_return_metrics=return_metrics["total_return"] is not None,
                has_policy_metrics=adaptive is not None and static is not None,
                overlay_count=int(group["overlay_count"].max()) if "overlay_count" in group else result.overlay_count,
                adaptive_return_total=adaptive,
                static_baseline_return_total=static,
                adaptive_vs_static_return_delta=delta,
                adaptive_vs_static_win=_bool_from_delta(delta),
                **return_metrics,
            )
            rows.append(_finalize_row(row, config))
    return rows


def _bootstrap_rows(
    simulation_run_id: str,
    results: list[BlockBootstrapResult],
    scenario_names: Mapping[str, str],
    config: StressMetricsConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        frame = pd.read_parquet(result.simulated_return_paths_path)
        catalog = pd.read_csv(result.bootstrap_path_catalog_path)
        path_index_by_id = dict(zip(catalog["path_id"], catalog["path_index"], strict=False))
        for path_id, group in frame.groupby("path_id", sort=True):
            return_metrics = _return_metrics(group.get("simulated_return"), config.tail_quantile)
            regime_metrics = _regime_metrics(group.get("regime_label"), config.stress_regimes)
            row = _base_row(
                simulation_run_id,
                result.scenario_id,
                scenario_names.get(result.scenario_id, result.scenario_name),
                "regime_block_bootstrap",
                "simulated_return_paths",
                config,
                path_id=str(path_id),
                path_index=_int_or_none(path_index_by_id.get(path_id)),
                row_count=len(group),
                path_length=len(group),
                policy_name="simulated_return_path",
                has_return_metrics=return_metrics["total_return"] is not None,
                has_regime_metrics=regime_metrics["unique_regime_count"] is not None,
                **return_metrics,
                **regime_metrics,
            )
            rows.append(_finalize_row(row, config))
    return rows


def _monte_carlo_rows(
    simulation_run_id: str,
    results: list[MonteCarloResult],
    scenario_names: Mapping[str, str],
    config: StressMetricsConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        frame = pd.read_parquet(result.regime_paths_path)
        for path_id, group in frame.groupby("path_id", sort=True):
            regime_metrics = _regime_metrics(group.get("regime_label"), config.stress_regimes)
            row = _base_row(
                simulation_run_id,
                result.scenario_id,
                scenario_names.get(result.scenario_id, result.scenario_name),
                "regime_transition_monte_carlo",
                "monte_carlo_regime_paths",
                config,
                path_id=str(path_id),
                path_index=_int_or_none(group["path_index"].iloc[0]) if "path_index" in group else None,
                row_count=len(group),
                path_length=len(group),
                policy_name="regime_path_only",
                has_regime_metrics=True,
                **regime_metrics,
            )
            rows.append(_finalize_row(row, config))
    return rows


def _base_row(
    simulation_run_id: str,
    scenario_id: str,
    scenario_name: str,
    simulation_type: str,
    source_artifact_type: str,
    config: StressMetricsConfig,
    **values: Any,
) -> dict[str, Any]:
    row = {column: None for column in PATH_METRIC_COLUMNS}
    row.update(
        {
            "simulation_run_id": simulation_run_id,
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "simulation_type": simulation_type,
            "source_artifact_type": source_artifact_type,
            "has_return_metrics": False,
            "has_policy_metrics": False,
            "has_regime_metrics": False,
            "tail_quantile": config.tail_quantile,
            "tail_quantile_return": None,
            "fallback_activation_count": 0,
            "overlay_count": 0,
        }
    )
    row.update(values)
    if row["annualized_return"] is None and row["total_return"] is not None and row["path_length"]:
        row["annualized_return"] = _annualized_return(row["total_return"], int(row["path_length"]))
    return row


def _finalize_row(row: dict[str, Any], config: StressMetricsConfig) -> dict[str, Any]:
    reasons = _failure_reasons(row, config.failure_thresholds)
    row["policy_failure"] = bool(reasons)
    row["failure_reason"] = ";".join(reasons)
    row["stress_score"] = _stress_score(row, config.failure_thresholds)
    return {column: row.get(column) for column in PATH_METRIC_COLUMNS}


def _return_metrics(values: Any, tail_quantile: float) -> dict[str, Any]:
    series = _numeric_series(values)
    if series.empty:
        return {
            "total_return": None,
            "annualized_return": None,
            "volatility": None,
            "max_drawdown": None,
            "min_period_return": None,
            "max_period_return": None,
            "mean_period_return": None,
            "tail_quantile_return": None,
        }
    total = float((1.0 + series).prod() - 1.0)
    return {
        "total_return": total,
        "annualized_return": _annualized_return(total, len(series)),
        "volatility": float(series.std(ddof=0)),
        "max_drawdown": _max_drawdown(series),
        "min_period_return": float(series.min()),
        "max_period_return": float(series.max()),
        "mean_period_return": float(series.mean()),
        "tail_quantile_return": float(series.quantile(tail_quantile)),
    }


def _regime_metrics(values: Any, stress_regimes: Iterable[str]) -> dict[str, Any]:
    if values is None:
        return {
            "regime_transition_count": None,
            "unique_regime_count": None,
            "stress_regime_share": None,
            "max_regime_duration": None,
            "mean_regime_duration": None,
        }
    series = pd.Series(values).dropna().astype(str).reset_index(drop=True)
    if series.empty:
        return {
            "regime_transition_count": None,
            "unique_regime_count": None,
            "stress_regime_share": None,
            "max_regime_duration": None,
            "mean_regime_duration": None,
        }
    transitions = int(series.ne(series.shift()).sum() - 1)
    stress_set = {value.lower() for value in stress_regimes}
    durations: list[int] = []
    current = series.iloc[0]
    count = 0
    for value in series:
        if value == current:
            count += 1
            continue
        durations.append(count)
        current = value
        count = 1
    durations.append(count)
    return {
        "regime_transition_count": transitions,
        "unique_regime_count": int(series.nunique()),
        "stress_regime_share": float(series.str.lower().isin(stress_set).mean()),
        "max_regime_duration": max(durations),
        "mean_regime_duration": float(sum(durations) / len(durations)),
    }


def _summary_rows(path_rows: list[dict[str, Any]], config: StressMetricsConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ranking_metric = _ranking_metric(config)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in path_rows:
        key = (row["scenario_id"], row["scenario_name"], row["simulation_type"])
        groups.setdefault(key, []).append(row)
    for (scenario_id, scenario_name, simulation_type), group in sorted(
        groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])
    ):
        total_returns = _numbers(row["total_return"] for row in group)
        drawdowns = _numbers(row["max_drawdown"] for row in group)
        volatilities = _numbers(row["volatility"] for row in group)
        deltas = _numbers(row["adaptive_vs_static_return_delta"] for row in group)
        transitions = _numbers(row["regime_transition_count"] for row in group)
        stress_shares = _numbers(row["stress_regime_share"] for row in group)
        scores = _numbers(row["stress_score"] for row in group)
        rows.append(
            {
                "simulation_run_id": group[0]["simulation_run_id"],
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "simulation_type": simulation_type,
                "path_count": len(group),
                "row_count": sum(int(row["row_count"] or 0) for row in group),
                "tail_quantile": config.tail_quantile,
                "paths_with_return_metrics": sum(bool(row["has_return_metrics"]) for row in group),
                "paths_with_policy_metrics": sum(bool(row["has_policy_metrics"]) for row in group),
                "paths_with_regime_metrics": sum(bool(row["has_regime_metrics"]) for row in group),
                "mean_total_return": _mean(total_returns),
                "median_total_return": _median(total_returns),
                "tail_quantile_total_return": _quantile(total_returns, config.tail_quantile),
                "worst_total_return": min(total_returns) if total_returns else None,
                "mean_max_drawdown": _mean(drawdowns),
                "worst_max_drawdown": min(drawdowns) if drawdowns else None,
                "mean_volatility": _mean(volatilities),
                "policy_failure_rate": _rate(row["policy_failure"] for row in group),
                "adaptive_vs_static_win_rate": _rate(
                    row["adaptive_vs_static_win"] for row in group if row["adaptive_vs_static_win"] is not None
                ),
                "mean_adaptive_vs_static_delta": _mean(deltas),
                "mean_regime_transition_count": _mean(transitions),
                "mean_stress_regime_share": _mean(stress_shares),
                "worst_stress_score": max(scores) if scores else None,
                "mean_stress_score": _mean(scores),
                "ranking_metric": ranking_metric,
                "notes": _summary_notes(group),
            }
        )
    return rows


def _leaderboard_rows(summary_rows: list[dict[str, Any]], config: StressMetricsConfig) -> list[dict[str, Any]]:
    ranking_metric = _ranking_metric(config)
    ascending = bool(config.leaderboard.get("ascending", True))

    def ranking_value(row: Mapping[str, Any]) -> float:
        value = _number(row.get(ranking_metric))
        if value is not None:
            return value
        fallback = _number(row.get("mean_stress_score"))
        return fallback if fallback is not None else float("inf")

    ordered = sorted(
        summary_rows,
        key=lambda row: (
            ranking_value(row) if ascending else -ranking_value(row),
            str(row["scenario_name"]),
            str(row["simulation_type"]),
            str(row["scenario_id"]),
        ),
    )
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ordered, start=1):
        value = ranking_value(row)
        failure_rate = _number(row.get("policy_failure_rate")) or 0.0
        rows.append(
            {
                "rank": rank,
                "simulation_run_id": row["simulation_run_id"],
                "scenario_id": row["scenario_id"],
                "scenario_name": row["scenario_name"],
                "simulation_type": row["simulation_type"],
                "policy_name": "all",
                "path_count": row["path_count"],
                "tail_quantile": row["tail_quantile"],
                "policy_failure_rate": row["policy_failure_rate"],
                "adaptive_vs_static_win_rate": row["adaptive_vs_static_win_rate"],
                "tail_quantile_total_return": row["tail_quantile_total_return"],
                "worst_max_drawdown": row["worst_max_drawdown"],
                "mean_stress_score": row["mean_stress_score"],
                "ranking_metric": ranking_metric,
                "ranking_value": value,
                "tie_breaker": f"{row['scenario_name']}|{row['simulation_type']}|{row['scenario_id']}",
                "decision_label": "review" if failure_rate > 0.0 else "monitor",
                "primary_reason": _leaderboard_reason(row),
            }
        )
    return rows


def _ranking_metric(config: StressMetricsConfig) -> str:
    metric = str(config.leaderboard.get("ranking_metric", "mean_stress_score"))
    if metric not in LEADERBOARD_RANKING_METRICS:
        expected = ", ".join(sorted(LEADERBOARD_RANKING_METRICS))
        raise ValueError(
            "stress_metrics.leaderboard.ranking_metric must be a numeric simulation_summary.csv "
            f"column. Got {metric!r}; expected one of: {expected}."
        )
    return metric


def _failure_reasons(row: Mapping[str, Any], thresholds: Mapping[str, float]) -> list[str]:
    checks = (
        ("max_drawdown_limit", "max_drawdown", lambda value, limit: value < limit),
        ("min_total_return", "total_return", lambda value, limit: value < limit),
        ("max_transition_count", "regime_transition_count", lambda value, limit: value > limit),
        ("max_stress_regime_share", "stress_regime_share", lambda value, limit: value > limit),
        (
            "max_policy_underperformance",
            "adaptive_vs_static_return_delta",
            lambda value, limit: value < limit,
        ),
    )
    reasons: list[str] = []
    for threshold_name, column, breached in checks:
        value = _number(row.get(column))
        limit = _number(thresholds.get(threshold_name))
        if value is not None and limit is not None and breached(value, limit):
            reasons.append(threshold_name)
    return reasons


def _stress_score(row: Mapping[str, Any], thresholds: Mapping[str, float]) -> float:
    score = 0.0
    drawdown = _number(row.get("max_drawdown"))
    drawdown_limit = abs(float(thresholds.get("max_drawdown_limit", -0.10)))
    if drawdown is not None:
        score += max(0.0, abs(min(drawdown, 0.0)) / drawdown_limit)
    delta = _number(row.get("adaptive_vs_static_return_delta"))
    under_limit = abs(float(thresholds.get("max_policy_underperformance", -0.02)))
    if delta is not None and delta < 0:
        score += abs(delta) / under_limit
    transitions = _number(row.get("regime_transition_count"))
    max_transitions = float(thresholds.get("max_transition_count", 50.0))
    if transitions is not None and max_transitions > 0:
        score += max(0.0, transitions / max_transitions)
    stress_share = _number(row.get("stress_regime_share"))
    max_stress_share = float(thresholds.get("max_stress_regime_share", 0.50))
    if stress_share is not None and max_stress_share > 0:
        score += max(0.0, stress_share / max_stress_share)
    if row.get("policy_failure"):
        score += 1.0
    return float(score)


def _policy_failure_summary(
    *,
    simulation_run_id: str,
    path_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    leaderboard_rows: list[dict[str, Any]],
    metrics_config: StressMetricsConfig,
) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    for row in path_rows:
        for reason in str(row.get("failure_reason") or "").split(";"):
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    failures = sum(bool(row["policy_failure"]) for row in path_rows)
    return canonicalize_value(
        {
            "simulation_run_id": simulation_run_id,
            "total_scenarios_evaluated": len(summary_rows),
            "total_paths_evaluated": len(path_rows),
            "failure_thresholds": dict(metrics_config.failure_thresholds),
            "scenario_failure_rates": {
                row["scenario_id"]: row["policy_failure_rate"] for row in summary_rows
            },
            "worst_failure_scenarios": [
                {
                    "scenario_id": row["scenario_id"],
                    "scenario_name": row["scenario_name"],
                    "policy_failure_rate": row["policy_failure_rate"],
                    "mean_stress_score": row["mean_stress_score"],
                }
                for row in sorted(
                    summary_rows,
                    key=lambda item: (
                        -float(item["policy_failure_rate"] or 0.0),
                        -float(item["mean_stress_score"] or 0.0),
                        str(item["scenario_name"]),
                    ),
                )[:5]
            ],
            "failure_reasons": reason_counts,
            "policy_failure_count": failures,
            "policy_failure_rate": failures / len(path_rows) if path_rows else 0.0,
            "generated_files": {
                "simulation_path_metrics_csv": "simulation_path_metrics.csv",
                "simulation_summary_csv": "simulation_summary.csv",
                "simulation_leaderboard_csv": "simulation_leaderboard.csv",
                "policy_failure_summary_json": "policy_failure_summary.json",
                "simulation_metric_config_json": "simulation_metric_config.json",
                "manifest_json": "manifest.json",
            },
            "leaderboard_row_count": len(leaderboard_rows),
            "limitations": _limitations(),
        }
    )


def _manifest_payload(
    *,
    simulation_run_id: str,
    output_dir_name: str,
    paths: Mapping[str, Path],
    path_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    leaderboard_rows: list[dict[str, Any]],
    metrics_config: StressMetricsConfig,
) -> dict[str, Any]:
    return canonicalize_value(
        {
            "artifact_type": "simulation_stress_metrics",
            "schema_version": SIMULATION_STRESS_METRICS_SCHEMA_VERSION,
            "simulation_run_id": simulation_run_id,
            "generated_files": {key: path.name for key, path in paths.items()},
            "row_counts": {
                "simulation_path_metrics_csv": len(path_rows),
                "simulation_summary_csv": len(summary_rows),
                "simulation_leaderboard_csv": len(leaderboard_rows),
            },
            "source_scenarios_consumed": [
                {
                    "scenario_id": row["scenario_id"],
                    "scenario_name": row["scenario_name"],
                    "simulation_type": row["simulation_type"],
                }
                for row in summary_rows
            ],
            "thresholds": dict(metrics_config.failure_thresholds),
            "ranking_settings": dict(metrics_config.leaderboard),
            "relative_paths": {
                "metrics_dir": output_dir_name,
                **{key: path.name for key, path in paths.items()},
            },
            "limitations": _limitations(),
        }
    )


def _limitations() -> list[str]:
    return [
        "Metrics summarize robustness under configured simulated or replayed scenarios and are not forecasts.",
        "Monte Carlo contributes regime-path metrics only; no synthetic returns are inferred.",
        "Higher stress_score means a worse stress outcome.",
    ]


def _summary_notes(rows: list[dict[str, Any]]) -> str:
    notes = []
    if not any(row["has_return_metrics"] for row in rows):
        notes.append("return_metrics_unavailable")
    if not any(row["has_policy_metrics"] for row in rows):
        notes.append("policy_metrics_unavailable")
    if not any(row["has_regime_metrics"] for row in rows):
        notes.append("regime_metrics_unavailable")
    return ";".join(notes)


def _leaderboard_reason(row: Mapping[str, Any]) -> str:
    if float(row.get("policy_failure_rate") or 0.0) > 0.0:
        return "Policy failure threshold breached in one or more paths."
    if row.get("notes"):
        return str(row["notes"])
    return "No configured failure threshold breached."


def _series_total(values: Any) -> float | None:
    series = _numeric_series(values)
    if series.empty:
        return None
    return float((1.0 + series).prod() - 1.0)


def _numeric_series(values: Any) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return series.astype(float)


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.astype(float)).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _annualized_return(total_return: float, path_length: int) -> float | None:
    if path_length <= 0 or total_return <= -1.0:
        return None
    return float((1.0 + total_return) ** (252.0 / path_length) - 1.0)


def _bool_from_delta(value: float | None) -> bool | None:
    if value is None:
        return None
    return value > 0.0


def _numbers(values: Iterable[Any]) -> list[float]:
    return [number for number in (_number(value) for value in values) if number is not None]


def _number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def _string_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(pd.Series(values).median())


def _quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(pd.Series(values).quantile(quantile))


def _rate(values: Iterable[Any]) -> float:
    items = [bool(value) for value in values if value is not None]
    return sum(items) / len(items) if items else 0.0


def _safe_dir_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in value)
    return safe.strip("._") or "simulation_metrics"


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
