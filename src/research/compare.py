from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.cli.run_strategy import (
    StrategyRunResult,
    get_strategy_config,
    run_strategy_experiment,
)
from src.research import experiment_tracker
from src.research.registry import default_registry_path, load_registry
from src.research.strategies import build_strategy
from src.research.walk_forward import WalkForwardRunResult, run_walk_forward_experiment

DEFAULT_LEADERBOARD_PATH = Path("artifacts") / "strategies" / "leaderboard.csv"
DEFAULT_METRIC = "sharpe_ratio"


@dataclass(frozen=True)
class LeaderboardEntry:
    rank: int
    strategy_name: str
    run_id: str
    evaluation_mode: str
    selected_metric_name: str
    selected_metric_value: float | None
    cumulative_return: float | None
    total_return: float | None
    sharpe_ratio: float | None
    max_drawdown: float | None
    annualized_return: float | None
    annualized_volatility: float | None
    volatility: float | None
    win_rate: float | None
    hit_rate: float | None
    profit_factor: float | None
    turnover: float | None
    exposure_pct: float | None


@dataclass(frozen=True)
class ComparisonResult:
    metric: str
    evaluation_mode: str
    selection_mode: str
    selection_rule: str
    leaderboard: list[LeaderboardEntry]
    csv_path: Path
    json_path: Path


def compare_strategies(
    strategies: Sequence[str],
    *,
    metric: str = DEFAULT_METRIC,
    evaluation_path: Path | None = None,
    top_k: int | None = None,
    from_registry: bool = False,
    output_path: Path | None = None,
) -> ComparisonResult:
    """Compare multiple strategies using fresh execution or registry-backed results."""

    strategy_names = _normalize_strategy_names(strategies)
    evaluation_mode = "walk_forward" if evaluation_path is not None else "single"

    if from_registry:
        rows = _load_rows_from_registry(
            strategy_names,
            metric=metric,
            evaluation_mode=evaluation_mode,
            evaluation_path=evaluation_path,
        )
        selection_mode = "registry"
        selection_rule = (
            "latest run per strategy filtered by evaluation mode"
            if evaluation_path is None
            else "latest run per strategy filtered by evaluation mode and evaluation config path"
        )
    else:
        rows = _execute_rows(
            strategy_names,
            evaluation_path=evaluation_path,
        )
        selection_mode = "fresh"
        selection_rule = "freshly executed run per strategy"

    leaderboard = _rank_rows(rows, metric=metric, top_k=top_k)
    csv_path, json_path = write_leaderboard_artifacts(
        leaderboard,
        metric=metric,
        evaluation_mode=evaluation_mode,
        selection_mode=selection_mode,
        selection_rule=selection_rule,
        output_path=output_path,
    )
    return ComparisonResult(
        metric=metric,
        evaluation_mode=evaluation_mode,
        selection_mode=selection_mode,
        selection_rule=selection_rule,
        leaderboard=leaderboard,
        csv_path=csv_path,
        json_path=json_path,
    )


def write_leaderboard_artifacts(
    leaderboard: Sequence[LeaderboardEntry],
    *,
    metric: str,
    evaluation_mode: str,
    selection_mode: str,
    selection_rule: str,
    output_path: Path | None = None,
) -> tuple[Path, Path]:
    """Persist leaderboard CSV and JSON artifacts using a stable schema."""

    csv_path = resolve_output_csv_path(output_path)
    json_path = csv_path.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    leaderboard_frame = pd.DataFrame(
        [asdict(entry) for entry in leaderboard],
        columns=list(LeaderboardEntry.__dataclass_fields__),
    )
    leaderboard_frame.to_csv(csv_path, index=False)
    payload = {
        "metric": metric,
        "evaluation_mode": evaluation_mode,
        "selection_mode": selection_mode,
        "selection_rule": selection_rule,
        "leaderboard": [asdict(entry) for entry in leaderboard],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path


def resolve_output_csv_path(output_path: Path | None) -> Path:
    """Resolve the CSV path for leaderboard outputs."""

    if output_path is None:
        return DEFAULT_LEADERBOARD_PATH
    if output_path.suffix.lower() == ".csv":
        return output_path
    return output_path / "leaderboard.csv"


def render_leaderboard_table(leaderboard: Sequence[LeaderboardEntry]) -> str:
    """Render a compact plain-text table for console output."""

    columns = [
        ("rank", "rank"),
        ("strategy_name", "strategy"),
        ("run_id", "run_id"),
        ("evaluation_mode", "mode"),
        ("selected_metric_value", "metric"),
        ("total_return", "total_return"),
        ("sharpe_ratio", "sharpe_ratio"),
        ("max_drawdown", "max_drawdown"),
    ]
    rows = [
        {
            "rank": str(entry.rank),
            "strategy_name": entry.strategy_name,
            "run_id": entry.run_id,
            "evaluation_mode": entry.evaluation_mode,
            "selected_metric_value": _format_metric(entry.selected_metric_value),
            "total_return": _format_metric(entry.total_return),
            "sharpe_ratio": _format_metric(entry.sharpe_ratio),
            "max_drawdown": _format_metric(entry.max_drawdown),
        }
        for entry in leaderboard
    ]

    widths: dict[str, int] = {}
    for key, label in columns:
        values = [len(label), *(len(row[key]) for row in rows)]
        widths[key] = max(values)

    header = "  ".join(label.ljust(widths[key]) for key, label in columns)
    divider = "  ".join("-" * widths[key] for key, _label in columns)
    body = [
        "  ".join(row[key].ljust(widths[key]) for key, _label in columns)
        for row in rows
    ]
    return "\n".join([header, divider, *body]) if body else "\n".join([header, divider])


def _execute_rows(
    strategies: Sequence[str],
    *,
    evaluation_path: Path | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy_name in strategies:
        if evaluation_path is None:
            result: StrategyRunResult | WalkForwardRunResult = run_strategy_experiment(strategy_name)
            evaluation_mode = "single"
        else:
            config = get_strategy_config(strategy_name)
            strategy = build_strategy(strategy_name, config)
            result = run_walk_forward_experiment(
                strategy_name,
                strategy,
                evaluation_path=evaluation_path,
                strategy_config=config,
            )
            evaluation_mode = "walk_forward"

        rows.append(
            {
                "strategy_name": strategy_name,
                "run_id": result.run_id,
                "evaluation_mode": evaluation_mode,
                "metrics_summary": dict(result.metrics),
            }
        )
    return rows


def _load_rows_from_registry(
    strategies: Sequence[str],
    *,
    metric: str,
    evaluation_mode: str,
    evaluation_path: Path | None,
) -> list[dict[str, Any]]:
    registry_path = default_registry_path(experiment_tracker.ARTIFACTS_ROOT)
    entries = load_registry(registry_path)
    rows: list[dict[str, Any]] = []
    for strategy_name in strategies:
        candidates = [
            entry
            for entry in entries
            if entry.get("strategy_name") == strategy_name
            and entry.get("evaluation_mode") == evaluation_mode
            and _matches_evaluation_path(entry, evaluation_path)
        ]
        if not candidates:
            raise ValueError(
                f"No registry runs found for strategy '{strategy_name}' in evaluation mode '{evaluation_mode}'."
            )

        selected = max(
            candidates,
            key=lambda entry: (str(entry.get("timestamp") or ""), str(entry.get("run_id") or "")),
        )
        rows.append(
            {
                "strategy_name": strategy_name,
                "run_id": str(selected["run_id"]),
                "evaluation_mode": str(selected.get("evaluation_mode") or evaluation_mode),
                "metrics_summary": dict(selected.get("metrics_summary") or {}),
                "selected_metric_name": metric,
            }
        )
    return rows


def _matches_evaluation_path(entry: dict[str, Any], evaluation_path: Path | None) -> bool:
    if evaluation_path is None:
        return True
    entry_path = entry.get("evaluation_config_path")
    if entry_path is None:
        return False
    return Path(str(entry_path)).as_posix() == evaluation_path.as_posix()


def _rank_rows(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    top_k: int | None,
) -> list[LeaderboardEntry]:
    ranked_rows = sorted(rows, key=lambda row: _sort_key(row, metric=metric))
    leaderboard: list[LeaderboardEntry] = []
    for index, row in enumerate(ranked_rows, start=1):
        metrics_summary = dict(row.get("metrics_summary") or {})
        leaderboard.append(
            LeaderboardEntry(
                rank=index,
                strategy_name=str(row["strategy_name"]),
                run_id=str(row["run_id"]),
                evaluation_mode=str(row.get("evaluation_mode") or "single"),
                selected_metric_name=metric,
                selected_metric_value=_coerce_metric(metrics_summary.get(metric)),
                cumulative_return=_coerce_metric(metrics_summary.get("cumulative_return")),
                total_return=_coerce_metric(metrics_summary.get("total_return")),
                sharpe_ratio=_coerce_metric(metrics_summary.get("sharpe_ratio")),
                max_drawdown=_coerce_metric(metrics_summary.get("max_drawdown")),
                annualized_return=_coerce_metric(metrics_summary.get("annualized_return")),
                annualized_volatility=_coerce_metric(metrics_summary.get("annualized_volatility")),
                volatility=_coerce_metric(metrics_summary.get("volatility")),
                win_rate=_coerce_metric(metrics_summary.get("win_rate")),
                hit_rate=_coerce_metric(metrics_summary.get("hit_rate")),
                profit_factor=_coerce_metric(metrics_summary.get("profit_factor")),
                turnover=_coerce_metric(metrics_summary.get("turnover")),
                exposure_pct=_coerce_metric(metrics_summary.get("exposure_pct")),
            )
        )
    return leaderboard[:top_k] if top_k is not None else leaderboard


def _sort_key(row: dict[str, Any], *, metric: str) -> tuple[bool, float, str, str]:
    metrics_summary = dict(row.get("metrics_summary") or {})
    metric_value = _coerce_metric(metrics_summary.get(metric))
    return (
        metric_value is None,
        0.0 if metric_value is None else -metric_value,
        str(row["strategy_name"]),
        str(row["run_id"]),
    )


def _normalize_strategy_names(strategies: Sequence[str]) -> list[str]:
    names = [strategy.strip() for strategy in strategies if strategy.strip()]
    if not names:
        raise ValueError("At least one strategy name must be provided.")
    return names


def _coerce_metric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"
