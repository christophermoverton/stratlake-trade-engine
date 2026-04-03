from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.alpha_eval.artifacts import DEFAULT_ALPHA_EVAL_ARTIFACTS_ROOT
from src.research.alpha_eval.registry import load_alpha_evaluation_registry
from src.research.registry import canonicalize_value, serialize_canonical_json

DEFAULT_ALPHA_COMPARISONS_ROOT = Path("artifacts") / "alpha_comparisons"
DEFAULT_ALPHA_COMPARISON_VIEW = "forecast"
DEFAULT_ALPHA_COMPARISON_METRIC = "ic_ir"
DEFAULT_ALPHA_SLEEVE_COMPARISON_METRIC = "sharpe_ratio"

_FORECAST_METRICS = {
    "mean_ic": "desc",
    "ic_ir": "desc",
    "mean_rank_ic": "desc",
    "rank_ic_ir": "desc",
    "n_periods": "desc",
}
_SLEEVE_METRICS = {
    "sharpe_ratio": "desc",
    "annualized_return": "desc",
    "total_return": "desc",
    "cumulative_return": "desc",
    "win_rate": "desc",
    "hit_rate": "desc",
    "profit_factor": "desc",
    "exposure_pct": "desc",
    "trade_count": "desc",
    "max_drawdown": "asc",
    "turnover": "asc",
    "average_turnover": "asc",
    "total_turnover": "asc",
    "total_transaction_cost": "asc",
    "total_slippage_cost": "asc",
    "total_execution_friction": "asc",
}
_SUMMARY_KEYS = ("mean_ic", "ic_ir", "mean_rank_ic", "rank_ic_ir", "n_periods")
_SLEEVE_SUMMARY_KEYS = (
    "sharpe_ratio",
    "annualized_return",
    "total_return",
    "cumulative_return",
    "max_drawdown",
    "turnover",
    "average_turnover",
    "total_turnover",
    "win_rate",
    "hit_rate",
    "profit_factor",
    "trade_count",
    "exposure_pct",
    "total_transaction_cost",
    "total_slippage_cost",
    "total_execution_friction",
)
_LEADERBOARD_COLUMNS = [
    "rank",
    "alpha_name",
    "run_id",
    "dataset",
    "timeframe",
    "evaluation_horizon",
    "mapping_name",
    "mean_ic",
    "ic_ir",
    "mean_rank_ic",
    "rank_ic_ir",
    "n_periods",
    "sharpe_ratio",
    "annualized_return",
    "total_return",
    "cumulative_return",
    "max_drawdown",
    "turnover",
    "average_turnover",
    "total_turnover",
    "win_rate",
    "hit_rate",
    "profit_factor",
    "trade_count",
    "exposure_pct",
    "total_transaction_cost",
    "total_slippage_cost",
    "total_execution_friction",
    "artifact_path",
]


class AlphaEvaluationComparisonError(ValueError):
    """Raised when alpha-evaluation leaderboard inputs are invalid or incomplete."""


@dataclass(frozen=True)
class AlphaEvaluationLeaderboardEntry:
    rank: int
    alpha_name: str
    run_id: str
    dataset: str
    timeframe: str
    evaluation_horizon: int
    mapping_name: str | None
    mean_ic: float | None
    ic_ir: float | None
    mean_rank_ic: float | None
    rank_ic_ir: float | None
    n_periods: int | None
    sharpe_ratio: float | None
    annualized_return: float | None
    total_return: float | None
    cumulative_return: float | None
    max_drawdown: float | None
    turnover: float | None
    average_turnover: float | None
    total_turnover: float | None
    win_rate: float | None
    hit_rate: float | None
    profit_factor: float | None
    trade_count: float | None
    exposure_pct: float | None
    total_transaction_cost: float | None
    total_slippage_cost: float | None
    total_execution_friction: float | None
    artifact_path: str


@dataclass(frozen=True)
class AlphaEvaluationComparisonResult:
    comparison_id: str
    view: str
    metric: str
    forecast_metric: str
    sleeve_metric: str
    filters: dict[str, Any]
    leaderboard: list[AlphaEvaluationLeaderboardEntry]
    csv_path: Path
    json_path: Path


def compare_alpha_evaluation_runs(
    *,
    metric: str = DEFAULT_ALPHA_COMPARISON_METRIC,
    view: str = DEFAULT_ALPHA_COMPARISON_VIEW,
    sleeve_metric: str = DEFAULT_ALPHA_SLEEVE_COMPARISON_METRIC,
    alpha_name: str | None = None,
    dataset: str | None = None,
    timeframe: str | None = None,
    evaluation_horizon: int | None = None,
    mapping_name: str | None = None,
    artifacts_root: str | Path = DEFAULT_ALPHA_EVAL_ARTIFACTS_ROOT,
    output_path: str | Path | None = None,
) -> AlphaEvaluationComparisonResult:
    """Build and persist one deterministic alpha-evaluation leaderboard from the registry only."""

    normalized_view = _normalize_view(view)
    normalized_forecast_metric = _normalize_forecast_metric(metric)
    normalized_sleeve_metric = _normalize_sleeve_metric(sleeve_metric)
    filters = _normalized_filters(
        alpha_name=alpha_name,
        dataset=dataset,
        timeframe=timeframe,
        evaluation_horizon=evaluation_horizon,
        mapping_name=mapping_name,
    )
    entries = load_alpha_evaluation_registry(artifacts_root)
    if not entries:
        raise AlphaEvaluationComparisonError("Alpha evaluation registry is empty.")

    normalized_rows = [_normalize_registry_entry(entry) for entry in entries]
    filtered_rows = _filter_rows(normalized_rows, filters)
    if not filtered_rows:
        raise AlphaEvaluationComparisonError("No alpha evaluation runs matched the requested filters.")

    ranked_leaderboard = _rank_rows(
        filtered_rows,
        view=normalized_view,
        forecast_metric=normalized_forecast_metric,
        sleeve_metric=normalized_sleeve_metric,
    )
    comparison_id = build_alpha_comparison_id(
        metric=normalized_forecast_metric,
        view=normalized_view,
        sleeve_metric=normalized_sleeve_metric,
        filters=filters,
        run_ids=[entry.run_id for entry in ranked_leaderboard],
    )
    resolved_output_path = (
        default_alpha_comparison_output_path(comparison_id)
        if output_path is None
        else Path(output_path)
    )
    csv_path, json_path = write_alpha_comparison_artifacts(
        leaderboard=ranked_leaderboard,
        comparison_id=comparison_id,
        view=normalized_view,
        forecast_metric=normalized_forecast_metric,
        sleeve_metric=normalized_sleeve_metric,
        filters=filters,
        output_path=resolved_output_path,
    )
    return AlphaEvaluationComparisonResult(
        comparison_id=comparison_id,
        view=normalized_view,
        metric=_result_metric(view=normalized_view, forecast_metric=normalized_forecast_metric, sleeve_metric=normalized_sleeve_metric),
        forecast_metric=normalized_forecast_metric,
        sleeve_metric=normalized_sleeve_metric,
        filters=filters,
        leaderboard=ranked_leaderboard,
        csv_path=csv_path,
        json_path=json_path,
    )


def write_alpha_comparison_artifacts(
    *,
    leaderboard: list[AlphaEvaluationLeaderboardEntry],
    comparison_id: str,
    view: str,
    forecast_metric: str,
    sleeve_metric: str,
    filters: dict[str, Any],
    output_path: str | Path,
) -> tuple[Path, Path]:
    """Persist one deterministic alpha leaderboard as CSV and JSON artifacts."""

    csv_path = resolve_alpha_comparison_csv_path(output_path)
    json_path = csv_path.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    persisted_rows = [_persisted_entry(entry) for entry in leaderboard]
    frame = pd.DataFrame(persisted_rows, columns=_LEADERBOARD_COLUMNS)
    with csv_path.open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")

    payload = {
        "comparison_id": comparison_id,
        "filters": canonicalize_value(filters),
        "forecast_metric": forecast_metric,
        "leaderboard": persisted_rows,
        "metric": _result_metric(view=view, forecast_metric=forecast_metric, sleeve_metric=sleeve_metric),
        "run_ids": [entry["run_id"] for entry in persisted_rows],
        "sleeve_metric": sleeve_metric,
        "view": view,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path


def build_alpha_comparison_id(
    *,
    metric: str,
    filters: dict[str, Any],
    run_ids: list[str],
    view: str = DEFAULT_ALPHA_COMPARISON_VIEW,
    sleeve_metric: str = DEFAULT_ALPHA_SLEEVE_COMPARISON_METRIC,
) -> str:
    """Return a deterministic comparison id for one logical alpha leaderboard request."""

    normalized_view = _normalize_view(view)
    normalized_metric = _normalize_forecast_metric(metric)
    normalized_sleeve_metric = _normalize_sleeve_metric(sleeve_metric)
    payload = {
        "filters": canonicalize_value(filters),
        "forecast_metric": normalized_metric,
        "run_ids": sorted(_normalize_non_empty_string(run_id, field_name="run_id") for run_id in run_ids),
        "sleeve_metric": normalized_sleeve_metric,
        "view": normalized_view,
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"registry_alpha_{normalized_view}_{digest}"


def default_alpha_comparison_output_path(comparison_id: str) -> Path:
    """Return the default deterministic leaderboard CSV path for one alpha comparison."""

    return DEFAULT_ALPHA_COMPARISONS_ROOT / comparison_id / "leaderboard.csv"


def resolve_alpha_comparison_csv_path(output_path: str | Path) -> Path:
    """Resolve the concrete CSV path used for persisted leaderboard outputs."""

    resolved_output_path = Path(output_path)
    if resolved_output_path.suffix.lower() == ".csv":
        return resolved_output_path
    return resolved_output_path / "leaderboard.csv"


def render_alpha_leaderboard_table(
    leaderboard: list[AlphaEvaluationLeaderboardEntry],
    *,
    view: str = DEFAULT_ALPHA_COMPARISON_VIEW,
    forecast_metric: str = DEFAULT_ALPHA_COMPARISON_METRIC,
    sleeve_metric: str = DEFAULT_ALPHA_SLEEVE_COMPARISON_METRIC,
) -> str:
    """Render a compact plain-text leaderboard summary for CLI output."""

    normalized_view = _normalize_view(view)
    if normalized_view == "forecast":
        columns = [
            ("rank", "rank"),
            ("alpha_name", "alpha"),
            ("run_id", "run_id"),
            ("mapping_name", "mapping"),
            (forecast_metric, forecast_metric),
            ("mean_ic", "mean_ic"),
            ("timeframe", "timeframe"),
        ]
    elif normalized_view == "sleeve":
        columns = [
            ("rank", "rank"),
            ("alpha_name", "alpha"),
            ("run_id", "run_id"),
            ("mapping_name", "mapping"),
            (sleeve_metric, sleeve_metric),
            ("total_return", "total_ret"),
            ("turnover", "turnover"),
        ]
    else:
        columns = [
            ("rank", "rank"),
            ("alpha_name", "alpha"),
            ("run_id", "run_id"),
            ("mapping_name", "mapping"),
            (forecast_metric, f"forecast_{forecast_metric}"),
            (sleeve_metric, f"sleeve_{sleeve_metric}"),
            ("total_return", "sleeve_total_ret"),
        ]

    rows = [_render_table_row(entry, columns=columns) for entry in leaderboard]
    widths = {key: max(len(label), *(len(row[key]) for row in rows)) for key, label in columns}
    header = "  ".join(label.ljust(widths[key]) for key, label in columns)
    divider = "  ".join("-" * widths[key] for key, _label in columns)
    body = ["  ".join(row[key].ljust(widths[key]) for key, _label in columns) for row in rows]
    return "\n".join([header, divider, *body]) if body else "\n".join([header, divider])


def _render_table_row(
    entry: AlphaEvaluationLeaderboardEntry,
    *,
    columns: list[tuple[str, str]],
) -> dict[str, str]:
    rendered: dict[str, str] = {}
    for key, _label in columns:
        value = getattr(entry, key)
        if isinstance(value, float) or value is None:
            rendered[key] = _format_metric(value)
        elif isinstance(value, int):
            rendered[key] = str(value)
        else:
            rendered[key] = "-" if value is None else str(value)
    return rendered


def _normalized_filters(
    *,
    alpha_name: str | None,
    dataset: str | None,
    timeframe: str | None,
    evaluation_horizon: int | None,
    mapping_name: str | None,
) -> dict[str, Any]:
    return {
        "alpha_name": _normalize_optional_string(alpha_name, field_name="alpha_name"),
        "dataset": _normalize_optional_string(dataset, field_name="dataset"),
        "evaluation_horizon": None if evaluation_horizon is None else int(evaluation_horizon),
        "mapping_name": _normalize_optional_string(mapping_name, field_name="mapping_name"),
        "timeframe": _normalize_optional_string(timeframe, field_name="timeframe"),
    }


def _normalize_registry_entry(entry: dict[str, Any]) -> AlphaEvaluationLeaderboardEntry:
    if not isinstance(entry, dict):
        raise AlphaEvaluationComparisonError("Registry entry must be a mapping.")

    metrics_summary = entry.get("metrics_summary")
    if not isinstance(metrics_summary, dict):
        raise AlphaEvaluationComparisonError(
            f"Alpha evaluation run '{entry.get('run_id', '<unknown>')}' is missing metrics_summary."
        )
    missing_summary_keys = [key for key in _SUMMARY_KEYS if key not in metrics_summary]
    if missing_summary_keys:
        formatted = ", ".join(missing_summary_keys)
        raise AlphaEvaluationComparisonError(
            f"Alpha evaluation run '{entry.get('run_id', '<unknown>')}' is missing metrics_summary fields: {formatted}."
        )

    sleeve_metrics = _extract_sleeve_metrics(entry)
    return AlphaEvaluationLeaderboardEntry(
        rank=0,
        alpha_name=_normalize_non_empty_string(entry.get("alpha_name"), field_name="alpha_name"),
        run_id=_normalize_non_empty_string(entry.get("run_id"), field_name="run_id"),
        dataset=_normalize_non_empty_string(entry.get("dataset"), field_name="dataset"),
        timeframe=_normalize_non_empty_string(entry.get("timeframe"), field_name="timeframe"),
        evaluation_horizon=_normalize_required_int(entry.get("evaluation_horizon"), field_name="evaluation_horizon"),
        mapping_name=_extract_mapping_name(entry),
        mean_ic=_coerce_metric(metrics_summary.get("mean_ic"), field_name="mean_ic", run_id=entry.get("run_id")),
        ic_ir=_coerce_metric(metrics_summary.get("ic_ir"), field_name="ic_ir", run_id=entry.get("run_id")),
        mean_rank_ic=_coerce_metric(
            metrics_summary.get("mean_rank_ic"),
            field_name="mean_rank_ic",
            run_id=entry.get("run_id"),
        ),
        rank_ic_ir=_coerce_metric(
            metrics_summary.get("rank_ic_ir"),
            field_name="rank_ic_ir",
            run_id=entry.get("run_id"),
        ),
        n_periods=_coerce_optional_int(metrics_summary.get("n_periods"), field_name="n_periods", run_id=entry.get("run_id")),
        sharpe_ratio=_coerce_metric(sleeve_metrics.get("sharpe_ratio"), field_name="sharpe_ratio", run_id=entry.get("run_id")),
        annualized_return=_coerce_metric(
            sleeve_metrics.get("annualized_return"),
            field_name="annualized_return",
            run_id=entry.get("run_id"),
        ),
        total_return=_coerce_metric(sleeve_metrics.get("total_return"), field_name="total_return", run_id=entry.get("run_id")),
        cumulative_return=_coerce_metric(
            sleeve_metrics.get("cumulative_return"),
            field_name="cumulative_return",
            run_id=entry.get("run_id"),
        ),
        max_drawdown=_coerce_metric(
            sleeve_metrics.get("max_drawdown"),
            field_name="max_drawdown",
            run_id=entry.get("run_id"),
        ),
        turnover=_coerce_metric(sleeve_metrics.get("turnover"), field_name="turnover", run_id=entry.get("run_id")),
        average_turnover=_coerce_metric(
            sleeve_metrics.get("average_turnover"),
            field_name="average_turnover",
            run_id=entry.get("run_id"),
        ),
        total_turnover=_coerce_metric(
            sleeve_metrics.get("total_turnover"),
            field_name="total_turnover",
            run_id=entry.get("run_id"),
        ),
        win_rate=_coerce_metric(sleeve_metrics.get("win_rate"), field_name="win_rate", run_id=entry.get("run_id")),
        hit_rate=_coerce_metric(sleeve_metrics.get("hit_rate"), field_name="hit_rate", run_id=entry.get("run_id")),
        profit_factor=_coerce_metric(
            sleeve_metrics.get("profit_factor"),
            field_name="profit_factor",
            run_id=entry.get("run_id"),
        ),
        trade_count=_coerce_metric(
            sleeve_metrics.get("trade_count"),
            field_name="trade_count",
            run_id=entry.get("run_id"),
        ),
        exposure_pct=_coerce_metric(
            sleeve_metrics.get("exposure_pct"),
            field_name="exposure_pct",
            run_id=entry.get("run_id"),
        ),
        total_transaction_cost=_coerce_metric(
            sleeve_metrics.get("total_transaction_cost"),
            field_name="total_transaction_cost",
            run_id=entry.get("run_id"),
        ),
        total_slippage_cost=_coerce_metric(
            sleeve_metrics.get("total_slippage_cost"),
            field_name="total_slippage_cost",
            run_id=entry.get("run_id"),
        ),
        total_execution_friction=_coerce_metric(
            sleeve_metrics.get("total_execution_friction"),
            field_name="total_execution_friction",
            run_id=entry.get("run_id"),
        ),
        artifact_path=_normalize_non_empty_string(entry.get("artifact_path"), field_name="artifact_path"),
    )


def _extract_mapping_name(entry: Mapping[str, Any]) -> str | None:
    config = entry.get("config")
    signal_mapping = config.get("signal_mapping") if isinstance(config, Mapping) else None
    if not isinstance(signal_mapping, Mapping):
        return None

    metadata = signal_mapping.get("metadata")
    if isinstance(metadata, Mapping):
        raw_name = metadata.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            return raw_name.strip()

    policy = signal_mapping.get("policy")
    quantile = signal_mapping.get("quantile")
    if not isinstance(policy, str) or not policy.strip():
        return None
    if quantile is None:
        return policy.strip()
    try:
        normalized_quantile = float(quantile)
    except (TypeError, ValueError):
        return policy.strip()
    if math.isnan(normalized_quantile) or math.isinf(normalized_quantile):
        return policy.strip()
    return f"{policy.strip()}[q={normalized_quantile:g}]"


def _extract_sleeve_metrics(entry: Mapping[str, Any]) -> dict[str, Any]:
    manifest = entry.get("manifest")
    if not isinstance(manifest, Mapping):
        return {}
    sleeve = manifest.get("sleeve")
    if not isinstance(sleeve, Mapping):
        return {}
    metric_summary = sleeve.get("metric_summary")
    if not isinstance(metric_summary, Mapping):
        return {}
    return dict(metric_summary)


def _filter_rows(
    rows: list[AlphaEvaluationLeaderboardEntry],
    filters: dict[str, Any],
) -> list[AlphaEvaluationLeaderboardEntry]:
    filtered = rows
    if filters["alpha_name"] is not None:
        filtered = [row for row in filtered if row.alpha_name == filters["alpha_name"]]
    if filters["dataset"] is not None:
        filtered = [row for row in filtered if row.dataset == filters["dataset"]]
    if filters["timeframe"] is not None:
        filtered = [row for row in filtered if row.timeframe == filters["timeframe"]]
    if filters["evaluation_horizon"] is not None:
        filtered = [row for row in filtered if row.evaluation_horizon == filters["evaluation_horizon"]]
    if filters["mapping_name"] is not None:
        filtered = [row for row in filtered if row.mapping_name == filters["mapping_name"]]
    return filtered


def _rank_rows(
    rows: list[AlphaEvaluationLeaderboardEntry],
    *,
    view: str,
    forecast_metric: str,
    sleeve_metric: str,
) -> list[AlphaEvaluationLeaderboardEntry]:
    ranked = sorted(
        rows,
        key=lambda row: _sort_key(row, view=view, forecast_metric=forecast_metric, sleeve_metric=sleeve_metric),
    )
    return [
        AlphaEvaluationLeaderboardEntry(
            rank=index,
            alpha_name=row.alpha_name,
            run_id=row.run_id,
            dataset=row.dataset,
            timeframe=row.timeframe,
            evaluation_horizon=row.evaluation_horizon,
            mapping_name=row.mapping_name,
            mean_ic=row.mean_ic,
            ic_ir=row.ic_ir,
            mean_rank_ic=row.mean_rank_ic,
            rank_ic_ir=row.rank_ic_ir,
            n_periods=row.n_periods,
            sharpe_ratio=row.sharpe_ratio,
            annualized_return=row.annualized_return,
            total_return=row.total_return,
            cumulative_return=row.cumulative_return,
            max_drawdown=row.max_drawdown,
            turnover=row.turnover,
            average_turnover=row.average_turnover,
            total_turnover=row.total_turnover,
            win_rate=row.win_rate,
            hit_rate=row.hit_rate,
            profit_factor=row.profit_factor,
            trade_count=row.trade_count,
            exposure_pct=row.exposure_pct,
            total_transaction_cost=row.total_transaction_cost,
            total_slippage_cost=row.total_slippage_cost,
            total_execution_friction=row.total_execution_friction,
            artifact_path=row.artifact_path,
        )
        for index, row in enumerate(ranked, start=1)
    ]


def _sort_key(
    row: AlphaEvaluationLeaderboardEntry,
    *,
    view: str,
    forecast_metric: str,
    sleeve_metric: str,
) -> tuple[Any, ...]:
    if view == "forecast":
        ordered_metrics = [
            forecast_metric,
            *[metric_name for metric_name in ("ic_ir", "mean_ic", "mean_rank_ic", "rank_ic_ir", "n_periods") if metric_name != forecast_metric],
        ]
    elif view == "sleeve":
        ordered_metrics = [
            sleeve_metric,
            *[
                metric_name
                for metric_name in ("sharpe_ratio", "total_return", "annualized_return", "max_drawdown", "turnover", "trade_count")
                if metric_name != sleeve_metric
            ],
        ]
    else:
        ordered_metrics = [
            forecast_metric,
            sleeve_metric,
            *[metric_name for metric_name in ("ic_ir", "mean_ic", "rank_ic_ir", "n_periods") if metric_name != forecast_metric],
            *[metric_name for metric_name in ("sharpe_ratio", "total_return", "max_drawdown", "turnover") if metric_name != sleeve_metric],
        ]

    sort_parts: list[Any] = []
    for metric_name in ordered_metrics:
        sort_parts.extend(_metric_sort_components(getattr(row, metric_name), metric_name=metric_name))
    sort_parts.extend(
        [
            row.mapping_name is None,
            "" if row.mapping_name is None else row.mapping_name,
            row.alpha_name,
            row.run_id,
        ]
    )
    return tuple(sort_parts)


def _metric_sort_components(value: float | int | None, *, metric_name: str) -> tuple[bool, float]:
    direction = _metric_direction(metric_name)
    if value is None:
        return (True, 0.0)
    normalized = float(value)
    if direction == "desc":
        return (False, -normalized)
    return (False, normalized)


def _metric_direction(metric_name: str) -> str:
    if metric_name in _FORECAST_METRICS:
        return _FORECAST_METRICS[metric_name]
    if metric_name in _SLEEVE_METRICS:
        return _SLEEVE_METRICS[metric_name]
    raise AlphaEvaluationComparisonError(f"Unsupported metric {metric_name!r}.")


def _persisted_entry(entry: AlphaEvaluationLeaderboardEntry) -> dict[str, Any]:
    payload = asdict(entry)
    return {column: payload.get(column) for column in _LEADERBOARD_COLUMNS}


def _normalize_view(view: str) -> str:
    normalized = _normalize_non_empty_string(view, field_name="view")
    if normalized not in {"forecast", "sleeve", "combined"}:
        raise AlphaEvaluationComparisonError("view must be one of: forecast, sleeve, combined.")
    return normalized


def _normalize_forecast_metric(metric: str) -> str:
    normalized = _normalize_non_empty_string(metric, field_name="metric")
    if normalized not in _FORECAST_METRICS:
        raise AlphaEvaluationComparisonError(
            "metric must be one of: ic_ir, mean_ic, mean_rank_ic, rank_ic_ir, n_periods."
        )
    return normalized


def _normalize_sleeve_metric(metric: str) -> str:
    normalized = _normalize_non_empty_string(metric, field_name="sleeve_metric")
    if normalized not in _SLEEVE_METRICS:
        formatted = ", ".join(sorted(_SLEEVE_METRICS))
        raise AlphaEvaluationComparisonError(f"sleeve_metric must be one of: {formatted}.")
    return normalized


def _result_metric(*, view: str, forecast_metric: str, sleeve_metric: str) -> str:
    if view == "forecast":
        return forecast_metric
    if view == "sleeve":
        return sleeve_metric
    return f"{forecast_metric}+{sleeve_metric}"


def _normalize_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise AlphaEvaluationComparisonError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise AlphaEvaluationComparisonError(f"{field_name} must be a non-empty string.")
    return normalized


def _normalize_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _normalize_non_empty_string(value, field_name=field_name)


def _normalize_required_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or value is None:
        raise AlphaEvaluationComparisonError(f"{field_name} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AlphaEvaluationComparisonError(f"{field_name} must be an integer.") from exc


def _coerce_optional_int(
    value: object,
    *,
    field_name: str,
    run_id: object,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise AlphaEvaluationComparisonError(
            f"Alpha evaluation run '{run_id}' has a non-numeric {field_name} value."
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AlphaEvaluationComparisonError(
            f"Alpha evaluation run '{run_id}' has a non-numeric {field_name} value."
        ) from exc


def _coerce_metric(
    value: object,
    *,
    field_name: str,
    run_id: object,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise AlphaEvaluationComparisonError(
            f"Alpha evaluation run '{run_id}' has a non-numeric {field_name} value."
        )
    try:
        metric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise AlphaEvaluationComparisonError(
            f"Alpha evaluation run '{run_id}' has a non-numeric {field_name} value."
        ) from exc
    if math.isnan(metric_value) or math.isinf(metric_value):
        return None
    return metric_value


def _format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


__all__ = [
    "AlphaEvaluationComparisonError",
    "AlphaEvaluationComparisonResult",
    "AlphaEvaluationLeaderboardEntry",
    "DEFAULT_ALPHA_COMPARISON_METRIC",
    "DEFAULT_ALPHA_COMPARISON_VIEW",
    "DEFAULT_ALPHA_COMPARISONS_ROOT",
    "DEFAULT_ALPHA_SLEEVE_COMPARISON_METRIC",
    "build_alpha_comparison_id",
    "compare_alpha_evaluation_runs",
    "default_alpha_comparison_output_path",
    "render_alpha_leaderboard_table",
    "resolve_alpha_comparison_csv_path",
    "write_alpha_comparison_artifacts",
]
