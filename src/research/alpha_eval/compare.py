from __future__ import annotations

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
DEFAULT_ALPHA_COMPARISON_METRIC = "ic_ir"
_SUMMARY_KEYS = ("mean_ic", "ic_ir", "mean_rank_ic", "rank_ic_ir", "n_periods")
_LEADERBOARD_COLUMNS = [
    "rank",
    "alpha_name",
    "run_id",
    "dataset",
    "timeframe",
    "evaluation_horizon",
    "mean_ic",
    "ic_ir",
    "mean_rank_ic",
    "rank_ic_ir",
    "n_periods",
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
    mean_ic: float | None
    ic_ir: float | None
    mean_rank_ic: float | None
    rank_ic_ir: float | None
    n_periods: int | None
    artifact_path: str


@dataclass(frozen=True)
class AlphaEvaluationComparisonResult:
    comparison_id: str
    metric: str
    filters: dict[str, Any]
    leaderboard: list[AlphaEvaluationLeaderboardEntry]
    csv_path: Path
    json_path: Path


def compare_alpha_evaluation_runs(
    *,
    metric: str = DEFAULT_ALPHA_COMPARISON_METRIC,
    alpha_name: str | None = None,
    dataset: str | None = None,
    timeframe: str | None = None,
    evaluation_horizon: int | None = None,
    artifacts_root: str | Path = DEFAULT_ALPHA_EVAL_ARTIFACTS_ROOT,
    output_path: str | Path | None = None,
) -> AlphaEvaluationComparisonResult:
    """Build and persist one deterministic alpha-evaluation leaderboard from the registry only."""

    normalized_metric = _normalize_metric(metric)
    filters = _normalized_filters(
        alpha_name=alpha_name,
        dataset=dataset,
        timeframe=timeframe,
        evaluation_horizon=evaluation_horizon,
    )
    entries = load_alpha_evaluation_registry(artifacts_root)
    if not entries:
        raise AlphaEvaluationComparisonError("Alpha evaluation registry is empty.")

    normalized_rows = [_normalize_registry_entry(entry) for entry in entries]
    filtered_rows = _filter_rows(normalized_rows, filters)
    if not filtered_rows:
        raise AlphaEvaluationComparisonError("No alpha evaluation runs matched the requested filters.")

    ranked_leaderboard = _rank_rows(filtered_rows, metric=normalized_metric)
    comparison_id = build_alpha_comparison_id(
        metric=normalized_metric,
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
        metric=normalized_metric,
        filters=filters,
        output_path=resolved_output_path,
    )
    return AlphaEvaluationComparisonResult(
        comparison_id=comparison_id,
        metric=normalized_metric,
        filters=filters,
        leaderboard=ranked_leaderboard,
        csv_path=csv_path,
        json_path=json_path,
    )


def write_alpha_comparison_artifacts(
    *,
    leaderboard: list[AlphaEvaluationLeaderboardEntry],
    comparison_id: str,
    metric: str,
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
        "leaderboard": persisted_rows,
        "metric": metric,
        "run_ids": [entry["run_id"] for entry in persisted_rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path


def build_alpha_comparison_id(
    *,
    metric: str,
    filters: dict[str, Any],
    run_ids: list[str],
) -> str:
    """Return a deterministic comparison id for one logical alpha leaderboard request."""

    payload = {
        "filters": canonicalize_value(filters),
        "metric": _normalize_metric(metric),
        "run_ids": sorted(_normalize_non_empty_string(run_id, field_name="run_id") for run_id in run_ids),
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"registry_alpha_{_normalize_metric(metric)}_{digest}"


def default_alpha_comparison_output_path(comparison_id: str) -> Path:
    """Return the default deterministic leaderboard CSV path for one alpha comparison."""

    return DEFAULT_ALPHA_COMPARISONS_ROOT / comparison_id / "leaderboard.csv"


def resolve_alpha_comparison_csv_path(output_path: str | Path) -> Path:
    """Resolve the concrete CSV path used for persisted leaderboard outputs."""

    resolved_output_path = Path(output_path)
    if resolved_output_path.suffix.lower() == ".csv":
        return resolved_output_path
    return resolved_output_path / "leaderboard.csv"


def render_alpha_leaderboard_table(leaderboard: list[AlphaEvaluationLeaderboardEntry]) -> str:
    """Render a compact plain-text leaderboard summary for CLI output."""

    columns = [
        ("rank", "rank"),
        ("alpha_name", "alpha"),
        ("run_id", "run_id"),
        ("ic_ir", "ic_ir"),
        ("mean_ic", "mean_ic"),
        ("timeframe", "timeframe"),
    ]
    rows = [
        {
            "rank": str(entry.rank),
            "alpha_name": entry.alpha_name,
            "run_id": entry.run_id,
            "ic_ir": _format_metric(entry.ic_ir),
            "mean_ic": _format_metric(entry.mean_ic),
            "timeframe": entry.timeframe,
        }
        for entry in leaderboard
    ]
    widths = {
        key: max(len(label), *(len(row[key]) for row in rows))
        for key, label in columns
    }
    header = "  ".join(label.ljust(widths[key]) for key, label in columns)
    divider = "  ".join("-" * widths[key] for key, _label in columns)
    body = [
        "  ".join(row[key].ljust(widths[key]) for key, _label in columns)
        for row in rows
    ]
    return "\n".join([header, divider, *body]) if body else "\n".join([header, divider])


def _normalized_filters(
    *,
    alpha_name: str | None,
    dataset: str | None,
    timeframe: str | None,
    evaluation_horizon: int | None,
) -> dict[str, Any]:
    return {
        "alpha_name": _normalize_optional_string(alpha_name, field_name="alpha_name"),
        "dataset": _normalize_optional_string(dataset, field_name="dataset"),
        "timeframe": _normalize_optional_string(timeframe, field_name="timeframe"),
        "evaluation_horizon": None if evaluation_horizon is None else int(evaluation_horizon),
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

    return AlphaEvaluationLeaderboardEntry(
        rank=0,
        alpha_name=_normalize_non_empty_string(entry.get("alpha_name"), field_name="alpha_name"),
        run_id=_normalize_non_empty_string(entry.get("run_id"), field_name="run_id"),
        dataset=_normalize_non_empty_string(entry.get("dataset"), field_name="dataset"),
        timeframe=_normalize_non_empty_string(entry.get("timeframe"), field_name="timeframe"),
        evaluation_horizon=_normalize_required_int(entry.get("evaluation_horizon"), field_name="evaluation_horizon"),
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
        artifact_path=_normalize_non_empty_string(entry.get("artifact_path"), field_name="artifact_path"),
    )


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
    return filtered


def _rank_rows(
    rows: list[AlphaEvaluationLeaderboardEntry],
    *,
    metric: str,
) -> list[AlphaEvaluationLeaderboardEntry]:
    ranked = sorted(rows, key=lambda row: _sort_key(row, metric=metric))
    return [
        AlphaEvaluationLeaderboardEntry(
            rank=index,
            alpha_name=row.alpha_name,
            run_id=row.run_id,
            dataset=row.dataset,
            timeframe=row.timeframe,
            evaluation_horizon=row.evaluation_horizon,
            mean_ic=row.mean_ic,
            ic_ir=row.ic_ir,
            mean_rank_ic=row.mean_rank_ic,
            rank_ic_ir=row.rank_ic_ir,
            n_periods=row.n_periods,
            artifact_path=row.artifact_path,
        )
        for index, row in enumerate(ranked, start=1)
    ]


def _sort_key(
    row: AlphaEvaluationLeaderboardEntry,
    *,
    metric: str,
) -> tuple[bool, float, bool, float, str, str]:
    # Deterministic ranking order:
    # 1. primary metric descending with missing values last
    # 2. mean_ic descending with missing values last
    # 3. alpha_name ascending
    # 4. run_id ascending
    primary_metric = getattr(row, metric)
    tie_break_mean_ic = row.mean_ic
    return (
        primary_metric is None,
        0.0 if primary_metric is None else -primary_metric,
        tie_break_mean_ic is None,
        0.0 if tie_break_mean_ic is None else -tie_break_mean_ic,
        row.alpha_name,
        row.run_id,
    )


def _persisted_entry(entry: AlphaEvaluationLeaderboardEntry) -> dict[str, Any]:
    payload = asdict(entry)
    return {column: payload.get(column) for column in _LEADERBOARD_COLUMNS}


def _normalize_metric(metric: str) -> str:
    normalized = _normalize_non_empty_string(metric, field_name="metric")
    if normalized not in {"mean_ic", "ic_ir", "mean_rank_ic", "rank_ic_ir"}:
        raise AlphaEvaluationComparisonError(
            "metric must be one of: ic_ir, mean_ic, mean_rank_ic, rank_ic_ir."
        )
    return normalized


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
    "DEFAULT_ALPHA_COMPARISONS_ROOT",
    "build_alpha_comparison_id",
    "compare_alpha_evaluation_runs",
    "default_alpha_comparison_output_path",
    "render_alpha_leaderboard_table",
    "resolve_alpha_comparison_csv_path",
    "write_alpha_comparison_artifacts",
]
