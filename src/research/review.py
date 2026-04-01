from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.research.alpha_eval import DEFAULT_ALPHA_EVAL_ARTIFACTS_ROOT, load_alpha_evaluation_registry
from src.research.experiment_tracker import ARTIFACTS_ROOT as DEFAULT_STRATEGY_ARTIFACTS_ROOT
from src.research.promotion import (
    DEFAULT_PROMOTION_ARTIFACT_FILENAME,
    evaluate_promotion_gates,
    write_promotion_gate_artifact,
)
from src.research.registry import (
    PORTFOLIO_RUN_TYPE,
    STRATEGY_RUN_TYPE,
    canonicalize_value,
    default_registry_path,
    filter_by_run_type,
    load_registry,
    resolve_review_status,
    serialize_canonical_json,
)

ALPHA_REVIEW_RUN_TYPE = "alpha_evaluation"
SUPPORTED_REVIEW_RUN_TYPES = (ALPHA_REVIEW_RUN_TYPE, STRATEGY_RUN_TYPE, PORTFOLIO_RUN_TYPE)
DEFAULT_REVIEW_ROOT = Path("artifacts") / "reviews"
DEFAULT_STRATEGY_REVIEW_METRIC = "sharpe_ratio"
DEFAULT_PORTFOLIO_REVIEW_METRIC = "sharpe_ratio"
DEFAULT_ALPHA_REVIEW_METRIC = "ic_ir"
DEFAULT_PORTFOLIO_ARTIFACTS_ROOT = Path("artifacts") / "portfolios"
_LEADERBOARD_FILENAME = "leaderboard.csv"
_REVIEW_SUMMARY_FILENAME = "review_summary.json"
_MANIFEST_FILENAME = "manifest.json"
_LEADERBOARD_COLUMNS = [
    "run_type",
    "rank_within_type",
    "entity_name",
    "run_id",
    "selected_metric_name",
    "selected_metric_value",
    "secondary_metric_name",
    "secondary_metric_value",
    "timeframe",
    "evaluation_mode",
    "promotion_status",
    "passed_gate_count",
    "gate_count",
    "artifact_path",
]


class ResearchReviewError(ValueError):
    """Raised when a unified research review request cannot be fulfilled."""


@dataclass(frozen=True)
class ResearchReviewEntry:
    run_type: str
    rank_within_type: int
    entity_name: str
    run_id: str
    selected_metric_name: str
    selected_metric_value: float | None
    secondary_metric_name: str
    secondary_metric_value: float | None
    timeframe: str | None
    evaluation_mode: str | None
    promotion_status: str | None
    passed_gate_count: int | None
    gate_count: int | None
    artifact_path: str


@dataclass(frozen=True)
class ResearchReviewResult:
    review_id: str
    filters: dict[str, Any]
    entries: list[ResearchReviewEntry]
    csv_path: Path
    json_path: Path
    manifest_path: Path
    promotion_gate_path: Path | None


def compare_research_runs(
    *,
    run_types: Iterable[str] | None = None,
    timeframe: str | None = None,
    dataset: str | None = None,
    alpha_name: str | None = None,
    strategy_name: str | None = None,
    portfolio_name: str | None = None,
    top_k_per_type: int | None = None,
    strategy_artifacts_root: str | Path = DEFAULT_STRATEGY_ARTIFACTS_ROOT,
    portfolio_artifacts_root: str | Path = DEFAULT_PORTFOLIO_ARTIFACTS_ROOT,
    alpha_artifacts_root: str | Path = DEFAULT_ALPHA_EVAL_ARTIFACTS_ROOT,
    output_path: str | Path | None = None,
    promotion_gate_config: Mapping[str, Any] | None = None,
) -> ResearchReviewResult:
    """Build a registry-backed unified review surface across alpha, strategy, and portfolio runs."""

    normalized_filters = _normalize_filters(
        run_types=run_types,
        timeframe=timeframe,
        dataset=dataset,
        alpha_name=alpha_name,
        strategy_name=strategy_name,
        portfolio_name=portfolio_name,
        top_k_per_type=top_k_per_type,
    )
    entries = _collect_review_entries(
        filters=normalized_filters,
        strategy_artifacts_root=Path(strategy_artifacts_root),
        portfolio_artifacts_root=Path(portfolio_artifacts_root),
        alpha_artifacts_root=Path(alpha_artifacts_root),
    )
    if not entries:
        raise ResearchReviewError("No research runs matched the requested review filters.")

    review_id = build_research_review_id(
        filters=normalized_filters,
        entries=entries,
    )
    resolved_output_path = default_research_review_output_path(review_id) if output_path is None else Path(output_path)
    csv_path, json_path, manifest_path, promotion_gate_path = write_research_review_artifacts(
        entries=entries,
        review_id=review_id,
        filters=normalized_filters,
        output_path=resolved_output_path,
        promotion_gate_config=promotion_gate_config,
    )
    return ResearchReviewResult(
        review_id=review_id,
        filters=normalized_filters,
        entries=entries,
        csv_path=csv_path,
        json_path=json_path,
        manifest_path=manifest_path,
        promotion_gate_path=promotion_gate_path,
    )


def write_research_review_artifacts(
    *,
    entries: list[ResearchReviewEntry],
    review_id: str,
    filters: Mapping[str, Any],
    output_path: str | Path,
    promotion_gate_config: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, Path, Path | None]:
    """Persist one deterministic unified review artifact set with a manifest."""

    csv_path = resolve_research_review_csv_path(output_path)
    output_dir = csv_path.parent
    json_path = output_dir / _REVIEW_SUMMARY_FILENAME
    manifest_path = output_dir / _MANIFEST_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)

    persisted_rows = [_persisted_entry(entry) for entry in entries]
    frame = pd.DataFrame(persisted_rows, columns=_LEADERBOARD_COLUMNS)
    with csv_path.open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")

    summary_payload = _review_summary_payload(
        review_id=review_id,
        filters=filters,
        persisted_rows=persisted_rows,
    )
    json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    promotion_evaluation = evaluate_promotion_gates(
        run_type="review",
        config=promotion_gate_config,
        sources={
            "metrics": _review_metrics_payload(entries),
            "metadata": {
                "filters": canonicalize_value(dict(filters)),
                "review_id": review_id,
                "run_ids": [row["run_id"] for row in persisted_rows],
                "selected_metrics": _selected_metric_names_by_run_type(entries),
            },
            "aggregate_metrics": _aggregate_metrics_by_run_type(entries),
        },
    )
    promotion_gate_path = write_promotion_gate_artifact(output_dir, promotion_evaluation)

    manifest = _build_review_manifest(
        review_id=review_id,
        filters=filters,
        entries=entries,
        leaderboard_frame=frame,
        leaderboard_filename=csv_path.name,
        promotion_evaluation=promotion_evaluation,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path, manifest_path, promotion_gate_path


def _review_summary_payload(
    *,
    review_id: str,
    filters: Mapping[str, Any],
    persisted_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "review_id": review_id,
        "filters": canonicalize_value(dict(filters)),
        "counts_by_run_type": _counts_by_run_type_from_rows(persisted_rows),
        "entry_count": len(persisted_rows),
        "entries": persisted_rows,
        "run_ids": [row["run_id"] for row in persisted_rows],
    }
    return canonicalize_value(payload)


def build_research_review_id(
    *,
    filters: Mapping[str, Any],
    entries: Iterable[ResearchReviewEntry],
) -> str:
    """Return a deterministic identifier for one logical unified review request."""

    payload = {
        "filters": canonicalize_value(dict(filters)),
        "entries": [
            {
                "run_id": entry.run_id,
                "run_type": entry.run_type,
            }
            for entry in sorted(entries, key=lambda item: (item.run_type, item.run_id))
        ],
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"registry_review_{digest}"


def default_research_review_output_path(review_id: str) -> Path:
    """Return the default deterministic CSV path for one unified review."""

    return DEFAULT_REVIEW_ROOT / review_id / _LEADERBOARD_FILENAME


def resolve_research_review_csv_path(output_path: str | Path) -> Path:
    """Resolve the concrete CSV path used for persisted review outputs."""

    resolved_output_path = Path(output_path)
    if resolved_output_path.suffix.lower() == ".csv":
        return resolved_output_path
    return resolved_output_path / _LEADERBOARD_FILENAME


def render_research_review_table(entries: Iterable[ResearchReviewEntry]) -> str:
    """Render a compact plain-text review table for CLI output."""

    normalized_entries = list(entries)
    columns = [
        ("run_type", "type"),
        ("rank_within_type", "rank"),
        ("entity_name", "name"),
        ("selected_metric_name", "metric"),
        ("selected_metric_value", "value"),
        ("promotion_status", "promotion"),
        ("timeframe", "timeframe"),
        ("run_id", "run_id"),
    ]
    rows = [
        {
            "run_type": _display_run_type(entry.run_type),
            "rank_within_type": str(entry.rank_within_type),
            "entity_name": entry.entity_name,
            "selected_metric_name": entry.selected_metric_name,
            "selected_metric_value": _format_metric(entry.selected_metric_value),
            "promotion_status": entry.promotion_status or "-",
            "timeframe": entry.timeframe or "-",
            "run_id": entry.run_id,
        }
        for entry in normalized_entries
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


def _normalize_filters(
    *,
    run_types: Iterable[str] | None,
    timeframe: str | None,
    dataset: str | None,
    alpha_name: str | None,
    strategy_name: str | None,
    portfolio_name: str | None,
    top_k_per_type: int | None,
) -> dict[str, Any]:
    normalized_run_types = _normalize_run_types(run_types)
    if top_k_per_type is not None and top_k_per_type <= 0:
        raise ResearchReviewError("top_k_per_type must be a positive integer when provided.")
    return {
        "run_types": normalized_run_types,
        "timeframe": _normalize_optional_string(timeframe),
        "dataset": _normalize_optional_string(dataset),
        "alpha_name": _normalize_optional_string(alpha_name),
        "strategy_name": _normalize_optional_string(strategy_name),
        "portfolio_name": _normalize_optional_string(portfolio_name),
        "top_k_per_type": top_k_per_type,
    }


def _normalize_run_types(run_types: Iterable[str] | None) -> list[str]:
    if run_types is None:
        return list(SUPPORTED_REVIEW_RUN_TYPES)
    normalized: list[str] = []
    for run_type in run_types:
        if not isinstance(run_type, str):
            raise ResearchReviewError("run_types must contain only strings.")
        text = run_type.strip().lower()
        if text not in SUPPORTED_REVIEW_RUN_TYPES:
            formatted = ", ".join(SUPPORTED_REVIEW_RUN_TYPES)
            raise ResearchReviewError(f"run_types must contain only supported values: {formatted}.")
        if text not in normalized:
            normalized.append(text)
    if not normalized:
        raise ResearchReviewError("run_types must contain at least one supported run type.")
    return normalized


def _collect_review_entries(
    *,
    filters: Mapping[str, Any],
    strategy_artifacts_root: Path,
    portfolio_artifacts_root: Path,
    alpha_artifacts_root: Path,
) -> list[ResearchReviewEntry]:
    entries: list[ResearchReviewEntry] = []
    requested_run_types = set(filters["run_types"])
    if ALPHA_REVIEW_RUN_TYPE in requested_run_types:
        entries.extend(_alpha_review_entries(alpha_artifacts_root=alpha_artifacts_root, filters=filters))
    if STRATEGY_RUN_TYPE in requested_run_types:
        entries.extend(_strategy_review_entries(strategy_artifacts_root=strategy_artifacts_root, filters=filters))
    if PORTFOLIO_RUN_TYPE in requested_run_types:
        entries.extend(_portfolio_review_entries(portfolio_artifacts_root=portfolio_artifacts_root, filters=filters))
    return sorted(
        entries,
        key=lambda item: (_run_type_sort_key(item.run_type), item.rank_within_type, item.entity_name, item.run_id),
    )


def _alpha_review_entries(*, alpha_artifacts_root: Path, filters: Mapping[str, Any]) -> list[ResearchReviewEntry]:
    rows = load_alpha_evaluation_registry(alpha_artifacts_root)
    filtered = [
        row for row in rows
        if _matches_optional_string(row.get("timeframe"), filters["timeframe"])
        and _matches_optional_string(row.get("dataset"), filters["dataset"])
        and _matches_optional_string(row.get("alpha_name"), filters["alpha_name"])
    ]
    selected = _latest_by_name(filtered, name_field="alpha_name")
    ranked = sorted(
        selected,
        key=lambda row: _alpha_sort_key(row, metric_name=DEFAULT_ALPHA_REVIEW_METRIC),
    )
    return [
        ResearchReviewEntry(
            run_type=ALPHA_REVIEW_RUN_TYPE,
            rank_within_type=index,
            entity_name=str(row["alpha_name"]),
            run_id=str(row["run_id"]),
            selected_metric_name=DEFAULT_ALPHA_REVIEW_METRIC,
            selected_metric_value=_coerce_metric(_metrics_summary(row).get(DEFAULT_ALPHA_REVIEW_METRIC)),
            secondary_metric_name="mean_ic",
            secondary_metric_value=_coerce_metric(_metrics_summary(row).get("mean_ic")),
            timeframe=_coerce_optional_string(row.get("timeframe")),
            evaluation_mode=ALPHA_REVIEW_RUN_TYPE,
            promotion_status=_promotion_status(row),
            passed_gate_count=_promotion_gate_count(row, key="passed_gate_count"),
            gate_count=_promotion_gate_count(row, key="gate_count"),
            artifact_path=str(row["artifact_path"]),
        )
        for index, row in enumerate(_limit_rows(ranked, filters["top_k_per_type"]), start=1)
    ]


def _strategy_review_entries(*, strategy_artifacts_root: Path, filters: Mapping[str, Any]) -> list[ResearchReviewEntry]:
    rows = filter_by_run_type(load_registry(default_registry_path(strategy_artifacts_root)), STRATEGY_RUN_TYPE)
    filtered = [
        row for row in rows
        if _matches_optional_string(row.get("timeframe"), filters["timeframe"])
        and _matches_optional_string(row.get("dataset"), filters["dataset"])
        and _matches_optional_string(row.get("strategy_name"), filters["strategy_name"])
    ]
    selected = _latest_by_name(filtered, name_field="strategy_name")
    ranked = sorted(
        selected,
        key=lambda row: _strategy_sort_key(row, metric_name=DEFAULT_STRATEGY_REVIEW_METRIC),
    )
    return [
        ResearchReviewEntry(
            run_type=STRATEGY_RUN_TYPE,
            rank_within_type=index,
            entity_name=str(row["strategy_name"]),
            run_id=str(row["run_id"]),
            selected_metric_name=DEFAULT_STRATEGY_REVIEW_METRIC,
            selected_metric_value=_coerce_metric(_metrics_summary(row).get(DEFAULT_STRATEGY_REVIEW_METRIC)),
            secondary_metric_name="total_return",
            secondary_metric_value=_coerce_metric(_metrics_summary(row).get("total_return")),
            timeframe=_coerce_optional_string(row.get("timeframe")),
            evaluation_mode=_coerce_optional_string(row.get("evaluation_mode")),
            promotion_status=_promotion_status(row),
            passed_gate_count=_promotion_gate_count(row, key="passed_gate_count"),
            gate_count=_promotion_gate_count(row, key="gate_count"),
            artifact_path=str(row["artifact_path"]),
        )
        for index, row in enumerate(_limit_rows(ranked, filters["top_k_per_type"]), start=1)
    ]


def _portfolio_review_entries(*, portfolio_artifacts_root: Path, filters: Mapping[str, Any]) -> list[ResearchReviewEntry]:
    rows = filter_by_run_type(load_registry(default_registry_path(portfolio_artifacts_root)), PORTFOLIO_RUN_TYPE)
    filtered = [
        row for row in rows
        if _matches_optional_string(row.get("timeframe"), filters["timeframe"])
        and _matches_optional_string(row.get("portfolio_name"), filters["portfolio_name"])
    ]
    selected = _latest_by_name(filtered, name_field="portfolio_name")
    ranked = sorted(
        selected,
        key=lambda row: _portfolio_sort_key(row, metric_name=DEFAULT_PORTFOLIO_REVIEW_METRIC),
    )
    return [
        ResearchReviewEntry(
            run_type=PORTFOLIO_RUN_TYPE,
            rank_within_type=index,
            entity_name=str(row["portfolio_name"]),
            run_id=str(row["run_id"]),
            selected_metric_name=DEFAULT_PORTFOLIO_REVIEW_METRIC,
            selected_metric_value=_coerce_metric(_metrics_summary(row).get(DEFAULT_PORTFOLIO_REVIEW_METRIC)),
            secondary_metric_name="total_return",
            secondary_metric_value=_coerce_metric(_metrics_summary(row).get("total_return")),
            timeframe=_coerce_optional_string(row.get("timeframe")),
            evaluation_mode="walk_forward" if row.get("split_count") else "single",
            promotion_status=_promotion_status(row),
            passed_gate_count=_promotion_gate_count(row, key="passed_gate_count"),
            gate_count=_promotion_gate_count(row, key="gate_count"),
            artifact_path=str(row["artifact_path"]),
        )
        for index, row in enumerate(_limit_rows(ranked, filters["top_k_per_type"]), start=1)
    ]


def _latest_by_name(rows: Iterable[Mapping[str, Any]], *, name_field: str) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get(name_field)
        if not isinstance(name, str) or not name.strip():
            continue
        normalized_name = name.strip()
        candidate = dict(row)
        existing = latest.get(normalized_name)
        if existing is None or _timestamp_sort_value(candidate) > _timestamp_sort_value(existing):
            latest[normalized_name] = candidate
    return list(latest.values())


def _timestamp_sort_value(row: Mapping[str, Any]) -> str:
    value = row.get("timestamp")
    if not isinstance(value, str):
        return ""
    return value


def _metrics_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("metrics_summary")
    return payload if isinstance(payload, dict) else {}


def _strategy_sort_key(row: Mapping[str, Any], *, metric_name: str) -> tuple[bool, float, bool, float, str, str]:
    metrics = _metrics_summary(row)
    metric_value = _coerce_metric(metrics.get(metric_name))
    secondary_value = _coerce_metric(metrics.get("total_return"))
    return _descending_metric_sort_key(
        primary_value=metric_value,
        secondary_value=secondary_value,
        name=str(row.get("strategy_name") or ""),
        run_id=str(row.get("run_id") or ""),
    )


def _portfolio_sort_key(row: Mapping[str, Any], *, metric_name: str) -> tuple[bool, float, bool, float, str, str]:
    metrics = _metrics_summary(row)
    metric_value = _coerce_metric(metrics.get(metric_name))
    secondary_value = _coerce_metric(metrics.get("total_return"))
    return _descending_metric_sort_key(
        primary_value=metric_value,
        secondary_value=secondary_value,
        name=str(row.get("portfolio_name") or ""),
        run_id=str(row.get("run_id") or ""),
    )


def _alpha_sort_key(row: Mapping[str, Any], *, metric_name: str) -> tuple[bool, float, bool, float, str, str]:
    metrics = _metrics_summary(row)
    metric_value = _coerce_metric(metrics.get(metric_name))
    secondary_value = _coerce_metric(metrics.get("mean_ic"))
    return _descending_metric_sort_key(
        primary_value=metric_value,
        secondary_value=secondary_value,
        name=str(row.get("alpha_name") or ""),
        run_id=str(row.get("run_id") or ""),
    )


def _descending_metric_sort_key(
    *,
    primary_value: float | None,
    secondary_value: float | None,
    name: str,
    run_id: str,
) -> tuple[bool, float, bool, float, str, str]:
    return (
        primary_value is None,
        0.0 if primary_value is None else -primary_value,
        secondary_value is None,
        0.0 if secondary_value is None else -secondary_value,
        name,
        run_id,
    )


def _limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return rows[:limit] if limit is not None else rows


def _persisted_entry(entry: ResearchReviewEntry) -> dict[str, Any]:
    return asdict(entry)


def _promotion_status(row: Mapping[str, Any]) -> str | None:
    resolved_review = resolve_review_status(row)
    if resolved_review is not None:
        return resolved_review
    direct = row.get("promotion_status")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    summary = row.get("promotion_gate_summary")
    if isinstance(summary, dict):
        value = summary.get("promotion_status")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _promotion_gate_count(row: Mapping[str, Any], *, key: str) -> int | None:
    summary = row.get("promotion_gate_summary")
    if not isinstance(summary, dict):
        review_metadata = row.get("review_metadata")
        if isinstance(review_metadata, dict):
            nested_summary = review_metadata.get("promotion_gate_summary")
            if isinstance(nested_summary, dict):
                summary = nested_summary
    if not isinstance(summary, dict):
        return None
    value = summary.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _counts_by_run_type(entries: Iterable[ResearchReviewEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.run_type] = counts.get(entry.run_type, 0) + 1
    return counts


def _counts_by_run_type_from_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        run_type = row.get("run_type")
        if isinstance(run_type, str):
            counts[run_type] = counts.get(run_type, 0) + 1
    return counts


def _selected_metric_names_by_run_type(entries: Iterable[ResearchReviewEntry]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for entry in entries:
        selected.setdefault(entry.run_type, entry.selected_metric_name)
    return dict(sorted(selected.items()))


def _aggregate_metrics_by_run_type(entries: Iterable[ResearchReviewEntry]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ResearchReviewEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.run_type, []).append(entry)

    payload: dict[str, dict[str, Any]] = {}
    for run_type, run_entries in sorted(grouped.items()):
        selected_values = [entry.selected_metric_value for entry in run_entries if entry.selected_metric_value is not None]
        secondary_values = [entry.secondary_metric_value for entry in run_entries if entry.secondary_metric_value is not None]
        payload[run_type] = canonicalize_value(
            {
                "entry_count": len(run_entries),
                "selected_metric_name": run_entries[0].selected_metric_name,
                "secondary_metric_name": run_entries[0].secondary_metric_name,
                "best_selected_metric_value": max(selected_values) if selected_values else None,
                "best_secondary_metric_value": max(secondary_values) if secondary_values else None,
            }
        )
    return payload


def _review_metrics_payload(entries: Iterable[ResearchReviewEntry]) -> dict[str, Any]:
    normalized_entries = list(entries)
    status_counts: dict[str, int] = {}
    for entry in normalized_entries:
        if entry.promotion_status is None:
            continue
        normalized_status = entry.promotion_status.strip().lower()
        if normalized_status:
            status_counts[normalized_status] = status_counts.get(normalized_status, 0) + 1

    counts_by_run_type = _counts_by_run_type(normalized_entries)
    payload = {
        "entry_count": len(normalized_entries),
        "alpha_entry_count": counts_by_run_type.get(ALPHA_REVIEW_RUN_TYPE, 0),
        "strategy_entry_count": counts_by_run_type.get(STRATEGY_RUN_TYPE, 0),
        "portfolio_entry_count": counts_by_run_type.get(PORTFOLIO_RUN_TYPE, 0),
        "eligible_entry_count": status_counts.get("eligible", 0),
        "blocked_entry_count": status_counts.get("blocked", 0),
        "promoted_entry_count": status_counts.get("promoted", 0),
        "reviewed_entry_count": sum(status_counts.values()),
    }
    return canonicalize_value(payload)


def _build_review_manifest(
    *,
    review_id: str,
    filters: Mapping[str, Any],
    entries: list[ResearchReviewEntry],
    leaderboard_frame: pd.DataFrame,
    leaderboard_filename: str,
    promotion_evaluation: Any,
) -> dict[str, Any]:
    artifact_inventory = {
        leaderboard_filename: {
            "path": leaderboard_filename,
            "rows": int(len(leaderboard_frame)),
            "columns": leaderboard_frame.columns.tolist(),
        },
        _MANIFEST_FILENAME: {"path": _MANIFEST_FILENAME},
        _REVIEW_SUMMARY_FILENAME: {
            "path": _REVIEW_SUMMARY_FILENAME,
            "rows": len(entries),
        },
        **(
            {DEFAULT_PROMOTION_ARTIFACT_FILENAME: {"path": DEFAULT_PROMOTION_ARTIFACT_FILENAME}}
            if promotion_evaluation is not None
            else {}
        ),
    }
    review_group = sorted(
        [
            leaderboard_filename,
            _MANIFEST_FILENAME,
            _REVIEW_SUMMARY_FILENAME,
            *([DEFAULT_PROMOTION_ARTIFACT_FILENAME] if promotion_evaluation is not None else []),
        ]
    )
    payload = {
        "artifact_files": sorted(artifact_inventory),
        "artifacts": artifact_inventory,
        "artifact_groups": {
            "core": review_group,
            "leaderboard": [leaderboard_filename],
            "qa": (
                [DEFAULT_PROMOTION_ARTIFACT_FILENAME]
                if promotion_evaluation is not None
                else []
            ),
            "review": review_group,
            "summary": [_REVIEW_SUMMARY_FILENAME],
        },
        "counts_by_run_type": _counts_by_run_type(entries),
        "evaluation_mode": "registry_review",
        "files_written": len(artifact_inventory),
        "leaderboard_path": leaderboard_filename,
        "promotion_gate_summary": None if promotion_evaluation is None else promotion_evaluation.summary(),
        "review_filters": canonicalize_value(dict(filters)),
        "review_id": review_id,
        "review_metrics": _review_metrics_payload(entries),
        "review_summary_path": _REVIEW_SUMMARY_FILENAME,
        "row_count": len(entries),
        "selected_metrics": _selected_metric_names_by_run_type(entries),
        "timestamp": _utc_timestamp_from_review_id(review_id),
    }
    return canonicalize_value(payload)


def _run_type_sort_key(run_type: str) -> int:
    order = {
        ALPHA_REVIEW_RUN_TYPE: 0,
        STRATEGY_RUN_TYPE: 1,
        PORTFOLIO_RUN_TYPE: 2,
    }
    return order.get(run_type, 99)


def _display_run_type(run_type: str) -> str:
    if run_type == ALPHA_REVIEW_RUN_TYPE:
        return "alpha"
    return run_type


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_string(value: Any) -> str | None:
    return _normalize_optional_string(value)


def _matches_optional_string(value: Any, expected: str | None) -> bool:
    if expected is None:
        return True
    return _normalize_optional_string(value) == expected


def _coerce_metric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    return None


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def _utc_timestamp_from_review_id(review_id: str) -> str:
    digest = hashlib.sha256(review_id.encode("utf-8")).hexdigest()
    year = 2000 + int(digest[0:2], 16) % 25
    month = (int(digest[2:4], 16) % 12) + 1
    day = (int(digest[4:6], 16) % 28) + 1
    hour = int(digest[6:8], 16) % 24
    minute = int(digest[8:10], 16) % 60
    second = int(digest[10:12], 16) % 60
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"
